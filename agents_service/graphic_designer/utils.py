import requests
import json
import re
import time
import functools
import logging
import os
import threading
import contextvars
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import base64

# Cooperative cancellation. Imported (rather than defined here) so the registry
# has no dependency on this module and can be used from main.py's websocket
# layer without dragging the whole utils import graph in.
from .cancellation import RunCancelled, checkCancelled, isCancelled

# --- Profile logger setup ---
_profile_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(_profile_log_dir, exist_ok=True)
_profile_log_path = os.path.join(_profile_log_dir, 'profile.log')

_profile_logger = logging.getLogger('graphic_designer.profile')
_profile_logger.setLevel(logging.INFO)
_profile_logger.propagate = False
_profile_handler = logging.FileHandler(_profile_log_path, mode='a')
_profile_handler.setFormatter(logging.Formatter('%(message)s'))
_profile_logger.addHandler(_profile_handler)

# --- API call logger setup ---
# Every provider call in the product funnels through apiLog, so a single file
# answers "what did we actually call, on which model, and what did it cost?".
# Without it the only evidence of a fallback firing was a print line, and
# nothing at all recorded image-model calls — the expensive ones.
_api_log_path = os.path.join(_profile_log_dir, 'api_calls.log')

_api_logger = logging.getLogger('graphic_designer.api')
_api_logger.setLevel(logging.INFO)
_api_logger.propagate = False
_api_handler = logging.FileHandler(_api_log_path, mode='a')
_api_handler.setFormatter(logging.Formatter('%(message)s'))
_api_logger.addHandler(_api_handler)

# Per-process running total, so a runaway loop is visible as a climbing number
# in the log rather than only as a surprise on the provider's dashboard.
_api_totals = threading.local()

# --- Current graph node, for attribution ---
# Set by @profile (which wraps every node and knows its name) and read by
# apiLog, so each provider call says which node issued it. A ContextVar rather
# than a thread-local because it can be copied into worker threads explicitly
# (see the ThreadPoolExecutor call sites), which a thread-local cannot.
_current_node = contextvars.ContextVar('kognie_current_node', default='-')


def currentNode() -> str:
    """Name of the graph node currently executing, or '-' outside one."""
    try:
        return _current_node.get()
    except LookupError:
        return '-'


def setCurrentNode(name: str):
    """Set the executing node; returns the token to reset with."""
    return _current_node.set(name)


def resetCurrentNode(token) -> None:
    try:
        _current_node.reset(token)
    except Exception:
        pass

# --- Spend circuit breaker ---
# Two budgets over the same rolling window, because they catch different faults:
#
#   per-session  — a runaway loop inside ONE user's run. This is the limit that
#                  should be tight: a single generation is ~15 calls, so ~50 is
#                  already three runs' worth.
#   global       — the whole process. Must scale with concurrency, or the
#                  breaker becomes a concurrency ceiling: 100 users × ~15 calls
#                  is ~1,500 calls inside the window, which the old single
#                  process-wide 50-call limit would have killed instantly.
#
# Both are deliberately NOT thread-local: LangGraph runs nodes on a worker pool,
# so a thread-local budget resets between nodes and would never see a runaway.
SPEND_WINDOW_S = float(os.getenv("KOGNIE_SPEND_WINDOW_S", 300))
SPEND_LIMIT_USD = float(os.getenv("KOGNIE_SPEND_LIMIT_USD", 1.0))
SPEND_LIMIT_CALLS = int(os.getenv("KOGNIE_SPEND_LIMIT_CALLS", 50))
# Global ceiling: a backstop against "every session is looping at once", not a
# per-user budget. Defaults to 200x the per-session allowance so it never binds
# before the per-session limit does at realistic concurrency.
GLOBAL_SPEND_LIMIT_USD = float(os.getenv("KOGNIE_GLOBAL_SPEND_LIMIT_USD",
                                         SPEND_LIMIT_USD * 200))
GLOBAL_SPEND_LIMIT_CALLS = int(os.getenv("KOGNIE_GLOBAL_SPEND_LIMIT_CALLS",
                                         SPEND_LIMIT_CALLS * 200))
SPEND_BREAKER_ON = os.getenv("KOGNIE_SPEND_BREAKER", "on").lower() != "off"

_spend_lock = threading.Lock()
_spend_events = []          # [(timestamp, usd)] within the rolling window, all sessions
_spend_by_session = {}      # session key -> [(timestamp, usd)] within the window

# Which session the calls on this thread belong to. A ContextVar rather than a
# thread-local for the same reason as _current_node: it can be copied into
# worker threads explicitly, which is how the critic lenses and the suggestion
# executors keep their attribution.
_current_session = contextvars.ContextVar('kognie_spend_session', default='-')


def setSpendSession(session_key: str):
    """Attribute subsequent provider calls on this context to ``session_key``.
    Returns the token to reset with (see :func:`resetSpendSession`)."""
    return _current_session.set(session_key or '-')


def resetSpendSession(token) -> None:
    try:
        _current_session.reset(token)
    except Exception:
        pass


def currentSpendSession() -> str:
    try:
        return _current_session.get()
    except LookupError:
        return '-'


# --- "did this turn actually touch pixels?" marker ---
# Drives the `show_cost` boolean stamped on every interrupt (see
# custom_interrupt): the frontend renders a charge only when the money being
# billed bought real image work, not a chit-chat reply or a classification.
#
# Keyed by spend session rather than held in a thread-local because the node
# that PAYS (an image executor) and the node that INTERRUPTS are frequently
# different nodes, and LangGraph is free to run them on different worker
# threads. The session key is a ContextVar, so it follows the run into those
# threads the way `_current_node` already does.
_image_work_lock = threading.Lock()
_image_work_by_session = {}      # session key -> timestamp of the last image call
_IMAGE_WORK_TTL_S = 3600
"""Marks older than this are dropped. A run bills its image work seconds after
doing it, so anything this old belongs to a session that went away without a
billing interrupt (a drop mid-generation, a thread id that changed under the
same socket) — evicting on write keeps the dict from growing for the life of
the process, the way ``_spend_by_session`` does."""


def markImageWork() -> None:
    """Record that real image work happened on this session since the last
    cost-bearing interrupt.

    Called automatically for every image API call that goes through
    :func:`apiLog`, and directly by the pixel pipelines that don't (layering's
    detection and segmentation stages)."""
    now = time.time()
    with _image_work_lock:
        _image_work_by_session[currentSpendSession()] = now
        cutoff = now - _IMAGE_WORK_TTL_S
        for key in [k for k, ts in _image_work_by_session.items() if ts < cutoff]:
            _image_work_by_session.pop(key, None)


def imageWorkPending() -> bool:
    """Whether image work has happened since the last cost-bearing interrupt."""
    with _image_work_lock:
        marked = _image_work_by_session.get(currentSpendSession())
    return marked is not None and (time.time() - marked) < _IMAGE_WORK_TTL_S


def clearImageWork() -> None:
    """Forget the marker — called once the interrupt that bills the image work
    has been emitted, mirroring the ``cost`` reset in the same place."""
    with _image_work_lock:
        _image_work_by_session.pop(currentSpendSession(), None)


def isImageApiCall(model: str, purpose: str) -> bool:
    """Whether one :func:`apiLog` entry describes real image generation/editing.

    Purpose is checked first and is the authoritative half: ``gptExpandImage``
    bills its outpaint through a text model id, so a model-only test would
    read it as a chat call.
    """
    from .config import IMAGE_CALL_MODELS, IMAGE_CALL_PURPOSES

    if (purpose or '') in IMAGE_CALL_PURPOSES:
        return True
    name = (model or '').split('/')[-1]
    return name in IMAGE_CALL_MODELS


class SpendLimitExceeded(BaseException):
    """Raised when provider spend in the rolling window crosses the limit.

    Inherits :class:`BaseException`, **not** ``Exception``, on purpose. Every
    image executor wraps its call in a bare ``except Exception`` and treats
    anything it catches as "this model failed, try the next one" — so a breaker
    built on ``Exception`` gets swallowed and *feeds* the retry loop it exists to
    stop. It tripped 6,697 times in one run and never halted anything.

    Sitting beside ``KeyboardInterrupt`` in the hierarchy means it passes
    straight through those handlers and actually stops the run.
    """


def _evict(events, cutoff):
    while events and events[0][0] < cutoff:
        events.pop(0)


def _checkSpendBreaker(cost, status: str = 'ok') -> None:
    if not SPEND_BREAKER_ON:
        return
    # 'start' is an announcement, not a call: the matching ok/failed event is
    # what gets counted. Counting both would halve the effective call ceiling
    # and inflate the totals (the 6,690-call figure was start events double-counting).
    if status == 'start':
        return
    now = time.time()
    cutoff = now - SPEND_WINDOW_S
    session = currentSpendSession()

    with _spend_lock:
        _spend_events.append((now, cost or 0.0))
        _evict(_spend_events, cutoff)
        g_spent, g_calls = sum(c for _, c in _spend_events), len(_spend_events)

        own = _spend_by_session.setdefault(session, [])
        own.append((now, cost or 0.0))
        _evict(own, cutoff)
        s_spent, s_calls = sum(c for _, c in own), len(own)

        # Sessions go quiet and never come back; without this the dict is a slow
        # leak over the life of the process.
        for key in [k for k, v in _spend_by_session.items() if not v and k != session]:
            _spend_by_session.pop(key, None)

    scope = None
    if s_spent > SPEND_LIMIT_USD or s_calls > SPEND_LIMIT_CALLS:
        scope, spent, calls = f"session {session}", s_spent, s_calls
        lim_usd, lim_calls = SPEND_LIMIT_USD, SPEND_LIMIT_CALLS
    elif g_spent > GLOBAL_SPEND_LIMIT_USD or g_calls > GLOBAL_SPEND_LIMIT_CALLS:
        scope, spent, calls = "process", g_spent, g_calls
        lim_usd, lim_calls = GLOBAL_SPEND_LIMIT_USD, GLOBAL_SPEND_LIMIT_CALLS

    if scope:
        msg = (f"spend breaker tripped [{scope}]: ${spent:.2f} over {calls} calls in the last "
               f"{int(SPEND_WINDOW_S)}s (limits: ${lim_usd:.2f} / "
               f"{lim_calls} calls). Set KOGNIE_SPEND_BREAKER=off to disable.")
        _api_logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  BREAKER  |  {msg}")
        print(f"[api][BREAKER] {msg}")
        raise SpendLimitExceeded(msg)


def spendWindowTotals(session: str = None):
    """(usd, calls) currently inside the rolling window — for health endpoints.

    Process-wide by default; pass a session key for that session's slice.
    """
    now = time.time()
    with _spend_lock:
        events = _spend_by_session.get(session, []) if session else _spend_events
        live = [(t, c) for t, c in events if t >= now - SPEND_WINDOW_S]
    return round(sum(c for _, c in live), 4), len(live)


def resetSpendWindow() -> None:
    """Drop every recorded event. For tests and harness runs only — production
    never calls this, or the breaker would have nothing to measure."""
    with _spend_lock:
        _spend_events.clear()
        _spend_by_session.clear()


def providerCost(model: str, tokens=None, tokens_in=None, tokens_out=None):
    """What the provider will charge for one call, or None if the model has no
    rate on file.

    Priced from :data:`config.ACTUAL_RATES_PER_1K` rather than ``GEN_PRICES``,
    and deliberately does **not** touch the ``@profile`` cost accumulator the
    way :func:`calculateCost` does — this is observability, and must not move
    a number the customer gets billed.

    Returns None (rather than 0.0) for an unknown model, so the log can say
    "unpriced" instead of implying the call was free.
    """
    from .config import ACTUAL_RATES_PER_1K, FLAT_FEE_MODELS

    name = (model or '').split('/')[-1]          # 'models/gemini-3.1-flash-lite' -> id

    rates = ACTUAL_RATES_PER_1K.get(name)
    if not rates:
        return None
    rate_in, rate_out = rates
    if tokens_in is not None or tokens_out is not None:
        return round(((tokens_in or 0) * rate_in + (tokens_out or 0) * rate_out) / 1_000, 6)
    if tokens:
        # Only a total is available: price it at the input rate and mark it, so
        # the figure is a floor rather than a fabricated split -- except for
        # the flat-fee image models, whose (input, output) rate is the SAME
        # number on both sides (see ACTUAL_RATES_PER_1K), so this is exact for
        # them rather than a floor.
        return round(tokens * rate_in / 1_000, 6)
    # No usage reported at all -- the flat-fee image models (Flux, Stability's
    # upscaler, gpt-expand) are the only callers this happens for today, and
    # each of their table rows is defined as "1 call == 1,000 token-equivalents"
    # (see ACTUAL_RATES_PER_1K), so that's exactly right for them, not a guess.
    # A token-metered model reaching here instead means its call site doesn't
    # parse usage out of the response yet -- worth fixing at the call site,
    # not by inventing a number here.
    if name in FLAT_FEE_MODELS:
        return round(1000 * rate_in / 1_000, 6)
    return None


# --- apiLog observers ---
# apiLog is already the single funnel every provider call in the app passes
# through, so a run-scoped harness needs no instrumentation of its own: it
# registers a sink here and sees every call, including the fallback rungs and
# image models that a stage-function counter structurally cannot observe.
_api_sinks = []


def registerApiSink(fn) -> None:
    """Call ``fn(record: dict)`` for every subsequent provider call.

    The record carries the same fields the log line does. Sinks must not raise
    — a broken observer can never break a generation — and are dropped by
    :func:`clearApiSinks`.
    """
    _api_sinks.append(fn)


def clearApiSinks() -> None:
    """Remove every registered sink."""
    _api_sinks.clear()


def apiLog(provider: str, model: str, purpose: str, tokens=None, cost=None,
           status: str = 'ok', detail: str = '', tokens_in=None,
           tokens_out=None, role=None, actual_cost=None) -> None:
    """Record one provider API call.

    Called at every LLM / image-model call site. `status` is ``ok``, ``failed``
    or ``retry`` so a fallback chain shows every rung it tried, not just the one
    that answered.

    :param provider: ``google`` | ``openai`` | ``anthropic`` | ``stability`` | ``bfl``
    :param model: The exact model id sent to the provider.
    :param purpose: Call site — node or stage name.
    :param tokens: Total tokens where the provider reports them.
    :param cost: USD billed to the customer, when the caller has priced it
        (typically a :func:`calculateCost` / ``GEN_PRICES`` figure).
    :param actual_cost: USD the provider actually charges, when the caller has
        already priced it more precisely than the generic
        :func:`providerCost` split can (e.g. :func:`common_nodes.ragbotCost`,
        which applies the cache-read/write discount ``providerCost`` doesn't
        know about). Leave ``None`` to have it derived from
        ``PROVIDER_TEXT_RATES`` / ``PROVIDER_IMAGE_RATES`` here.
    :param role: The registry role this call was routed by (``classify_extract``,
        ``cot_reasoning``, ...), when it went through the model router. ``None``
        for call sites that still name a model directly.
    """
    if tokens is None and (tokens_in is not None or tokens_out is not None):
        tokens = (tokens_in or 0) + (tokens_out or 0)
    if cost is None and status == 'ok':
        cost = providerCost(model, tokens, tokens_in, tokens_out)

    # `cost` above is whatever the caller billed the customer (GEN_PRICES, or
    # a providerCost() estimate when the caller had no billed figure of its
    # own) -- callers like critiquepipeline/agents.py pass an explicit
    # calculateCost() result there, which is a *blended* price, not what the
    # provider actually charges. `actual_cost` is priced fresh every time from
    # PROVIDER_TEXT_RATES / PROVIDER_IMAGE_RATES -- the real per-token or
    # per-image rate -- independent of what `cost` was, unless the caller
    # already computed a more precise figure and passed it in directly. The
    # two are allowed to diverge from `cost`; that divergence is the reason to
    # track both.
    if actual_cost is None and status == 'ok':
        actual_cost = providerCost(model, tokens, tokens_in, tokens_out)
    elif status != 'ok':
        actual_cost = None

    # apiLog is the single funnel every provider call passes through, so it is
    # also the one place that can tell the frontend "the money about to be
    # billed bought an image". Only successful calls count: a failed rung of a
    # fallback chain costs nothing, so it must not light up the cost display.
    if status == 'ok' and isImageApiCall(model, purpose):
        markImageWork()

    running = getattr(_api_totals, 'cost', 0.0) + (cost or 0.0)
    _api_totals.cost = running
    calls = getattr(_api_totals, 'calls', 0) + 1
    _api_totals.calls = calls

    # Roll the real provider cost into whichever @profile node is currently
    # executing. This is what lets profile.log show a per-node actual_cost
    # figure without every one of the ~40 call sites needing to compute and
    # thread it through themselves -- apiLog is already the single funnel
    # every provider call passes through.
    if actual_cost is not None and getattr(_profile_local, 'active', False):
        _profile_local.actual_cost = getattr(_profile_local, 'actual_cost', 0.0) + actual_cost

    cost_str = f"${cost:.6f}" if cost is not None else "unpriced"
    actual_cost_str = f"${actual_cost:.6f}" if actual_cost is not None else "unpriced"
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    node = currentNode()
    line = (f"{ts}  |  {provider:<10} |  {model:<34} |  node={node:<28} |  {purpose:<22} |  "
            f"role={str(role or '-'):<17} |  "
            f"status={status:<7} |  tokens={str(tokens or '-'):>8} |  "
            f"cost={cost_str:>12}  |  actual_cost={actual_cost_str:>12}  |  "
            f"call#{calls} running=${running:.4f}")
    if detail:
        line += f"  |  {detail[:160]}"
    _api_logger.info(line)

    if _api_sinks:
        record = {"ts": ts, "provider": provider, "model": model, "role": role,
                  "node": node, "purpose": purpose, "status": status,
                  "tokens": tokens, "tokens_in": tokens_in, "tokens_out": tokens_out,
                  "cost": cost, "actual_cost": actual_cost, "detail": detail}
        for sink in list(_api_sinks):
            try:
                sink(record)
            except Exception:
                pass          # an observer must never break a generation
    if status == 'start':
        # Printed immediately BEFORE the request leaves, so a hung or slow call
        # is attributable to a provider and model instead of silence.
        print(f"[api →] [{node}] CALLING {provider}/{model}  ({purpose})"
              + (f"  [{detail}]" if detail else ""))
        # Cancellation chokepoint #2: every provider call in the codebase logs
        # a 'start' event immediately before the request leaves, which makes
        # this the one place that can stop a cancelled run from *paying* for
        # more work. Raised after the log line, so the trace shows the call
        # that was about to be made and then the cancel that prevented it.
        checkCancelled(f"apiLog:{provider}/{model}")
    else:
        print(f"[api] [{node}] {provider}/{model} {purpose} status={status} "
              f"tokens={tokens or '-'} cost={cost_str} running=${running:.4f}")

    # Last, so the call that trips the breaker is still recorded before raising.
    _checkSpendBreaker(cost, status)


def _callerName(depth: int = 3) -> str:
    """Name of the function that triggered the call, for the `purpose` column.
    Cheap frame walk — only runs once per API call, which is already network-bound."""
    import sys
    skip = {'wrapper', 'apiLog', '_callerName', '<lambda>', '<module>'}
    try:
        for up in range(depth, depth + 8):
            name = sys._getframe(up).f_code.co_name
            if name not in skip:
                return name
    except Exception:
        pass
    return 'unknown'


def resetApiTotals() -> None:
    """Zero the per-thread running total — call at the start of a request."""
    _api_totals.cost = 0.0
    _api_totals.calls = 0


# Thread-local accumulator for cost tracking inside @profile
_profile_local = threading.local()

# Patch langgraph.types.interrupt to capture interrupt calls and their costs
import sys
import langgraph.types

_orig_interrupt = langgraph.types.interrupt


def stampShowCost(value: dict) -> dict:
    """Stamp the frontend's ``show_cost`` boolean onto an outgoing interrupt.

    ``show_cost`` answers one question for the UI: should this frame display a
    charge? Two independent ways to earn a True:

    1. The interrupt *type* is a delivered image result
       (:data:`config.IMAGE_RESULT_INTERRUPTS` — ``layering_executed``,
       ``suggestions_executed``, ``generation_completed``,
       ``editing_successful``, ``expand_successful``). These are unconditional:
       the node emitting one has just handed the user a new or changed image,
       so the charge always belongs on screen. No marker lookup, no cost test —
       which also means a pixel path nobody remembered to wire into
       :func:`markImageWork` still reports correctly.
    2. Any other frame that carries money which bought real image work, per the
       :func:`markImageWork` marker.

    Text-only turns (chit-chat, FAQs, clarifying questions, classification)
    come back False even on the rare turn where their deferred pennies cross
    the one-cent billing threshold, because there is nothing the user would
    recognise as a paid result to attach the number to. So does
    ``execution_failed``: every executor fell over, nothing was delivered.

    The marker is cleared only when the cost is genuinely emitted, so it rides
    forward with deferred spend exactly the way ``state['cost']`` does: image
    work whose cost is passed on to a later interrupt lights up *that*
    interrupt instead.
    """
    from .config import IMAGE_RESULT_INTERRUPTS

    try:
        cost = float(value.get('cost') or 0.0)
    except (TypeError, ValueError):
        cost = 0.0

    billed = cost > 0
    is_result_frame = value.get('type') in IMAGE_RESULT_INTERRUPTS
    value['show_cost'] = bool(is_result_frame or (billed and imageWorkPending()))
    if billed:
        clearImageWork()
    return value


def custom_interrupt(value):
    if getattr(_profile_local, 'active', False):
        _profile_local.is_interrupt = True
        if isinstance(value, dict) and 'cost' in value:
            _profile_local.interrupt_cost = value['cost']
    if isinstance(value, dict):
        stampShowCost(value)
    return _orig_interrupt(value)

langgraph.types.interrupt = custom_interrupt

# Update any modules that have already imported 'interrupt'
for mod_name, mod in list(sys.modules.items()):
    if hasattr(mod, 'interrupt') and mod.interrupt is _orig_interrupt:
        setattr(mod, 'interrupt', custom_interrupt)

# Function for making API requests with timeout and retry logic

def makeApiRequest(method, url, **kwargs):
    """Make an HTTP request with automatic retry and exponential backoff.

    Wraps :func:`requests.get` / :func:`requests.post` with configurable
    timeout and up to 3 retries on timeout errors.  Non-timeout request
    failures are raised immediately.

    :param method: HTTP method — ``'GET'`` or ``'POST'``.
    :type method: str
    :param url: The target URL.
    :type url: str
    :param kwargs: Additional keyword arguments forwarded to
        :func:`requests.post` or :func:`requests.get`.  A ``timeout`` key
        (default ``(5, 30)``) is consumed and not forwarded.
    :returns: The :class:`requests.Response` object on success.
    :rtype: requests.Response
    :raises requests.exceptions.Timeout: If all retry attempts are
        exhausted.
    :raises requests.exceptions.RequestException: For non-timeout request
        failures.
    """
    max_retries = 3
    base_timeout = kwargs.pop('timeout', (5, 30))
    
    for attempt in range(max_retries):
        # A cancel during the exponential backoff below must not be followed by
        # another attempt.
        checkCancelled('makeApiRequest')
        try:
            if method == 'POST':
                response = requests.post(url, timeout=base_timeout, **kwargs)
            else:
                response = requests.get(url, timeout=base_timeout, **kwargs)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except requests.exceptions.RequestException as e:
            raise
    
    return None


# Function to robustly parse JSON from a string response given my LLM

def parseJson(response: str) -> dict:
    """Robustly parse a JSON object or array from a (possibly noisy) string.

    Attempts multiple parsing strategies in order:

    1. Direct :func:`json.loads` on the trimmed response.
    2. Regex extraction of the first ``{...}`` or ``[...]`` block.
    3. Trailing-comma cleanup followed by a second parse attempt.

    :param response: The raw string potentially containing JSON, such as
        an LLM response that may include markdown fences or commentary.
    :type response: str
    :returns: The parsed JSON as a Python ``dict`` or ``list``.
    :rtype: dict | list
    :raises ValueError: If no valid JSON can be extracted from the input.
    """

    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    
    if not match:
        raise ValueError("No JSON object or array found in response")
    
    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            cleaned = re.sub(r',\s*([\]}])', r'\1', json_str)
            # strict=False tolerates raw newlines inside strings (e.g. multi-line bulleted prompts)
            return json.loads(cleaned, strict=False)
        except:
            raise ValueError(f"Failed to parse identified JSON string: {e}")


_SELECTION_REFERENCE_RE = re.compile(
    r"(?:the\s+|these\s+|those\s+|my\s+|this\s+)?"
    r"(?:marked|selected|highlighted|chosen|drawn|boxed|circled)?\s*"
    r"(?:region|regions|area|areas|box|boxes|bbox|bboxes|part|parts|"
    r"selection|selections|section|sections|zone|zones)"
    r"(?:\s+i\s+(?:drew|marked|selected))?"
)
"""Phrases that name WHERE the user pointed, not WHAT they pointed at.

A canvas Isolate/Erase click sends the regions as coordinates and a prompt like
"isolate the marked regions". The layering classifier read that phrase as an
object name (measured: ``requested_objects: ['marked regions']``), which both
skipped the ask the drawn boxes used to arrive through AND scoped Gemini to
hunt the whole image for a thing called "marked regions" -- returning one
unrelated object and none of the marked ones.
"""


def requestedLayerTargets(response_data) -> tuple:
    """Read what the layering classifier says the user pointed at.

    Objects and texts come back as separate lists so that a headline's wording
    is never handed to Gemini as an object name to ground. Both the entry gate
    and the subgraph's classifier read the same reply shape, so they share this
    rather than each doing their own coercion and drifting apart.

    Phrases that name the user's SELECTION rather than a thing ("the marked
    regions") are dropped from the object list -- see the note at
    :data:`_SELECTION_REFERENCE_RE`.

    :returns: ``(requested_objects, requested_texts, detect_all)``
    """
    if not isinstance(response_data, dict):
        return [], [], False

    def _clean(key):
        return [str(v).strip() for v in (response_data.get(key) or [])
                if str(v).strip()]

    objects, texts = _clean("requested_objects"), _clean("requested_texts")
    # A name in both lists means the model could not decide what it was. The
    # text list wins: OCR either read those words or it did not, so a wrong
    # guess there costs nothing, while the same wrong guess in the object list
    # sends Gemini hunting for a physical object that was never in the image.
    if texts:
        lowered = {t.lower() for t in texts}
        objects = [o for o in objects if o.lower() not in lowered]
    # Deterministic backstop for selection-reference phrases. "isolate the
    # marked regions" measurably came back as requested_objects:
    # ['marked regions'], which reads as a named object, skips the "which
    # objects?" ask the drawn boxes arrive through, and sends Gemini looking
    # for a thing called "marked regions" across the whole image. The prompt
    # now rules these out too; this is the half that does not depend on the
    # model complying.
    objects = [o for o in objects if not _SELECTION_REFERENCE_RE.fullmatch(o.strip().lower())]
    return objects, texts, bool(response_data.get("detect_all"))


def calculateCost(model: str, total_tokens: int) -> float:
    """Calculate the monetary cost of a model invocation based on token usage.

    Looks up the per-1000-token price from :data:`config.GEN_PRICES` and
    computes the total cost rounded to 6 decimal places.

    :param model: The model identifier (e.g. ``'gpt-5-mini'``,
        ``'gemini-2.5-flash'``).
    :type model: str
    :param total_tokens: The total number of tokens consumed.
    :type total_tokens: int
    :returns: The computed cost in USD.
    :rtype: float
    """
    from .config import GEN_PRICES

    model_cost_per_thousand = GEN_PRICES.get(model)

    if model_cost_per_thousand is None:
        # Used to be a TypeError on the None multiply, which took down the whole
        # node -- an unpriced model must never be able to fail a generation. It
        # is a billing gap, so bill nothing and make the gap loud: routing can
        # now move a call to a model nobody added to GEN_PRICES.
        _api_logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  UNPRICED  |  "
                         f"{model} has no GEN_PRICES entry; billed $0 for {total_tokens} tokens")
        print(f"[api][UNPRICED] no GEN_PRICES entry for {model!r} — billed $0. Add one.")
        return 0.0

    total_cost = (total_tokens / 1000) * model_cost_per_thousand
    total_cost = round(total_cost, 6)

    # Accumulate into the profile tracker if active
    if getattr(_profile_local, 'active', False):
        _profile_local.cost += total_cost

    return total_cost


def deferredInterruptCost(cost: float) -> float:
    """Expose deferred non-image spend only once it reaches one cent."""
    try:
        accumulated = float(cost or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return accumulated if accumulated >= 0.01 else 0.0


def profile(func):
    """Profiling decorator: logs function name, execution time, and whether it is an interrupt to logs/profile.log."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _start = time.time()

        # Save and reset the thread-local accumulator
        _prev_active = getattr(_profile_local, 'active', False)
        _prev_cost = getattr(_profile_local, 'cost', 0.0)
        _prev_actual_cost = getattr(_profile_local, 'actual_cost', 0.0)
        _prev_is_interrupt = getattr(_profile_local, 'is_interrupt', False)
        _prev_interrupt_cost = getattr(_profile_local, 'interrupt_cost', None)

        _profile_local.active = True
        _profile_local.cost = 0.0
        # Real provider spend for this node, summed from every apiLog('ok')
        # call made while it runs (see apiLog) — unlike `cost`/`interrupt_cost`
        # below, this is populated on every node, not only ones that interrupt,
        # because "what did this node actually cost" doesn't depend on whether
        # it happened to pause for user input.
        _profile_local.actual_cost = 0.0
        _profile_local.is_interrupt = False
        _profile_local.interrupt_cost = None

        # Name every provider call this node makes. @profile already wraps each
        # node, so this is the one place that reliably knows which node is
        # running — including through the pipeline modules it calls into.
        _node_token = setCurrentNode(func.__name__)

        # Cancellation chokepoint #1: the node boundary. @profile wraps every
        # node in the main graph and in all three subgraphs, so a cancel
        # requested while the previous node ran stops the run here instead of
        # paying for the next stage. Raised BEFORE any of this node's work, so
        # nothing half-done is written to the checkpoint.
        checkCancelled(func.__name__)

        from langgraph.errors import GraphInterrupt

        try:
            result = func(*args, **kwargs)
            _elapsed = time.time() - _start
            _is_interrupt = _profile_local.is_interrupt
            _cost_val = _profile_local.interrupt_cost if _is_interrupt else None
            _actual_cost_val = _profile_local.actual_cost

            _ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _is_interrupt_str = "True" if _is_interrupt else "False"
            _cost_str = f"${_cost_val:.6f}" if _cost_val is not None else "N/A"
            _line = (f"{_ts}  |  {func.__name__:<35}  |  time={_elapsed:>8.3f}s  |  "
                     f"is_interrupt={_is_interrupt_str:<5}  |  cost={_cost_str}  |  "
                     f"actual_cost=${_actual_cost_val:.6f}")
            _profile_logger.info(_line)

            # Restore parent frame's accumulator (for nested @profile calls)
            _profile_local.active = _prev_active
            _profile_local.cost = _prev_cost + (_cost_val if _cost_val is not None else 0.0)
            _profile_local.actual_cost = _prev_actual_cost + _actual_cost_val
            _profile_local.is_interrupt = _prev_is_interrupt or _is_interrupt
            _profile_local.interrupt_cost = _prev_interrupt_cost
            resetCurrentNode(_node_token)

            return result

        except GraphInterrupt as e:
            _elapsed = time.time() - _start
            _is_interrupt = True

            # Try to get the cost from the exception or the thread local
            _cost_val = _profile_local.interrupt_cost
            if _cost_val is None:
                try:
                    if hasattr(e, 'interrupts') and len(e.interrupts) > 0:
                        val = e.interrupts[0].value
                        if isinstance(val, dict) and 'cost' in val:
                            _cost_val = val['cost']
                except Exception:
                    pass
            _actual_cost_val = _profile_local.actual_cost

            _ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _is_interrupt_str = "True"
            _cost_str = f"${_cost_val:.6f}" if _cost_val is not None else "N/A"
            _line = (f"{_ts}  |  {func.__name__:<35}  |  time={_elapsed:>8.3f}s  |  "
                     f"is_interrupt={_is_interrupt_str:<5}  |  cost={_cost_str}  |  "
                     f"actual_cost=${_actual_cost_val:.6f}")
            _profile_logger.info(_line)

            # Restore parent frame's accumulator
            _profile_local.active = _prev_active
            _profile_local.cost = _prev_cost + (_cost_val if _cost_val is not None else 0.0)
            _profile_local.actual_cost = _prev_actual_cost + _actual_cost_val
            _profile_local.is_interrupt = _prev_is_interrupt or _is_interrupt
            _profile_local.interrupt_cost = _prev_interrupt_cost
            resetCurrentNode(_node_token)

            raise e

        except RunCancelled as e:
            # The user cancelled mid-node. Log the partial spend the same way
            # the crash path below does — the provider still charged for
            # whatever went out before the cancel landed — then let the
            # BaseException travel all the way out of ainvoke() to main.py.
            _elapsed = time.time() - _start
            _actual_cost_val = _profile_local.actual_cost
            _ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _line = (f"{_ts}  |  {func.__name__:<35}  |  time={_elapsed:>8.3f}s  |  "
                     f"is_interrupt=False  |  cost=N/A  |  "
                     f"actual_cost=${_actual_cost_val:.6f}  |  cancelled={e.scope}")
            _profile_logger.info(_line)

            _profile_local.active = _prev_active
            _profile_local.cost = _prev_cost
            _profile_local.actual_cost = _prev_actual_cost + _actual_cost_val
            _profile_local.is_interrupt = _prev_is_interrupt
            _profile_local.interrupt_cost = _prev_interrupt_cost
            resetCurrentNode(_node_token)
            raise

        except Exception as e:
            _elapsed = time.time() - _start
            _actual_cost_val = _profile_local.actual_cost
            # Whatever calls did fire before the crash still cost real money —
            # log a line for it (there wasn't one before) and carry the spend
            # up to the parent frame instead of silently dropping it.
            if _actual_cost_val:
                _ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                _line = (f"{_ts}  |  {func.__name__:<35}  |  time={_elapsed:>8.3f}s  |  "
                         f"is_interrupt=False  |  cost=N/A  |  "
                         f"actual_cost=${_actual_cost_val:.6f}  |  raised={type(e).__name__}")
                _profile_logger.info(_line)

            _profile_local.active = _prev_active
            _profile_local.cost = _prev_cost
            _profile_local.actual_cost = _prev_actual_cost + _actual_cost_val
            _profile_local.is_interrupt = _prev_is_interrupt
            _profile_local.interrupt_cost = _prev_interrupt_cost
            resetCurrentNode(_node_token)
            raise e

    return wrapper


def messageText(response) -> str:
    """The plain text of a LangChain chat response, whatever shape it arrives in.

    Thinking models (Gemini 3 and later) return ``content`` as a LIST of typed
    blocks -- ``[{"type": "text", "text": "...", "extras": {"signature": ...}}]``,
    interleaved with reasoning blocks -- where every model before them returned a
    plain string. Every call site here wants the text, and a bare
    ``response.content.strip()`` raises ``AttributeError: 'list' object has no
    attribute 'strip'`` against a block list. Non-text blocks (reasoning, thought
    signatures) are dropped.

    :param response: A LangChain message (or a raw content value).
    :returns: The concatenated text, ``""`` if there is none.
    :rtype: str
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)
    return "" if content is None else str(content)


def normalizeMessageContent(response):
    """Flatten a response's block-list ``content`` to plain text, in place.

    Applied at the bottom of every model-fallback loop so callers keep working
    with ``.content`` as the string they have always assumed -- see
    :func:`messageText` for why block lists show up at all.
    """
    try:
        if not isinstance(getattr(response, "content", None), str):
            response.content = messageText(response)
    except Exception:
        pass
    return response


# ── Text-LLM fallback chain (ordered by speed / cost) ──────────────────────

TEXT_MODEL_FALLBACK_CHAIN = [
    {"name": "gemini-3-flash-preview",  "provider": "google"},
    # {"name": "deepseek-chat",     "provider": "deepseek"},
    {"name": "gpt-5-mini",      "provider": "openai"}
    
]
"""list[dict]: Ordered list of text-generation models to try.

Each entry is ``{"name": "<model>", "provider": "google"|"deepseek"|"openai"}``.
The first model is attempted; on failure the next one in line is tried,
and so on until one succeeds or the chain is exhausted.
"""


# Per-attempt bound for a single LLM call in the fallback chain. Text/JSON
# calls (classification, compilation, ...) normally finish in a few seconds;
# 60s is generous headroom while still guaranteeing a stalled provider raises
# instead of hanging the calling node (and its awaiting websocket request)
# forever with no error.
LLM_CALL_TIMEOUT_SECONDS = 60


def invoke_with_fallback(messages, temperature=0, thinking_budget=None,
                         fallback_chain=None, role=None):
    """Invoke an LLM with automatic fallback to the next fastest model.

    Iterates through ``fallback_chain`` (defaults to the chain
    :mod:`graphic_designer.model_router` resolves for ``role``, or to
    :data:`TEXT_MODEL_FALLBACK_CHAIN` when neither is given). For each model it
    instantiates the appropriate LangChain chat class, calls
    ``.invoke(messages)``, and returns on success. On any exception the next
    model is tried.

    :param messages: The message list to send (LangChain message format).
    :type messages: list[dict]
    :param temperature: Sampling temperature (default ``0``).
    :type temperature: float
    :param thinking_budget: Google-specific thinking-budget parameter.
        Ignored for OpenAI models.
    :type thinking_budget: int | None
    :param fallback_chain: Override the default chain.
    :type fallback_chain: list[dict] | None
    :param role: Registry role to route by (``classify_extract``, ...). The
        router decides which provider serves this call and in what fallback
        order; an explicit ``fallback_chain`` still wins.
    :type role: str | None
    :returns: A tuple ``(response, billing_model)`` where ``response`` is the
        LangChain AIMessage and ``billing_model`` is the :data:`config.GEN_PRICES`
        key for whichever model answered — **not** necessarily the id sent on
        the wire. Every caller feeds this straight to :func:`calculateCost`, so
        returning the wire id would reprice the customer whenever routing moved
        a call to a different model.
    :rtype: tuple
    :raises Exception: Re-raises the last exception if every model in the
        chain fails.
    """
    from .config import GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY   # deferred to avoid circular import

    if fallback_chain is None:
        if role:
            from .model_router import resolveRole
            fallback_chain = resolveRole(role)
        else:
            fallback_chain = TEXT_MODEL_FALLBACK_CHAIN

    last_error = None
    for model_cfg in fallback_chain:
        # The `except Exception` below turns any failure into "try the next
        # model". RunCancelled is a BaseException so it passes straight
        # through, but check up front anyway so a cancel that lands between two
        # rungs doesn't buy one more attempt.
        checkCancelled('invoke_with_fallback')
        # Registry entries carry api_id/billing_id; the legacy literal chain
        # carries only 'name'. Accept both so the two can coexist during the
        # migration and a caller that still passes its own chain keeps working.
        api_id = model_cfg.get("api_id") or model_cfg["name"]
        billing_id = model_cfg.get("billing_id") or api_id
        call_role = model_cfg.get("role") or role
        try:
            if model_cfg["provider"] == "google":
                kwargs = {
                    # The COT modules prefix ids with 'models/'; the LangChain
                    # google client wants the bare id.
                    "model": api_id.split("/")[-1],
                    "google_api_key": GEMINI_API_KEY,
                    # Without this, a stalled connection to the provider hangs
                    # this call (and the whole node awaiting it) forever with
                    # no exception -- the fallback chain below only advances
                    # on an exception, so a silent stall never falls through
                    # to the next model either.
                    "timeout": LLM_CALL_TIMEOUT_SECONDS,
                }
                if thinking_budget is not None:
                    kwargs["thinking_budget"] = thinking_budget
                llm = ChatGoogleGenerativeAI(**kwargs)
            elif model_cfg["provider"] == "deepseek":
                llm = ChatOpenAI(
                    model=api_id,
                    openai_api_key=DEEPSEEK_API_KEY,
                    openai_api_base="https://api.deepseek.com",
                    timeout=LLM_CALL_TIMEOUT_SECONDS,
                )
            elif model_cfg["provider"] == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model=api_id, temperature=temperature, timeout=LLM_CALL_TIMEOUT_SECONDS)
            else:
                llm = ChatOpenAI(
                    model=api_id,
                    openai_api_key=OPENAI_API_KEY,
                    timeout=LLM_CALL_TIMEOUT_SECONDS,
                )

            apiLog(model_cfg["provider"], api_id,
                   'invoke_with_fallback', status='start', role=call_role)
            response = llm.invoke(messages)
            _tokens_in = _tokens_out = None
            try:
                # LangChain normalizes usage_metadata to {input_tokens,
                # output_tokens, total_tokens} across Google/OpenAI/Anthropic,
                # so this split works the same for every provider in the chain.
                _usage = response.usage_metadata or {}
                _tokens_in = _usage.get('input_tokens')
                _tokens_out = _usage.get('output_tokens')
            except Exception:
                pass
            apiLog(model_cfg["provider"], api_id, 'invoke_with_fallback',
                   tokens_in=_tokens_in, tokens_out=_tokens_out, role=call_role)
            return normalizeMessageContent(response), billing_id

        except Exception as e:
            last_error = e
            apiLog(model_cfg["provider"], api_id,
                   'invoke_with_fallback', status='failed', detail=str(e), role=call_role)
            logging.getLogger('graphic_designer.fallback').warning(
                "Model %s failed (%s), falling back…", api_id, e
            )
            continue

    raise last_error


def invokeMultimodalWithFallback(content_parts, temperature=0, thinking_budget=None,
                                 role="classify_extract", fallback_chain=None):
    """Invoke a vision-capable model on ``[{type: text|image_url}, ...]`` parts,
    with the same provider fallback as :func:`invoke_with_fallback`.

    Exists because four call sites hand-rolled their own copy of the fallback
    loop for exactly one reason: they build image parts, and text-only providers
    in the chain need those stripped. That per-provider content shaping is the
    only real difference, so it lives here once instead of four times.

    :param content_parts: OpenAI-style content blocks. Image parts are dropped
        for providers that cannot accept them, rather than failing the call.
    :returns: ``(response, billing_model)``, same contract as
        :func:`invoke_with_fallback`.
    """
    from .model_router import resolveRole

    chain = fallback_chain or (resolveRole(role) if role else TEXT_MODEL_FALLBACK_CHAIN)
    text_only_providers = {"deepseek"}

    last_error = None
    for entry in chain:
        provider = entry.get("provider")
        parts = ([p for p in content_parts if not isinstance(p, dict) or p.get("type") != "image_url"]
                 if provider in text_only_providers else content_parts)
        try:
            return invoke_with_fallback([{"role": "user", "content": parts}],
                                        temperature=temperature,
                                        thinking_budget=thinking_budget,
                                        fallback_chain=[entry])
        except Exception as e:
            last_error = e
            continue
    raise last_error


# ── Per-process criticism detector (the "individual navigator") ────────────

def detectCriticism(state, user_text, origin):
    """Check a post-process user message for non-actionable criticism.

    This is the per-process navigator: every flow that shows the user a
    finished graphic calls it on the resume payload before doing its normal
    routing.  It answers one narrow question — *is this dissatisfaction that
    doesn't say what to fix?* — so the global ``navigator`` keeps its existing
    class set and new flows can be added without retraining it.

    A concrete instruction ("make the background darker") is deliberately
    **not** criticism; it already names the change and belongs on the normal
    editing route.  Only vague dissatisfaction ("it feels cheap", "eh") routes
    to ``criticismHandler``.

    Fails **closed**: any exception returns ``False`` so the caller falls
    through to its existing behaviour rather than trapping the user in the
    criticism loop.

    :param state: The current graph state (read for ``conversation_history``).
    :type state: GraphicDesigner
    :param user_text: The user's raw message from the resume payload.
    :type user_text: str
    :param origin: Flow that produced the graphic — ``'generation'``,
        ``'editing'``, ``'expand'``, ``'suggestion'`` or ``'layering'``.
    :type origin: str
    :returns: ``(is_criticism, cost)`` — whether to route to
        ``criticismHandler``, and the cost of the classification call.
    :rtype: tuple[bool, float]
    """
    from .prompts import CRITICISM_DETECTOR_PROMPT   # deferred to avoid circular import

    if not user_text or not user_text.strip():
        return False, 0.0

    try:
        prompt = CRITICISM_DETECTOR_PROMPT.format(
            origin=origin,
            user_response=user_text,
            chat_history=json.dumps(state.get("conversation_history", [])[-6:])
        )
        llm_response, model_used = invoke_with_fallback(
            [{"role": "user", "content": prompt}], temperature=0
        )
        cost = calculateCost(model_used, llm_response.usage_metadata['total_tokens'])
        parsed = parseJson(llm_response.content.strip())
        return bool(parsed.get("criticism", False)), cost

    except Exception as e:
        logging.getLogger('graphic_designer.criticism').warning(
            "[detectCriticism] %s flow: classification failed (%s) — treating as non-criticism",
            origin, e
        )
        return False, 0.0


# Changing png base64 into webp base64 for better performance and smaller size

def convert_png_to_webp_base64(base64image: str) -> str:
    """Convert a base64-encoded PNG image to a base64-encoded WEBP image.

    This function decodes the input PNG image, converts it to WEBP format
    using Pillow, and then re-encodes it as a base64 string.

    :param base64image: The input image as a base64-encoded PNG string.
    :type base64image: str
    :returns: The converted image as a base64-encoded WEBP string.
    :rtype: str
    """
    from PIL import Image
    from io import BytesIO
    import base64

    image_bytes = base64.b64decode(base64image)
    img = Image.open(BytesIO(image_bytes))
    buffered = BytesIO()
    img.save(buffered, format="WEBP")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

 

def convert_urls_to_b64(urls: list[str]) -> list[str]:
    """Convert a list of image URLs to base64-encoded strings.

    This function takes a list of image URLs, fetches each image, and converts
    it to a base64-encoded string. If any URL cannot be fetched or processed,
    it is skipped and a warning is logged.

    :param urls: A list of image URLs to convert.
    :type urls: list[str]
    :returns: A list of base64-encoded image strings corresponding to the input URLs.
    :rtype: list[str]
    """
    b64_images = []
    for url in urls:
        try:
            response = makeApiRequest('GET', url)
            if response and response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    img_format = content_type.split('/')[-1].upper()
                    img_bytes = response.content
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    b64_images.append(f"data:{content_type};base64,{img_b64}")
                else:
                    logging.getLogger('graphic_designer.utils').warning(
                        "URL %s did not return an image (Content-Type: %s)", url, content_type
                    )
            else:
                logging.getLogger('graphic_designer.utils').warning(
                    "Failed to fetch URL %s (status code: %s)", url, response.status_code if response else 'No Response'
                )
        except Exception as e:
            logging.getLogger('graphic_designer.utils').warning(
                "Error processing URL %s: %s", url, e
            )
    return b64_images
def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_palette_reference_image(color_palettes_data: dict) -> str:
    """Generate a base64 encoded WEBP image visualizing the top 2 brand color palettes.

    :param color_palettes_data: The color palettes dictionary from brand DNA.
    :type color_palettes_data: dict
    :returns: A base64-encoded data URI representing the generated palette image, or None if invalid.
    :rtype: str
    """
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    import base64

    if not color_palettes_data or 'ranked_palettes' not in color_palettes_data:
        return None
    
    palettes = sorted(color_palettes_data['ranked_palettes'], key=lambda x: x.get('rank', 999))
    top_palettes = palettes[:2]
    
    if not top_palettes:
        return None
        
    width = 800
    height = 400
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        
    y_offset = 20
    for i, palette in enumerate(top_palettes):
        draw.text((20, y_offset), f"Palette {i+1}: {palette.get('name', 'Unnamed')}", fill=(0,0,0), font=title_font)
        y_offset += 40
        
        colors = palette.get('colors', {})
        x_offset = 20
        swatch_width = 150
        swatch_height = 100
        
        for role, color_data in colors.items():
            hex_code = color_data.get('hex', '#FFFFFF')
            try:
                rgb = hex_to_rgb(hex_code)
            except Exception:
                rgb = (255, 255, 255)
            
            draw.rectangle([x_offset, y_offset, x_offset + swatch_width, y_offset + swatch_height], fill=rgb)
            
            # Draw label below
            draw.text((x_offset, y_offset + swatch_height + 10), role.capitalize(), fill=(0,0,0), font=font)
            draw.text((x_offset, y_offset + swatch_height + 30), hex_code, fill=(50,50,50), font=font)
            
            x_offset += swatch_width + 20
            
        y_offset += swatch_height + 70
        
    buffer = BytesIO()
    img.save(buffer, format="WEBP")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/webp;base64,{img_str}"


def compositeLogo(base_image_bytes: bytes, logo_b64: str, zone_pct: dict, pad_frac: float = 0.12) -> bytes:
    """Deterministically paste a logo into a percentage-defined zone of a base image.

    Mirrors ``generation_pipeline/avenue2_generation_pipeline/image_generation.py``'s
    ``composite_logo()`` math (zone rect from ``zone_pct``, aspect-preserving fit
    with padding, centered, alpha-composited), adapted to operate on in-memory
    bytes/base-64 instead of filesystem paths since production stores images in S3.

    :param base_image_bytes: The generated image's raw bytes.
    :type base_image_bytes: bytes
    :param logo_b64: The logo image as a base-64 string, with or without a
        ``data:image/...;base64,`` prefix.
    :type logo_b64: str
    :param zone_pct: ``{"x_pct": [x0, x1], "y_pct": [y0, y1]}`` percentages (0-100)
        defining the logo zone rectangle on the base image.
    :type zone_pct: dict
    :param pad_frac: Fraction of the zone reserved as padding around the logo.
    :type pad_frac: float
    :returns: The composited image, re-encoded as WEBP bytes.
    :rtype: bytes
    """
    from PIL import Image
    from io import BytesIO

    if ',' in logo_b64 and logo_b64.strip().startswith('data:'):
        logo_b64 = logo_b64.split(',', 1)[1]

    base = Image.open(BytesIO(base_image_bytes)).convert("RGBA")
    bw, bh = base.size
    logo = Image.open(BytesIO(base64.b64decode(logo_b64))).convert("RGBA")

    xp = zone_pct.get("x_pct", [0, 100])
    yp = zone_pct.get("y_pct", [0, 100])
    zx0, zx1 = bw * xp[0] / 100.0, bw * xp[1] / 100.0
    zy0, zy1 = bh * yp[0] / 100.0, bh * yp[1] / 100.0
    zw, zh = max(1.0, zx1 - zx0), max(1.0, zy1 - zy0)

    avail_w, avail_h = zw * (1 - pad_frac), zh * (1 - pad_frac)
    scale = min(avail_w / logo.width, avail_h / logo.height)
    new_w, new_h = max(1, int(logo.width * scale)), max(1, int(logo.height * scale))
    logo_r = logo.resize((new_w, new_h), Image.LANCZOS)

    px = int(zx0 + (zw - new_w) / 2)
    py = int(zy0 + (zh - new_h) / 2)
    base.alpha_composite(logo_r, (px, py))

    buffer = BytesIO()
    base.convert("RGB").save(buffer, format="WEBP", quality=95)
    return buffer.getvalue()
