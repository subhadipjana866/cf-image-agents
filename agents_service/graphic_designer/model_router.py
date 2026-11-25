"""Resolve a *role* to the concrete models that should serve it.

Call sites ask for what kind of work they are doing — ``classify_extract``,
``cot_reasoning``, ``copywriting``, ``image`` — and this module answers with an
ordered list of providers to try. Which model that is stays in
:data:`config.MODEL_ROLES`; how one is picked stays here.

Why the split:

* ``config.py`` is data-only by convention (flat literals, each followed by a
  docstring explaining *why*; no classes, no logic). A round-robin cursor under
  a lock is not data.
* ``config.py`` imports boto3 and settings at module scope, and ``utils.py``
  already defers ``from .config import ...`` *inside* functions to dodge the
  resulting import cycle. Putting resolution logic there would deepen it.
* ``llm_fallback.py`` must import this standalone. The avenue2 pipeline is not
  a package — ``nodes.py`` puts its directory on ``sys.path`` and its modules
  import each other by bare name — so **this module must have no heavy
  module-level imports** (no langchain, no boto3) and must degrade gracefully
  when the parent package is not importable, exactly as ``llm_fallback._apiLog``
  already does.

Two mechanisms, deliberately distinct:

* **strategy** (proactive) — which pool member to send this call to, chosen
  *before* it is made, so load spreads across providers instead of piling onto
  one limit until it 429s.
* **fallback_chain** (reactive) — where to go *after* a failure. This is the
  safety net that already existed; the strategy sits on top of it.
"""

from __future__ import annotations

import os
import threading

# Master switch for the proactive half. Off => every role resolves to its first
# pool member followed by the rest of the pool and then its fallback chain,
# which reproduces the single-primary-with-fallbacks behavior the app had before
# the router existed. That makes "did the refactor change anything?" answerable
# by flipping one env var, and makes rollback a config change rather than a
# revert.
ROUTER_STRATEGIES_ON = os.getenv("KOGNIE_ROUTER_STRATEGIES", "on").lower() != "off"

_rr_lock = threading.Lock()
_rr_cursors: dict[str, int] = {}

# Fallback registry used only when the parent package cannot be imported (the
# standalone/sys.path pipeline runs). Mirrors the shape of config.MODEL_ROLES
# with a single conservative pool member per role.
_STANDALONE_ROLES = {
    "cot_reasoning": {
        "strategy": "pinned",
        "pool": [{"provider": "google", "api_id": os.getenv("COT_MODEL", "models/gemini-3.1-flash-lite"),
                  "billing_id": "gemini-3-flash-preview", "weight": 1}],
        "fallback_chain": [
            {"provider": "openai", "api_id": os.getenv("COT_OPENAI_FALLBACK_MODEL", "gpt-5.6-terra"),
             "billing_id": "gpt-5.6-terra"},
            {"provider": "anthropic",
             "api_id": os.getenv("COT_ANTHROPIC_FALLBACK_MODEL", "claude-haiku-4-5-20251001"),
             "billing_id": "claude-haiku-4-5-20251001"},
        ],
    },
}


def _roles() -> dict:
    try:
        from .config import MODEL_ROLES
        return MODEL_ROLES
    except Exception:
        try:
            from graphic_designer.config import MODEL_ROLES
            return MODEL_ROLES
        except Exception:
            return _STANDALONE_ROLES


def knownRoles() -> list[str]:
    return sorted(_roles().keys())


def _nextIndex(role: str, size: int) -> int:
    """Round-robin cursor for `role`.

    Process-wide under a lock rather than thread-local, for the same reason the
    spend breaker is: LangGraph runs nodes on a worker pool, so a per-thread
    cursor would hand every thread the same first pick and spread nothing.
    """
    if size <= 1:
        return 0
    with _rr_lock:
        i = _rr_cursors.get(role, -1) + 1
        _rr_cursors[role] = i
        return i % size


def resolveRole(role: str, pin: str = None, exclude=None) -> list[dict]:
    """Ordered ``[{provider, api_id, billing_id, role}, ...]`` to try for `role`.

    The list is returned in ``TEXT_MODEL_FALLBACK_CHAIN``'s own shape (it also
    carries ``name``), so :func:`utils.invoke_with_fallback` consumes it as a
    plain ``fallback_chain`` override and grows no second loop — it has always
    accepted that argument and no caller ever used it.

    :param pin: an ``api_id`` to force to the head, for a caller that already
        needs a specific provider for this call. A pin that is no longer in
        the registry is ignored rather than trusted.
    :param exclude: ``api_id``s to drop entirely (already tried this run).
    """
    cfg = _roles().get(role)
    if not cfg:
        raise KeyError(f"unknown model role {role!r}; known roles: {knownRoles()}")

    pool = [dict(e) for e in cfg.get("pool", [])]
    chain = [dict(e) for e in cfg.get("fallback_chain", [])]
    if not pool:
        raise ValueError(f"model role {role!r} has an empty pool")

    strategy = cfg.get("strategy", "pinned")
    if pin:
        # Searches the fallback chain too, not just the pool: a run that already
        # fell through to a fallback provider must stay there for its remaining
        # calls, which is the entire point of stickiness.
        candidates = pool + chain
        head = [e for e in candidates if e["api_id"] == pin]
        if head:
            ordered = head + [e for e in pool if e["api_id"] != pin]
            chain = [e for e in chain if e["api_id"] != pin]
        else:
            ordered = pool
    elif strategy == "alternating" and ROUTER_STRATEGIES_ON and len(pool) > 1:
        # Whichever member the cursor lands on, the others follow it: the
        # mirror IS the fallback, so a 429 on one provider lands on the other
        # before touching the role's own chain.
        i = _nextIndex(role, len(pool))
        ordered = pool[i:] + pool[:i]
    else:
        ordered = pool

    out = ordered + chain
    if exclude:
        blocked = set(exclude)
        out = [e for e in out if e["api_id"] not in blocked] or ordered[:1]

    for e in out:
        e["role"] = role
        e.setdefault("name", e["api_id"])   # TEXT_MODEL_FALLBACK_CHAIN compatibility

    # Demote saturated providers. A no-op until PROVIDER_LIMITS is filled in,
    # so the mechanism can ship before the numbers are known.
    out = orderByCapacity(out)
    return out


def entryForRole(role: str, pin: str = None) -> dict:
    """Just the first entry — for call sites that pick a model rather than
    iterate a chain (the image executors, the critic agents)."""
    return resolveRole(role, pin=pin)[0]


def billingIdFor(api_id: str, default=None):
    """The GEN_PRICES key for a wire model id, or `default`.

    Exists so a call site that only knows what it sent can still bill correctly;
    without it, routing a call to a new model silently reprices the customer.
    """
    for cfg in _roles().values():
        for entry in list(cfg.get("pool", [])) + list(cfg.get("fallback_chain", [])):
            if entry.get("api_id") == api_id:
                return entry.get("billing_id", default)
    return default


def chatModelForRole(role: str, temperature=0, thinking_budget=None, index: int = 0):
    """A constructed LangChain chat model for `role`.

    For the few call sites that need a model *object* rather than a response —
    ``helperAgent`` binds tools to it, so ``invoke_with_fallback`` cannot serve
    it. Langchain is imported inside, mirroring ``critiquepipeline/agents.py``'s
    ``_model()`` factory, so this module stays importable without it.
    """
    entry = resolveRole(role)[index]
    return chatModelForEntry(entry, temperature=temperature, thinking_budget=thinking_budget)


def chatModelForEntry(entry: dict, temperature=0, thinking_budget=None):
    """Construct the LangChain chat model one registry entry describes."""
    from .config import GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY

    provider = entry.get("provider")
    model = entry["api_id"].split("/")[-1] if provider == "google" else entry["api_id"]

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": model, "google_api_key": GEMINI_API_KEY, "temperature": temperature}
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        return ChatGoogleGenerativeAI(**kwargs)

    from langchain_openai import ChatOpenAI
    if provider == "deepseek":
        return ChatOpenAI(model=model, openai_api_key=DEEPSEEK_API_KEY,
                          openai_api_base="https://api.deepseek.com", temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    return ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY, temperature=temperature)


# ── Backpressure ───────────────────────────────────────────────────────────
# A token bucket and a concurrency semaphore per provider. This is what turns
# the proactive spread above into something that reacts to real pressure: the
# round-robin alone splits load by count, which says nothing about whether the
# provider it just picked is already saturated.
#
# On saturation the caller is told to move to the next pool member rather than
# being made to wait, because a queue that grows is a latency regression and
# this pipeline is user-facing. Only when NO member has room does it block, and
# only briefly.
_bucket_lock = threading.Lock()
_buckets: dict[str, dict] = {}     # provider -> {"tokens": float, "ts": float, "inflight": int}


def _limits(provider: str) -> dict:
    try:
        from .config import PROVIDER_LIMITS
    except Exception:
        return {}
    return PROVIDER_LIMITS.get(provider) or {}


def hasCapacity(provider: str, estimated_tokens: int = 0) -> bool:
    """Whether `provider` can take one more call right now.

    Always True when that provider has no configured limits — an unfilled
    :data:`config.PROVIDER_LIMITS` disables backpressure rather than throttling
    at a guessed number.
    """
    import time
    lim = _limits(provider)
    rpm, tpm, conc = lim.get("rpm"), lim.get("tpm"), lim.get("concurrency")
    if not any((rpm, tpm, conc)):
        return True

    now = time.time()
    with _bucket_lock:
        b = _buckets.setdefault(provider, {"tokens": 0.0, "calls": 0.0, "ts": now, "inflight": 0})
        # Refill both buckets for the elapsed fraction of a minute.
        elapsed = max(0.0, now - b["ts"])
        b["ts"] = now
        if rpm:
            b["calls"] = max(0.0, b["calls"] - rpm * elapsed / 60.0)
        if tpm:
            b["tokens"] = max(0.0, b["tokens"] - tpm * elapsed / 60.0)

        if conc and b["inflight"] >= conc:
            return False
        if rpm and b["calls"] + 1 > rpm:
            return False
        if tpm and b["tokens"] + estimated_tokens > tpm:
            return False
        return True


def acquire(provider: str, estimated_tokens: int = 0) -> bool:
    """Reserve capacity for one call. Returns False when the provider is
    saturated — the caller should try the next pool member, not queue."""
    import time
    lim = _limits(provider)
    if not any((lim.get("rpm"), lim.get("tpm"), lim.get("concurrency"))):
        return True
    if not hasCapacity(provider, estimated_tokens):
        return False
    with _bucket_lock:
        b = _buckets.setdefault(provider, {"tokens": 0.0, "calls": 0.0, "ts": time.time(), "inflight": 0})
        b["calls"] += 1
        b["tokens"] += estimated_tokens
        b["inflight"] += 1
    return True


def release(provider: str) -> None:
    """Give back the concurrency slot :func:`acquire` took. The token/rate
    portions are not returned — they decay with time, which is what a rate
    limit measures."""
    with _bucket_lock:
        b = _buckets.get(provider)
        if b and b["inflight"] > 0:
            b["inflight"] -= 1


def orderByCapacity(entries: list[dict], estimated_tokens: int = 0) -> list[dict]:
    """Reorder a resolved chain so providers with room come first.

    Stable within each group, so the strategy's choice still wins among
    providers that are equally free — backpressure only demotes the saturated.
    """
    if not entries:
        return entries
    free = [e for e in entries if hasCapacity(e.get("provider"), estimated_tokens)]
    busy = [e for e in entries if e not in free]
    return free + busy


def bucketState() -> dict:
    """Current bucket contents per provider — for a health endpoint."""
    with _bucket_lock:
        return {k: dict(v) for k, v in _buckets.items()}


def preflight(verbose: bool = True) -> dict:
    """One tiny call per distinct model in the registry; returns
    ``{api_id: True|"error text"}``.

    Turns "this key cannot serve the model we pinned" from a runtime mystery
    into a line at boot. Every result is also written to the api log, so a
    quota problem is visible in the same place the calls are.
    """
    from .utils import apiLog

    seen, results = [], {}
    for role, cfg in _roles().items():
        for entry in list(cfg.get("pool", [])) + list(cfg.get("fallback_chain", [])):
            key = (entry.get("provider"), entry.get("api_id"))
            if key in seen or entry.get("provider") not in ("google", "openai", "anthropic"):
                continue
            seen.append(key)
            api_id = entry["api_id"]
            # Image models cannot answer a text probe; skip rather than report
            # a false failure.
            if "image" in api_id or "banana" in api_id:
                continue
            try:
                llm = chatModelForEntry(entry, temperature=0)
                llm.invoke([{"role": "user", "content": "ok"}])
                results[api_id] = True
                apiLog(entry["provider"], api_id, "preflight", status="ok", role=role)
            except Exception as e:
                results[api_id] = str(e)[:200]
                apiLog(entry["provider"], api_id, "preflight", status="failed",
                       detail=str(e)[:200], role=role)

    if verbose:
        for api_id, ok in results.items():
            print(f"[preflight] {api_id:38s} {'OK' if ok is True else 'UNSERVABLE: ' + str(ok)[:90]}")
        broken = [m for m, ok in results.items() if ok is not True]
        if broken:
            print(f"[preflight] {len(broken)} registry model(s) this key cannot serve: {broken}. "
                  f"Calls routed to them will fail over to their fallback chain.")
    return results


if __name__ == "__main__":
    print("roles:", knownRoles())
    for r in knownRoles():
        print(f"\n{r}:")
        for e in resolveRole(r):
            print(f"   {e['provider']:<10} {e['api_id']:<38} bills as {e['billing_id']}")
    preflight()
