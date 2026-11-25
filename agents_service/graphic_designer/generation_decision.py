"""Is this generation request a new design, or more of the previous one?

A project keeps one thread, so everything collected for the last design -- the
manifest field answers (product name, offer, dates...), the product photo, the
chat -- is still in state when the user asks for the next one. That is right
when the new request is about the same thing ("same post for Diwali", "another
one with a bolder headline") and wrong when it is not ("now one for our new
running shoes"): the old answers end up printed on the new design and the
questions that would have collected the right ones are never asked.

:func:`judge_new_generation` asks OpenAI's Decisions API (``POST
/v1/decisions``) one predicate question about the previous and the new
request. ``generationManager`` flushes the previous design's inputs when it
says "new" (see :func:`fresh_generation_reset`).

The call is plain HTTP rather than ``client.decisions.create``: the pinned
openai SDK (2.36.0) predates the endpoint, and bumping it for one call is not
worth the risk to every other OpenAI call site.
"""
from __future__ import annotations

import json
import time

import httpx

import settings
from .config import OPENAI_API_KEY
from .utils import apiLog, calculateCost

_ENDPOINT = "https://api.openai.com/v1/decisions"
_QUESTION_NAME = "same_design"

_INSTRUCTIONS = (
    "The user is making marketing graphics in one design project. PREVIOUS is "
    "the last graphic they had generated, NEW is the graphic they are asking for "
    "now. Is NEW about the same subject as PREVIOUS -- the same product, offer, "
    "event, announcement or message -- so the content details given for PREVIOUS "
    "(product name, prices, discount, dates, venue, call to action...) still "
    "apply to NEW? Answer true for variations, refinements and follow-ups of "
    "PREVIOUS (another version, a different style or format, the same promotion "
    "for another occasion), and when NEW refers back to the earlier design. "
    "Answer false when NEW is about a different product, offer, event or "
    "message, even if it is the same kind of graphic."
)


def _format_answers(answers: dict) -> str:
    lines = [f"- {k.replace('_', ' ')}: {v}" for k, v in (answers or {}).items()
             if str(v).strip() and v != "skipped"]
    return "\n".join(lines) or "(none)"


def _decision_input(prev: dict, prev_answers: dict, new_prompt: str, design_goal: str) -> str:
    return (
        "PREVIOUS\n"
        f"Design goal: {prev.get('design_goal') or '(unknown)'}\n"
        f"Request: {prev.get('prompt', '')}\n"
        f"Content details provided:\n{_format_answers(prev_answers)}\n\n"
        "NEW\n"
        f"Design goal: {design_goal or '(unknown)'}\n"
        f"Request: {new_prompt}"
    )


def post_decision(text: str, questions: list, purpose: str) -> tuple:
    """POST one Decisions API request and bill it.

    :returns: ``(body, tokens, cost)``; ``body`` is None on any HTTP or
        network failure (already logged), so callers fall back on their own.
    """
    model = settings.GENERATION_DECISION_MODEL
    try:
        resp = httpx.post(
            _ENDPOINT,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model, "input": text, "questions": questions},
            timeout=settings.GENERATION_DECISION_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        apiLog(provider="openai", model=model, purpose=purpose,
               status="failed", detail=str(e)[:200])
        print(f"[{purpose}] Decisions API call failed ({e})")
        return None, 0, 0.0

    usage = body.get("usage") or {}
    tokens = int(usage.get("input_tokens") or usage.get("total_tokens") or len(text) // 4)
    cost = calculateCost(model, tokens)
    apiLog(provider="openai", model=model, purpose=purpose,
           tokens=tokens, tokens_in=tokens, tokens_out=0, cost=cost)
    return body, tokens, cost


def judge_new_generation(prev: dict, prev_answers: dict, new_prompt: str,
                         design_goal: str) -> tuple:
    """Ask the Decisions API whether ``new_prompt`` starts a new design.

    :param prev: ``generation_subject`` of the last generation --
        ``{"prompt", "design_goal"}``.
    :param prev_answers: the manifest field answers collected for it.
    :param new_prompt: this turn's request.
    :param design_goal: this turn's design goal.
    :returns: ``(is_new, probability_same, cost)``. ``is_new`` is False on
        any failure or refusal, so an outage keeps today's behaviour (reuse)
        instead of re-asking questions the user already answered.
    """
    model = settings.GENERATION_DECISION_MODEL
    text = _decision_input(prev, prev_answers, new_prompt, design_goal)
    print(f"[generationDecision] asking {model}: previous={_short(prev.get('prompt'))!r} "
          f"answers={sorted((prev_answers or {}).keys())} new={_short(new_prompt)!r}")
    started = time.monotonic()
    body, tokens, cost = post_decision(
        text, [{"type": "predicate", "name": _QUESTION_NAME,
                "instructions": _INSTRUCTIONS}],
        purpose="generationDecision")
    if body is None:
        print(f"[generationDecision] Decisions API failed after "
              f"{time.monotonic() - started:.2f}s -- treating as related")
        return False, None, 0.0

    answer = next((a for a in body.get("answers") or []
                   if a.get("name") == _QUESTION_NAME), None)
    if not answer or answer.get("type") != "predicate":
        print(f"[generationDecision] no predicate answer ({json.dumps(answer)[:200]}) "
              f"-- treating as related")
        return False, None, cost
    p_same = float(answer.get("probability", 1.0))
    is_new = p_same < settings.GENERATION_DECISION_THRESHOLD
    print(f"[generationDecision] p(same)={p_same:.3f} threshold="
          f"{settings.GENERATION_DECISION_THRESHOLD} -> {'NEW' if is_new else 'SAME'} design "
          f"({time.monotonic() - started:.2f}s, {tokens} tokens, ${cost:.6f})")
    return is_new, p_same, cost


def _short(text, limit: int = 120) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


# asset_prefs keys that belong to the brand, not to one design: kept across a
# new generation. Everything else (the product photo, combo-specific reference
# assets) was supplied for the previous design.
_BRAND_ASSET_KEYS = ("logo", "brand_logo")


def fresh_generation_reset(state: dict) -> dict:
    """State update that forgets the previous design's inputs.

    Flushed: manifest field answers (including "skipped" -- a field declined
    for the old design may matter for the new one), the product photo and
    other per-design assets, and the chat. Kept: the logo and brand kit, and
    this turn's own inputs -- the prompt, attachments, and any field answers
    or assets the editor sent with this very request (``turn_editor_inputs``).
    """
    turn = state.get("turn_editor_inputs") or {}
    prefs = {k: v for k, v in (state.get("asset_prefs") or {}).items()
             if k in _BRAND_ASSET_KEYS}
    prefs.update(turn.get("asset_prefs") or {})
    history = state.get("conversation_history") or []
    # The navigator already appended this turn's message; it is the only part
    # of the chat that belongs to the new design.
    current = [history[-1]] if history and history[-1].get("role") == "user" else []
    reset = {
        "field_answers": dict(turn.get("field_answers") or {}),
        "asset_prefs": prefs,
        "extra_assets": {},
        "manifest_pending_fields": [],
        "manifest_pending_assets": [],
        "conversation_history": current,
    }
    old_prefs = state.get("asset_prefs") or {}
    print(f"[newDesignFlush] field_answers {sorted((state.get('field_answers') or {}).keys())} "
          f"-> {sorted(reset['field_answers'].keys())} | asset_prefs {sorted(old_prefs.keys())} "
          f"-> {sorted(prefs.keys())} | conversation_history {len(history)} -> {len(current)} msgs"
          f" | kept from this turn: {sorted(turn.keys()) or 'nothing'}")
    return reset


def unsaved_turn_inputs(turn: dict, prior: dict) -> dict:
    """``turn_editor_inputs`` minus what the saved state already held.

    A thread rebuilt on reconnect has no last frame to diff against, so every
    field answer and asset in the frame looks new. Values identical to the
    saved state's belong to the previous design, not to this turn, and must
    not survive a new-design flush (:func:`fresh_generation_reset`).

    :param turn: ``{"field_answers": {...}, "asset_prefs": {...}}`` for this frame.
    :param prior: the saved state the thread was re-seeded from.
    """
    prior = prior or {}
    out = {}
    for key in ("field_answers", "asset_prefs"):
        saved = prior.get(key) or {}
        fresh = {k: v for k, v in ((turn or {}).get(key) or {}).items() if saved.get(k) != v}
        if fresh:
            out[key] = fresh
    return out
