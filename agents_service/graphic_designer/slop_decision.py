"""Reading the user's replies in the AI slop remover branch.

Two ``choice`` questions to OpenAI's Decisions API, the same call the
navigator uses (see :mod:`graphic_designer.navigator_decision`):

* :func:`judge_slop_offer` -- the reply to the generation_completed offer to
  remove AI slop: did the user agree, decline, or ask for something else?
* :func:`classify_slop_reply` -- a reply inside the slop subgraph that is not
  the exact "create more": a question, chit-chat, a new request, or anything
  else (which carries on to the Critic Agent).

Both return None on any failure so the caller falls back on its own rule.
"""
from __future__ import annotations

import json
import time

import settings
from .generation_decision import post_decision, _short

_OFFER_QUESTION = "slop_offer"
_REPLY_QUESTION = "slop_reply"

_OFFER_INSTRUCTIONS = (
    "The user just had a marketing graphic generated and was asked whether they "
    "want the AI slop removed from it (the tell-tale artifacts that make a graphic "
    "look AI-made). Read the USER REPLY, using the recent chat for context, and "
    "pick how they answered."
)

_OFFER_CHOICES = [
    {"value": "accept",
     "description": "Any agreement to remove the AI slop: \"yes\", \"sure\", \"go ahead\", "
                    "\"please do\", \"yeah clean it up\", \"ok\"."},
    {"value": "decline",
     "description": "Any refusal or wish to move on without it: \"no\", \"no thanks\", "
                    "\"skip\", \"it's fine as is\", \"not now\"."},
    {"value": "other",
     "description": "Neither: a different request or message, e.g. an edit (\"make the "
                    "background blue\"), a new graphic, a question or a complaint."},
]

_REPLY_INSTRUCTIONS = (
    "The user had the AI slop removed from a marketing graphic and was asked whether "
    "they want to create more. Read the USER REPLY, using the recent chat for "
    "context, and pick what it is."
)

_REPLY_CHOICES = [
    {"value": "question",
     "description": "A question that asks for information, not a change: about AI slop, "
                    "the result, Kognie's services, pricing, models, or anything general."},
    {"value": "chit-chat",
     "description": "Casual conversation with no task intent: \"hi\", \"cool\", \"thanks!\"."},
    {"value": "unclear",
     "description": "Vague or ambiguous input with no clear intent."},
    {"value": "new_request",
     "description": "A request for another operation: an edit to the graphic, a new graphic, "
                    "more variations, resizing for platforms, layering, suggestions."},
    {"value": "continue",
     "description": "Anything that just moves on: \"no\", \"no thanks\", \"looks good\", "
                    "\"done\", \"that's all\"."},
]


def _decision_input(reply: str, history: list) -> str:
    return (
        f"USER REPLY\n{reply}\n\n"
        f"RECENT CHAT\n{json.dumps((history or [])[-6:])}"
    )


def _ask(reply: str, history: list, name: str, instructions: str, choices: list,
         purpose: str) -> tuple:
    text = _decision_input(reply, history)
    print(f"[{purpose}] asking {settings.GENERATION_DECISION_MODEL}: reply={_short(reply)!r}")
    started = time.monotonic()
    body, tokens, cost = post_decision(
        text, [{"type": "choice", "name": name,
                "instructions": instructions, "choices": choices}],
        purpose=purpose)
    if body is None:
        return None, 0.0

    answer = next((a for a in body.get("answers") or [] if a.get("name") == name), None)
    verdict = (answer or {}).get("choice")
    if not answer or answer.get("type") != "choice" or verdict not in {c["value"] for c in choices}:
        print(f"[{purpose}] no usable choice answer ({json.dumps(answer)[:200]})")
        return None, cost
    print(f"[{purpose}] -> {verdict} (confidence={answer.get('confidence')}, "
          f"{time.monotonic() - started:.2f}s, {tokens} tokens, ${cost:.6f})")
    return verdict, cost


def judge_slop_offer(reply: str, history: list) -> tuple:
    """Did the user agree to have the AI slop removed?

    :returns: ``(verdict, cost)``; ``verdict`` is ``accept`` / ``decline`` /
        ``other``, or None on any failure or refusal.
    """
    return _ask(reply, history, _OFFER_QUESTION, _OFFER_INSTRUCTIONS, _OFFER_CHOICES,
                "slopOfferDecision")


def classify_slop_reply(reply: str, history: list) -> tuple:
    """What is this reply to the slop subgraph's "create more?" question?

    :returns: ``(verdict, cost)``; ``verdict`` is one of ``question`` /
        ``chit-chat`` / ``unclear`` / ``new_request`` / ``continue``, or None
        on any failure or refusal.
    """
    return _ask(reply, history, _REPLY_QUESTION, _REPLY_INSTRUCTIONS, _REPLY_CHOICES,
                "slopReplyDecision")
