"""Did the user's reply to an edit say it went to the wrong image?

An edit with nothing selected runs on the project's latest graphic (see
``nodes.navigatorClassify``), which is not always the one the user meant. A
reply like "oh no, I wanted to edit another image, not this one" names no
route of its own, so the navigator used to answer it as unclear.

:func:`judge_wrong_image` asks OpenAI's Decisions API one predicate question
about the reply to ``editing_successful``. ``successfulEditHandler`` sends a
true answer to ``editWrongImageHandler``, which apologises and asks the user
to select the right image and send the request again.
"""
from __future__ import annotations

import json
import time

import settings
from .generation_decision import post_decision, _short

_QUESTION_NAME = "wrong_image"

_INSTRUCTIONS = (
    "The user asked for the EDIT REQUEST and the assistant applied it to one of "
    "the user's images. Does the USER REPLY say the edit was applied to the wrong "
    "image -- that they meant a different image or design than the one that was "
    "edited, so editing this one was a mistake? True for replies like \"not this "
    "one\", \"I wanted to edit another image\", \"wrong picture\", \"I meant the "
    "other design\", \"why did you edit this one?\". False when the user accepts "
    "this edit and asks for more work, including on another image (\"now edit the "
    "second image too\", \"do the same on the other one\"), and when the reply is "
    "about the edited image itself: a complaint about the result, a further change, "
    "undoing the edit, a question, or anything else."
)


def judge_wrong_image(edit_request: str, reply: str, history: list) -> tuple:
    """Ask whether ``reply`` says the edit went to the wrong image.

    :returns: ``(is_wrong_image, cost)``. ``is_wrong_image`` is False on any
        failure or refusal, so an outage keeps today's routing.
    """
    text = (f"EDIT REQUEST\n{edit_request}\n\n"
            f"USER REPLY\n{reply}\n\n"
            f"RECENT CHAT\n{json.dumps((history or [])[-6:])}")
    print(f"[editReplyDecision] asking {settings.GENERATION_DECISION_MODEL}: "
          f"reply={_short(reply)!r}")
    started = time.monotonic()
    body, tokens, cost = post_decision(
        text, [{"type": "predicate", "name": _QUESTION_NAME, "instructions": _INSTRUCTIONS}],
        purpose="editReplyDecision")
    if body is None:
        return False, 0.0

    answer = next((a for a in body.get("answers") or []
                   if a.get("name") == _QUESTION_NAME), None)
    try:
        probability = float((answer or {}).get("probability"))
    except (TypeError, ValueError):
        print(f"[editReplyDecision] no usable predicate answer ({json.dumps(answer)[:200]})")
        return False, cost
    is_wrong = probability >= settings.EDIT_WRONG_IMAGE_THRESHOLD
    print(f"[editReplyDecision] -> wrong_image={is_wrong} (p={probability:.3f}, "
          f"{time.monotonic() - started:.2f}s, {tokens} tokens, ${cost:.6f})")
    return is_wrong, cost
