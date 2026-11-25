"""Navigator routing through OpenAI's Decisions API.

:func:`classify_request` asks one ``choice`` question whose options are the
navigator's routes, so the category and the non-image-related subtype
(question / chit-chat / unclear) come back from a single call. The option
descriptions mirror NAVIGATOR_PHASE1_PROMPT and NAVIGATOR_PHASE2_PROMPT; keep
them in step with those prompts. ``navigator`` falls back to its LLM
classifier whenever this returns None.
"""
from __future__ import annotations

import json
import time

import settings
from .generation_decision import post_decision, _short

_QUESTION_NAME = "route"

_INSTRUCTIONS = (
    "The user is chatting with Kognie, an assistant that creates and edits "
    "marketing graphics. Read the USER MESSAGE, using the image state and the "
    "recent chat for context, and pick what the user is asking for now."
)

_CHOICES = [
    {"value": "generation",
     "description": "Create a new image that does not build on an existing image."},
    {"value": "generate_more",
     "description": "Another/more version(s) of the SAME design that was just generated, "
                    "without describing new content, subjects or changes: \"generate more\", "
                    "\"another one\", \"more like this\", \"more variations\". If the user "
                    "describes new content it is generation; if they ask for changes it is editing."},
    {"value": "regenerate",
     "description": "The SAME design produced again from the same brief, a fresh attempt: "
                    "\"regenerate\", \"try again\", \"redo it\", also with a newly attached "
                    "reference image or art style. It names no specific change to make."},
    {"value": "editing",
     "description": "A specific modification to an existing image: \"make the background "
                    "darker\", \"remove the logo\", \"change the text to 'Sale Now'\". A question "
                    "about an edit (its price, a term, how to undo) is question instead; pulling "
                    "an object out as its own layer is layering instead."},
    {"value": "expand",
     "description": "Resize, adapt or convert an existing image for social platforms or other "
                    "output dimensions: \"convert for instagram\", \"resize for reel\", "
                    "\"export for linkedin\", \"generate for multiple platforms\"."},
    {"value": "layering",
     "description": "Turn an image, or specific objects in it, into separate layers: \"turn "
                    "this into layers\", \"isolate the vase\", \"separate the person from the "
                    "background\", \"extract the logo as its own layer\"."},
    {"value": "suggestion",
     "description": "Asking for ideas, recommendations or improvements for an image or design."},
    {"value": "slop_removal",
     "description": "Remove AI slop -- the tell-tale artifacts that make a graphic look AI-made -- "
                    "from an existing image: \"remove all ai slop from this image\", \"make it "
                    "look less AI\", \"clean up the AI look\". Removing a named object or "
                    "element (\"remove the logo\") is editing instead."},
    {"value": "question",
     "description": "A question not asking to create or change an image: about Kognie's "
                    "services, pricing, models, process, recommendations, or anything general."},
    {"value": "chit-chat",
     "description": "Casual conversation with no task intent: \"hi\", \"cool\", \"how are you\"."},
    {"value": "unclear",
     "description": "Vague or ambiguous input with no clear intent."},
]


def _decision_input(prompt: str, current_image_url, last_generated_image_url,
                    history: list) -> str:
    return (
        f"USER MESSAGE\n{prompt}\n\n"
        "IMAGE STATE\n"
        f"Image currently selected: {'yes' if current_image_url else 'no'}\n"
        f"A design was generated earlier in this session: "
        f"{'yes' if last_generated_image_url else 'no'}\n\n"
        f"RECENT CHAT\n{json.dumps(history or [])}"
    )


def classify_request(prompt: str, current_image_url, last_generated_image_url,
                     history: list) -> tuple:
    """Pick the navigator route for this turn with the Decisions API.

    :returns: ``(route, cost)``. ``route`` is one of the choice values above,
        or None on any failure or refusal so the caller can use its LLM path.
    """
    text = _decision_input(prompt, current_image_url, last_generated_image_url, history)
    # slop_removal is only a route while its branch is switched on.
    choices = [c for c in _CHOICES
               if c["value"] != "slop_removal" or settings.SLOP_REMOVAL_ENABLED]
    print(f"[navigatorDecision] asking {settings.GENERATION_DECISION_MODEL}: "
          f"prompt={_short(prompt)!r}")
    started = time.monotonic()
    body, tokens, cost = post_decision(
        text, [{"type": "choice", "name": _QUESTION_NAME,
                "instructions": _INSTRUCTIONS, "choices": choices}],
        purpose="navigatorDecision")
    if body is None:
        return None, 0.0

    answer = next((a for a in body.get("answers") or []
                   if a.get("name") == _QUESTION_NAME), None)
    route = (answer or {}).get("choice")
    valid = {c["value"] for c in choices}
    if not answer or answer.get("type") != "choice" or route not in valid:
        print(f"[navigatorDecision] no usable choice answer ({json.dumps(answer)[:200]}) "
              f"-- falling back to the LLM classifier")
        return None, cost
    top = sorted(answer.get("probabilities") or [],
                 key=lambda p: p.get("probability", 0), reverse=True)[:3]
    print(f"[navigatorDecision] -> {route} (confidence={answer.get('confidence')}, "
          f"top={[(p.get('value'), round(p.get('probability', 0), 3)) for p in top]}, "
          f"{time.monotonic() - started:.2f}s, {tokens} tokens, ${cost:.6f})")
    return route, cost
