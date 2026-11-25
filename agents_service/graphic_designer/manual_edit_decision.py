"""Should this edit be done by the AI image model or by a manual control?

The frontend has manual controls (brightness, corner radius, border width...)
next to the canvas. An edit such as "increase the brightness to 46%" is exact
work those controls do perfectly and the image model does badly, so the edit
graph hands it to the frontend instead of running it (see
:func:`nodes.editModeClassifier` / :func:`nodes.manualEditHandler`).

The controls live in :data:`OPERATIONS_PATH` (``manual_operations.json``)::

    {"operations": [
        {"id": "brightness", "label": "Brightness",
         "aliases": ["bright", "darker"], "mode": "specific",
         "description": "Overall brightness of the whole graphic."},
        ...
    ]}

``id`` is what the frontend gets back as ``manual_op``. ``mode`` is the rule:

* ``always``   -- any request to change this control is manual, with or
  without a value ("increase the corner radius").
* ``specific`` -- manual only when the request gives an exact value or amount
  ("brightness to 46%", "brightness up by 20%"). Without one ("increase the
  brightness") the edit stays with the AI.

:func:`classify_edit_mode` sends ONE Decisions API request with at most five
questions, whatever the size of the catalog: a ``choice`` for the control the
request is about, and predicates for "an exact value is given", "a control
whose mode is always is asked for", "several controls" and "anything a
control cannot do". The rules above are then applied here, in code, to those
probabilities -- the model is never asked to apply them itself. Any failure,
refusal or empty catalog answers "ai", which is exactly how editing behaved
before this branch existed.

The question count no longer grows with the catalog: it used to carry one
predicate per control plus one per ``specific`` control, and every question
repeats its instructions, so 15 controls meant 26 questions and ~7.9k input
tokens. Five questions are ~2.3k tokens and about twice as fast.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import settings
from .generation_decision import post_decision, _short

OPERATIONS_PATH = Path(__file__).with_name("manual_operations.json")

MODES = ("always", "specific")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_OPERATION_QUESTION = "operation"
_SEVERAL_QUESTION = "several_controls"
_OTHER_QUESTION = "other_edit"
_EXACT_QUESTION = "exact_value"
_ALWAYS_QUESTION = "always_control"
_NONE = "none"

_CONTEXT = (
    "The user is editing a marketing graphic in an editor that has the MANUAL "
    "CONTROLS listed in the input next to the canvas. Read the EDIT REQUEST. "
)

_OPERATION_INSTRUCTIONS = (
    "Which of the listed manual controls does the request ask to change? Pick the "
    "control whose setting the request adjusts, using its name, aliases and "
    "description; when it asks for several, pick the first one mentioned. Pick "
    "\"none\" when the request does not ask to change any listed control -- for "
    "example adding, removing, moving or replacing objects, changing text, "
    "colours of individual elements, the background, the style or the mood."
)

_EXACT_INSTRUCTIONS = (
    "Does the request give an exact numeric value or amount for a listed manual "
    "control it adjusts -- a target (\"to 46%\", \"set it to 12\", \"at 0.8\") or "
    "an exact change (\"by 20%\", \"+15\")? Numbers may be written in words "
    "(\"to fifty percent\"). False for vague amounts with no number (\"a bit\", "
    "\"slightly\", \"more\", \"a lot\", \"double\") and when the number belongs "
    "to something else in the request."
)

_ALWAYS_INSTRUCTIONS = (
    "Does the request ask to change any of these manual controls: {labels}? True "
    "only when one of these settings itself is to be adjusted, whatever the "
    "wording -- see each control's description and aliases in the input."
)

_SEVERAL_INSTRUCTIONS = (
    "Does the request ask to change two or more DIFFERENT listed manual controls "
    "(e.g. brightness and border width)? Asking to change one control is false."
)

_OTHER_INSTRUCTIONS = (
    "Besides adjusting listed manual controls, does the request ALSO ask for a "
    "change none of those controls can make -- adding, removing, moving or "
    "replacing objects, changing text, recolouring individual elements, the "
    "background, the style or the mood? False when the whole request is "
    "covered by the listed controls."
)


def load_operations(path: Path = OPERATIONS_PATH) -> list:
    """The manual controls from ``manual_operations.json``.

    Malformed entries (no or non-snake_case ``id``, duplicate ``id``, unknown
    ``mode``) are skipped with a log line rather than failing the edit. A
    missing or unreadable file means "no manual controls".
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[manualEditDecision] cannot read {path.name} ({e}) -- no manual controls")
        return []

    ops, seen = [], set()
    for entry in (raw or {}).get("operations") or []:
        op_id = str((entry or {}).get("id") or "").strip()
        mode = str((entry or {}).get("mode") or "").strip().lower()
        if not _ID_RE.match(op_id) or op_id in seen or op_id == _NONE or mode not in MODES:
            print(f"[manualEditDecision] skipping invalid operation {json.dumps(entry)[:160]}")
            continue
        seen.add(op_id)
        ops.append({
            "id": op_id,
            "label": str(entry.get("label") or op_id.replace("_", " ")).strip(),
            "aliases": [str(a).strip() for a in entry.get("aliases") or [] if str(a).strip()],
            "mode": mode,
            "description": str(entry.get("description") or "").strip(),
        })
    return ops


def _describe(op: dict) -> str:
    parts = [op["label"]]
    if op["description"]:
        parts.append(op["description"])
    if op["aliases"]:
        parts.append("Also called: " + ", ".join(op["aliases"]) + ".")
    return " -- ".join(parts)


def _decision_input(prompt: str, ops: list) -> str:
    # The controls go in the shared input once, not into every question.
    controls = "\n".join(f"- {op['id']}: {_describe(op)}" for op in ops)
    return f"MANUAL CONTROLS\n{controls}\n\nEDIT REQUEST\n{prompt}"


def _questions(ops: list) -> list:
    context = _CONTEXT
    questions = [
        {"type": "choice", "name": _OPERATION_QUESTION,
         "instructions": context + _OPERATION_INSTRUCTIONS,
         "choices": [{"value": op["id"], "description": _describe(op)} for op in ops]
                    + [{"value": _NONE,
                        "description": "The request does not ask to change any listed control."}]},
        {"type": "predicate", "name": _SEVERAL_QUESTION,
         "instructions": context + _SEVERAL_INSTRUCTIONS},
        {"type": "predicate", "name": _OTHER_QUESTION,
         "instructions": context + _OTHER_INSTRUCTIONS},
    ]
    if any(op["mode"] == "specific" for op in ops):
        questions.append({"type": "predicate", "name": _EXACT_QUESTION,
                          "instructions": context + _EXACT_INSTRUCTIONS})
    always = [op["label"] for op in ops if op["mode"] == "always"]
    if always:
        questions.append({"type": "predicate", "name": _ALWAYS_QUESTION,
                          "instructions": context + _ALWAYS_INSTRUCTIONS.format(
                              labels=", ".join(always))})
    return questions


def decide(ops: list, answers: dict, threshold: float) -> dict:
    """Apply the manual/AI rules to the Decisions API answers.

    :param answers: ``{question name: answer}`` from the response.
    :returns: ``{"mode": "ai" | "manual" | "manual_multiple", "op": id | None,
        "ops": [ids]}``. ``ops`` is ``[op]`` for a manual edit and, for
        ``manual_multiple``, the first control the request names (the only
        one the request identifies).
    """
    def prob(name: str) -> float:
        a = answers.get(name) or {}
        try:
            return float(a.get("probability")) if a.get("type") == "predicate" else 0.0
        except (TypeError, ValueError):
            return 0.0

    by_id = {op["id"]: op for op in ops}
    exact = prob(_EXACT_QUESTION) >= threshold
    always = prob(_ALWAYS_QUESTION) >= threshold

    ai = {"mode": "ai", "op": None, "ops": []}
    # A request that also needs real image work (add a cat AND round the
    # corners) stays with the AI: the frontend cannot do the rest of it.
    if prob(_OTHER_QUESTION) >= threshold:
        return ai

    chosen = (answers.get(_OPERATION_QUESTION) or {}).get("choice")
    if chosen not in by_id:
        return ai

    if prob(_SEVERAL_QUESTION) >= threshold:
        # "brightness to 20% and increase the border width": several controls
        # at once are pointed at the side menu instead of picked one by one.
        # Only when at least one of them is manual by the rules -- an exact
        # value, or an always control -- "brighter and more tint" is vague on
        # both, so it is still an AI edit.
        return {"mode": "manual_multiple", "op": None, "ops": [chosen]} if exact or always else ai

    # One control. An always control needs the always predicate to agree with
    # the choice; a specific one needs an exact value. When they do not, the
    # request is ambiguous and the AI edit -- today's behaviour -- is the safe
    # side to err on.
    if (always if by_id[chosen]["mode"] == "always" else exact):
        return {"mode": "manual", "op": chosen, "ops": [chosen]}
    return ai


def classify_edit_mode(prompt: str, ops: list = None) -> tuple:
    """Decide whether this edit request goes to a manual control.

    :param prompt: the edit request.
    :param ops: the manual controls; read from ``manual_operations.json``
        when omitted.
    :returns: ``(decision, cost)`` -- ``decision`` as :func:`decide` returns
        it; ``{"mode": "ai", ...}`` on an empty catalog or any failure.
    """
    ai = {"mode": "ai", "op": None, "ops": []}
    ops = load_operations() if ops is None else ops
    if not ops or not (prompt or "").strip():
        return ai, 0.0

    print(f"[manualEditDecision] asking {settings.GENERATION_DECISION_MODEL}: "
          f"prompt={_short(prompt)!r} ({len(ops)} controls)")
    started = time.monotonic()
    body, tokens, cost = post_decision(
        _decision_input(prompt, ops), _questions(ops), purpose="manualEditDecision")
    if body is None:
        return ai, 0.0

    answers = {a.get("name"): a for a in body.get("answers") or [] if a.get("name")}
    if (answers.get(_OPERATION_QUESTION) or {}).get("type") != "choice":
        print(f"[manualEditDecision] no usable operation answer "
              f"({json.dumps(answers.get(_OPERATION_QUESTION))[:200]}) -- AI edit")
        return ai, cost

    decision = decide(ops, answers, settings.MANUAL_EDIT_THRESHOLD)
    print(f"[manualEditDecision] -> {decision['mode']} op={decision['op']} ops={decision['ops']} "
          f"(choice={answers[_OPERATION_QUESTION].get('choice')}, "
          f"{time.monotonic() - started:.2f}s, {tokens} tokens, ${cost:.6f})")
    return decision, cost
