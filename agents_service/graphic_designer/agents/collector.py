"""The input-collector: runs after selection, before any agent, for every task.

It is the only place that looks at the image, validates the request against it,
and talks to the user. It returns one of:

* ready  -- everything the agent needs is settled; the task goes to the agent
            (optionally with notices to tell the user first)
* ask    -- a question for the user; the task becomes the orchestrator's pending
            task and nothing runs. The user's reply goes through interpretation
            again, so it may answer, ask something else, start something new or
            cancel.
* cannot -- the request cannot be done on this image; say why, nothing runs.

Agents never ask the user anything; a gap they find mid-run comes back here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from image_agents.registry import Card

from .session import Brief, DesignSession
from .vision import vision_json

MAX_ROUNDS = 2            # questions per task before defaults or a polite stop
IMAGE_AGENTS = {"image-editor", "critic-agent", "image-evaluator"}
# Agents that work on an existing image (and may fall back to the project's latest graphic).
NEEDS_IMAGE = IMAGE_AGENTS | {"manual-ops", "layering-agent"}
NO_CHECK_AGENTS = {"question-handler", "chitchat-handler", "custom-agent"}


@dataclass
class CollectTask:
    id: str
    agent_id: str
    goal: str
    desired_result: str
    performed_by: str = "system"
    constraints: list[str] = field(default_factory=list)
    record: list[dict] = field(default_factory=list)     # [{"q": ..., "a": ...}]
    rounds: int = 0
    notices_shown: set = field(default_factory=set)       # each notice is told once per task


@dataclass
class Collected:
    status: str                                   # ready | ask | cannot
    goal: str = ""
    notices: list[str] = field(default_factory=list)
    question: str = ""
    options: list[str] = field(default_factory=list)
    ask_kind: str = "text"                        # text | select_image | form
    fields: list[dict] = field(default_factory=list)
    reason: str = ""
    brief: dict = field(default_factory=dict)
    cost: float = 0.0
    checks: list[str] = field(default_factory=list)  # what was checked, for the terminal


# --------------------------------------------------------------------- checklists
CHECKLISTS = {
    "image-editor": """\
- Every element the task names must be visible in the image. Missing -> ask which element
  they mean, saying briefly what you do see. If clearly nothing like it exists and no
  sensible element matches, still ask; never let an edit run on nothing.
- More than one thing matches and the task does not say which -> ask which one.
- The change swaps one value for another but gives no new value ("change the button
  colour", "different font", "change the headline" with no wording) -> ask, offering
  "suggest some" and "you choose". If the user already said "you choose", status ready
  and note that the agent picks.
- Relative amounts ("brighter", "bigger") are fine: the agent picks the amount.
- A colour or style clearly outside the brand palette -> notice (not a blocker).
- A change that will not come out well (exact pixel demands, relayout of many elements,
  flipping a design with text) -> notice explaining the risk; still ready.""",
    "image-generator": """\
- Decide whether this is a brand-new design or more of / a redo of the current one.
  Variations and redos need an existing image.
- Pick the catalog design goal and platform when the request matches one; otherwise
  freeform (design_goal null) with an aspect ratio if one is implied.
- Put every content detail the user gave (offer, dates, names, venue, prices...) into the
  brief's "prompt" in their words. Never invent content details and never mark required
  details as handled by the agent; the code checks and asks for required fields.
- Do not ask about things the pipeline can decide (layout, colours, fonts, wording).
- Do not ask for a platform or size: when none is given, the code picks one and tells the
  user. Do not ask for the catalog's content fields either; the code asks for those.""",
    "critic-agent": """\
- The design's purpose is needed for an uploaded design that was not made here and whose
  goal is unknown -> ask what it is for (goal and platform), offering "just review it".
- A named area ("review the typography") must exist in the image.""",
    "suggestion-agent": """\
- Options about an element of the image: that element must be visible in it.
- Options about the brief's content (which offer, what wording, which colours for a design
  not made yet): no image needed; ready.
- Options about the image when there is no image: ask them to select one.""",
    "image-evaluator": """\
- A named area to evaluate must exist in the image. Comparing versions needs two versions.""",
    "manual-ops": """\
- A control that edits a design needs an image. Account screens (add funds, plan,
  billing) need nothing.""",
}

COLLECTOR_PROMPT = """\
You are the input-collector of Kognie, a graphic-design assistant. A task has been routed
to an agent. Before the agent runs, check the task against the image (attached when there
is one) and the session, and decide: ready, ask, or cannot. The agent will NOT be able to
ask the user anything, so anything only the user can answer must be asked now.

TARGET AGENT: {agent_name} ({agent_id})
{agent_description}
Its required inputs:
{required_inputs}

TASK {task_id}: {goal}
Needs result: {desired_result}; performed by: {performed_by}
Constraints: {constraints}

ALREADY ASKED FOR THIS TASK (question -> user's answer):
{record}
Questions asked so far: {rounds} of at most {max_rounds}.

SESSION
{session}

CHECKLIST FOR THIS AGENT
{checklist}

Rules:
- Ask only what the agent truly cannot decide itself, all in ONE short question, with 2-4
  options when the answer is a choice. Be friendly and specific; say what you see in the
  image when it helps.
- Accept the user's earlier answers as given and do not ask the same thing twice, EXCEPT
  when an answer does not fit its question (a bare number like "1", "ok", or text about
  something else for "what's the event called?"): that is not an answer. Do not use it as
  a value; ask that one thing again, or treat it as skipped if this is the last round.
- If this is the last allowed question round and something is still missing: if the agent
  can proceed with a sensible default, return ready with a notice saying what was assumed;
  otherwise return cannot with a polite explanation.
- notices are short sentences to tell the user before the agent runs (brand clash, risk).
- goal: rewrite the task as one complete instruction including every answer the user gave
  (e.g. "Remove the banana at the bottom of the image").
{extra}
Return a single JSON object and nothing else:
{{"status": "ready | ask | cannot", "checks": ["what you checked, one line each"],
  "notices": [], "question": "", "options": [], "goal": "", "reason": ""{brief_schema}}}
"""

GENERATOR_EXTRA = """\
CATALOG design goals: {goals}
CATALOG platforms: {platforms}
Brief so far: {brief}
For this agent also return "brief": {{"new_design": true|false, "design_goal": catalog goal
or null, "platform": catalog platform or null, "aspect_ratio": "W:H" or null, "style": null
or preset/free text, "prompt": "the request in the user's words plus every detail they
gave"}}. The catalog's required fields are checked in code after you
answer; do not ask about them yourself.
"""


def _card_text(card: Card) -> tuple[str, str]:
    inputs = "\n".join(
        f"- {r.name}{' (required' + (', when ' + r.applies_when if r.applies_when else '') + ')' if r.required else ' (optional)'}: {r.meaning}"
        for r in card.required_inputs) or "- none"
    return " ".join(card.description.split()), inputs


def _session_text(session: DesignSession, show_image: bool = True) -> str:
    lines = []
    v = session.current if show_image else None
    if session.current and not show_image:
        lines.append("No image is selected on the canvas.")
    elif v:
        lines.append(f"Current image: version {v.number}, made by {v.made_by} via {v.tool} "
                     f"({v.instruction[:120]!r}); it is attached.")
        if v.meta.get("design_goal"):
            lines.append(f"It was generated here as {v.meta.get('design_goal')} for "
                         f"{v.meta.get('platform')}.")
        elif v.made_by in ("image-generator", "image-editor") and session.brief:
            lines.append(f"It was generated here from this brief: {session.brief.prompt[:200]!r}; "
                         "its purpose is known.")
        elif v.made_by in ("upload", "user"):
            lines.append("It was uploaded or selected by the user; its purpose is unknown "
                         "unless the chat says so.")
    else:
        lines.append("No image in this session.")
    if session.brand_name or session.brand_dna:
        dna = session.brand_dna or {}
        colors = [c for c in [dna.get("color_primary"), *(dna.get("color_secondary") or []),
                              dna.get("color_accent"), dna.get("color_background")] if c]
        lines.append(f"Brand: {session.brand_name or dna.get('brand_name')}; colours: "
                     f"{', '.join(map(str, colors)) or 'unknown'}; tone: {dna.get('tone') or 'unknown'}; "
                     f"logo on file: {'yes' if dna.get('logo') else 'no'}")
    else:
        lines.append("No brand attached.")
    return "\n".join(lines)


def _select_image(goal: str) -> Collected:
    return Collected(status="ask", ask_kind="select_image", goal=goal,
                     question="Please select the image you want me to work on.",
                     checks=["no image in the session: the user must select one"])


def collect(task: CollectTask, session: DesignSession, card: Card) -> Collected:
    """Validate one task and settle its inputs. Never runs the agent."""
    full_goal = task.goal + (f" ({'; '.join(task.constraints)})" if task.constraints else "")
    if task.agent_id in NO_CHECK_AGENTS:
        return Collected(status="ready", goal=full_goal, checks=["nothing to collect"])
    if task.agent_id == "manual-ops" and task.desired_result == "account_action":
        return Collected(status="ready", goal=full_goal, checks=["account screen: no image needed"])
    if task.agent_id in NEEDS_IMAGE and not session.current:
        return _select_image(task.goal)
    if task.agent_id == "layering-agent":
        # The layering agent asks "which elements?" itself, with today's layering_started
        # frame (names or drawn boxes), as layeringManager did; nothing else to settle here.
        return Collected(status="ready", goal=full_goal, checks=["image present; the layering "
                                                                 "agent asks for its targets"])

    # A new design with nothing selected on the canvas does not look at the session's
    # image; it is only relevant to the generator when the user selected it.
    show_image = task.agent_id != "image-generator" or session.canvas_selected
    desc, inputs = _card_text(card)
    record = "\n".join(f"- {r['q']} -> {r['a']}" for r in task.record) or "(nothing yet)"
    extra, brief_schema = "", ""
    if task.agent_id == "image-generator":
        from .. import v33_bridge
        b = session.brief
        extra = GENERATOR_EXTRA.format(
            goals="; ".join(v33_bridge.v33.goals()), platforms="; ".join(v33_bridge.v33.platforms()),
            brief=json.dumps(b.__dict__, default=str)[:1500] if b else "(none)")
        brief_schema = ', "brief": {}'
    prompt = COLLECTOR_PROMPT.format(
        agent_name=card.name, agent_id=card.id, agent_description=desc, required_inputs=inputs,
        task_id=task.id, goal=task.goal, desired_result=task.desired_result,
        performed_by=task.performed_by, constraints="; ".join(task.constraints) or "(none)",
        record=record, rounds=task.rounds, max_rounds=MAX_ROUNDS,
        session=_session_text(session, show_image), checklist=CHECKLISTS.get(task.agent_id, "- (none)"),
        extra=extra, brief_schema=brief_schema)
    images = [session.current.url] if session.current and show_image else []
    raw, cost, _ = vision_json(prompt, images)

    out = Collected(status=str(raw.get("status") or "ask").lower(),
                    goal=raw.get("goal") or task.goal,
                    notices=[str(n) for n in raw.get("notices") or [] if str(n).strip()],
                    question=raw.get("question") or "", options=list(raw.get("options") or []),
                    reason=raw.get("reason") or "", brief=raw.get("brief") or {}, cost=cost,
                    checks=[str(c) for c in raw.get("checks") or []])
    if out.status not in ("ready", "ask", "cannot"):
        out.status = "ask"
    if task.agent_id == "image-generator" and out.status == "ready":
        out = _generator_fields(task, session, out)
    if out.status == "ask" and task.rounds >= MAX_ROUNDS:
        # The model ignored the cap: stop asking.
        out = Collected(status="cannot", goal=out.goal, cost=out.cost, checks=out.checks,
                        reason="I still don't have what I need for this, so I've left it for "
                               "now. Tell me when you'd like to pick it up again.")
    if out.status == "ask" and not out.question:
        out.question = "Could you tell me a bit more about what you'd like?"
    return out


def _generator_fields(task: CollectTask, session: DesignSession, out: Collected) -> Collected:
    """Apply the brief, then ask for the catalog's required fields that are still empty."""
    b = out.brief or {}
    prev = session.brief
    new_design = bool(b.get("new_design", True))
    same_task = task.rounds > 0          # re-checking after the user answered this task's question
    keep = prev and (not new_design or same_task)
    answers = dict(prev.field_answers) if keep else {}
    # The model's own field_answers are NOT trusted for catalog fields: it sometimes fills
    # them with values it made up ("the agent can write it"), which skipped the questions.
    # Required fields are filled only by _fill_fields (what the user actually said), the
    # form, or answers already given for this same design.
    goal, platform = b.get("design_goal"), b.get("platform")
    from .. import v33_bridge
    if goal and not platform:
        # A catalog goal with no platform: pick one by aspect ratio (else the square post)
        # and say so, rather than dropping to freeform and losing the catalog fields.
        catalog_goal = next((g for g in v33_bridge.v33.goals() if g.lower() == str(goal).lower()), None)
        if catalog_goal:
            ratio = b.get("aspect_ratio") or (prev.aspect_ratio if prev else None)
            by_ratio = {"1:1": "Instagram Square Post", "4:5": "Instagram Portrait Post",
                        "9:16": "Instagram Story", "16:9": "Facebook Feed Landscape",
                        "5:4": "LinkedIn Feed Post", "3:4": "Pinterest Pin", "4:3": "Presentation Slide"}
            platform = by_ratio.get(ratio or "", "Instagram Square Post")
            out.notices.append(f"I'll make it as an {platform} — tell me if you need another size.")
    combo = v33_bridge.resolve_combo(goal, platform) if goal and platform else None
    if goal and platform and not combo:
        out.notices.append(f"'{goal}' on '{platform}' isn't one of our catalog layouts, so I'll "
                           "design it freely.")
        goal = platform = None
    elif combo:
        goal, platform = combo
    session.brief = Brief(
        prompt=b.get("prompt") or out.goal or task.goal, design_goal=goal, platform=platform,
        aspect_ratio=b.get("aspect_ratio") or (prev.aspect_ratio if keep else None),
        field_answers=answers, style=b.get("style") or (prev.style if keep else None),
        photo_url=(session.attachments[0] if session.attachments else (prev.photo_url if prev else None)),
        photo_mode=("reference" if session.attachments else (prev.photo_mode if prev else None)),
        logo_url=prev.logo_url if prev else None, logo_mode=prev.logo_mode if prev else "none")
    if new_design and not same_task and prev and prev.prompt and prev.prompt != session.brief.prompt:
        out.checks.append("new design: answers collected for the previous design were cleared")
    if not combo:
        out.checks.append("freeform design: no catalog fields to collect")
        return out
    missing = _missing_fields(goal, platform, answers)
    if missing:
        filled, cost = _fill_fields(task, session, missing)
        out.cost += cost
        answers.update(filled)
        session.brief.field_answers = answers
        missing = [m for m in missing if m["field_id"] not in filled]
        if filled:
            out.checks.append(f"filled from the conversation: {', '.join(f'{k}={v}' for k, v in filled.items())}")
    out.checks.append(f"catalog {goal} x {platform}: "
                      f"{'all required fields present' if not missing else str(len(missing)) + ' required field(s) missing'}")
    if not missing:
        return out
    if task.rounds >= MAX_ROUNDS:
        out.notices.append("Going ahead without: " + ", ".join(m["label"] for m in missing) + ".")
        return out
    out.status, out.ask_kind, out.fields = "ask", "form", missing
    out.question = ("To make this right I need a few details:\n" + "\n".join(
        f"- {m['question']}" + (f" (e.g. {m['example']})" if m.get("example") else "") for m in missing)
        + "\nAnswer what you like; say 'skip' for anything you'd rather leave out.")
    return out


def _fill_fields(task: CollectTask, session: DesignSession, fields: list[dict]) -> tuple[dict, float]:
    """Map what the user already said (request, answers, brand) onto catalog field ids."""
    said = [task.goal, *task.constraints, *(f"{r['q']} -> {r['a']}" for r in task.record)]
    brand = session.brand_dna or {}
    field_lines = "\n".join(f"- {f['field_id']}: {f['label']} ({f['question']})" for f in fields)
    said_lines = "\n".join(f"- {x}" for x in said if x)
    brand_name = brand.get("brand_name") or session.brand_name or "unknown"
    prompt = (
        "Fill these design-brief fields ONLY from what the user said or the brand data. Do not "
        "invent values. If the user said to skip something ('skip', 'skip the rest', 'leave "
        "it out', 'you decide'), use 'skipped' for each field that covers. Leave a field out "
        "when nothing covers it. A value must make sense for its field: a bare number, 'ok', "
        "'yes' or unrelated text is not an event name, venue or headline; leave such a field "
        "out. Headline, tagline or other wording fields count only when "
        "the user gave the exact words to print; a description of the design is not a "
        "headline.\n"
        f"FIELDS:\n{field_lines}\n"
        f"WHAT THE USER SAID:\n{said_lines}\n"
        f"BRAND: name={brand_name}\n"
        'Return JSON: {"field_id": "value or skipped", ...}')
    raw, cost, _ = vision_json(prompt, [])
    ids = {f["field_id"] for f in fields}
    return {k: str(v).strip() for k, v in raw.items() if k in ids and str(v).strip()}, cost


# Wording the v33 copywriter writes itself (headline, tagline...): never asked while it is
# on. Facts only the user knows (offer, deadline, brand, venue, date) are still asked.
_WORDING_FIELD = re.compile(r"head_?line|tag_?line|slogan|hook|call_?to_?action|\bcta\b",
                            re.IGNORECASE)


def _copywriter_on() -> bool:
    import settings
    from .. import v33_bridge
    return bool(settings.COPYWRITER_ENABLED and v33_bridge.v33.COPYWRITER)


def _missing_fields(goal: str, platform: str, answers: dict) -> list[dict]:
    """The combo's mandatory fields (``frontend_manifest``) with no answer yet, minus the
    wording fields the copywriter writes."""
    from .. import v33_bridge
    manifest = v33_bridge.frontend_manifest(goal, platform) or {}
    given = {k.lower().replace(" ", "_") for k, v in answers.items() if str(v).strip()}
    copywriter = _copywriter_on()
    missing = []
    for f in manifest.get("mandatory_fields") or []:
        fid = str(f.get("field_id") or "")
        label = str(f.get("label") or fid)
        if fid.lower() in given or label.lower().replace(" ", "_") in given:
            continue
        if copywriter and (_WORDING_FIELD.search(fid) or _WORDING_FIELD.search(label)):
            continue
        missing.append({"field_id": fid, "label": label,
                        "question": f.get("question") or f"What should the {label} be?",
                        "example": f.get("example") or ""})
    return missing
