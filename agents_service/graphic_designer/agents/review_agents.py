"""Review agents: Critic, Suggestion, Image Evaluator.

The critic and suggestion tools wrap today's code (critiquepipeline.suggestionCreator,
common_nodes.specificSuggestionCreator), which run without interrupts when the collector
has settled the design's purpose. The evaluator tools are vision calls.
(The slop terminator is deliberately not built: its detection and repair come later.)
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional


from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool

from .build import see_current_image
from .common import base_state, make_agent, safe_tool
from .session import DesignSession
from .vision import vision_json

DEFAULT_MODEL = "openai:gpt-5.4"


def _provenance(session: DesignSession) -> tuple[str, Optional[str], Optional[str]]:
    """(generated|uploaded, goal, platform) for the current image."""
    for v in reversed(session.versions):
        if v.meta.get("design_goal"):
            return "generated", v.meta.get("design_goal"), v.meta.get("platform")
        if v.made_by == "image-generator":           # freeform: generated, no catalog goal
            return "generated", None, None
        if v.made_by in ("upload", "user"):
            break
    return "uploaded", None, None


def _numbered(text: str) -> list[str]:
    return [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip() for ln in text.splitlines()
            if re.match(r"^\s*\d+[.)]\s+", ln)]


# ---------------------------------------------------------------------- Critic
@tool
@safe_tool
def critique_design(runtime: ToolRuntime[DesignSession], purpose: str,
                    focus: Optional[str] = None) -> str:
    """Run the full critique (integrity, composition, aesthetic and content lenses, brand-aware,
    then an arbiter) on the current image. Returns up to three suggestions, each an executable
    change. `purpose` is what the design is for (from the request or the collector);
    `focus` narrows it ("the typography", "why it feels unbalanced")."""
    from ..critiquepipeline.nodes import suggestionCreator
    s = runtime.context
    v = s.get(None)
    started = time.monotonic()
    provenance, goal, platform = _provenance(s)
    state = base_state(s) | {
        "current_image_url": v.url, "attached_images": [], "active_graphic": None,
        "critique_provenance": provenance, "critique_intent": purpose or "general design review",
        "design_goal": goal or "", "platform_canonical": platform or "", "platform": platform or "",
        "prompt": focus or purpose or "Review this design", "purpose": purpose,
    }
    out = suggestionCreator(state)
    update = out.update or {}
    stack = update.get("s_stack") or []
    cost = float(update.get("cost") or 0.0)
    s.suggestions = [{"n": i + 1, "text": u.get("user_friendly_text") or u.get("topic", ""),
                      "type": u.get("type", "EDIT"), "instruction": u.get("nano_prompt", "")}
                     for i, u in enumerate(stack)]
    s.record("critique_design", ok=True, detail=f"{len(stack)} suggestion(s)", cost=cost,
             started=started)
    if not stack:
        return "OK: the critique found nothing worth changing."
    return "OK: suggestions:\n" + "\n".join(f"{x['n']}. [{x['type']}] {x['text']}"
                                            for x in s.suggestions)


CRITIC_PROMPT = """\
You are the critic of Kognie, a graphic-design assistant. You review a design and propose
improvements; you never change it. Call critique_design once with the design's purpose (from
the request) and any focus the user named, then present EXACTLY the suggestions the tool
returned -- same number, same order, same numbering -- in plain words, each phrased as a
change they could ask for. Never add suggestions of your own to the list. Add one line saying they
can ask you to apply any of them (e.g. "apply 1 and 3"). If the critique found nothing, say
the design works and why in one or two sentences. Never invent suggestions the tool did not
return, but you may add a one-line overall impression based on the image you see.
"""


def build_critic_agent(model: str = DEFAULT_MODEL):
    return make_agent(init_chat_model(model), [critique_design], system_prompt=CRITIC_PROMPT,
                        middleware=[see_current_image], context_schema=DesignSession,
                        name="critic-agent")


# ------------------------------------------------------------------ Suggestion
@tool
@safe_tool
def suggest_options(runtime: ToolRuntime[DesignSession], contemplated_change: str) -> str:
    """Offer 2-4 options for a specific change the user is considering ("colours for the
    button", "a catchier headline", "where to put the logo"), looking at the current image."""
    from ..common_nodes import specificSuggestionCreator
    s = runtime.context
    started = time.monotonic()
    if not s.current:
        # Options about the brief's content (offer, wording) for a design not made yet.
        raw, cost, _ = vision_json(
            "Offer 2-4 concrete, ready-to-use options for this, as a marketing designer would, "
            f"each one line: {contemplated_change}\nReturn JSON: {{\"options\": [\"...\"]}}", [])
        options = [str(o) for o in raw.get("options") or []]
        s.suggestions = [{"n": i + 1, "text": o, "type": "BRIEF", "instruction": o}
                         for i, o in enumerate(options)]
        s.record("suggest_options", ok=bool(options), detail=contemplated_change[:120], cost=cost,
                 started=started)
        return "OK:\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    v = s.get(None)
    out = specificSuggestionCreator({"suggestion_request": contemplated_change, "prompt": contemplated_change,
                                     "current_image_url": v.url, "cost": 0.0,
                                     "conversation_history": []})
    update = out.update or {}
    text = str(((update.get("conversation_history") or [{}])[-1]).get("content", ""))
    cost = float(update.get("cost") or 0.0)
    options = _numbered(text)
    if options:
        s.suggestions = [{"n": i + 1, "text": o, "type": "EDIT", "instruction": o}
                         for i, o in enumerate(options)]
    s.record("suggest_options", ok=bool(text), detail=contemplated_change[:120], cost=cost,
             started=started)
    return f"OK:\n{text}" if text else "FAILED: no options came back."


SUGGESTION_PROMPT = """\
You are the suggestion agent of Kognie. The user is considering a specific change and wants
options. Call suggest_options with that change, then present the options as a short numbered
list, each something they could ask for as-is, and say they can pick one ("the second one")
or ask you to choose. If the tool says the thing is not visible in the image, say so. You
never apply a change yourself.
"""


def build_suggestion_agent(model: str = DEFAULT_MODEL):
    return make_agent(init_chat_model(model), [suggest_options], system_prompt=SUGGESTION_PROMPT,
                        middleware=[see_current_image], context_schema=DesignSession,
                        name="suggestion-agent")


# ------------------------------------------------------------------- Evaluator
def _vision_tool(s: DesignSession, name: str, prompt: str, urls: list[str]) -> dict:
    started = time.monotonic()
    raw, cost, _ = vision_json(prompt, urls)
    s.record(name, ok=True, detail=json.dumps(raw)[:200], cost=cost, started=started)
    return raw


@tool
@safe_tool
def score_design(runtime: ToolRuntime[DesignSession], brief: Optional[str] = None,
                 version: Optional[int] = None) -> str:
    """Score a design 1-10 overall and per dimension (composition, typography, colour,
    legibility, brief fit, technical quality), pass or fail, and list concrete defects."""
    s = runtime.context
    v = s.get(version)
    raw = _vision_tool(s, "score_design", (
        "Score this marketing design strictly and fairly as a senior art director. "
        f"Brief: {brief or 'unknown; judge general quality'}.\n"
        'Return JSON: {"overall": 1-10, "scores": {"composition": n, "typography": n, "colour": n, '
        '"legibility": n, "brief_fit": n, "technical": n}, "passed": true|false, '
        '"defects": ["specific defect with location"], "strengths": ["..."]}'), [v.url])
    return f"OK: version {v.number}: {json.dumps(raw)}"


@tool
@safe_tool
def compare_versions(runtime: ToolRuntime[DesignSession], version_a: int, version_b: int,
                     criterion: Optional[str] = None) -> str:
    """Compare two versions and say which is better executed (or for a stated criterion), and why."""
    s = runtime.context
    a, b = s.get(version_a), s.get(version_b)
    raw = _vision_tool(s, "compare_versions", (
        f"Image 1 is version {a.number}, image 2 is version {b.number}. Which is better "
        f"{'for: ' + criterion if criterion else 'executed as a design'}? Return JSON: "
        '{"better": 1|2|"tie", "why": "two sentences", "differences": ["..."]}'), [a.url, b.url])
    return f"OK: {json.dumps(raw)}"


@tool
@safe_tool
def verify_change(runtime: ToolRuntime[DesignSession], before_version: int, after_version: int,
                  instruction: str) -> str:
    """Check whether an edit actually happened as asked between two versions, and whether
    anything else got worse."""
    s = runtime.context
    a, b = s.get(before_version), s.get(after_version)
    raw = _vision_tool(s, "verify_change", (
        f"Image 1 is before, image 2 is after this edit: {instruction!r}. Return JSON: "
        '{"applied": true|false, "note": "...", "regressions": ["anything else that got worse"]}'),
        [a.url, b.url])
    return f"OK: {json.dumps(raw)}"


EVALUATOR_PROMPT = """\
You are the image evaluator of Kognie. You judge quality and report defects; you never
suggest redesigns at length and never change the image. Pick the tool: score_design for "rate
it" / "is it good" / quality checks, compare_versions for "which is better", verify_change
for "did the edit work". Report the result plainly: the score or verdict, then the main
defects or reasons in a short list.
"""


def build_evaluator_agent(model: str = DEFAULT_MODEL):
    return make_agent(init_chat_model(model), [score_design, compare_versions, verify_change],
                        system_prompt=EVALUATOR_PROMPT, middleware=[see_current_image],
                        context_schema=DesignSession, name="image-evaluator")
