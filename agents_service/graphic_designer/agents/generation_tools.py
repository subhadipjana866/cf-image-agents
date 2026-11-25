"""Tools of the image-generation agent.

Each tool runs one of today's generation pipelines, unchanged and in order,
on the brief already collected in ``runtime.context.brief``:

* catalog design: ``v32Generation`` (v33: GROUND, SUBJECT + COPYWRITER,
  TYPOGRAPHER / ART DIRECTOR / COMPOSITOR, TYPESET, DIRECTOR, SLOP GUARD,
  RENDER, TEXT QA) then ``logoCompositor``
* freeform design: ``runGptImageModel`` then ``logoCompositor``
* variations: ``v32CreateMore`` then ``logoCompositor``

Nothing here asks the user. A gap the pipeline cannot fill comes back as
NEEDS_INPUT for the input-collector.
"""
from __future__ import annotations

import time
from typing import Optional

from langchain.tools import ToolRuntime, tool

from .common import base_state, describe, safe_tool
from .runner import run_chain
from .session import DesignSession

AGENT_ID = "image-generator"

# How the image services word a refusal (OpenAI's safety system; Gemini returns no image).
_SAFETY_WORDS = ("safety system", "content_policy", "moderation_blocked", "safety_violation",
                 "content policy")


def _safety_block(state: dict) -> bool:
    """True when the render was refused on safety grounds, not a technical failure: the
    primary renderer said so, and no renderer produced an image."""
    attempts = state.get("v33_render_attempts") or []
    details = [str(a.get("detail") or "").lower() for a in attempts if isinstance(a, dict)]
    return any(w in d for d in details for w in _SAFETY_WORDS)


def _bridge():
    from .. import v33_bridge   # loads the 170-combo catalog on first use
    return v33_bridge


def _finish(session: DesignSession, chain, tool_name: str, instruction: str,
            started: float, kind_note: str = "") -> str:
    st = chain.state
    cost = float(st.get("cost") or 0.0)
    if chain.outcome == "blocked":
        if chain.stopped_at == "v32NeedsInputAsk":
            question = st.get("v32_question") or "the brief is missing details"
            reply = (f"NEEDS_INPUT: the design pipeline cannot continue without this from the "
                     f"user: {question}")
        else:
            reply = f"NEEDS_INPUT: the pipeline stopped at {chain.stopped_at}, which needs the user"
        session.record(tool_name, ok=False, detail=reply, cost=cost, started=started,
                       path=chain.path)
        return reply
    if chain.outcome != "done" and _safety_block(st):
        reply = ("BLOCKED_BY_SAFETY: the image services refused to create this image under "
                 "their content rules. Nothing was created.")
        session.record(tool_name, ok=False, detail=reply, cost=cost, started=started,
                       path=chain.path, blocked="safety")
        return reply
    if chain.outcome != "done":
        reply = (f"FAILED: every model in the {tool_name} pipeline failed "
                 f"(stopped at {chain.stopped_at}). Nothing was created.")
        session.record(tool_name, ok=False, detail=reply, cost=cost, started=started,
                       path=chain.path)
        return reply

    urls = [u for u in str(st.get("current_image_url") or "").split("|") if u]
    models = st.get("image_models") or {}
    parent = session.current.number if session.current and tool_name == "create_more_variations" else None
    made = []
    for url in urls:
        made.append(session.add_version(
            url, made_by=AGENT_ID, tool=tool_name, instruction=instruction, parent=parent,
            model=models.get(url), meta={"text_layers": st.get("text_layers"),
                                         "design_goal": st.get("design_goal"),
                                         "platform": st.get("platform_canonical")}))
    # Keep what create-more needs for the next variation.
    if st.get("v32_prev"):
        session.v33_prev = st["v32_prev"]
    session.lineage = list(st.get("v32_lineage") or session.lineage)
    session.lineage_urls = list(st.get("v32_lineage_urls") or session.lineage_urls)

    notices = ((st.get("text_layers") or {}).get("notices") or [])
    lines = [f"OK: created {len(made)} image(s){kind_note}, ${cost:.4f}:"]
    lines += [f"- {describe(v)}" for v in made]
    if notices:
        lines.append("Typography notices to pass on: " + "; ".join(map(str, notices)))
    if st.get("v32_exhausted"):
        lines.append("Note: every composition in the menu has now been used; "
                     "further variations will repeat earlier layouts.")
    session.record(tool_name, ok=True, detail=lines[0], cost=cost, started=started,
                   path=chain.path, versions=[v.number for v in made])
    return "\n".join(lines)


@tool
@safe_tool
def list_catalog(runtime: ToolRuntime[DesignSession]) -> str:
    """List the design goals and platforms the catalog pipeline supports.

    Use when you need to know whether a requested goal or platform has a
    catalog layout (then use generate_catalog_design) or not (then use
    generate_freeform_design), or when the user asks what kinds of designs or
    sizes are supported. Read-only and free.
    """
    b = _bridge()
    return ("Goals: " + "; ".join(b.v33.goals()) + "\nPlatforms: " + "; ".join(b.v33.platforms()))


@tool
@safe_tool
def generate_catalog_design(runtime: ToolRuntime[DesignSession]) -> str:
    """Create a new design with the full catalog pipeline (copy, typography, layout, render).

    Use when the brief names a design goal AND platform from the catalog
    (e.g. "Festive / Seasonal Sale" on "Instagram Portrait Post"). This is the
    highest-quality path: real fonts, editable text layers, layout rules for
    that goal and platform. If the pair is not in the catalog the pipeline
    itself switches to the freeform renderer.

    Do not use for variations of an existing design (create_more_variations)
    or for goals outside the catalog (generate_freeform_design).
    """
    session = runtime.context
    if not session.brief:
        return "FAILED: no brief was provided; the input-collector must supply one."
    if not session.brief.design_goal or not session.brief.platform:
        return ("FAILED: the brief has no catalog design goal and platform. Use "
                "generate_freeform_design, or ask for them.")
    started = time.monotonic()
    chain = run_chain("generation", base_state(session))
    return _finish(session, chain, "generate_catalog_design", session.brief.prompt, started)


@tool
@safe_tool
def generate_freeform_design(runtime: ToolRuntime[DesignSession],
                             num_images: Optional[int] = None) -> str:
    """Create a new design for any purpose outside the catalog, in one render.

    Use when the request is not one of the catalog goals (a wedding invitation,
    a menu, a certificate, an illustration...) or no platform is given. Uses the
    brief's aspect ratio, style, photo and logo. `num_images` (1-4) overrides
    how many alternatives to render at once.

    Do not use when the brief has a catalog goal and platform
    (generate_catalog_design gives better text and layout there).
    """
    session = runtime.context
    if not session.brief:
        return "FAILED: no brief was provided; the input-collector must supply one."
    started = time.monotonic()
    state = base_state(session) | {"is_freeform": True}
    if num_images:
        state["num_images"] = max(1, min(int(num_images), 4))
    chain = run_chain("runGptImageModel", state)
    return _finish(session, chain, "generate_freeform_design", session.brief.prompt, started)


@tool
@safe_tool
def create_more_variations(runtime: ToolRuntime[DesignSession], count: int = 1,
                           version: Optional[int] = None) -> str:
    """Make new variations of an existing design: same brief and text, a new composition.

    Use for "more like this", "another version", "3 more options", "a different
    layout". Each variation is a new image; the original is unchanged. Works on
    the latest version unless `version` is given, including uploaded designs
    (the brief is read back from the image). `count` is 1-4.

    Do not use when the user wants a specific change to the design (that is
    editing), or a fresh take on the same brief from scratch (regenerate_design).
    """
    session = runtime.context
    source = session.get(version)
    count = max(1, min(int(count), 4))
    replies = []
    for i in range(count):
        started = time.monotonic()
        state = base_state(session) | {"current_image_url": source.url,
                                       "last_generated_image_url": source.url}
        chain = run_chain("v32CreateMore", state)
        reply = _finish(session, chain, "create_more_variations", f"variation of v{source.number}",
                        started, kind_note=f" (variation {i + 1} of {count})")
        replies.append(reply)
        if not reply.startswith("OK"):
            break
    return "\n".join(replies)


@tool
@safe_tool
def regenerate_design(runtime: ToolRuntime[DesignSession], extra_instruction: str = "") -> str:
    """Run the same brief again from scratch for a fresh take.

    Use for "regenerate", "try again", "redo it", optionally with a one-off
    instruction such as "brighter colours this time" in `extra_instruction`.
    Uses the catalog pipeline when the brief has a catalog goal and platform,
    otherwise the freeform renderer.

    Do not use for variations that keep the text and change the layout
    (create_more_variations), or for edits to the existing image.
    """
    session = runtime.context
    if not session.brief:
        return "FAILED: no brief to regenerate from."
    started = time.monotonic()
    state = base_state(session)
    if extra_instruction.strip():
        state["prompt"] = (f"{session.brief.prompt}\n\nADDITIONAL INSTRUCTION FOR THIS RUN: "
                           f"{extra_instruction.strip()}")
    start = "generation" if session.brief.design_goal and session.brief.platform else "runGptImageModel"
    chain = run_chain(start, state)
    return _finish(session, chain, "regenerate_design", state["prompt"], started)


GENERATION_TOOLS = [list_catalog, generate_catalog_design, generate_freeform_design,
                    create_more_variations, regenerate_design]
