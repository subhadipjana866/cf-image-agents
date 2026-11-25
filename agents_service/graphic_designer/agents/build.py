"""The image-generation and image-edit agents (LangChain ``create_agent``).

Each agent is a tool-calling loop: the model reads the request, picks tools by
their descriptions, sees each result, and decides the next step. Tool calls run
one at a time (``parallel_tool_calls=False``) because every edit works on the
image the previous step produced.
"""
from __future__ import annotations

from typing import Any


from langchain.agents.middleware import dynamic_prompt, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from .common import make_agent
from .edit_tools import EDIT_TOOLS
from .generation_tools import GENERATION_TOOLS
from .session import DesignSession

DEFAULT_MODEL = "openai:gpt-5.4"


@wrap_model_call
def one_tool_at_a_time(request, handler):
    """Ask for tool calls one per turn, so each edit sees the previous result."""
    return handler(request.override(
        model_settings={**request.model_settings, "parallel_tool_calls": False}))


def _brief_block(session: DesignSession) -> str:
    b = session.brief
    if not b:
        return "BRIEF: none was provided."
    lines = [f"Request: {b.prompt}",
             f"Design goal: {b.design_goal or '(none: not a catalog goal)'}",
             f"Platform: {b.platform or '(none)'}"]
    if b.aspect_ratio:
        lines.append(f"Aspect ratio: {b.aspect_ratio}")
    if b.field_answers:
        lines += [f"Detail - {k}: {v}" for k, v in b.field_answers.items()]
    lines.append(f"Product photo: {'yes, used ' + (b.photo_mode or 'as reference') if b.photo_url else 'no'}")
    lines.append(f"Logo: {'yes' if b.logo_url and b.logo_mode == 'provided' else 'no'}")
    if b.style:
        lines.append(f"Style: {b.style}")
    return "BRIEF (already collected; do not ask for it again):\n" + "\n".join(lines)


def _versions_block(session: DesignSession) -> str:
    if not session.versions:
        return "IMAGES IN THIS SESSION: none yet."
    rows = [f"- version {v.number} ({v.tool}): {v.instruction[:100]!r}" for v in session.versions]
    return ("IMAGES IN THIS SESSION (the last one is the current image):\n" + "\n".join(rows[-12:]))


GENERATION_PROMPT = """\
You are the image-generation agent of Kognie, an assistant that designs marketing graphics.
You create NEW designs. The brief (goal, platform, copy details, photo, logo, style) has
already been collected and is attached to this run; never ask for it again.

Choose the tool:
- Catalog goal AND platform in the brief -> generate_catalog_design (best text and layout).
- No catalog goal, or a purpose outside the catalog -> generate_freeform_design.
- "more like this", "another version", "3 more options" of an existing design ->
  create_more_variations (with count).
- "regenerate", "try again", "redo it" -> regenerate_design (pass any one-off note).
- Unsure whether a goal or platform is in the catalog -> list_catalog first.
One request normally needs one tool call. Do not call a generation tool twice for the same
request unless the user asked for that many designs.

Results:
- OK: tell the user what you made, briefly, with the image link(s). Pass on any typography
  notices (for example a font that was substituted). Mention when variations are exhausted.
- NEEDS_INPUT: do not guess and do not ask; report plainly which detail is missing so the
  collector can ask for it.
- FAILED: say plainly that it did not work and nothing was created; never claim an image
  exists when it does not.
- BLOCKED_BY_SAFETY: the image services refused the request under their content rules.
  Tell the user that plainly, name the most likely cause in the request (a copyrighted or
  trademarked character, a brand or logo they don't own, a real or famous person, or
  violent or adult content), and suggest an original rewording that would work (e.g.
  "Iron Man suit" -> "a sleek red-and-gold armored suit"). Do not offer to try the same
  request again; ask them how they would like to reword it.

Outside your job (say so kindly in one sentence, and say what you can do instead):
- Changing an existing design (colours, text, objects, crop, resize): the edit agent does that.
- Critique, ratings, fixing AI artifacts, separating layers: other agents handle those.
- Video, animation, printing, shipping, 3D models, anything that is not a still image:
  not supported.
Write like a helpful designer: short, plain, no internal tool names or jargon.
"""

EDIT_PROMPT = """\
You are the image-edit agent of Kognie, an assistant that designs marketing graphics.
You change an EXISTING design. Every tool works on the latest version unless you pass
`version`, and every change makes a new version, so earlier versions are never lost.

The input-collector has already checked the request against the image and settled the
values it could. Carry it out; ask the user only for what is still truly missing.

How to work:
0. The current image is attached to every turn. If, despite the checks, an element the
   request names is not in it, ask the user what they meant (say what you see), then do it.
1. Split the request into individual changes. A single message may hold several
   ("make the headline bigger, background navy, round the corners, then square for
   Instagram").
2. Pick the best tool for each change by its description. Prefer an exact tool
   (crop_image, adjust_tone, rotate_image, add_border...) whenever the change is
   mechanical or numeric; use generative tools for content changes.
3. Order the work:
   a. Content changes first. Batch two to five INDEPENDENT generative changes to
      different parts of the image into one apply_multiple_edits call; use separate
      calls when one change depends on another, or when a specialised tool
      (replace_background, apply_art_style, edit_text_or_graphic...) handles it better.
   b. Then place_logo, if asked.
   c. Then exact edits in this order: tone; crop, rotate, flip, resize; border; round
      corners; opacity last.
   d. Platform resizes, upscale and export last, on the finished design.
   The user's wording ("then", "after that") sets the order only where it changes the
   result they want. When their order would undo an earlier change (cropping after
   adding a border cuts the border off; a border after rounding the corners squares
   them again), use the order that keeps every change and say so in one line.
4. Call one tool at a time and read each result before the next call.
5. Relative amounts ("brighter", "a bit bigger"): pick a sensible value and say which.
   A missing new value ("you choose", or none given): pick one that suits the design and
   say what you picked.
6. Exact text changes need the exact wording. Never invent new copy; if the wording is
   missing, ask the user for it.
7. If two requested changes contradict each other, ask the user which one they want
   before making either. Make the other, non-conflicting changes either way.
8. "undo", "go back", "previous one was better" -> restore_version. Questions about the
   image's size or history -> get_image_info.

Results:
- After a generative edit, glance at the new image: if the change clearly did not happen,
  say so instead of reporting success.
- A tool reply starting OK succeeded. FAILED means nothing changed for that step: say so
  honestly, continue with independent changes, and never claim a failed change was made.
- Finish with a short summary: each change made and the values you chose, which changes
  failed or were skipped and why, and the link to the final version.

Outside your job (say so kindly in one sentence, do the parts you can):
- New designs from scratch, or "more versions/variations": the generation agent does those.
- Cutting a subject out onto a transparent background, or separating elements into layers:
  the layering agent does that.
- Critique ("what's wrong with it"), ratings, or repairing malformed AI hands/faces/text:
  other agents handle those.
- Video, animation, GIFs, printing, 3D, or anything that is not a still image: not supported.
Write like a helpful designer: short, plain, no internal tool names or jargon.
"""


@wrap_model_call
def see_current_image(request, handler):
    """Show the model the current version on every call, so it sees what each edit
    produced (checking the request against the image is the collector's job)."""
    session = request.runtime.context
    v = session.current if session else None
    if v is None:
        return handler(request)
    from .images import to_data_uri
    url = v.url if v.url.startswith(("http://", "https://", "data:")) else to_data_uri(v.url)
    look = HumanMessage(content=[
        {"type": "text", "text": f"[Current image: version {v.number}, shown by the app, "
                                 f"not sent by the user.]"},
        {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
    ])
    return handler(request.override(messages=[*request.messages, look]))


@dynamic_prompt
def generation_prompt(request) -> str:
    session = request.runtime.context
    return f"{GENERATION_PROMPT}\n{_brief_block(session)}\n\n{_versions_block(session)}"


@dynamic_prompt
def edit_prompt(request) -> str:
    return f"{EDIT_PROMPT}\n{_versions_block(request.runtime.context)}"


def _model(name: str) -> Any:
    return init_chat_model(name) if isinstance(name, str) else name


def build_generation_agent(model: str = DEFAULT_MODEL):
    return make_agent(_model(model), GENERATION_TOOLS,
                        middleware=[generation_prompt, one_tool_at_a_time],
                        context_schema=DesignSession,
                        name="image-generator")


def build_edit_agent(model: str = DEFAULT_MODEL):
    return make_agent(_model(model), EDIT_TOOLS,
                        middleware=[edit_prompt, see_current_image, one_tool_at_a_time],
                        context_schema=DesignSession,
                        name="image-editor")
