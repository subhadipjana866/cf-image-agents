"""Small agents: ManualOps, ChitChat Handler, Question Handler.

Each is a LangChain tool-calling agent like the generation and edit agents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional


from langchain.agents.middleware import dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool

from .common import make_agent, safe_tool
from .session import DesignSession

DEFAULT_MODEL = "openai:gpt-5.4"
_MANUAL_OPS = Path(__file__).resolve().parents[1] / "manual_operations.json"
CARDS_DIR = Path(__file__).resolve().parents[2] / "agent_navigation" / "cards"
NOT_BUILT = {"slop-terminator": "finding and fixing AI artifacts (coming later)"}


def _controls() -> list[dict]:
    return json.loads(_MANUAL_OPS.read_text())["operations"]


def _capabilities() -> str:
    from image_agents.registry import Registry
    lines = []
    for card in Registry.load(CARDS_DIR).all():
        skills = "; ".join(s.summary for s in card.skills)
        note = f" [not available yet: {NOT_BUILT[card.id]}]" if card.id in NOT_BUILT else ""
        lines.append(f"- {card.name}: {' '.join(card.description.split())} Skills: {skills}{note}")
    lines.append("- Not supported at all: video, animation, GIFs, printing or shipping, 3D models, "
                 "anything that is not a still image.")
    return "\n".join(lines)


# ------------------------------------------------------------------ ManualOps
@tool
@safe_tool
def list_canvas_controls(runtime: ToolRuntime[DesignSession]) -> str:
    """List the canvas controls the user can set themselves in the app, with what each does."""
    return "\n".join(f"- {c['id']} ({c['mode']}): {c['description']}" for c in _controls())


@tool
@safe_tool
def open_interface(runtime: ToolRuntime[DesignSession],
                   interface: Literal["canvas_control", "crop_tool", "corner_radius_control",
                                      "transform_handles", "text_editor", "box_picker",
                                      "image_picker", "add_funds", "account_settings"],
                   control: Optional[str] = None, value: Optional[str] = None,
                   target: Optional[str] = None, amount: Optional[str] = None,
                   section: Optional[str] = None) -> str:
    """Ask the app to open an interface where the USER makes the change or account action.

    - canvas_control: one of the canvas controls (see list_canvas_controls); `control` is its
      id, `value` a starting value if the user gave one ("+20", "80%").
    - crop_tool, corner_radius_control, transform_handles (move/resize an element: `target`),
      text_editor (`target` = which text block), box_picker (draw boxes for layers),
      image_picker (choose which image to work on).
    - add_funds (`amount` if given), account_settings (`section`: plan, billing, profile).
    Nothing is changed by opening it; the user acts in it.
    """
    session = runtime.context
    if interface == "canvas_control":
        ids = {c["id"] for c in _controls()}
        if control not in ids:
            return f"FAILED: unknown control {control!r}; controls are {', '.join(sorted(ids))}"
    action = {"type": "open_interface", "interface": interface,
              "prefill": {k: v for k, v in {"control": control, "value": value, "target": target,
                                             "amount": amount, "section": section}.items() if v},
              "version": session.current.number if session.current else None}
    session.client_actions.append(action)
    session.record("open_interface", ok=True, detail=json.dumps(action))
    return f"OK: the app will open {interface} with {action['prefill'] or 'no prefill'}."


MANUAL_OPS_PROMPT = """\
You are ManualOps in Kognie, a graphic-design assistant. You open an interface in the app so
the USER can make a change themselves, or do an account action (add funds, plan, billing).
You never make the change and never claim it is done.
- Pick the right interface with open_interface; prefill any value the user gave.
- For canvas controls (brightness, contrast, exposure, temperature, tint, opacity, corner
  radius, border width or colour, rotation, flip, width, height, position, crop, shrink),
  use interface canvas_control with the control id; list_canvas_controls if unsure.
- Reply in one or two short sentences: what you opened and what they can do there.
- If the request is something the app should do for them rather than by hand, say that you
  can do it for them instead if they ask (do not do it yourself).
"""


def build_manual_ops_agent(model: str = DEFAULT_MODEL):
    return make_agent(init_chat_model(model), [list_canvas_controls, open_interface],
                        system_prompt=MANUAL_OPS_PROMPT, context_schema=DesignSession,
                        name="manual-ops")


# ------------------------------------------------------------ ChitChat Handler
@tool
@safe_tool
def explain_capabilities(runtime: ToolRuntime[DesignSession]) -> str:
    """What Kognie can and cannot do, read from the agent cards. Use for "what can you do?"."""
    return _capabilities()


CHITCHAT_PROMPT = """\
You are Kognie, a friendly graphic-design assistant, handling small talk: greetings, thanks,
reactions, jokes, how-are-you. Reply warmly in one or two short sentences, then, when it
fits, invite them to create or improve a design. Do not invent facts about their designs.
If they ask what you can do, use explain_capabilities and summarise briefly.
If the message turns out to be a real request, say what you would do and ask them to go
ahead (another part of the system will handle it).
"""


def build_chitchat_agent(model: str = DEFAULT_MODEL):
    return make_agent(init_chat_model(model), [explain_capabilities],
                        system_prompt=CHITCHAT_PROMPT, context_schema=DesignSession,
                        name="chitchat-handler")


# ------------------------------------------------------------ Question Handler
def _ragbot(question: str, session: DesignSession, pricing: bool) -> str:
    from ..common_nodes import askRagbot   # today's ragbot entry point (no interrupt)
    ctx = (f"- Process the user is in right now: chat with the design assistant\n"
           f"- Spent so far in this session (USD): {round(session.cost, 4)}")
    answer, cost = askRagbot(question, ctx, pricing=pricing)
    session.record("ragbot_pricing" if pricing else "ragbot_product", ok=bool(answer),
                   detail=question[:120], cost=cost)
    return answer or "NO_ANSWER: the knowledge base had nothing on this."


@tool
@safe_tool
def rag_pricing(runtime: ToolRuntime[DesignSession], question: str) -> str:
    """Answer a pricing question from Kognie's pricing knowledge base (plans, credits, what a
    generation or edit costs, refunds). Use for any question about price or billing."""
    return _ragbot(question, runtime.context, pricing=True)


@tool
@safe_tool
def rag_product(runtime: ToolRuntime[DesignSession], question: str) -> str:
    """Answer a question about Kognie itself (features, how something works, supported formats,
    models, limits) from the product knowledge base."""
    return _ragbot(question, runtime.context, pricing=False)


@tool
@safe_tool
def describe_current_image(runtime: ToolRuntime[DesignSession]) -> str:
    """Facts about the current image and its history: versions, how each was made, sizes."""
    from .images import has_transparency, load_image
    s = runtime.context
    if not s.current:
        return "There is no image in this session."
    img = load_image(s.current.url)
    rows = "\n".join(f"- version {v.number}: {v.tool} by {v.made_by} ({v.instruction[:90]!r})"
                     for v in s.versions)
    return (f"Current: version {s.current.number}, {img.width}x{img.height}, {img.format}, "
            f"transparent={has_transparency(img)}\n{rows}")


@tool
@safe_tool
def explain_session_costs(runtime: ToolRuntime[DesignSession]) -> str:
    """What each step in this session cost, and the total."""
    s = runtime.context
    rows = [f"- {t['tool']}: ${t['cost']:.4f}" for t in s.trace if t.get("cost")]
    return "\n".join(rows + [f"Total: ${s.cost:.4f}"]) if rows else "Nothing has been charged yet."


@tool
@safe_tool
def explain_last_actions(runtime: ToolRuntime[DesignSession], count: int = 5) -> str:
    """The most recent steps the system took and their results, to explain why something
    happened ("why did you change the background?")."""
    s = runtime.context
    rows = [f"- {t['tool']} ({'ok' if t['ok'] else 'failed'}): {t['detail'][:200]}"
            for t in s.trace[-max(1, min(count, 20)):]]
    return "\n".join(rows) or "No steps yet."


QUESTION_PROMPT = """\
You are the question handler of Kognie, a graphic-design assistant. You give information;
you never change, create or assess a design.
- Pricing, plans, credits, billing -> rag_pricing. Kognie's features and how it works ->
  rag_product. What it can and cannot do in this chat -> explain_capabilities.
- The current image or its history -> describe_current_image. What things cost in this
  session -> explain_session_costs. Why something happened -> explain_last_actions.
- General graphic-design knowledge (kerning, contrast, hierarchy...) -> answer yourself,
  briefly and accurately.
- If a knowledge-base tool has no answer, say you don't have that information rather than
  guessing; never invent prices.
Answer in a few short sentences, plainly. If they seem to want something done, say you can
do it if they ask.
"""


def build_question_agent(model: str = DEFAULT_MODEL):
    return make_agent(init_chat_model(model),
                        [rag_pricing, rag_product, explain_capabilities, describe_current_image,
                         explain_session_costs, explain_last_actions],
                        system_prompt=QUESTION_PROMPT, context_schema=DesignSession,
                        name="question-handler")


# ---------------------------------------------------------------- Custom Agent
CUSTOM_PROMPT = """\
You are the custom agent of Kognie, a graphic-design assistant. You handle requests, or
parts of requests, that none of the specialist agents cover, by answering directly.
The current image, if there is one, is shown to you.

What you do:
- Write text that goes with a design: captions, hashtags, alt text, ad copy, post
  descriptions, emails, translations of copy. Base it on what is actually in the image
  and the conversation; never invent facts like prices or dates.
- Brainstorm ideas and plans (campaign ideas, posting schedules, offer ideas).
- Answer general questions outside graphic design, briefly and accurately.
- When the request asks for something this system cannot do at all (video, animation,
  GIFs, sound, printing or shipping, 3D models, booking, payments), say plainly and kindly
  that it isn't supported, then offer the closest thing that is (for example a still
  image with a motion-style effect, a set of frames, or a print-ready file).

Rules:
- You return text only. Never claim you created, edited, exported or sent an image or file.
  If an image change would help, say they can ask for it and it will be done.
- Keep it short and useful; use a list only for several parallel items.
"""


def build_custom_agent(model: str = DEFAULT_MODEL):
    from .build import see_current_image
    return make_agent(init_chat_model(model), [], system_prompt=CUSTOM_PROMPT,
                        middleware=[see_current_image], context_schema=DesignSession,
                        name="custom-agent")
