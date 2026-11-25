"""Shared pieces for agent tools: the node state built from a session, and replies.

Tools never raise into the agent. A failure comes back as a "FAILED: ..." string
the agent can explain to the user; a gap only the user can fill comes back as
"NEEDS_INPUT: ...". Every outcome is also recorded in ``session.trace``.
"""
from __future__ import annotations

import functools
import time
import traceback
from typing import Any, Callable

from .images import to_data_uri
from .session import DesignSession, Version


def logo_data_uri(session: DesignSession) -> str | None:
    brief = session.brief
    src = brief.logo_url if brief and brief.logo_url else None
    if not src and session.brand_dna:
        logos = session.brand_dna.get("logo") or []      # fetch_brand_dna: list of logo URLs
        src = logos[0] if isinstance(logos, list) and logos else None
    return to_data_uri(src) if src else None


def base_state(session: DesignSession) -> dict[str, Any]:
    """The GraphicDesigner state the existing nodes read, built from the session."""
    b = session.brief
    current = session.current
    state: dict[str, Any] = {
        "name": "main_graph",
        "user_id": session.user_id,
        "session_id": session.session_id,
        "cost": 0.0,
        "conversation_history": [],
        "brand_dna": session.brand_dna,
        "brand_name": session.brand_name,
        "brand_id": None,
        "current_image_url": current.url if current else "",
        "image_models": {},
        "generation_count": 1,
        "edit_logo_mode": "none",
        "edit_reference_images": [],
        "attached_images": [],
        "v32_prev": session.v33_prev,
        "v32_lineage": list(session.lineage),
        "v32_lineage_urls": list(session.lineage_urls),
        "model": "gpt-image",
        "num_images": 1,
        "style": None,
    }
    if b:
        details = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in b.field_answers.items()
                            if str(v).strip() and v != "skipped")
        state.update({
            "prompt": b.prompt,
            "clarified_intent": b.prompt + (f"\n\nDetails provided:\n{details}" if details else ""),
            "field_answers": dict(b.field_answers),
            "design_goal": b.design_goal or "",
            "platform": b.platform or "",
            "platform_canonical": b.platform or "",
            "aspect_ratio": b.aspect_ratio or "1:1",
            "style": b.style,
            "num_images": max(1, min(int(b.num_images or 1), 4)),
            "photo_mode": b.photo_mode if b.photo_url else None,
            "logo_mode": b.logo_mode,
            "is_freeform": not b.design_goal,
        })
        if b.photo_url:
            state["attached_images"] = [to_data_uri(b.photo_url)]
        logo = logo_data_uri(session) if b.logo_mode == "provided" else None
        if logo:
            state["logo_asset"] = logo
            # Same rule as intentClarifier: a photo means the logo rides along as a
            # reference in one call; otherwise it is pasted in afterwards.
            state["logo_insertion_mode"] = "single_call" if b.photo_url else "strict_paste"
        else:
            state["logo_mode"] = "omitted" if b.logo_mode == "provided" else b.logo_mode
    return state


def meta_state(session: DesignSession, parent: Version | None) -> dict[str, Any]:
    """What ``image_metadata.tagged`` reads, for images a tool makes itself."""
    state = base_state(session)
    state["current_image_url"] = parent.url if parent else ""
    return state


def describe(v: Version) -> str:
    model = f", {v.model}" if v.model else ""
    return f"version {v.number} ({v.tool}{model}): {v.url}"


def safe_tool(fn: Callable[..., str]) -> Callable[..., str]:
    """Turn any exception into a FAILED reply and record it, so the agent can explain it."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        started = time.monotonic()
        runtime = kwargs.get("runtime")
        try:
            return fn(*args, **kwargs)
        except LookupError as err:          # bad version number, no image yet
            reply = f"FAILED: {err}"
        except Exception as err:            # noqa: BLE001 - reported to the agent, not raised
            traceback.print_exc()
            reply = f"FAILED: {type(err).__name__}: {err}"
        if runtime is not None:
            runtime.context.record(fn.__name__, ok=False, detail=reply, started=started)
        return reply
    return wrapper


# ------------------------------------------------------------------ asking the user
# Any agent may stop and ask the user. ask_user raises a LangGraph interrupt; the run is
# checkpointed under the task's thread, the chat shows the question, and the orchestrator
# resumes the same run with the reply (Command(resume=...)) once it decides the reply
# answers this task. Nothing before the interrupt runs again.
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

# One saver for the process; every run uses its own thread id ("<chat>:<task>:<attempt>")
# and the chat deletes the thread when the task finishes.
CHECKPOINTER = InMemorySaver()

ASK_RULE = """
Asking the user: when you truly need something only the user can give to do this job well
(a choice between real alternatives, a value you cannot sensibly pick yourself, a request
with two reasonable readings, a thing they named that is not in the image), call ask_user
with one short, plain question, plus options when there are a few clear choices. The work
pauses until they reply, and their reply comes back as the tool result. Never ask for
something already given in the request or the brief, and never ask for permission to do
what was asked."""


@tool
def ask_user(question: str, options: list[str] | None = None, multiple: bool = False,
             fields: list[dict] | None = None, allow_free_text: bool = True) -> str:
    """Ask the user one question and wait for their reply.

    Use only when you cannot do the job well without their answer.
    - `options`: two to six short choices when the answer is one of a few (shown as
      buttons). `multiple=True` lets them pick several.
    - `fields`: when you need typed values, a list of {"id", "label", "type"} with type one
      of text, textarea, number, color, url, date, select (select also takes "options").
    - `allow_free_text`: whether they may type their own answer instead.
    Returns what the user replied.
    """
    reply = interrupt({"question": question, "options": list(options or []),
                       "multiple": bool(multiple), "fields": list(fields or []),
                       "allow_free_text": bool(allow_free_text)})
    return f"USER REPLIED: {reply}"


@wrap_model_call
def ask_rule(request, handler):
    """Append the asking rule to whatever system prompt the agent has (static or dynamic)."""
    current = request.system_message.content if request.system_message else ""
    return handler(request.override(system_message=SystemMessage(content=f"{current}\n{ASK_RULE}")))


def make_agent(model: Any, tools: list, *, middleware: list | tuple = (), **kwargs: Any):
    """create_agent with ask_user, the asking rule and the shared checkpointer."""
    return create_agent(model, [*tools, ask_user], middleware=[*middleware, ask_rule],
                        checkpointer=CHECKPOINTER, **kwargs)
