"""One user turn for an agent, including frontend replies that only carry a selected image.

    result = run_turn(agent, session, history, text="remove the banana")
    if result.needs_image:            # show the frontend's image picker
        ...
    result = run_turn(agent, session, result.history, selected_image=url)   # no text needed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .session import DesignSession


@dataclass
class TurnResult:
    reply: str
    history: list = field(default_factory=list)
    new_messages: list = field(default_factory=list)
    needs_image: bool = False          # the frontend should ask the user to pick an image


def run_turn(agent: Any, session: DesignSession, history: list, *, text: str = "",
             selected_image: Optional[str] = None, recursion_limit: int = 60) -> TurnResult:
    text = (text or "").strip()
    if selected_image:
        session.select_image(selected_image)
        pending = session.pending_request if session.awaiting_image else None
        session.awaiting_image = False
        if not text:
            # A reply with only an image: carry on with what the user asked before.
            text = (f"I've selected the image. Please go ahead with my request: {pending}"
                    if pending else "I've selected this image to work on.")
        else:
            text = f"(I've selected an image in the app.) {text}"
    if not text:
        return TurnResult(reply="", history=history)

    start = len(history) + 1
    out = agent.invoke({"messages": history + [("user", text)]}, context=session,
                       config={"recursion_limit": recursion_limit})
    messages = out["messages"]
    return TurnResult(reply=str(messages[-1].content), history=messages,
                      new_messages=messages[start:], needs_image=session.awaiting_image)
