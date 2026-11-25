"""The entry point your application calls.

    assistant = Assistant(llm=..., workdir="./assets")
    reply = assistant.send("session-1", "Make the headline bigger")
    reply = assistant.ui_event("session-1", "action_completed", task_id="T3", image=path)

`reply` is a dict: text, assets (references), client_actions (interfaces for
the client to open) and request (what the user is being asked, if anything).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from .agents import ImageEditorAgent, ManualOpsAgent, default_stubs
from .agents.base import AgentHandler
from .assets import AssetStore
from .config import Config
from .llm import LLM
from .orchestrator import build_orchestrator
from .registry import Registry
from .schemas import AssetRef


class Assistant:
    def __init__(
        self,
        llm: LLM,
        workdir: str | Path,
        *,
        registry: Optional[Registry] = None,
        agents: Optional[list[AgentHandler]] = None,
        config: Optional[Config] = None,
        checkpointer: Any = None,
    ):
        self.config = config or Config()
        self.registry = registry or Registry.load()
        self.store = AssetStore(workdir)
        # Real agents replace the stubs by using the same agent id.
        handlers: dict[str, AgentHandler] = {h.agent_id: h for h in default_stubs()}
        for handler in [ImageEditorAgent(), ManualOpsAgent(), *(agents or [])]:
            handlers[handler.agent_id] = handler
        self.agents = handlers
        # InMemorySaver loses state on restart. Use a durable checkpointer
        # (for example PostgresSaver) in production: users may answer hours later.
        self.graph = build_orchestrator(
            llm=llm, registry=self.registry, store=self.store, agents=handlers,
            config=self.config, checkpointer=checkpointer or InMemorySaver())

    def _thread(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id},
                "recursion_limit": self.config.limits.graph_recursion_limit}

    def send(self, session_id: str, text: str) -> dict:
        """Handle one user message."""
        state = self.graph.invoke(
            {"session_id": session_id, "source": "user", "input": text},
            self._thread(session_id))
        return state["reply"]

    def add_image(self, session_id: str, image: Image.Image | str | Path,
                  description: str) -> AssetRef:
        """Register an image the user uploaded. It becomes the focus asset."""
        ref = self.store.new_asset(session_id, image, description=description,
                                   made_by="user", method="upload")
        self.graph.update_state(self._thread(session_id),
                                {"session_id": session_id, "focus": ref.model_dump()})
        return ref

    def ui_event(self, session_id: str, event_type: str, *, task_id: str,
                 image: Image.Image | str | Path | None = None) -> dict:
        """Report what the user did in an interface that ManualOps opened.

        `event_type` is "action_completed" or "interface_closed". For a
        completed edit, pass the resulting image; it is stored as a new
        version made by the user and becomes the focus asset.
        """
        event: dict = {"type": event_type, "task_id": task_id}
        if event_type == "action_completed" and image is not None:
            snapshot = self.graph.get_state(self._thread(session_id)).values
            task = (snapshot.get("tasks") or {}).get(task_id) or {}
            base = (task.get("target_assets") or [snapshot.get("focus")])[0]
            if base:
                parent = self.store.latest(base["asset_id"])
                ref = self.store.new_version(parent, image, made_by="user",
                                             tool=(task.get("result") or {}).get("data", {})
                                             .get("interface"), method="manual")
                event["asset"] = ref.model_dump()
        state = self.graph.invoke(
            {"session_id": session_id, "source": "ui_event", "event": event},
            self._thread(session_id))
        return state["reply"]

    def tasks(self, session_id: str) -> dict:
        """Current task table for a session, for inspection and tracing."""
        return self.graph.get_state(self._thread(session_id)).values.get("tasks") or {}
