"""External agents.

An external agent is called through a `transport` that you supply: a function
that sends one request to the remote agent and returns its reply. This module
does not implement the A2A wire protocol. Write the transport against the
A2A client library you use, and map its task states onto the reply shape
below.

What is enforced here, whatever the transport:

* Only the task fields and asset versions named in the selection step's
  `external_handoff.share` leave the system. If none are named, only the
  goal is sent.
* The external agent never addresses the user. Its request for input comes
  back as an `input_required` outcome, which the orchestrator relays.
* Returned images are written to the asset store by this side.
"""
from __future__ import annotations

from typing import Any, Callable

from ..schemas import Outcome, TaskEnvelope
from .base import AgentContext

# request: {"goal": str, "constraints": [...], "inputs": {...},
#           "assets": [{"asset_id", "version", "path"}], "resume": {...} | None}
# reply:   {"state": "completed" | "input-required" | "failed",
#           "text": str, "options": [...], "image_path": str | None,
#           "data": {...}, "reason": str, "retryable": bool}
Transport = Callable[[dict], dict]


class ExternalAgent:
    def __init__(self, agent_id: str, transport: Transport, share: list[str] | None = None):
        self.agent_id = agent_id
        self._transport = transport
        self._share = set(share or [])

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        request: dict[str, Any] = {"goal": env.goal, "resume": env.resume, "assets": []}
        if "constraints" in self._share:
            request["constraints"] = env.constraints
        if "inputs" in self._share:
            request["inputs"] = env.inputs
        if "assets" in self._share:
            request["assets"] = [
                {**ref.model_dump(), "path": str(ctx.store.get(ref).path)}
                for ref in env.target_assets
            ]

        reply = self._transport(request)
        state = reply.get("state")
        if state == "input-required":
            return Outcome(kind="input_required", request=reply.get("text", ""),
                           options=reply.get("options") or [])
        if state != "completed":
            return Outcome(kind="failed", reason=reply.get("reason") or "the external agent failed",
                           retryable=bool(reply.get("retryable")))

        assets = []
        if reply.get("image_path"):
            if env.target_assets:
                assets = [ctx.store.new_version(env.target_assets[0], reply["image_path"],
                                                made_by=self.agent_id, instruction=env.goal,
                                                method="generative")]
            else:
                assets = [ctx.store.new_asset(env.session_id, reply["image_path"],
                                              description=env.goal[:80], made_by=self.agent_id,
                                              instruction=env.goal)]
        # Results from outside are treated as generative, so they pass the quality gate.
        return Outcome(kind="completed", assets=assets, generative=bool(assets),
                       result=reply.get("data") or {}, summary=reply.get("text", ""))
