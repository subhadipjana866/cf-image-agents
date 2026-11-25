"""ManualOps: opens an interface for the user to act in.

It performs no change. Its task is complete the moment the interface is
opened; whether the user then acts comes back later as a UI event.

The interface is chosen by matching the task goal against the skills on the
agent's own card, so adding an interface means adding a skill to the card.
The matcher here is a word-overlap score; swap in a small model call if the
list of interfaces grows.
"""
from __future__ import annotations

import re

from ..schemas import Outcome, TaskEnvelope
from .base import AgentContext

_STOP = {
    "the", "a", "an", "to", "of", "on", "in", "for", "so", "can", "it", "its", "their", "them",
    "themselves", "user", "users", "open", "let", "by", "with", "and", "or", "that", "this",
    "v1", "v2", "v3", "v4", "v5", "screen", "tool", "control", "hand", "manual", "manually",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


class ManualOpsAgent:
    agent_id = "manual-ops"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        card = ctx.registry.get(self.agent_id)
        goal = _words(env.goal)
        best, best_score = None, 0
        for skill in card.skills:
            vocabulary = _words(" ".join([skill.summary, skill.interface or "", *skill.examples])
                                .replace("_", " "))
            score = len(goal & vocabulary)
            if score > best_score:
                best, best_score = skill, score
        if best is None or not best.interface:
            return Outcome(kind="failed", reason="there is no interface for this action")

        asset = env.target_assets[0].model_dump() if env.target_assets else None
        action = {
            "type": "open_interface",
            "interface": best.interface,
            "task_id": env.task_id,          # the client echoes this in its UI event
            "asset": asset,
            "prefill": env.constraints,      # values the user mentioned; confirmed in the UI
        }
        label = best.interface.replace("_", " ")
        return Outcome(kind="completed", client_action=action,
                       result={"interface": best.interface},
                       summary=f"Opened the {label} for you.")
