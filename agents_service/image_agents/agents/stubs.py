"""Placeholder agents.

These exist so the orchestrator can run end to end before your real
sub-graphs are connected. Each returns a plausible outcome and does no real
work. Replace each with a `SubgraphAdapter` around your existing sub-graph,
keeping the same agent id as on its card.
"""
from __future__ import annotations

from PIL import Image

from ..schemas import Outcome, TaskEnvelope
from .base import AgentContext


class StubGenerator:
    agent_id = "image-generator"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        image = Image.new("RGB", (1200, 628), "#d9d4c7")
        ref = ctx.store.new_asset(env.session_id, image, description=env.goal[:80],
                                  made_by=self.agent_id, instruction=env.goal,
                                  method="generative")
        return Outcome(kind="completed", assets=[ref], generative=True,
                       summary="[stub] Generated a placeholder design.")


class StubLayering:
    agent_id = "layering-agent"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        if not env.target_assets:
            return Outcome(kind="failed", reason="no design was given")
        ref = env.target_assets[0]
        w, h = ctx.store.open(ref).size
        layer = {"element": env.goal, "box": [0, 0, w, h], "mask": None, "source": "stub"}
        ctx.store.add_layer(ref, layer)
        return Outcome(kind="completed", result={"layer": layer},
                       summary="[stub] Separated the element as a layer.")


class StubEvaluator:
    agent_id = "image-evaluator"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        report = {"scores": {"overall": 0.8}, "passed": True, "defects": []}
        if env.target_assets:
            ctx.store.attach_report(env.target_assets[0], "evaluation", report)
        return Outcome(kind="completed", result=report,
                       summary="[stub] Quality check passed.")


class StubCritic:
    agent_id = "critic-agent"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        suggestions = ["[stub] Increase contrast between headline and background."]
        return Outcome(kind="completed", result={"suggestions": suggestions},
                       summary="Suggestions: " + " ".join(suggestions))


class StubSlopTerminator:
    agent_id = "slop-terminator"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        return Outcome(kind="completed", result={"defects": []},
                       summary="[stub] No generation artifacts found.")


class StubQuestionHandler:
    agent_id = "question-handler"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        return Outcome(kind="completed", result={"answer": "[stub] answer"},
                       summary="[stub] Here is the information you asked for.")


class StubSuggestion:
    agent_id = "suggestion-agent"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        options = ["[stub] option one", "[stub] option two", "[stub] option three"]
        return Outcome(kind="completed", result={"options": options},
                       summary="Options: " + "; ".join(options))


class StubChitChat:
    agent_id = "chitchat-handler"

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        return Outcome(kind="completed", summary="[stub] Happy to help.")


def default_stubs() -> list:
    return [StubGenerator(), StubLayering(), StubEvaluator(), StubCritic(),
            StubSlopTerminator(), StubQuestionHandler(), StubSuggestion(), StubChitChat()]
