"""The agent contract.

Every agent, internal or external, is called the same way: it receives a
`TaskEnvelope` and returns one `Outcome`. An agent never calls another agent
or the user. If it needs something, it says so in its outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..assets import AssetStore
from ..config import Config
from ..llm import LLM
from ..registry import Registry
from ..schemas import Outcome, TaskEnvelope


@dataclass
class AgentContext:
    store: AssetStore
    llm: LLM
    registry: Registry
    config: Config


class AgentHandler(Protocol):
    agent_id: str

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome: ...


class SubgraphAdapter:
    """Wraps an existing LangGraph sub-graph so it honours the agent contract.

    The sub-graph itself is unchanged. You supply two small functions:

    * `to_state(envelope, ctx)` builds the sub-graph's own input state.
    * `from_state(final_state, envelope, ctx)` maps its final state to one of
      the four outcomes, storing any image it produced in `ctx.store`.

    For a sub-graph that must pause, return an `input_required` or
    `assistance_required` outcome carrying whatever it needs to continue in
    `agent_state`. The orchestrator hands that back in `envelope.agent_state`
    together with `envelope.resume` on the next step.
    """

    def __init__(
        self,
        agent_id: str,
        graph: Any,
        to_state: Callable[[TaskEnvelope, AgentContext], dict],
        from_state: Callable[[dict, TaskEnvelope, AgentContext], Outcome],
    ):
        self.agent_id = agent_id
        self._graph = graph
        self._to_state = to_state
        self._from_state = from_state

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        final = self._graph.invoke(self._to_state(env, ctx))
        return self._from_state(final, env, ctx)
