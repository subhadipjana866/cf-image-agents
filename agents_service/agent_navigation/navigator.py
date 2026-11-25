"""Central navigation only: interpret one input, then pick an agent per task.

This runs the first two steps of the `image_agents` orchestrator, unchanged:
`steps.interpret` (interpretation.txt) and `steps.select` (selection.txt),
with the registry built from our cards in `agent_navigation/cards`. Tasks are
built with the orchestrator's own `_add_tasks`, and candidates come from
`Registry.candidates`, so the hard rules (user-only results, user-performed
tasks only to interface agents) apply exactly as they do in the full graph.

Nothing is dispatched. The result says, for each task, which agent the
orchestrator would hand it to, and why.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from image_agents import steps
from image_agents.config import Config
from image_agents.llm import LLM
from image_agents.orchestrator import _add_tasks
from image_agents.registry import Registry

CARDS_DIR = Path(__file__).parent / "cards"

# One generated poster in focus, shaped like `AssetStore.describe()` output, so
# edit-style inputs ("make it brighter") have something to refer to.
DEFAULT_ASSETS: list[dict] = [
    {
        "asset_id": "a1",
        "latest_version": 1,
        "description": "AI-generated Instagram post for a coffee shop grand opening",
        "produced_by": "image-generator",
        "method": "generative",
        "in_focus": True,
        "text_blocks": [
            {"id": "tb1", "text": "Grand Opening", "role": "headline"},
            {"id": "tb2", "text": "Saturday 10am, 12 Baker Street", "role": "body"},
            {"id": "tb3", "text": "Visit us", "role": "cta"},
        ],
        "layers": [],
    }
]


@dataclass
class RoutedTask:
    id: str
    goal: str
    desired_result: str
    performed_by: str
    depends_on: list[str]
    supports_task_id: Optional[str]
    decision: str                       # assign | missing_inputs | no_capable_agent
    agent_id: Optional[str]
    confidence: Optional[str] = None
    analysis: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    capability_gap: Optional[str] = None
    candidates: list[str] = field(default_factory=list)   # cards the selection step could pick
    constraints: list[str] = field(default_factory=list)
    alternates: list[dict] = field(default_factory=list)


@dataclass
class Route:
    input_type: str
    analysis: str = ""
    tasks: list[RoutedTask] = field(default_factory=list)
    clarification: Optional[dict] = None
    pending_input: Optional[dict] = None     # input_for_pending_task: {task_id, input, agent_id}
    affected_task_id: Optional[str] = None   # amendment / cancellation
    conflicts: list[dict] = field(default_factory=list)

    @property
    def agents(self) -> list[str]:
        """The routing outcome in one list, for comparing against expectations."""
        if self.clarification:
            return ["clarify"]
        if self.input_type == "input_for_pending_task":
            return ["pending"]
        if self.input_type == "cancellation":
            return ["cancel"]
        # missing_inputs still names the agent; readiness would ask for the input.
        return [t.agent_id if t.decision != "no_capable_agent" and t.agent_id else "none"
                for t in self.tasks] or ["none"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"agents": self.agents}


class Navigator:
    def __init__(self, llm: LLM, *, cards_dir: str | Path = CARDS_DIR,
                 config: Optional[Config] = None, disabled: tuple[str, ...] = ()):
        self.llm = llm
        self.config = config or Config()
        registry = Registry.load(cards_dir)
        # Feature flags: a switched-off agent is simply not in the registry.
        self.registry = Registry([c for c in registry.all() if c.id not in disabled])

    def route(self, text: str, *, history: Optional[list[dict]] = None,
              assets: Optional[list[dict]] = None, pending_task: Optional[dict] = None,
              active_tasks: Optional[list[dict]] = None) -> Route:
        """Interpret `text` and select an agent for every task it produces.

        `pending_task` is the orchestrator's pending view ({id, goal, requested,
        options, supporting_tasks}); an extra `agent_id` key, if given, is not
        sent to the model and is reported when the input answers that task. An
        `also_waiting` list holds older tasks still waiting on the user (the pending
        stack); a reply that answers one of them is routed to it.
        """
        assets = DEFAULT_ASSETS if assets is None else assets
        pending_view = None
        if pending_task:
            pending_view = {k: v for k, v in pending_task.items() if k != "agent_id"}
            pending_view.setdefault("supporting_tasks", [])
            pending_view["if_reply_also_asks_for_more"] = (
                "If the input answers this task AND also asks for something new (e.g. '70% off, "
                "and after this make another image with a dog'), it is input_for_pending_task "
                "with the answer in pending_task_input, and the new request goes in tasks.")

        # A reply that is valid JSON but breaks the schema (e.g. input_type
        # "conversation") is asked again once, as malformed JSON already is.
        for attempt in range(2):
            try:
                interp = steps.interpret(
                    self.llm, source="user", text=text, history=history or [], assets=assets,
                    pending_task=pending_view, active_tasks=active_tasks or [], config=self.config)
                break
            except ValidationError:
                if attempt:
                    raise

        route = Route(input_type=interp.input_type, analysis=interp.analysis,
                      affected_task_id=interp.affected_task_id,
                      conflicts=[c.model_dump() for c in interp.conflicts])
        if interp.clarification:
            route.clarification = interp.clarification.model_dump()
            return route
        if interp.input_type == "input_for_pending_task" and pending_task:
            supplied = interp.pending_task_input.input if interp.pending_task_input else text
            # The reply may answer an older waiting task (pending stack), not only the top one.
            waiting = {w["id"] for w in pending_task.get("also_waiting") or []}
            named = interp.pending_task_input.task_id if interp.pending_task_input else None
            task_id = named if named in waiting else pending_task["id"]
            route.pending_input = {"task_id": task_id, "input": supplied,
                                   "agent_id": pending_task.get("agent_id")}
            if not interp.tasks:
                return route
            # The reply also asked for something new ("70% off, and after this make a dog
            # image"): those tasks are routed too and run once the answered task is done.
            for intent in interp.tasks:
                intent.supports_task_id = None
        if interp.input_type == "cancellation" or not interp.tasks:
            return route

        # Same task bookkeeping as the orchestrator's interpret node.
        pending_id = pending_task["id"] if pending_task else None
        tasks: dict[str, dict] = {}
        if pending_task:   # so a supporting task's supports_task_id resolves
            tasks[pending_id] = {"id": pending_id}
        if not (interp.input_type == "supporting_task" and pending_id):
            for intent in interp.tasks:
                intent.supports_task_id = None
        order: list[str] = []
        _, new_ids = _add_tasks(tasks, order, 0, interp.tasks, pending_id=pending_id)
        batch = [tasks[i] for i in new_ids]

        candidates = {
            t["id"]: self.registry.candidates(excluded=t["excluded_agents"],
                                              user_performed=t["performed_by"] == "user",
                                              policy=self.config.policy)
            for t in batch
        }
        selectable = [t for t in batch if candidates[t["id"]]]
        selections = steps.select(
            self.llm, tasks=selectable, candidates={t["id"]: candidates[t["id"]] for t in selectable},
            assets=assets, config=self.config) if selectable else {}

        for t in batch:
            sel = selections.get(t["id"])
            route.tasks.append(RoutedTask(
                id=t["id"], goal=t["goal"], desired_result=t["desired_result"],
                performed_by=t["performed_by"], depends_on=t["depends_on"],
                supports_task_id=t["supports_task_id"],
                decision=sel.decision if sel else "no_capable_agent",
                agent_id=sel.agent_id if sel else None,
                confidence=sel.confidence if sel else None,
                analysis=sel.analysis if sel else "",
                missing_inputs=sel.missing_inputs if sel else [],
                capability_gap=(sel.capability_gap if sel
                                else "no agent in the registry may take this kind of task"),
                candidates=[c.id for c in candidates[t["id"]]],
                constraints=list(t["constraints"]),
                alternates=[a.model_dump() for a in sel.alternates] if sel else [],
            ))
        return route
