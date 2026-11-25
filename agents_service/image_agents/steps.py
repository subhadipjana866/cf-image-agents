"""The three model-backed orchestrator steps: interpret, select, readiness.

Each renders a prompt file, makes one model call and validates the reply.
None of them changes state; the orchestrator graph applies the results.

`select` and `check_readiness` take a list of tasks and make one call for
all of them. If the reply leaves a task out, that task alone is asked about
again, so a weak batched reply costs an extra call and never a wrong result.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from .config import Config
from .llm import LLM
from .registry import Card
from .schemas import Interpretation, Readiness, Selection

PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def _template(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text()


def render(name: str, **values: Any) -> str:
    """Fill {{placeholders}}. Non-string values are rendered as JSON."""
    text = _template(name)
    for key, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        text = text.replace("{{" + key + "}}", value)
    return text


def _history_block(history: list[dict], turns: int) -> str:
    recent = history[-2 * turns:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent) or "(none)"


def _items(raw: dict, key: str, single_marker: str, tasks: list[dict]) -> list[dict]:
    """The per-task entries of a batched reply.

    Also accepts the bare single-task shape (no wrapping list) when only one
    task was sent, which small models sometimes return.
    """
    items = raw.get(key)
    if items is None and single_marker in raw and len(tasks) == 1:
        items = [raw]
    if len(tasks) == 1 and items and len(items) == 1:
        items = [{**items[0], "task_id": tasks[0]["id"]}]
    return [i for i in items or [] if isinstance(i, dict)]


def interpret(
    llm: LLM,
    *,
    source: str,
    text: str,
    history: list[dict],
    assets: list[dict],
    pending_task: Optional[dict],
    active_tasks: list[dict],
    config: Config,
) -> Interpretation:
    prompt = render(
        "interpretation",
        source=source,
        input=text,
        conversation_history=_history_block(history, config.history_turns_in_prompt),
        assets=assets or "(none)",
        pending_task=pending_task or "(none)",
        active_tasks=active_tasks or "(none)",
    )
    interp = Interpretation.model_validate(llm.complete_json("interpret", prompt))

    # Rules enforced in code, whatever the model returned.
    for task in interp.tasks:
        if task.desired_result in config.policy.user_only_results:
            task.performed_by = "user"          # user-only actions
        if source != "user":
            task.performed_by = "system"        # an agent cannot open an interface for the user
    if any(c.kind == "contradiction" for c in interp.conflicts) and interp.clarification:
        interp.tasks = []                       # a contradiction holds the whole input
    return interp


def select(
    llm: LLM,
    *,
    tasks: list[dict],
    candidates: dict[str, list[Card]],
    assets: list[dict],
    config: Config,
) -> dict[str, Selection]:
    """Choose an agent for each task in one call.

    `candidates[task_id]` is the list of cards the registry allows for that
    task. An assignment outside it is rejected.
    """
    union: dict[str, Card] = {}
    for cards in candidates.values():
        union.update({c.id: c for c in cards})
    views = [
        {**{k: t[k] for k in ("id", "goal", "desired_result", "performed_by", "target_assets",
                              "constraints", "assumptions", "depends_on", "supports_task_id")},
         "origin": t.get("origin", "user"),
         "candidate_agents": [c.id for c in candidates[t["id"]]]}
        for t in tasks
    ]
    # Stable material first, the tasks last, so the long prefix can be cached.
    prompt = render("selection", registry=[c.prompt_view() for c in union.values()],
                    policy=config.policy.for_prompt(), assets=assets or "(none)", tasks=views)
    raw = llm.complete_json("select", prompt)

    out: dict[str, Selection] = {}
    for item in _items(raw, "assignments", "decision", tasks):
        try:
            sel = Selection.model_validate(item)
        except ValidationError:
            continue
        if sel.task_id not in candidates or sel.task_id in out:
            continue
        allowed = {c.id for c in candidates[sel.task_id]}
        if sel.decision != "no_capable_agent" and sel.agent_id not in allowed:
            sel = Selection(
                task_id=sel.task_id, analysis=sel.analysis, decision="no_capable_agent",
                capability_gap=f"selection chose an agent not allowed for this task: {sel.agent_id}")
        out[sel.task_id] = sel

    missing = [t for t in tasks if t["id"] not in out]
    if missing and len(tasks) > 1:               # ask again, one task at a time
        for task in missing:
            out.update(select(llm, tasks=[task], candidates={task["id"]: candidates[task["id"]]},
                              assets=assets, config=config))
    for task in tasks:
        out.setdefault(task["id"], Selection(
            task_id=task["id"], decision="no_capable_agent",
            capability_gap="selection returned no assignment for this task"))
    return out


def check_readiness(
    llm: LLM,
    *,
    tasks: list[dict],
    card: Card,
    asset_analysis: dict,
) -> dict[str, Readiness]:
    """Model-backed part of the readiness check, for tasks given to one agent.

    `target_asset` is checked in code and is not sent. Tasks the model does
    not report on are absent from the result.
    """
    inputs = [r.model_dump(exclude_none=True) for r in card.required_inputs
              if r.name != "target_asset"]
    if not inputs:
        return {t["id"]: Readiness(task_id=t["id"]) for t in tasks}
    views = [
        {**{k: t[k] for k in ("id", "goal", "desired_result", "constraints", "assumptions",
                              "target_assets")},
         "task_record": {"record": t.get("record", []),
                         "supporting_results": t.get("supporting_results", [])}}
        for t in tasks
    ]
    prompt = render("readiness", required_inputs=inputs, tasks=views,
                    asset_analysis=asset_analysis or "(none)")
    raw = llm.complete_json("readiness", prompt)

    ids = {t["id"] for t in tasks}
    out: dict[str, Readiness] = {}
    for item in _items(raw, "results", "resolved", tasks):
        try:
            result = Readiness.model_validate(item)
        except ValidationError:
            continue
        if result.task_id in ids and result.task_id not in out:
            out[result.task_id] = result

    missing = [t for t in tasks if t["id"] not in out]
    if missing and len(tasks) > 1:               # ask again, one task at a time
        for task in missing:
            out.update(check_readiness(llm, tasks=[task], card=card,
                                       asset_analysis=asset_analysis))
    return out
