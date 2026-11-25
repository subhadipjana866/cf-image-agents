"""The orchestrator graph.

One graph invocation handles one input: a user message or a UI event. The
graph runs until the work is done or the user is needed, replies, and ends.
Everything that must survive between inputs (tasks, the pending task, parked
tasks, the focus asset, history) is in graph state, saved by the
checkpointer under the session's thread id.

    entry -> interpret -> schedule -> select -> readiness -> dispatch -> outcome
                             ^                                              |
                             +----------------------------------------------+

Calls are batched where tasks allow it: `select` assigns every unassigned
task of a plan in one call, and for an agent whose card says `batchable`,
tasks chained one after another are checked for readiness in one call and
planned in one call. A batch that fails falls back to one task per call.
                          schedule -> finalize -> reply -> END

The run ends at `reply` whenever a task needs the user. The user's next
message always passes through `interpret` first, because it may supply the
requested input, ask for help deciding, change the subject or cancel. That is
why a pending task is modelled as state and not as a LangGraph `interrupt()`.
"""
from __future__ import annotations

import copy
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import steps
from .agents.base import AgentContext, AgentHandler
from .assets import AssetStore
from .config import Config
from .llm import LLM
from .registry import Registry
from .schemas import FINAL_STATUSES, AssetRef, Outcome, Task, TaskEnvelope, TaskIntent


class OState(TypedDict, total=False):
    session_id: str
    source: str                      # "user" | "ui_event"
    input: str
    event: dict
    history: list
    tasks: dict                      # task id -> Task as a dict
    order: list                      # task ids in creation order
    counter: int
    pending_task_id: Optional[str]
    focus: Optional[dict]            # AssetRef as a dict
    gated_plans: list
    # per turn
    current_task_id: Optional[str]
    last_outcomes: list              # [{task_id, outcome}] from the last dispatch
    outbox: list
    turn_steps: int
    route: str
    reply: dict


# ------------------------------------------------------------------ task helpers
def _children(tasks: dict, tid: str) -> list[dict]:
    return [t for t in tasks.values() if t["parent_id"] == tid]


def _dependents(tasks: dict, tid: str) -> list[dict]:
    return [t for t in tasks.values() if tid in t["depends_on"]]


def _cancel_tree(tasks: dict, tid: str) -> None:
    """Cancel a task, the child tasks it raised and the tasks that depend on it."""
    stack, seen = [tid], set()
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in tasks:
            continue
        seen.add(cur)
        if tasks[cur]["status"] not in FINAL_STATUSES:
            tasks[cur]["status"] = "cancelled"
        stack += [t["id"] for t in _children(tasks, cur)]
        stack += [t["id"] for t in _dependents(tasks, cur)]


def _produced(tasks: dict, tid: str, seen: set) -> Optional[dict]:
    """The latest asset produced by a task, or failing that by what it depended on."""
    if tid in seen or tid not in tasks:
        return None
    seen.add(tid)
    task = tasks[tid]
    assets = (task.get("result") or {}).get("assets") or []
    if assets:
        return assets[-1]
    for dep in reversed(task["depends_on"]):
        found = _produced(tasks, dep, seen)
        if found:
            return found
    return None


def _effective_assets(tasks: dict, task: dict) -> list[dict]:
    """A chained task works on the result of the task before it."""
    seen: set = set()
    for dep in reversed(task["depends_on"]):
        found = _produced(tasks, dep, seen)
        if found:
            return [found]
    return task["target_assets"]


def _pending_view(tasks: dict, pending_id: Optional[str]) -> Optional[dict]:
    if not pending_id or pending_id not in tasks:
        return None
    t = tasks[pending_id]
    request = t.get("request") or {}
    return {
        "id": t["id"],
        "goal": t["goal"],
        "requested": request.get("text", ""),
        "options": request.get("options", []),
        "supporting_tasks": t.get("supporting_results", []),
    }


def _active_view(tasks: dict, pending_id: Optional[str]) -> list[dict]:
    out = []
    for t in tasks.values():
        if t["id"] == pending_id or t["status"] in FINAL_STATUSES:
            continue
        if t["status"] == "waiting_on_task":
            status = f"waiting on task {t['waiting_on']}"
        elif t["status"] == "parked":
            status = "parked"
        else:
            status = "in progress"
        out.append({"id": t["id"], "goal": t["goal"], "status": status})
    return out


def _add_tasks(
    tasks: dict,
    order: list,
    counter: int,
    intents: list[TaskIntent],
    *,
    origin: str = "user",
    parent: Optional[dict] = None,
    pending_id: Optional[str] = None,
    fallback_assets: Optional[list[dict]] = None,
) -> tuple[int, list[str]]:
    """Turn interpreted intents into tracked tasks. Returns (counter, new ids)."""
    plan_id = parent["plan_id"] if parent else f"P{counter + 1}"
    mapping = {intent.id: f"T{counter + i + 1}" for i, intent in enumerate(intents)}
    new_ids = []
    for intent in intents:
        counter += 1
        tid = mapping[intent.id]
        supports = intent.supports_task_id
        if supports and supports not in tasks:
            supports = pending_id
        task = Task(
            id=tid,
            plan_id=plan_id,
            parent_id=parent["id"] if parent else None,
            origin=origin,
            depth=parent["depth"] + 1 if parent else 0,
            goal=intent.goal,
            desired_result=intent.desired_result,
            performed_by=intent.performed_by,
            target_assets=intent.target_assets or [AssetRef(**a) for a in fallback_assets or []],
            constraints=intent.constraints,
            assumptions=intent.assumptions,
            depends_on=[mapping.get(d, d) for d in intent.depends_on
                        if d in mapping or d in tasks],
            supports_task_id=supports,
            excluded_agents=(parent["excluded_agents"] + [parent["agent_id"]]) if parent else [],
        )
        tasks[tid] = task.model_dump()
        order.append(tid)
        new_ids.append(tid)
    return counter, new_ids


def _chain_batch(tasks: dict, order: list, head: dict, *, inputs_ready: bool,
                 limit: int) -> list[dict]:
    """The head task plus the tasks chained directly after it for the same agent.

    Used to check or plan several chained tasks in one call. A task joins
    only if it is fresh (never run, nothing to resume), is system-performed,
    depends on exactly the task before it, and has not had a batched call
    fail before.
    """
    batch = [head]
    if not head["batch_ok"] or head["resume"] is not None or head["steps"]:
        return batch
    while len(batch) < limit:
        prev = batch[-1]
        following = [
            t for t in (tasks[i] for i in order)
            if t["status"] == "ready" and t["agent_id"] == head["agent_id"]
            and t["plan_id"] == head["plan_id"] and t["parent_id"] == head["parent_id"]
            and t["depends_on"] == [prev["id"]] and t["performed_by"] == "system"
            and t["batch_ok"] and t["resume"] is None and not t["steps"]
            and (t["inputs_ready"] if inputs_ready
                 else not t["inputs_ready"] and not t["readiness_cache"])
        ]
        if len(following) != 1:
            break
        batch.append(following[0])
    return batch


def _affirmative(text: str) -> bool:
    return text.strip().lower().startswith(("yes", "ok", "sure", "go ahead", "share", "fine"))


# ----------------------------------------------------------------------- the graph
def build_orchestrator(
    *,
    llm: LLM,
    registry: Registry,
    store: AssetStore,
    agents: dict[str, AgentHandler],
    config: Config,
    checkpointer: Any = None,
):
    ctx = AgentContext(store=store, llm=llm, registry=registry, config=config)
    limits, policy = config.limits, config.policy

    def focus_ref(state: OState) -> Optional[AssetRef]:
        return AssetRef(**state["focus"]) if state.get("focus") else None

    def assets_block(state: OState) -> list[dict]:
        return store.describe(state["session_id"], focus_ref(state))

    def envelope(task: dict, tasks: dict, state: OState) -> TaskEnvelope:
        return TaskEnvelope(
            task_id=task["id"],
            parent_task_id=task["parent_id"],
            plan_id=task["plan_id"],
            session_id=state["session_id"],
            origin=task["origin"],
            goal=task["goal"],
            desired_result=task["desired_result"],
            performed_by=task["performed_by"],
            constraints=task["constraints"],
            assumptions=task["assumptions"],
            target_assets=[AssetRef(**a) for a in _effective_assets(tasks, task)],
            inputs=task["inputs"],
            context=task["record"],
            limits={"depth_remaining": limits.max_stack_depth - task["depth"],
                    "steps_remaining": limits.max_steps_per_task - task["steps"]},
            resume=task["resume"],
            agent_state=task["agent_state"],
        )

    def run_agent(task: dict, tasks: dict, state: OState) -> Outcome:
        """One agent step. Any exception becomes a failed outcome."""
        handler = agents.get(task["agent_id"])
        if handler is None:
            return Outcome(kind="failed", reason=f"no handler is registered for {task['agent_id']}")
        try:
            return handler.run(envelope(task, tasks, state), ctx)
        except Exception as err:  # an agent must never take the orchestrator down
            return Outcome(kind="failed", reason=f"{type(err).__name__}: {err}")

    def fail(tasks: dict, task: dict, reason: str, outbox: list) -> None:
        """Mark a task failed and tell whoever was waiting on it."""
        task["status"], task["failure"] = "failed", reason
        parent = tasks.get(task["parent_id"]) if task["parent_id"] else None
        if parent and parent["waiting_on"] == task["id"]:
            parent.update(status="ready", waiting_on=None,
                          resume={"assistance_failed": reason})
        elif not parent:
            outbox.append({"type": "notice", "task_id": task["id"],
                           "text": f"I could not do this: {task['goal']} ({reason})"})

    def ask_user(tasks: dict, task: dict, text: str, options: list, origin: str,
                 outbox: list) -> str:
        """Suspend a task on the user. Returns its id, the new pending task."""
        task["status"] = "pending"
        task["request"] = {"text": text, "options": options, "from": origin}
        outbox.append({"type": "request", "task_id": task["id"], "text": text,
                       "options": options})
        return task["id"]

    # ------------------------------------------------------------------- nodes
    def entry(state: OState) -> dict:
        update: dict = {"outbox": [], "turn_steps": 0, "reply": {}, "current_task_id": None,
                        "last_outcomes": []}
        if state.get("source") == "ui_event":
            update["route"] = "ui_event"
        else:
            update["history"] = list(state.get("history") or []) + [
                {"role": "user", "content": state["input"]}]
            update["route"] = "interpret"
        return update

    def interpret(state: OState) -> dict:
        tasks = copy.deepcopy(state.get("tasks") or {})
        order = list(state.get("order") or [])
        outbox = list(state.get("outbox") or [])
        counter = state.get("counter") or 0
        pending_id = state.get("pending_task_id")
        if pending_id and (pending_id not in tasks or tasks[pending_id]["status"] != "pending"):
            pending_id = None

        try:
            interp = steps.interpret(
                llm, source="user", text=state["input"], history=state["history"][:-1],
                assets=assets_block(state), pending_task=_pending_view(tasks, pending_id),
                active_tasks=_active_view(tasks, pending_id), config=config)
        except Exception as err:
            outbox.append({"type": "notice", "text": f"I could not interpret that ({err})."})
            return {"outbox": outbox, "route": "reply"}

        update: dict = {"route": "schedule"}
        kind = interp.input_type

        if interp.clarification:
            outbox.append({"type": "clarification", "text": interp.clarification.request,
                           "options": interp.clarification.options})
            update["route"] = "reply"

        elif kind == "input_for_pending_task" and pending_id:
            task = tasks[pending_id]
            supplied = interp.pending_task_input.input if interp.pending_task_input else state["input"]
            origin = task["request"]["from"]
            task["record"].append({"kind": "user_input", "request": task["request"]["text"],
                                   "value": supplied})
            task.update(status="ready", request=None)
            pending_id = None
            if origin == "agent":                    # an agent asked: resume it directly
                task["resume"] = {"user_input": supplied}
                update.update(current_task_id=task["id"], route="dispatch")
            elif origin == "consent":
                if _affirmative(supplied):
                    task["consent"] = True
                    update.update(current_task_id=task["id"], route="readiness")
                else:
                    fail(tasks, task, "you chose not to share the design", outbox)
            else:                                    # the readiness check asked: check again
                update.update(current_task_id=task["id"], route="readiness")

        elif kind == "cancellation":
            target = interp.affected_task_id if interp.affected_task_id in tasks else pending_id
            if target:
                _cancel_tree(tasks, target)
                outbox.append({"type": "notice", "text": "Cancelled: " + tasks[target]["goal"]})
                if target == pending_id:
                    pending_id = None
            else:
                outbox.append({"type": "notice", "text": "There is nothing in progress to cancel."})

        elif (kind == "amendment" and interp.tasks and interp.affected_task_id in tasks
              and tasks[interp.affected_task_id]["status"] not in FINAL_STATUSES):
            task, revised = tasks[interp.affected_task_id], interp.tasks[0]
            for child in _children(tasks, task["id"]):
                _cancel_tree(tasks, child["id"])
            task.update(
                goal=revised.goal, desired_result=revised.desired_result,
                performed_by=revised.performed_by, constraints=revised.constraints,
                assumptions=revised.assumptions, status="ready", request=None, resume=None,
                waiting_on=None, agent_id=None, agent_state={}, inputs={}, inputs_ready=False)
            if revised.target_assets:
                task["target_assets"] = [a.model_dump() for a in revised.target_assets]
            if task["id"] == pending_id:
                pending_id = None

        elif interp.tasks:                           # new task(s) or a supporting task
            is_supporting = kind == "supporting_task" and pending_id
            if pending_id and not is_supporting:     # the user moved on: park the pending task
                tasks[pending_id]["status"] = "parked"
                pending_id = None
            if not is_supporting:
                for intent in interp.tasks:
                    intent.supports_task_id = None
            counter, _ = _add_tasks(tasks, order, counter, interp.tasks, pending_id=pending_id)

        else:
            outbox.append({"type": "notice", "text": "I did not find anything to do in that."})
            update["route"] = "reply"

        update.update(tasks=tasks, order=order, counter=counter, outbox=outbox,
                      pending_task_id=pending_id)
        return update

    def ui_event(state: OState) -> dict:
        """The user acted (or not) in an interface. State is updated; nothing is interpreted."""
        tasks = copy.deepcopy(state.get("tasks") or {})
        event = state.get("event") or {}
        tid = event.get("task_id")
        update: dict = {"route": "reply"}
        if tid in tasks:
            tasks[tid]["record"].append({"kind": "ui_event", "event": event.get("type")})
        if event.get("type") == "action_completed":
            if event.get("asset"):
                update["focus"] = event["asset"]
                if tid in tasks:
                    result = tasks[tid].get("result") or {}
                    tasks[tid]["result"] = {**result, "assets": [event["asset"]]}
            for task in tasks.values():              # tasks that waited for this action
                if task["status"] == "parked" and task["waiting_on"] == f"ui:{tid}":
                    task.update(status="ready", waiting_on=None)
                    update["route"] = "schedule"
        update["tasks"] = tasks
        return update

    def schedule(state: OState) -> dict:
        tasks = copy.deepcopy(state.get("tasks") or {})
        outbox = list(state.get("outbox") or [])
        turn_steps = (state.get("turn_steps") or 0) + 1
        if turn_steps > limits.max_steps_per_turn:
            outbox.append({"type": "notice",
                           "text": "I stopped here because this request reached its step limit."})
            return {"tasks": tasks, "outbox": outbox, "turn_steps": turn_steps, "route": "finalize"}

        runnable = []
        for tid in state.get("order") or []:
            task = tasks[tid]
            if task["status"] != "ready":
                continue
            deps = [tasks[d] for d in task["depends_on"] if d in tasks]
            if any(d["status"] not in FINAL_STATUSES for d in deps):
                continue
            broken = [d for d in deps if d["status"] in ("failed", "cancelled")]
            if broken and (not policy.continue_on_failed_link
                           or any(d["status"] == "cancelled" for d in broken)):
                task["status"] = "cancelled"
                continue
            runnable.append(task)

        if not runnable:
            return {"tasks": tasks, "outbox": outbox, "turn_steps": turn_steps,
                    "current_task_id": None, "route": "finalize"}
        task = max(runnable, key=lambda t: t["depth"])   # help requested by an agent runs first
        if task["agent_id"] and task["inputs_ready"]:
            route = "dispatch"
        elif task["agent_id"]:
            route = "readiness"
        else:
            route = "select"
        return {"tasks": tasks, "outbox": outbox, "turn_steps": turn_steps,
                "current_task_id": task["id"], "route": route}

    def select(state: OState) -> dict:
        """Choose agents for every unassigned task of the current plan in one call."""
        tasks = copy.deepcopy(state["tasks"])
        outbox = list(state.get("outbox") or [])
        current = tasks[state["current_task_id"]]

        batch = [t for t in (tasks[i] for i in state.get("order") or [])
                 if t["status"] == "ready" and not t["agent_id"]
                 and t["plan_id"] == current["plan_id"] and t["parent_id"] == current["parent_id"]]
        candidates = {
            t["id"]: registry.candidates(excluded=t["excluded_agents"],
                                         user_performed=t["performed_by"] == "user",
                                         policy=policy)
            for t in batch
        }
        for task in batch:
            if not candidates[task["id"]]:
                fail(tasks, task, "no agent is available for this kind of task", outbox)
        batch = [t for t in batch if candidates[t["id"]]]

        if batch:
            try:
                selections = steps.select(
                    llm, tasks=batch, candidates={t["id"]: candidates[t["id"]] for t in batch},
                    assets=assets_block(state), config=config)
            except Exception as err:
                selections = {}
                for task in batch:
                    fail(tasks, task, f"selection failed: {err}", outbox)
            for task in batch:
                sel = selections.get(task["id"])
                if sel is None:
                    continue
                task["selection"] = sel.model_dump()
                if sel.decision == "no_capable_agent":
                    fail(tasks, task, sel.capability_gap or "no agent can do this", outbox)
                    continue
                task["agent_id"] = sel.agent_id
                if sel.confidence == "low":
                    task["assumptions"].append(
                        f"agent chosen with low confidence: {sel.agent_id}")

        route = "readiness" if current["status"] == "ready" and current["agent_id"] else "schedule"
        return {"tasks": tasks, "outbox": outbox, "route": route}

    def readiness(state: OState) -> dict:
        tasks = copy.deepcopy(state["tasks"])
        outbox = list(state.get("outbox") or [])
        task = tasks[state["current_task_id"]]
        pending_id = state.get("pending_task_id")
        card = registry.get(task["agent_id"])

        def suspend(text: str, options: list, origin: str) -> dict:
            if pending_id and pending_id != task["id"]:
                # Only one task may wait on the user at a time.
                fail(tasks, task, "it needed your input while another task was waiting on you",
                     outbox)
                return {"tasks": tasks, "outbox": outbox, "route": "schedule"}
            new_pending = ask_user(tasks, task, text, options, origin, outbox)
            return {"tasks": tasks, "outbox": outbox, "pending_task_id": new_pending,
                    "route": "reply"}

        # 0. Nothing leaves the system for an outside agent without consent.
        if (card.location == "external" and policy.external_consent_required
                and not task["consent"]):
            return suspend(
                f"To do this I would send your design to {card.name}, an outside service"
                f"{' run by ' + card.provider if card.provider else ''}. Shall I go ahead?",
                ["Yes, share it", "No"], "consent")

        missing: list[dict] = task["readiness_cache"] or []
        task["readiness_cache"] = None
        if not missing:
            # 1. The target asset is resolved in code: the task, then the focus asset.
            assets = _effective_assets(tasks, task)
            if not assets and state.get("focus") and any(
                    r.name == "target_asset" for r in card.required_inputs):
                if card.needs_asset() or task["desired_result"] != "account_action":
                    task["target_assets"] = assets = [state["focus"]]
            if card.needs_asset() and not assets:
                missing.append({"name": "target_asset", "options": [],
                                "request": "Which design should I work on? There is none in "
                                           "this conversation yet."})

            # 2. The card's other inputs, with a model where the card asks for one.
            #    Tasks chained after this one for the same agent are checked in the same call.
            if not missing and card.behaviour.readiness_model:
                batch = [task]
                if card.behaviour.batchable:
                    batch = _chain_batch(tasks, state.get("order") or [], task,
                                         inputs_ready=False, limit=limits.max_batch_size)
                analysis = store.analysis(AssetRef(**assets[0])) if assets else {}
                try:
                    results = steps.check_readiness(llm, tasks=batch, card=card,
                                                    asset_analysis=analysis)
                    if task["id"] not in results:
                        raise ValueError("no result for this task")
                except Exception as err:
                    fail(tasks, task, f"readiness check failed: {err}", outbox)
                    return {"tasks": tasks, "outbox": outbox, "route": "schedule"}
                for member in batch:
                    result = results.get(member["id"])
                    if result is None:
                        continue
                    member["inputs"] = result.resolved
                    if result.defaults_used:
                        member["assumptions"].append(
                            "left to the agent's default: " + ", ".join(result.defaults_used))
                    gaps = [m.model_dump() for m in result.missing]
                    if member is task:
                        missing += gaps
                    elif gaps:
                        member["readiness_cache"] = gaps     # asked when its turn comes
                    else:
                        member["inputs_ready"] = True

        if missing:
            return suspend(" ".join(m["request"] for m in missing),
                           [o for m in missing for o in m["options"]], "readiness")
        task["inputs_ready"] = True
        return {"tasks": tasks, "outbox": outbox, "route": "dispatch"}

    def dispatch(state: OState) -> dict:
        tasks = copy.deepcopy(state["tasks"])
        task = tasks[state["current_task_id"]]
        handler = agents.get(task["agent_id"])
        results: list[tuple[dict, Outcome]] = []

        # Chained tasks for an agent that can plan several at once go in one call.
        if (hasattr(handler, "run_batch") and task["agent_id"] in registry
                and registry.get(task["agent_id"]).behaviour.batchable):
            batch = _chain_batch(tasks, state.get("order") or [], task, inputs_ready=True,
                                 limit=limits.max_batch_size)
            if len(batch) > 1:
                try:
                    outcomes = handler.run_batch([envelope(t, tasks, state) for t in batch], ctx)
                    if not outcomes or len(outcomes) > len(batch):
                        raise ValueError("unusable batch result")
                    results = list(zip(batch, outcomes))
                except Exception:
                    for member in batch:             # fall back to one task per call
                        member["batch_ok"] = False

        if not results:
            if task["steps"] >= limits.max_steps_per_task:
                results = [(task, Outcome(kind="failed", reason="this task reached its step limit"))]
            else:
                results = [(task, run_agent(task, tasks, state))]

        for member, _ in results:
            member["steps"] += 1
            member["status"] = "in_progress"
            member["resume"] = None
        return {"tasks": tasks, "route": "outcome",
                "last_outcomes": [{"task_id": m["id"], "outcome": o.model_dump()}
                                  for m, o in results]}

    def outcome(state: OState) -> dict:
        tasks = copy.deepcopy(state["tasks"])
        order = list(state.get("order") or [])
        outbox = list(state.get("outbox") or [])
        counter = state.get("counter") or 0
        pending_id = state.get("pending_task_id")
        update: dict = {"route": "schedule"}

        for item in state.get("last_outcomes") or []:
            task = tasks[item["task_id"]]
            out = Outcome(**item["outcome"])
            task["agent_state"] = out.agent_state
            task["record"] += out.findings

            if out.kind == "completed":
                assets = [a.model_dump() for a in out.assets]
                task["status"] = "completed"
                task["result"] = {"data": out.result, "assets": assets, "summary": out.summary,
                                  "generative": out.generative}
                if assets:
                    update["focus"] = assets[-1]
                parent = tasks.get(task["parent_id"]) if task["parent_id"] else None
                if parent and parent["waiting_on"] == task["id"]:
                    parent.update(status="ready", waiting_on=None,
                                  resume={"assistance": {"need": task["goal"],
                                                         "result": task["result"]}})
                elif not parent:
                    if out.summary:
                        outbox.append({"type": "message", "task_id": task["id"],
                                       "text": out.summary})
                    if assets:
                        outbox.append({"type": "asset", "task_id": task["id"],
                                       "asset": assets[-1]})
                if out.client_action:
                    outbox.append({"type": "client_action", "action": out.client_action})
                supported = (tasks.get(task["supports_task_id"])
                             if task["supports_task_id"] else None)
                if supported:                        # results a pending task's reply may refer to
                    supported["supporting_results"].append(
                        {"task_id": task["id"], "goal": task["goal"], "result": out.result})
                if task["performed_by"] == "user":   # dependents wait for the user to act
                    for dep in _dependents(tasks, task["id"]):
                        if dep["status"] == "ready":
                            dep.update(status="parked", waiting_on=f"ui:{task['id']}")

            elif out.kind == "input_required":
                if pending_id and pending_id != task["id"]:
                    fail(tasks, task, "it needed input while another task was waiting on you",
                         outbox)
                else:
                    pending_id = ask_user(tasks, task, out.request, out.options, "agent", outbox)
                    update["route"] = "reply"

            elif out.kind == "assistance_required":
                if task["depth"] + 1 > limits.max_stack_depth:
                    task.update(status="ready",
                                resume={"assistance_failed": "delegation depth limit reached"})
                else:
                    need = out.need + (f"\nAcceptance: {out.acceptance}" if out.acceptance else "")
                    try:
                        interp = steps.interpret(
                            llm, source=task["agent_id"], text=need, history=[],
                            assets=assets_block(state), pending_task=None, active_tasks=[],
                            config=config)
                        intents = interp.tasks
                    except Exception:
                        intents = []
                    if not intents:
                        task.update(status="ready", resume={
                            "assistance_failed": "the need could not be stated as a task"})
                    else:
                        fallback = (out.need_inputs.get("target_assets")
                                    or _effective_assets(tasks, task))
                        counter, new_ids = _add_tasks(tasks, order, counter, intents,
                                                      origin=task["agent_id"], parent=task,
                                                      fallback_assets=fallback)
                        task.update(status="waiting_on_task", waiting_on=new_ids[-1])

            else:  # failed
                task["attempts"] += 1
                if out.retryable and task["attempts"] < limits.max_attempts_per_task:
                    task["status"] = "ready"         # same agent, same inputs, once more
                else:
                    fail(tasks, task, out.reason or "the agent failed", outbox)

        update.update(tasks=tasks, order=order, counter=counter, outbox=outbox,
                      pending_task_id=pending_id)
        return update

    def finalize(state: OState) -> dict:
        tasks = copy.deepcopy(state.get("tasks") or {})
        outbox = list(state.get("outbox") or [])
        gated = list(state.get("gated_plans") or [])
        pending_id = state.get("pending_task_id")

        # Quality gate: once per plan, on its final version, if any step was generative.
        evaluator = agents.get(policy.evaluator_agent_id)
        if policy.quality_gate == "generative" and evaluator:
            plans: dict[str, list[dict]] = {}
            for tid in state.get("order") or []:
                plans.setdefault(tasks[tid]["plan_id"], []).append(tasks[tid])
            for plan_id, members in plans.items():
                if plan_id in gated or any(t["status"] not in FINAL_STATUSES for t in members):
                    continue
                gated.append(plan_id)
                done = [t for t in members if t["status"] == "completed" and not t["parent_id"]
                        and (t["result"] or {}).get("assets")]
                if not done or not any(t["result"]["generative"] for t in done):
                    continue
                final_asset = done[-1]["result"]["assets"][-1]
                gate = Task(id=f"gate-{plan_id}", plan_id=plan_id, origin="orchestrator",
                            goal=f"Evaluate the quality of {final_asset['asset_id']} "
                                 f"v{final_asset['version']}.",
                            desired_result="quality_assessment",
                            target_assets=[AssetRef(**final_asset)],
                            agent_id=policy.evaluator_agent_id).model_dump()
                report = run_agent(gate, tasks, state)
                if report.kind == "completed" and report.result.get("passed") is False:
                    defects = "; ".join(map(str, report.result.get("defects") or [])) or "see report"
                    outbox.append({"type": "notice",
                                   "text": f"Quality check found problems: {defects}."})

        # Nothing is waiting on the user: bring back the most recently parked request.
        if not pending_id:
            parked = [tasks[tid] for tid in state.get("order") or []
                      if tasks[tid]["status"] == "parked" and tasks[tid]["request"]]
            if parked:
                task = parked[-1]
                task["status"] = "pending"
                pending_id = task["id"]
                outbox.append({"type": "request", "task_id": task["id"],
                               "text": "Coming back to your earlier request. "
                                       + task["request"]["text"],
                               "options": task["request"]["options"]})
        return {"tasks": tasks, "outbox": outbox, "gated_plans": gated,
                "pending_task_id": pending_id, "route": "reply"}

    def reply(state: OState) -> dict:
        """Merge everything said this turn into one reply. The single voice to the user."""
        outbox = state.get("outbox") or []
        latest: dict[str, dict] = {}
        for item in outbox:
            if item["type"] == "asset":
                latest[item["asset"]["asset_id"]] = item["asset"]   # last version per asset
        requests = [i for i in outbox if i["type"] in ("request", "clarification")]
        text = "\n".join(i["text"] for i in outbox
                         if i["type"] in ("message", "notice", "request", "clarification")
                         and i.get("text"))
        result = {
            "text": text,
            "assets": list(latest.values()),
            "client_actions": [i["action"] for i in outbox if i["type"] == "client_action"],
            "request": ({"text": requests[-1]["text"], "options": requests[-1].get("options", [])}
                        if requests else None),
        }
        update: dict = {"reply": result}
        if state.get("source") != "ui_event" or text:
            update["history"] = list(state.get("history") or []) + [
                {"role": "assistant", "content": text or "(no reply)"}]
        return update

    # ------------------------------------------------------------------ wiring
    graph = StateGraph(OState)
    for name, fn in [("entry", entry), ("interpret", interpret), ("ui_event", ui_event),
                     ("schedule", schedule), ("select", select), ("readiness", readiness),
                     ("dispatch", dispatch), ("outcome", outcome), ("finalize", finalize),
                     ("reply", reply)]:
        graph.add_node(name, fn)

    def by_route(state: OState) -> str:
        return state["route"]

    graph.add_edge(START, "entry")
    graph.add_conditional_edges("entry", by_route, ["interpret", "ui_event"])
    graph.add_conditional_edges("interpret", by_route,
                                ["schedule", "readiness", "dispatch", "reply"])
    graph.add_conditional_edges("ui_event", by_route, ["schedule", "reply"])
    graph.add_conditional_edges("schedule", by_route,
                                ["select", "readiness", "dispatch", "finalize"])
    graph.add_conditional_edges("select", by_route, ["readiness", "schedule"])
    graph.add_conditional_edges("readiness", by_route, ["dispatch", "schedule", "reply"])
    graph.add_edge("dispatch", "outcome")
    graph.add_conditional_edges("outcome", by_route, ["schedule", "reply"])
    graph.add_edge("finalize", "reply")
    graph.add_edge("reply", END)
    return graph.compile(checkpointer=checkpointer)
