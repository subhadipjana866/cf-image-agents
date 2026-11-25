"""Terminal chat through the whole path, with the orchestrator's task stack.

Every message (including a reply to a question) goes:

  [1] INTERPRET  -> input type and tasks          (agent_navigation, image_agents prompts)
  [2] SELECT     -> one agent card per task
  [3] COLLECT    -> the input-collector sees the image, validates, asks or informs
  [4] AGENT      -> the agent runs; it may stop and ask the user (ask_user interrupt)

Every task waiting on the user sits on the pending stack; the newest question is on top.
A reply is interpreted first, so it can answer the top question or an older one (that
task resumes where it stopped), ask something on the side (the question stays open),
start something new (the waiting tasks stay on the stack), amend or cancel. When a turn
ends with nothing new asked, the newest waiting task is offered again: the user
continues it or drops it.
"""
from __future__ import annotations

import re
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

from agent_navigation.llm import TrackedChatLLM
from agent_navigation.navigator import Navigator, Route, RoutedTask
from image_agents.registry import Registry

from .build import DEFAULT_MODEL
from . import asks
from .collector import NEEDS_IMAGE, Collected, CollectTask, collect
from .session import Brief, DesignSession
from .vision import vision_json

# Today's automatic post-generation flow (nodes.interruptBeforeFidelitySuggestion ->
# suggestionManager -> suggestionHandler -> suggestionExecutors -> suggestionMoreAsk), copied:
#   generation done -> GENERATION_COMPLETED -> automatic critique (asks which graphic first
#   when several were just made) -> suggestion mode (accept / reject / modify) -> accepted
#   ones are applied -> "Want more suggestions?" once -> yes: another round.
# When the suggestion loop ends (all rejected, "no more", or rounds used up) the layering
# offer follows, as nodes.interruptBeforeLayering did; yes starts the layering agent.
GENERATION_COMPLETED = ("Your graphic has been generated. Now our Critic Agent is doing a "
                        "professional review of the graphic and will give you some suggestions "
                        "to improve it. Please wait for those...")
SUGGESTION_TARGET_ASK = ("Please select the graphic you want to work with. Our Critic Agent will "
                         "analyze it to explore ways to make it more brilliant.")
SUGGESTION_MODE = ("Please choose a suggestion to accept it, reject it or modify it to better "
                   "suit you (e.g. \"apply 1 and 3\", \"2 but in blue\", \"none\").")
SUGGESTION_MORE_ASK = "Want more suggestions to improve this graphic?"
SUGGESTION_DECLINED = "No problem! What can we help you with next? 😊"
SUGGESTION_EDITS_COMPLETE = "Great! All your edits are complete. What would you like to do next? 🎉"
MAX_MORE_SUGGESTION_ROUNDS = 1        # nodes.MAX_SUGGESTION_MORE_ASKS
# A job that fails is not dropped: the bot asks whether to try again, and a yes runs the
# same task again (once; a second failure is only reported).
RETRY_ASK = "That didn't go through. Want me to try again?"
RETRY_DECLINED = "Okay, I've left it. What would you like to do next?"
RETRY_NEW_DETAILS = "Sure — tell me what you'd like me to make instead."
MAX_RETRIES = 1
# A waiting task offered again after other work: continue it or drop it.
RESUME_ASK = "You still have an unfinished request: \"{goal}\". Do you want to continue it or drop it?"
RESUME_DROPPED = "Okay, I've dropped it."
# Agents whose job is a new or changed image: finishing with no new version after a
# failed tool call means the job failed.
IMAGE_RESULT_AGENTS = {"image-generator", "image-editor"}
# Edits the frontend's own controls do exactly (manual_operations.json): "always" controls
# for any change, "specific" ones when an exact value is given. Decided by today's
# manual_edit_decision.classify_edit_mode; the frontend gets manual_op and applies it.
import os as _os
MANUAL_CONTROLS = _os.getenv("KOGNIE_ENGINE_MANUAL_CONTROLS", "on").lower() != "off"

RULE = "-" * 78
# Agents with a card but no implementation yet. Their tasks get a polite "not available yet".
NOT_BUILT = {
    "slop-terminator",   # TODO: integrate later, when the senior dev's slop detection/repair lands
}


@dataclass
class TaskState:
    id: str
    routed: RoutedTask
    collect: CollectTask
    status: str = "queued"     # queued | asking | done | failed | cancelled | skipped
    depends_on: list[str] = field(default_factory=list)
    ask: Optional[Collected] = None
    kind: str = "task"         # task | offers: suggestions | more_suggestions | pick_graphic | retry | resume
    origin: str = "user"       # who asked for the task: "user", "orchestrator" or an agent id
    payload: dict = field(default_factory=dict)


def _merge_layering(tasks: list[RoutedTask]) -> list[RoutedTask]:
    """One layering run per message: today's layering takes every element in one pass, so
    several layering tasks from one message ("the headline and the racks") become one."""
    lay = [t for t in tasks if t.agent_id == "layering-agent" and not t.depends_on]
    if len(lay) < 2:
        return tasks
    first = lay[0]
    first.goal = "; ".join(t.goal for t in lay)
    first.constraints = [c for t in lay for c in t.constraints]
    drop = {t.id for t in lay[1:]}
    for t in tasks:                       # anything that waited on a merged task waits on the one
        t.depends_on = [first.id if d in drop else d for d in t.depends_on]
    return [t for t in tasks if t.id not in drop]


_PLAIN_YES = {"y", "yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead", "yes please",
              "continue", "carry on", "go on", "resume", "retry", "try again", "more", "do it"}
_PLAIN_NO = {"n", "no", "nope", "no thanks", "no thank you", "nah", "drop it", "drop", "skip",
             "leave it", "stop", "cancel", "not now"}


_INTERNAL_ID = re.compile(r"\b(?:the\s+)?(?:image\s+|design\s+)?(?:asset\s+)?a\d+\s+"
                          r"(?:v|version\s*)\d+\b", re.IGNORECASE)


def user_text(text: str) -> str:
    """Strip internal asset ids ("asset a1 version 2") from anything the user will read."""
    return re.sub(r"\s{2,}", " ", _INTERNAL_ID.sub("the image", text or "")).strip()


def _llm_cost(usage_by_model: dict) -> float:
    """Price LangChain usage metadata ({model: {total_tokens}}) with the app's rate table."""
    from ..utils import calculateCost
    total = 0.0
    for model, u in (usage_by_model or {}).items():
        name = str(model).split("/")[-1]
        for known in ("gpt-5.4-mini", "gpt-5.4", "gpt-5-mini", "gemini-3-flash-preview"):
            if name.startswith(known):
                name = known
                break
        total += calculateCost(name, int((u or {}).get("total_tokens") or 0))
    return total


def print_agent_steps(new_messages: list, session: DesignSession, cost_before: float,
                      indent: str = "      ") -> None:
    step = 0
    for m in new_messages:
        for call in getattr(m, "tool_calls", None) or []:
            step += 1
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            print(f"{indent}{step}. {call['name']}({args})")
        if m.type == "tool":
            first = str(m.content).splitlines()[0] if m.content else ""
            print(f"{indent}     -> {first[:220]}")
    if new_messages:
        print(f"\n{indent}agent: {new_messages[-1].content}\n")
    if session.current:
        print(f"{indent}latest image: version {session.current.number}  {session.current.url}")
    print(f"{indent}cost: ${session.cost - cost_before:.4f}   session total: ${session.cost:.4f}")


def _assets(session: DesignSession) -> list[dict]:
    v = session.current
    if not v and session.fallback_image:
        # Not loaded yet, but a request that works on an image will use it.
        return [{"asset_id": "a1", "latest_version": 1,
                 "description": "the project's latest graphic (nothing selected on the canvas)",
                 "produced_by": "unknown", "method": "generative", "in_focus": False,
                 "text_blocks": [], "layers": []}]
    if not v:
        return []
    layers = (v.meta.get("text_layers") or {}).get("layers") or {}
    blocks = [{"id": e.get("id"), "text": " ".join(e.get("lines") or []) or e.get("text"),
               "role": e.get("role")} for e in (layers.get("elements") or [])][:8]
    return [{"asset_id": "a1", "latest_version": v.number,
             "description": f"{'generated design' if v.made_by == 'image-generator' else 'design'}"
                            f"; last step: {v.tool} ({v.instruction[:80]})",
             "produced_by": v.made_by, "method": "generative", "in_focus": True,
             "text_blocks": blocks,
             "layers": [{"layers": len(x["segmentation_result"]), "of_version": x["version"]}
                        for x in session.layers[-3:]]}]


def print_route(route: Route, usage: dict) -> None:
    print(f"  [1] INTERPRET  input_type = {route.input_type}")
    if route.analysis:
        print(f"      reading: {route.analysis}")
    for c in route.conflicts:
        print(f"      conflict ({c.get('kind')}): {c.get('explanation')}")
    if route.clarification:
        print(f"      clarify: {route.clarification.get('request')}  "
              f"options={route.clarification.get('options') or []}")
    if route.pending_input:
        print(f"      answers the pending question: {route.pending_input['input']}")
    for t in route.tasks:
        extra = f"  depends_on={t.depends_on}" if t.depends_on else ""
        extra += f"  supports={t.supports_task_id}" if t.supports_task_id else ""
        print(f"      {t.id}: {t.goal}")
        print(f"          result={t.desired_result}  performed_by={t.performed_by}{extra}")
    if route.tasks:
        print("  [2] SELECT")
        for t in route.tasks:
            print(f"      {t.id}: candidates = {', '.join(t.candidates) or '(none allowed)'}")
            print(f"          chosen = {t.agent_id or 'none'}  ({t.decision}, confidence {t.confidence})")
            if t.analysis:
                print(f"          why: {t.analysis}")
            if t.capability_gap:
                print(f"          gap: {t.capability_gap}")
    print(f"      (navigation so far: {usage['calls']} calls, {usage['input_tokens']} tokens in)")


class Chat:
    def __init__(self, session: DesignSession, *, model: str = DEFAULT_MODEL,
                 nav_model: str = DEFAULT_MODEL, preset_brief: Optional[Brief] = None,
                 verbose: bool = True, defer_auto_critique: bool = False):
        self.session = session
        self.verbose = verbose
        # Websocket mode: today's frontend answers generation_completed with "yes" to start
        # the critique, so the critique waits for that reply instead of running at once.
        self.defer_auto_critique = defer_auto_critique
        self.pending_auto_critique = False
        # Called with each progress event as it happens (the websocket sends spinners live).
        self.on_progress = None
        # Extra instruction appended to every agent request (the app shows images itself).
        self.reply_style = "Reply to the person directly, as 'you'."
        # Structured record of what this turn said and did, for the websocket bridge:
        # message | ask | progress | result | client_action.
        self.events: list[dict] = []
        self.model = model
        self.nav_llm = TrackedChatLLM({"default": nav_model})
        self.navigator = Navigator(self.nav_llm)
        self.cards = {c.id: c for c in self.navigator.registry.all()}
        self._agents: dict = {}
        self.histories: dict[str, list] = {}
        self.nav_history: list[dict] = []
        self.tasks: dict[str, TaskState] = {}
        self.queue: list[str] = []
        self.stack: list[str] = []   # tasks waiting on the user; the last one is the open question
        self.asked_this_turn = False
        self.supporting_turn = False
        self.chat_key = uuid.uuid4().hex[:12]   # agent checkpoint threads: <chat>:<task>:<attempt>
        self.ask_seq = 0                        # ids of structured questions (kognie.ask.v1)
        self.turn_no = 0
        self.auto_no = 0
        self.more_rounds = 0
        if preset_brief is not None:
            session.brief = preset_brief

    # -------------------------------------------------------------- agents
    def agent(self, agent_id: str):
        if agent_id not in self._agents:
            from . import build, layering, review_agents, support_agents
            builders = {
                "image-generator": build.build_generation_agent,
                "image-editor": build.build_edit_agent,
                "manual-ops": support_agents.build_manual_ops_agent,
                "chitchat-handler": support_agents.build_chitchat_agent,
                "question-handler": support_agents.build_question_agent,
                "critic-agent": review_agents.build_critic_agent,
                "suggestion-agent": review_agents.build_suggestion_agent,
                "image-evaluator": review_agents.build_evaluator_agent,
                "custom-agent": support_agents.build_custom_agent,
                "layering-agent": layering.build_layering_agent,
            }
            self._agents[agent_id] = builders[agent_id](self.model)
        return self._agents[agent_id]

    # Every message carries who sent it, which task it belongs to and who it is for.
    # Senders: an agent id, "input-collector" or "orchestrator". Recipients: "user",
    # "frontend" (an instruction for the app) or an agent id.
    def _emit(self, event: dict, *, sender: str, task: Optional[TaskState] = None,
              recipient: str = "user") -> dict:
        event.update(sender=sender, task_id=task.id if task else None, recipient=recipient)
        self.events.append(event)
        return event

    @staticmethod
    def _tag(sender: str, task: Optional[TaskState], recipient: str = "user") -> str:
        return f"[{sender} -> {recipient}" + (f" | task {task.id}]" if task else "]")

    def _say(self, text: str, *, sender: str = "orchestrator",
             task: Optional[TaskState] = None) -> None:
        text = user_text(text)
        print(f"      {self._tag(sender, task)} {text}")
        self._emit({"type": "message", "text": text}, sender=sender, task=task)
        self.nav_history.append({"role": "assistant", "content": text[:500]})

    def _asker(self, task: TaskState) -> str:
        """Who asked this task's open question."""
        if task.kind != "task":
            return "orchestrator"
        if task.payload.get("waiting_agent"):
            return task.collect.agent_id or "orchestrator"
        return "input-collector"

    def _ask_event(self, task: TaskState, question: str, *,
                   sender: Optional[str] = None) -> None:
        a = task.ask
        question = user_text(question)
        sender = sender or self._asker(task)
        event = {"type": "ask", "kind": task.kind, "question": question,
                 "ask_kind": a.ask_kind if a else "text", "options": list(a.options) if a else [],
                 "fields": list(a.fields) if a else []}
        if task.kind == "suggestions":
            event["suggestions"] = task.payload.get("suggestions") or []
        if task.payload.get("ask_extra"):
            event["extra"] = dict(task.payload["ask_extra"])
        if task.payload.get("ask_spec"):
            spec = dict(task.payload["ask_spec"])
            spec["prompt"] = question     # "Back to my question: ..." when shown again
            event["ask_spec"] = spec
        print(f"      {self._tag(sender, task)} asks: {question}")
        self._emit(event, sender=sender, task=task)

    # -------------------------------------------------------- pending stack
    @property
    def pending_id(self) -> Optional[str]:
        """The task whose question is on top of the pending stack (the latest asked)."""
        return self.stack[-1] if self.stack else None

    def _push(self, task: TaskState) -> None:
        if task.id in self.stack:
            self.stack.remove(task.id)
        self.stack.append(task.id)
        self.asked_this_turn = True

    def _unstack(self, task: TaskState) -> None:
        """Take a task off the stack, with any 'continue or drop?' offer made for it."""
        for tid in list(self.stack):
            t = self.tasks[tid]
            if tid == task.id or (t.kind == "resume" and t.payload.get("task_id") == task.id):
                self.stack.remove(tid)
                if tid != task.id:
                    t.status = "cancelled"

    def _spec(self, task: TaskState, asked_by: str, purpose: str, question: str,
              options, **kw) -> dict:
        self.ask_seq += 1
        return asks.build(ask_id=f"q{self.ask_seq}", task_id=task.id, asked_by=asked_by,
                          purpose=purpose, prompt=user_text(question), options=options, **kw)

    def _flow_open(self) -> bool:
        """A suggestion question from the post-generation flow is open (suggestions, more?,
        layering offer). A design that is finished and only waiting for the frontend's
        go-ahead to start the critique does not hold the next request: the job asked for
        is done, so the next one starts."""
        flow = ("pick_graphic", "suggestions", "more_suggestions", "layering_offer")
        return any(self.tasks[t].kind in flow for t in self.stack)

    def _thread(self, task: TaskState) -> str:
        return f"{self.chat_key}:{task.id}:{task.payload.get('retries', 0)}"

    def _release(self, task: TaskState) -> None:
        """Forget a finished task's agent checkpoint."""
        from .common import CHECKPOINTER
        try:
            CHECKPOINTER.delete_thread(self._thread(task))
        except Exception:  # noqa: BLE001 - nothing was saved for it
            pass

    # ---------------------------------------------------------------- turn
    def cancel_running(self) -> None:
        """A cancel arrived mid-run: drop queued work and everything waiting on the user."""
        for tid in self.queue + self.stack:
            self.tasks[tid].status = "cancelled"
            self._release(self.tasks[tid])
        self.queue, self.stack = [], []
        self.pending_auto_critique = False

    def turn(self, text: str, selected_image: Optional[str] = None, *,
             field_answers: Optional[dict] = None, suggestions: Optional[list] = None,
             attachments: Optional[list] = None, bboxes: Optional[list] = None,
             ask_reply: Optional[dict] = None) -> None:
        """One user message. Structured parts (a filled form, suggestion card statuses, an
        image selection) answer the open question directly, as today's frontend sends them."""
        s = self.session
        self.turn_no += 1
        self.events = []
        self.asked_this_turn = self.supporting_turn = False
        print(RULE)
        print(f"you: {text}" + (f"   [selected image: {selected_image}]" if selected_image else "")
              + (f"   [form: {field_answers}]" if field_answers else "")
              + (f"   [suggestion cards: {len(suggestions)}]" if suggestions else ""))
        pending = self.tasks.get(self.pending_id) if self.pending_id else None
        s.attachments = list(attachments or [])
        spec = pending.payload.get("ask_spec") if pending else None
        if ask_reply and spec and str(ask_reply.get("id")) == spec["id"]:
            answer = asks.reply_text(spec, ask_reply, text)
            print(f"  [1] INTERPRET  skipped: ask_reply {spec['id']} answers the open question "
                  f"({answer})")
            if pending.kind != "task":
                self._resolve_offer(pending, answer)
            else:
                self._answer(pending, answer)
            self._drain()
            self._finalize()
            return
        if ask_reply and not text.strip():
            # A stale or unknown reply: use its words as an ordinary message.
            text = "; ".join(str(v) for v in (ask_reply.get("values") or {}).values()) or text
        # Boxes drawn on the image (normalised by ws_bridge) belong to this turn only.
        s.manual_bboxes = list(bboxes or [])
        if s.manual_bboxes and pending and pending.ask \
                and pending.ask.ask_kind == "layering_targets":
            print(f"  [1] INTERPRET  skipped: {len(s.manual_bboxes)} drawn box(es) answer the "
                  "pending layering question")
            self._answer(pending, (text.strip() + " " if text.strip() else "")
                         + f"(drew {len(s.manual_bboxes)} box(es) on the image)")
            self._drain()
            self._finalize()
            return
        if s.attachments and not s.current and not selected_image:
            # As editingManager did: with no canvas, an attachment becomes the image to work on
            # for edits; generation still reads it as a photo via session.attachments.
            selected_image = s.attachments[0]
        if field_answers and pending and pending.ask and pending.ask.ask_kind == "form":
            print("  [1] INTERPRET  skipped: the form answers the pending question")
            answers = {k: (str(v).strip() or "skipped") for k, v in field_answers.items()}
            if s.brief:
                s.brief.field_answers.update(answers)
            self._answer(pending, "form: " + ", ".join(f"{k}={v}" for k, v in answers.items())
                         + (f"; also said: {text}" if text.strip() else ""))
            self._drain()
            self._finalize()
            return
        if suggestions and pending and pending.kind == "suggestions":
            print("  [1] INTERPRET  skipped: the suggestion cards answer the pending question")
            self._resolve_cards(pending, suggestions, text)
            self._drain()
            self._finalize()
            return
        if selected_image:
            s.select_image(selected_image)
        if (selected_image and not text.strip() and pending and pending.ask
                and pending.ask.ask_kind == "select_image"):
            print("  [1] INTERPRET  skipped: a reply carrying only a selected image answers the "
                  "pending 'select an image' question")
            if pending.kind == "pick_graphic":
                self._resolve_offer(pending, f"selected image version {s.current.number}")
            else:
                self._answer(pending, f"selected image version {s.current.number}")
            self._drain()
            self._finalize()
            return
        if selected_image and not text.strip():
            text = "I've selected this image to work on."

        pending_view = None
        if pending and pending.ask:
            pending_view = {"id": pending.id, "goal": pending.collect.goal,
                            "requested": pending.ask.question, "options": pending.ask.options}
            older = [self.tasks[t] for t in self.stack[:-1] if self.tasks[t].ask]
            if older:
                pending_view["also_waiting"] = [
                    {"id": t.id, "goal": t.collect.goal, "requested": t.ask.question}
                    for t in older]
                pending_view["note"] = (
                    "also_waiting lists older requests still waiting on the user's answer, "
                    "newest last. If the input answers one of them instead of this one, it is "
                    "input_for_pending_task with that task's id.")
        calls_before = len(self.nav_llm.usage.calls)
        route = self.navigator.route(text, history=self.nav_history, assets=_assets(s),
                                     pending_task=pending_view)
        from ..utils import calculateCost
        nav_tokens = sum(c.input_tokens + c.output_tokens
                         for c in self.nav_llm.usage.calls[calls_before:])
        s.record("navigation", ok=True, detail=route.input_type,
                 cost=calculateCost("gpt-5.4", nav_tokens))
        print_route(route, self.nav_llm.usage.totals())
        self.nav_history.append({"role": "user", "content": text})
        self._apply(route, text, pending)
        self._drain()
        self._finalize()

    def _apply(self, route: Route, text: str, pending: Optional[TaskState]) -> None:
        if route.clarification:
            print("  [3] COLLECT  (none: the orchestrator asks first)")
            q = user_text(route.clarification.get("request", ""))
            opts = route.clarification.get("options") or []
            if opts:
                self.ask_seq += 1
                spec = asks.build(ask_id=f"q{self.ask_seq}", task_id=None, asked_by="orchestrator",
                                  purpose="clarification", prompt=q, options=opts)
                print(f"      {self._tag('orchestrator', None)} asks: {q}  options: {' / '.join(opts)}")
                self._emit({"type": "ask", "kind": "clarification", "question": q,
                            "ask_kind": "text", "options": list(opts), "fields": [],
                            "ask_spec": spec}, sender="orchestrator")
                self.nav_history.append({"role": "assistant", "content": q[:500]})
            else:
                self._say(q, sender="orchestrator")
            if pending:
                print(f"      (still open: {pending.ask.question})")
            return
        if route.input_type == "input_for_pending_task" and pending:
            answer = route.pending_input["input"] if route.pending_input else text
            target = self.tasks.get((route.pending_input or {}).get("task_id", "")) or pending
            if target.id not in self.stack:
                target = pending
            if target.kind != "task":
                self._resolve_offer(target, answer)
            else:
                self._answer(target, answer)
            if route.tasks:
                # The reply also asked for something new: it waits until the answered task
                # (and anything that task queued) is done, then runs, asking whatever it needs.
                later = self._add(_merge_layering(route.tasks))
                self.queue.extend(later)
                for tid in later:
                    self.tasks[tid].payload["after"] = target.id
                    print(f"      {tid}: queued after {target.id} ({self.tasks[tid].collect.goal})")
            return
        if route.input_type == "cancellation":
            target = self.tasks.get(route.affected_task_id or "") or pending
            if target:
                self._cancel(target)
                self._say("Okay, I've cancelled that.")
            else:
                self._say("There's nothing in progress to cancel.")
            return
        if route.input_type == "amendment" and route.tasks:
            target = self.tasks.get(route.affected_task_id or "") or pending
            if target and target.status in ("asking", "queued"):
                revised = route.tasks[0]
                target.collect.goal, target.collect.constraints = revised.goal, revised.constraints
                target.collect.record, target.collect.rounds = [], 0
                target.status, target.ask = "queued", None
                target.payload.pop("waiting_agent", None)     # an amended task starts over
                self._release(target)
                self._unstack(target)
                self.queue.insert(0, target.id)
                print(f"      amended {target.id}: {revised.goal}")
                return
        new_ids = self._add(_merge_layering(route.tasks))
        supporting = route.input_type == "supporting_task" and pending is not None
        self.supporting_turn = supporting
        if pending and not supporting and pending.kind != "task":
            # As today: a new request exits suggestion mode (and other offers).
            pending.status = "cancelled"
            self._unstack(pending)
            print(f"      left {pending.kind} (the user moved on)")
        elif pending and not supporting:
            print(f"      {pending.id} stays on the pending stack (offered again after this)")
        self.queue[0:0] = new_ids

    def _add(self, routed: list[RoutedTask]) -> list[str]:
        mapping = {t.id: f"{self.turn_no}.{t.id}" for t in routed}
        ids = []
        for t in routed:
            tid = mapping[t.id]
            self.tasks[tid] = TaskState(
                id=tid, routed=t, depends_on=[mapping.get(d, d) for d in t.depends_on],
                collect=CollectTask(id=tid, agent_id=t.agent_id or "", goal=t.goal,
                                    desired_result=t.desired_result,
                                    performed_by=t.performed_by, constraints=t.constraints))
            ids.append(tid)
        return ids

    def _answer(self, task: TaskState, answer: str) -> None:
        task.collect.record.append({"q": task.ask.question if task.ask else "", "a": answer})
        task.status, task.ask = "queued", None
        self._unstack(task)
        self.queue.insert(0, task.id)

    def _cancel(self, task: TaskState) -> None:
        task.status = "cancelled"
        self._unstack(task)
        self._release(task)
        for t in self.tasks.values():
            if task.id in t.depends_on and t.status in ("queued", "asking"):
                self._cancel(t)
        self.queue = [q for q in self.queue if self.tasks[q].status == "queued"]

    # --------------------------------------------------------------- drain
    def _drain(self) -> None:
        held = self.tasks.get(self.pending_id) if self.pending_id else None
        ran_something = False
        waiting: list[str] = []
        from ..cancellation import checkCancelled
        while self.queue:
            checkCancelled("chat engine: between tasks")
            task = self.tasks[self.queue.pop(0)]
            if task.status != "queued":
                continue
            if any(self.tasks[d].status in ("failed", "cancelled", "skipped")
                   for d in task.depends_on if d in self.tasks):
                task.status = "skipped"
                print(f"  {task.id}: skipped (an earlier step it depends on did not complete)")
                continue
            if any(self.tasks[d].status != "done" for d in task.depends_on if d in self.tasks):
                waiting.append(task.id)   # runs once its dependency's question is answered
                print(f"  {task.id}: waiting for an earlier step")
                continue
            if task.payload.get("after") and self._flow_open():
                waiting.append(task.id)   # asked for "after this": the current flow comes first
                print(f"  {task.id}: waiting until the current flow is finished")
                continue
            if task.payload.pop("after", None):
                goal = " ".join(user_text(task.collect.goal).split())
                self._say(f"Now on to your next request: {goal}", task=task)
            ran_something = ran_something or task.id != (held.id if held else None)
            if not self._run_task(task):
                break                     # it asked the user: later tasks wait
        self.queue.extend(waiting)
        if (held and held.status == "asking" and self.supporting_turn and ran_something
                and not self.asked_this_turn):
            self._push(held)              # a side task ran; the earlier question is still open
            self._ask_event(held, f"Back to my question: {held.ask.question}")

    def _run_task(self, task: TaskState) -> bool:
        """Collect, then run. False when the task asked the user (stop draining)."""
        s, r = self.session, task.routed
        agent_id = task.collect.agent_id
        if not agent_id or r.decision == "no_capable_agent":
            # Nothing specialised covers it: the custom agent answers directly, or declines
            # gracefully and offers what is possible.
            agent_id = task.collect.agent_id = "custom-agent"
            print(f"  {task.id} -> custom-agent (fallback: {r.capability_gap or 'no capable agent'})")
        else:
            print(f"  {task.id} -> {agent_id}")
        if agent_id in NOT_BUILT:
            task.status = "skipped"
            what = {"layering-agent": "Separating designs into layers",
                    "slop-terminator": "Finding and fixing AI artifacts"}.get(agent_id, agent_id)
            self._say(f"{what} isn't available yet — it's coming soon.", task=task)
            return True

        if agent_id in NEEDS_IMAGE and not s.current and s.fallback_image \
                and not (agent_id == "manual-ops" and task.collect.desired_result == "account_action"):
            # Nothing selected and nothing made yet: work on the project's latest graphic.
            s.start_with_image(s.fallback_image, "the project's latest graphic")
            print(f"      using the project's latest graphic: {s.fallback_image}")

        if task.payload.get("waiting_agent"):
            # The agent stopped to ask; it continues from where it stopped with the reply.
            from langgraph.types import Command
            answer = task.collect.record[-1]["a"] if task.collect.record else ""
            print(f"  [4] AGENT {agent_id} resumes with the user's reply: {answer}")
            return self._invoke(task, agent_id, Command(resume=answer))

        if agent_id == "image-editor" and s.current and MANUAL_CONTROLS \
                and self._manual_control(task):
            return True

        print("  [3] COLLECT")
        result = collect(task.collect, s, self.cards[agent_id])
        s.record("collector", ok=result.status != "cannot", detail=result.status, cost=result.cost)
        for c in result.checks:
            print(f"      check: {c}")
        print(f"      status = {result.status}" + (f"  (${result.cost:.4f})" if result.cost else ""))
        for n in result.notices:
            if n not in task.collect.notices_shown:
                task.collect.notices_shown.add(n)
                self._say(n, sender="input-collector", task=task)
        if result.status == "cannot":
            task.status = "failed"
            self._say(result.reason or "I can't do that on this image.",
                      sender="input-collector", task=task)
            return True
        if result.status == "ask":
            task.status, task.ask = "asking", result
            task.collect.rounds += 1
            self._push(task)
            self.nav_history.append({"role": "assistant", "content": result.question[:500]})
            self._ask_event(task, result.question)
            if result.ask_kind == "select_image":
                print("      >> frontend: show the image picker (select_image)")
            elif result.ask_kind == "form":
                print(f"      >> frontend: show a form with {len(result.fields)} field(s)")
            return False

        print(f"  [4] AGENT {agent_id}: {result.goal}")
        history = self.histories.get(agent_id, [])
        # What the run started from, kept across any questions the agent stops to ask.
        task.payload["run"] = {"goal": result.goal, "start": len(history) + 1, "cost": s.cost,
                               "versions": len(s.versions), "actions": len(s.client_actions),
                               "trace": len(s.trace), "previous": s.current,
                               "layers": len(s.layers)}
        message = (f"{result.goal}\n({self.reply_style})\n"
                   f"[From {task.origin} via orchestrator | task {task.id} | to {agent_id}]")
        print(f"      {self._tag('orchestrator', task, agent_id)} {result.goal}")
        return self._invoke(task, agent_id, {"messages": history + [("user", message)]})

    def _invoke(self, task: TaskState, agent_id: str, inputs) -> bool:
        """Run (or resume) the agent on its own checkpoint thread. False when it asked the
        user, or when a failure opened the retry question."""
        s, run = self.session, task.payload["run"]
        progress = self._emit({"type": "progress", "agent_id": agent_id,
                               "desired_result": task.collect.desired_result},
                              sender=agent_id, task=task)
        if self.on_progress:
            try:
                self.on_progress(progress)
            except Exception as err:  # noqa: BLE001 - a spinner must never break the turn
                print(f"[chat] progress callback failed: {err}")
        from langchain_core.callbacks import UsageMetadataCallbackHandler
        usage = UsageMetadataCallbackHandler()
        config = {"recursion_limit": 60, "callbacks": [usage],
                  "configurable": {"thread_id": self._thread(task)}}
        try:
            out = self.agent(agent_id).invoke(inputs, context=s, config=config)
        except Exception as err:  # noqa: BLE001 - cancel / spend limit are BaseException
            traceback.print_exc()
            s.record(f"{agent_id}:llm", ok=False, detail=f"{type(err).__name__}: {err}",
                     cost=_llm_cost(usage.usage_metadata))
            self._say("Something went wrong while working on that.", task=task)
            task.payload.pop("waiting_agent", None)
            self._release(task)
            return self._failed(task)
        s.record(f"{agent_id}:llm", ok=True, detail="agent model calls",
                 cost=_llm_cost(usage.usage_metadata))

        stopped = out.get("__interrupt__") if isinstance(out, dict) else None
        if stopped:
            ask = getattr(stopped[0], "value", None) or {}
            question = str(ask.get("question") or "").strip() or "Could you tell me a bit more?"
            task.status, task.payload["waiting_agent"] = "asking", True
            task.ask = Collected(status="ask", question=question,
                                 options=[str(o) for o in ask.get("options") or []],
                                 ask_kind=str(ask.get("ask_kind") or "text"))
            # Anything else the agent sent with its question (e.g. what the layering
            # question accepts) rides along to the frontend frame.
            task.payload["ask_extra"] = {k: v for k, v in ask.items()
                                         if k not in ("question", "options", "ask_kind",
                                                      "show_cost", "multiple", "fields",
                                                      "allow_free_text")}
            if task.ask.ask_kind == "text":   # special asks (layering) keep their own frame
                task.payload["ask_spec"] = self._spec(
                    task, agent_id, "agent_question", question, ask.get("options"),
                    multiple=bool(ask.get("multiple")), fields=ask.get("fields"),
                    allow_free_text=ask.get("allow_free_text", True))
            self._push(task)
            self.nav_history.append({"role": "assistant", "content": question[:500]})
            self._ask_event(task, question)
            return False

        task.payload.pop("waiting_agent", None)
        self._release(task)
        self.histories[agent_id] = out["messages"]
        new = out["messages"][run["start"]:]
        print_agent_steps(new, s, run["cost"])
        if new:
            self.nav_history.append({"role": "assistant", "content": str(new[-1].content)[:500]})
        for action in s.client_actions[run["actions"]:]:
            self._emit({"type": "client_action", "action": action}, sender=agent_id, task=task,
                       recipient="frontend")
        reply = user_text(str(new[-1].content)) if new else ""
        if reply:
            print(f"      {self._tag(agent_id, task)} {reply[:300]}")
        self._emit({
            "type": "result", "agent_id": agent_id,
            "reply": reply, "goal": run["goal"],
            "new_versions": s.versions[run["versions"]:], "previous": run["previous"],
            "new_layers": s.layers[run.get("layers", len(s.layers)):],
            "cost": s.cost - run["cost"],
            "payload": {k: v for k, v in task.payload.items() if k != "run"}},
            sender=agent_id, task=task)
        new_versions = len(s.versions) - run["versions"]
        tool_failed = any(not e["ok"] for e in s.trace[run["trace"]:])
        if agent_id in IMAGE_RESULT_AGENTS and new_versions == 0 and tool_failed:
            if any(e.get("blocked") == "safety" for e in s.trace[run["trace"]:]):
                # Refused under the image services' content rules: the agent has explained
                # why and how to reword it; the same request would be refused again.
                task.status = "failed"
                return True
            return self._failed(task)
        task.status = "done"
        return self._after_agent(task, agent_id, new_versions)

    def _failed(self, task: TaskState) -> bool:
        """A job failed. Ask whether to try again (False: a question is open)."""
        task.status = "failed"
        attempts = task.payload.get("retries", 0)
        if attempts >= MAX_RETRIES:
            print(f"  {task.id}: failed again; not offering another retry")
            return True
        self._open_offer("retry", RETRY_ASK, ["Yes", "No"], payload={"task_id": task.id})
        return False

    def _manual_control(self, task: TaskState) -> bool:
        """True when the edit is one for the frontend's manual controls (sent, not run)."""
        from ..config import MANUAL_EDIT_MULTIPLE, MANUAL_EDIT_SINGLE
        from ..manual_edit_decision import classify_edit_mode, load_operations
        request = task.collect.goal + (f" ({'; '.join(task.collect.constraints)})"
                                       if task.collect.constraints else "")
        decision, cost = classify_edit_mode(request)
        self.session.record("manual_edit_decision", ok=True, detail=decision.get("mode", "ai"),
                            cost=cost)
        if decision.get("mode") not in ("manual", "manual_multiple"):
            return False
        op = decision.get("op") if decision.get("mode") == "manual" else None
        if op:
            label = next((o["label"] for o in load_operations() if o["id"] == op),
                         op.replace("_", " "))
            message = MANUAL_EDIT_SINGLE.format(label=label.lower())
        else:
            message = MANUAL_EDIT_MULTIPLE
        print(f"  [3] MANUAL CONTROL  {decision['mode']} manual_op={op} "
              f"(manual_operations.json; the frontend applies it)")
        print(f"      {self._tag('orchestrator', task, 'frontend')} {message}")
        self._emit({"type": "client_action", "action": {
            "type": "manual_edit", "manual_op": op, "multiple": op is None,
            "question": message, "prompt": task.collect.goal}},
            sender="orchestrator", task=task, recipient="frontend")
        self.nav_history.append({"role": "assistant", "content": message})
        task.status = "done"
        return True

    # ------------------------------------------------- automatic follow-ups
    def _after_agent(self, task: TaskState, agent_id: str, new_versions: int) -> bool:
        """Today's automatic flow after a result. False when it opened a question."""
        s = self.session
        if agent_id == "image-generator" and new_versions > 0:
            self._say(GENERATION_COMPLETED, task=task)
            self.more_rounds = 0
            if self.defer_auto_critique:
                self.pending_auto_critique = True
                return True
            if new_versions > 1:
                self._open_offer("pick_graphic", SUGGESTION_TARGET_ASK, [], ask_kind="select_image")
                print("      >> frontend: show the image picker (select_image)")
                return False
            self._queue_auto_critique()
            return True
        if agent_id == "critic-agent":
            if not s.suggestions:          # critique_design resets these on every run
                return True
            options = [f"{x['n']}. " + " ".join(str(x['text']).split()) for x in s.suggestions]
            self._open_offer("suggestions", SUGGESTION_MODE, options,
                             payload={"suggestions": list(s.suggestions)})
            return False
        if task.payload.get("from_suggestions"):
            if self.more_rounds < MAX_MORE_SUGGESTION_ROUNDS:
                self.more_rounds += 1
                self._open_offer("more_suggestions", SUGGESTION_MORE_ASK, ["Yes", "No"])
                return False
            self._offer_layering()
            return False
        return True

    def _offer_layering(self) -> None:
        """The suggestion loop is over: offer layering (nodes.interruptBeforeLayering)."""
        from ..config import LAYERING_OFFER
        self._open_offer("layering_offer", LAYERING_OFFER, ["Yes", "No"])

    def _new_task(self, agent_id: str, goal: str, result: str, payload: dict, *,
                  origin: str = "orchestrator") -> TaskState:
        self.auto_no += 1
        tid = f"{self.turn_no}.auto{self.auto_no}"
        routed = RoutedTask(id=tid, goal=goal, desired_result=result, performed_by="system",
                            depends_on=[], supports_task_id=None, decision="assign",
                            agent_id=agent_id, confidence="high")
        t = TaskState(id=tid, routed=routed, payload=payload, origin=origin,
                      collect=CollectTask(id=tid, agent_id=agent_id, goal=goal,
                                          desired_result=result))
        self.tasks[tid] = t
        return t

    def continue_auto_critique(self) -> None:
        """The frontend's "yes" after generation_completed: run the automatic critique."""
        self.events = []
        self.pending_auto_critique = False
        self.asked_this_turn = self.supporting_turn = False
        print(RULE)
        print("you: (auto-continue after generation_completed)")
        self._queue_auto_critique()
        self._drain()
        self._finalize()

    def _queue_auto_critique(self, focus: str = "") -> None:
        s = self.session
        purpose = (s.brief.prompt if s.brief else "") or "the design just created"
        t = self._new_task("critic-agent",
                           f"Review the design just created (version {s.current.number}) and "
                           f"suggest improvements. Its purpose: {purpose}"
                           + (f". The user wants the suggestions to focus on: {focus}" if focus else ""),
                           "design_advice", {"auto": True})
        print(f"  {t.id}: automatic critique queued")
        self.queue.insert(0, t.id)

    def _open_offer(self, kind: str, question: str, options: list[str], *,
                    ask_kind: str = "text", payload: Optional[dict] = None) -> None:
        t = self._new_task("", question, "other", payload or {})
        t.kind, t.status = kind, "asking"
        t.ask = Collected(status="ask", question=question, options=options, ask_kind=ask_kind)
        if ask_kind == "text" and kind != "suggestions":   # cards and the picker keep theirs
            t.payload["ask_spec"] = self._spec(t, "orchestrator", kind, question, options,
                                               allow_free_text=True)
        self._push(t)
        if options:
            print("        options: " + " / ".join(options))
        self.nav_history.append({"role": "assistant", "content": question[:500]})
        self._ask_event(t, question)

    def _resolve_cards(self, offer: TaskState, cards: list, text: str) -> None:
        """Suggestion cards sent back with statuses (accepted / rejected / pending, and a
        user_response for a modified one), as suggestionHandler received them."""
        items = {str(x["n"]): x for x in offer.payload.get("suggestions") or []}
        lines = []
        for c in cards:
            status = str(c.get("status") or "").lower()
            if status not in ("accepted", "accept", "modified"):
                continue
            item = items.get(str(c.get("s_id")))
            change = (c.get("user_response") or "").strip()
            base = (item or {}).get("instruction") or (item or {}).get("text") or c.get("nano_prompt") \
                or c.get("user_friendly_text") or ""
            lines.append(f"{c.get('s_id')}. {base}" + (f" (changed: {change})" if change else ""))
        answer = ("apply " + "; ".join(lines)) if lines else (text or "none")
        self._resolve_offer(offer, answer)

    def _read_reply(self, offer: TaskState, answer: str) -> tuple[str, str]:
        """What a reply to a yes/no offer means: (decision, change). A bare yes or no is read
        directly; anything longer ("yes, but with a different prompt I'll give you") is read
        by the model, so the extra words are not lost.

        decision: yes | no | yes_with_change (agrees and says now what to do differently)
                  | will_give_details (wants something different but has not said what yet)
        """
        plain = re.sub(r"[^a-z ]", "", answer.lower()).strip()
        if plain in _PLAIN_YES:
            return "yes", ""
        if plain in _PLAIN_NO:
            return "no", ""
        question = offer.ask.question if offer.ask else offer.collect.goal
        raw, cost, _ = vision_json(
            "The assistant asked the user a yes/no question and the user replied. Decide what "
            "the reply means.\n"
            f"QUESTION: {question}\nUSER REPLY: {answer}\n"
            "decision is one of:\n"
            "- yes: they agree, with nothing to change.\n"
            "- no: they decline.\n"
            "- yes_with_change: they agree and say, in this reply, what to do differently.\n"
            "- will_give_details: they want it done differently but have not said how yet "
            "(e.g. 'yes but with a different prompt I'll give you').\n"
            'Return JSON: {"decision": "...", "change": "the change in their words, or empty"}',
            [])
        self.session.record("offer_reply", ok=True, detail=answer[:120], cost=cost)
        decision = str(raw.get("decision") or "no").strip().lower()
        if decision not in ("yes", "no", "yes_with_change", "will_give_details"):
            decision = "no"
        print(f"      reply to '{offer.kind}' read as: {decision}"
              + (f" ({raw.get('change')})" if raw.get("change") else ""))
        return decision, str(raw.get("change") or "").strip()

    def _resolve_offer(self, offer: TaskState, answer: str) -> None:
        offer.status = "done"
        self._unstack(offer)
        if offer.kind in ("resume", "retry", "more_suggestions", "layering_offer"):
            decision, change = self._read_reply(offer, answer)
        if offer.kind == "resume":
            task = self.tasks.get(offer.payload.get("task_id", ""))
            if not task or task.status != "asking":
                return
            if decision == "yes_with_change" and change:
                self._answer(task, change)    # "continue, make it blue" answers its question
            elif decision != "no":
                self._push(task)          # its question is open again, on top
                question = task.ask.question if task.ask else task.collect.goal
                self.nav_history.append({"role": "assistant", "content": question[:500]})
                self._ask_event(task, question)
            else:
                self._cancel(task)
                self._say(RESUME_DROPPED, task=task)
            return
        if offer.kind == "pick_graphic":
            self._queue_auto_critique()
            return
        if offer.kind == "retry":
            task = self.tasks.get(offer.payload.get("task_id", ""))
            if task and decision in ("yes", "yes_with_change"):
                if decision == "yes_with_change" and change:
                    task.collect.goal += f"\nChange the user asked for before trying again: {change}"
                    task.collect.rounds = 0
                task.payload["retries"] = task.payload.get("retries", 0) + 1
                task.payload.pop("waiting_agent", None)
                task.status = "queued"
                self.queue.insert(0, task.id)
                print(f"      retrying {task.id}")
            elif decision == "will_give_details":
                self._say(RETRY_NEW_DETAILS)  # their next message is the new request
            else:
                self._say(RETRY_DECLINED)
            return
        if offer.kind == "more_suggestions":
            if decision in ("yes", "yes_with_change"):
                self._queue_auto_critique(focus=change)
            else:
                self._offer_layering()
            return
        if offer.kind == "layering_offer":
            if decision != "no":
                v = self.session.current
                t = self._new_task("layering-agent",
                                   "Separate the current design"
                                   + (f" (version {v.number})" if v else "")
                                   + " into layers the user can edit individually."
                                   + (f" What to separate: {change}" if change else ""),
                                   "extracted_element", {}, origin="user")
                self.queue.insert(0, t.id)
            else:
                self._say(SUGGESTION_DECLINED)
            return
        # suggestions: map the reply onto accept / reject / modify per suggestion
        items = offer.payload.get("suggestions") or []
        listing = "\n".join(f"{x['n']}. [{x['type']}] {x['text']} || {x.get('instruction', '')}"
                            for x in items)
        raw, cost, _ = vision_json(
            "The user was shown these design suggestions and replied. Decide which to apply. "
            "A modification ('2 but in blue') applies that suggestion with the change. 'all' "
            "applies all; 'none', 'no thanks' applies none.\n"
            f"SUGGESTIONS:\n{listing}\nUSER REPLY: {answer}\n"
            'Return JSON: {"apply": [{"n": 1, "type": "EDIT|UPSCALE|GENERATION", '
            '"instruction": "one complete edit instruction"}], '
            '"unknown": [numbers the user named that are not in the list]}', [])
        self.session.record("suggestion_resolver", ok=True, detail=answer[:120], cost=cost)
        missing = [str(n) for n in raw.get("unknown") or []]
        if missing:
            self._say(f"There's no suggestion {', '.join(missing)} — I had {len(items)}.")
        chosen = [a for a in raw.get("apply") or [] if a.get("instruction")]
        if not chosen:
            self._offer_layering()        # all rejected: the loop is over
            return
        edits = [a for a in chosen if str(a.get("type", "EDIT")).upper() != "GENERATION"]
        regens = [a for a in chosen if str(a.get("type", "")).upper() == "GENERATION"]
        print(f"      accepted: {[a.get('n') for a in chosen]}")
        new_ids = []
        if edits:
            goal = "Apply these improvements to the current design: " + "; ".join(
                a["instruction"] for a in edits)
            new_ids.append(self._new_task("image-editor", goal, "changed_image",
                                          {"from_suggestions": True}, origin="user").id)
        for a in regens:
            new_ids.append(self._new_task(
                "image-generator", f"Regenerate the design with this change: {a['instruction']}",
                "new_image", {"from_suggestions": True}, origin="user").id)
        self.queue[0:0] = new_ids

    # ------------------------------------------------------------ finalize
    def _finalize(self) -> None:
        """Nothing new was asked this turn: offer the newest task still waiting on the user
        again. The user continues it (its question comes back) or drops it."""
        if self.asked_this_turn or not self.stack or self.pending_auto_critique:
            return                        # (the critique flow after a new design comes first)
        top = self.tasks[self.stack[-1]]
        if top.kind != "task":            # an offer is still open: leave it
            return
        goal = " ".join(user_text(top.collect.goal).split())
        self._open_offer("resume", RESUME_ASK.format(goal=goal), ["Continue", "Drop it"],
                         payload={"task_id": top.id})
