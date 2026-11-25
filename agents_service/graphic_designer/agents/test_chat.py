"""Offline tests of the chat's task stack (navigation, collector and agents scripted).

    .venv/bin/python -m pytest graphic_designer/agents/test_chat.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_navigation.navigator import Route, RoutedTask
from graphic_designer.agents import chat as chat_mod
from graphic_designer.agents.collector import Collected
from graphic_designer.agents.session import DesignSession


def task(tid, agent, goal, result="changed_image", by="system", depends=()):
    return RoutedTask(id=tid, goal=goal, desired_result=result, performed_by=by,
                      depends_on=list(depends), supports_task_id=None, decision="assign",
                      agent_id=agent, confidence="high")


class FakeAgent:
    def __init__(self, log, agent_id):
        self.log, self.agent_id = log, agent_id

    def invoke(self, inputs, context=None, config=None):
        goal = inputs["messages"][-1][1].split("\n(Reply to the person")[0]
        self.log.append((self.agent_id, goal))
        return {"messages": [*inputs["messages"][:-1], SimpleNamespace(type="human", content=goal),
                             SimpleNamespace(type="ai", content=f"done: {goal}", tool_calls=[])]}


@pytest.fixture
def chat(monkeypatch):
    session = DesignSession()
    session.start_with_image("https://example.com/fruit.webp")
    c = chat_mod.Chat.__new__(chat_mod.Chat)
    c.session, c.model, c._agents, c.histories = session, "x", {}, {}
    c.nav_history, c.tasks, c.queue, c.stack, c.turn_no = [], {}, [], [], 0
    c.asked_this_turn = c.supporting_turn = False
    c.chat_key, c.ask_seq = "test", 0
    c.cards = {a: SimpleNamespace(id=a) for a in
               ("image-editor", "image-generator", "question-handler", "chitchat-handler")}
    c.nav_llm = SimpleNamespace(usage=SimpleNamespace(totals=lambda: {"calls": 0, "input_tokens": 0},
                                                      calls=[]))
    c.events, c.verbose, c.defer_auto_critique, c.pending_auto_critique = [], False, False, False
    c.on_progress, c.reply_style = None, "Reply to the person directly, as 'you'."
    c.ran, c.routes, c.collects = [], [], []
    c.more_rounds, c.auto_no = 0, 0
    c.agent = lambda a: FakeAgent(c.ran, a)
    c.navigator = SimpleNamespace(route=lambda *a, **k: c.routes.pop(0))
    monkeypatch.setattr(chat_mod, "collect", lambda t, s, card: c.collects.pop(0))
    monkeypatch.setattr(chat_mod, "MANUAL_CONTROLS", False)   # no Decisions API in unit tests
    return c


def ready(goal):
    return Collected(status="ready", goal=goal)


def ask(q, kind="text"):
    return Collected(status="ask", question=q, ask_kind=kind)


def test_missing_target_asks_side_question_keeps_it_then_answer_runs(chat):
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the cup")]),
                   Route("supporting_task", tasks=[task("T1", "question-handler", "what can you do",
                                                        result="information")]),
                   Route("input_for_pending_task", pending_input={"task_id": "1.T1", "input": "the banana"})]
    chat.collects = [ask("I don't see a cup. Which item?"), ready("what can you do"),
                     ready("remove the banana")]
    chat.turn("remove the cup")
    assert chat.pending_id == "1.T1" and chat.ran == []
    chat.turn("what can you do?")
    assert chat.ran == [("question-handler", "what can you do")]
    assert chat.pending_id == "1.T1"                       # the question is still open
    chat.turn("the banana")
    assert chat.ran[-1] == ("image-editor", "remove the banana")
    assert chat.tasks["1.T1"].collect.record[-1]["a"] == "the banana"
    assert chat.pending_id is None


def test_new_task_keeps_question_on_stack_then_offers_continue_or_drop(chat):
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "change the colour")]),
                   Route("new_task", tasks=[task("T1", "image-generator", "diwali post", "new_image")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "drop it"})]
    chat.collects = [ask("Which colour?"), ready("diwali post")]
    chat.agent = lambda a: FlowAgent(chat.ran, a, chat.session)   # the generator adds a version
    chat.more_rounds, chat.auto_no = 0, 0
    chat.defer_auto_critique = True
    chat.turn("change the colour")
    chat.turn("make a diwali post instead")
    assert ("image-generator", "diwali post") in chat.ran
    assert chat.stack == ["1.T1"]                 # still waiting; critique flow comes first
    chat.pending_auto_critique = False            # (the critique flow has finished)
    chat._finalize()
    offer = chat.tasks[chat.pending_id]
    assert offer.kind == "resume" and offer.payload["task_id"] == "1.T1"
    chat.turn("drop it")
    assert chat.tasks["1.T1"].status == "cancelled" and chat.stack == []
    assert chat.events[-1]["text"] == chat_mod.RESUME_DROPPED


def test_cancel_drops_pending_and_its_dependents(chat):
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "change the colour"),
                                            task("T2", "image-editor", "crop square", depends=["T1"])]),
                   Route("cancellation")]
    chat.collects = [ask("Which colour?")]
    chat.turn("change the colour then crop square")
    chat.turn("forget it")
    assert chat.tasks["1.T1"].status == "cancelled" and chat.tasks["1.T2"].status == "cancelled"
    assert chat.ran == [] and chat.pending_id is None


def test_image_only_reply_answers_select_image_without_navigation(chat):
    chat.session.versions.clear()
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the banana")])]
    chat.collects = [ask("Please select the image.", kind="select_image"), ready("remove the banana")]
    chat.turn("remove the banana")
    assert chat.pending_id == "1.T1"
    chat.turn("", selected_image="https://example.com/fruit.webp")    # no route consumed
    assert chat.ran == [("image-editor", "remove the banana")] and chat.routes == []


class FlowAgent(FakeAgent):
    """Generator adds a version; critic leaves two suggestions, like the real tools."""

    def __init__(self, log, agent_id, session):
        super().__init__(log, agent_id)
        self.session = session

    def invoke(self, inputs, context=None, config=None):
        if self.agent_id == "image-generator":
            self.session.add_version("https://example.com/new.webp", made_by="image-generator",
                                     tool="generate_catalog_design", instruction="diwali")
        if self.agent_id == "critic-agent":
            self.session.suggestions = [
                {"n": 1, "text": "Bigger headline", "type": "EDIT", "instruction": "enlarge headline"},
                {"n": 2, "text": "Warmer palette", "type": "EDIT", "instruction": "warm the colours"}]
        return super().invoke(inputs, context, config)


@pytest.fixture
def flow(chat, monkeypatch):
    chat.cards["critic-agent"] = SimpleNamespace(id="critic-agent")
    chat.agent = lambda a: FlowAgent(chat.ran, a, chat.session)
    chat.more_rounds, chat.auto_no = 0, 0
    monkeypatch.setattr(chat_mod, "vision_json", lambda prompt, images: (
        {"apply": [{"n": 1, "type": "EDIT", "instruction": "enlarge headline"}]}, 0.0, 0.0))
    return chat


def test_generation_runs_critique_then_suggestion_mode_then_more_ask(flow):
    flow.routes = [Route("new_task", tasks=[task("T1", "image-generator", "diwali post", "new_image")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "apply 1"}),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "no"})]
    flow.collects = [ready("diwali post"), ready("critique"), ready("apply edits")]
    flow.turn("make a diwali post")
    assert [a for a, _ in flow.ran] == ["image-generator", "critic-agent"]   # critique ran by itself
    assert flow.tasks[flow.pending_id].kind == "suggestions"
    flow.turn("apply 1")
    assert flow.ran[-1][0] == "image-editor"
    editor_task = next(t for t in flow.tasks.values() if t.collect.agent_id == "image-editor")
    assert "enlarge headline" in editor_task.collect.goal and editor_task.payload["from_suggestions"]
    assert flow.tasks[flow.pending_id].kind == "more_suggestions"
    flow.turn("no")
    assert flow.tasks[flow.pending_id].kind == "layering_offer" and len(flow.ran) == 3


def test_new_request_exits_suggestion_mode_instead_of_parking(flow):
    flow.routes = [Route("new_task", tasks=[task("T1", "image-generator", "diwali post", "new_image")]),
                   Route("new_task", tasks=[task("T1", "chitchat-handler", "hi", "conversation")])]
    flow.collects = [ready("diwali post"), ready("critique"), ready("hi")]
    flow.turn("make a diwali post")
    offer = flow.pending_id
    flow.turn("hi")
    assert flow.tasks[offer].status == "cancelled" and flow.pending_id is None   # not brought back


def test_exact_value_edit_goes_to_the_frontend_control_not_the_ai(chat, monkeypatch):
    import graphic_designer.manual_edit_decision as med
    monkeypatch.setattr(chat_mod, "MANUAL_CONTROLS", True)
    monkeypatch.setattr(med, "classify_edit_mode",
                        lambda prompt: ({"mode": "manual", "op": "rotation", "ops": ["rotation"]}, 0.0))
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "rotate the image 46 degrees")])]
    chat.turn("rotate the image 46 degrees")
    assert chat.ran == [] and chat.collects == []          # no collector, no AI edit
    action = next(e["action"] for e in chat.events if e["type"] == "client_action")
    assert action["manual_op"] == "rotation" and action["multiple"] is False


def test_vague_edit_still_goes_to_the_ai_editor(chat, monkeypatch):
    import graphic_designer.manual_edit_decision as med
    monkeypatch.setattr(chat_mod, "MANUAL_CONTROLS", True)
    monkeypatch.setattr(med, "classify_edit_mode",
                        lambda prompt: ({"mode": "ai", "op": None, "ops": []}, 0.0))
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "make it a bit brighter")])]
    chat.collects = [ready("make it a bit brighter")]
    chat.turn("make it a bit brighter")
    assert chat.ran == [("image-editor", "make it a bit brighter")]


class FailingAgent(FakeAgent):
    """Records a failed tool call and makes no new version, like an executor error."""
    def invoke(self, inputs, context=None, config=None):
        context.record("remove_object", ok=False, detail="FAILED: executor timeout")
        return super().invoke(inputs, context, config)


def test_failed_edit_offers_retry_and_yes_runs_it_again(chat):
    attempts = []
    chat.agent = lambda a: (FailingAgent if not attempts else FakeAgent)(attempts, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the banana")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "yes"})]
    chat.collects = [ready("remove the banana"), ready("remove the banana")]
    chat.turn("remove the banana")
    offer = chat.tasks[chat.pending_id]
    assert offer.kind == "retry" and chat.tasks["1.T1"].status == "failed"
    assert chat.events[-1]["question"] == chat_mod.RETRY_ASK
    chat.turn("yes")
    assert len(attempts) == 2 and chat.tasks["1.T1"].status == "done"
    assert chat.pending_id is None


def test_agent_crash_offers_retry_once_then_only_reports(chat):
    class Crash:
        def invoke(self, *a, **k):
            raise RuntimeError("model unavailable")
    chat.agent = lambda a: Crash()
    chat.routes = [Route("new_task", tasks=[task("T1", "image-generator", "make a poster",
                                                 result="new_image")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "retry"})]
    chat.collects = [ready("make a poster"), ready("make a poster")]
    chat.turn("make a poster")
    assert chat.tasks[chat.pending_id].kind == "retry"
    chat.turn("retry")
    assert chat.tasks["1.T1"].status == "failed" and chat.pending_id is None


def test_no_to_retry_leaves_it(chat):
    chat.agent = lambda a: FailingAgent([], a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the banana")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "no"})]
    chat.collects = [ready("remove the banana")]
    chat.turn("remove the banana")
    chat.turn("no")
    assert chat.events[-1]["text"] == chat_mod.RETRY_DECLINED
    assert chat.tasks["1.T1"].status == "failed" and chat.pending_id is None


class AskingAgent(FakeAgent):
    """Stops once to ask (an ask_user interrupt), then finishes with the reply."""
    def invoke(self, inputs, context=None, config=None):
        from langgraph.types import Command
        if isinstance(inputs, Command):
            self.log.append((self.agent_id, f"resumed: {inputs.resume}"))
            return {"messages": [SimpleNamespace(type="ai", content=f"done with {inputs.resume}",
                                                 tool_calls=[])]}
        self.log.append((self.agent_id, "asked"))
        return {"messages": inputs["messages"], "__interrupt__": [SimpleNamespace(
            value={"question": "Which cup, left or right?", "options": ["Left", "Right"]})]}


def test_agent_question_then_reply_resumes_the_same_run(chat):
    chat.agent = lambda a: AskingAgent(chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the cup")]),
                   Route("input_for_pending_task", pending_input={"task_id": "1.T1", "input": "left"})]
    chat.collects = [ready("remove the cup")]           # collector runs once, not on resume
    chat.turn("remove the cup")
    t = chat.tasks["1.T1"]
    assert t.status == "asking" and chat.stack == ["1.T1"]
    assert chat.events[-1]["question"] == "Which cup, left or right?"
    assert chat.events[-1]["options"] == ["Left", "Right"]
    chat.turn("the left one")
    assert chat.ran == [("image-editor", "asked"), ("image-editor", "resumed: left")]
    assert t.status == "done" and chat.stack == [] and chat.collects == []


def test_continue_brings_back_agent_question_and_reply_resumes(chat):
    agents = {"image-editor": AskingAgent, "chitchat-handler": FakeAgent}
    chat.agent = lambda a: agents[a](chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the cup")]),
                   Route("new_task", tasks=[task("T1", "chitchat-handler", "hi", "conversation")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "continue"}),
                   Route("input_for_pending_task", pending_input={"task_id": "1.T1", "input": "right"})]
    chat.collects = [ready("remove the cup"), ready("hi")]
    chat.turn("remove the cup")
    chat.turn("hi")
    assert chat.tasks[chat.pending_id].kind == "resume" and "1.T1" in chat.stack
    chat.turn("continue")
    assert chat.pending_id == "1.T1" and chat.events[-1]["question"] == "Which cup, left or right?"
    chat.turn("right")
    assert chat.ran[-1] == ("image-editor", "resumed: right") and chat.stack == []


def test_reply_can_answer_an_older_waiting_task(chat):
    chat.agent = lambda a: AskingAgent(chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the cup")]),
                   Route("new_task", tasks=[task("T1", "image-editor", "change the colour")]),
                   Route("input_for_pending_task", pending_input={"task_id": "1.T1", "input": "left"})]
    chat.collects = [ready("remove the cup"), ask("Which colour?")]
    chat.turn("remove the cup")
    chat.turn("change the colour")
    assert chat.stack == ["1.T1", "2.T1"]
    chat.turn("the left cup")
    assert chat.ran[-1] == ("image-editor", "resumed: left")
    reminder = chat.tasks[chat.pending_id]        # the colour request is offered again
    assert reminder.kind == "resume" and reminder.payload["task_id"] == "2.T1"


def test_every_message_says_sender_task_and_recipient(chat):
    chat.agent = lambda a: AskingAgent(chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the cup")]),
                   Route("input_for_pending_task", pending_input={"task_id": "1.T1", "input": "left"})]
    chat.collects = [Collected(status="ready", goal="remove the cup", notices=["Using version 1."])]
    chat.turn("remove the cup")
    notice, progress, question = chat.events
    assert (notice["sender"], notice["task_id"], notice["recipient"]) == ("input-collector", "1.T1", "user")
    assert (progress["sender"], progress["task_id"]) == ("image-editor", "1.T1")
    assert (question["sender"], question["task_id"], question["recipient"]) == ("image-editor", "1.T1", "user")
    chat.turn("left")
    result = [e for e in chat.events if e["type"] == "result"][0]
    assert (result["sender"], result["task_id"], result["recipient"]) == ("image-editor", "1.T1", "user")


def test_agent_request_names_sender_task_and_recipient(chat):
    seen = []

    class Spy(FakeAgent):
        def invoke(self, inputs, context=None, config=None):
            seen.append(inputs["messages"][-1][1])
            return super().invoke(inputs, context, config)
    chat.agent = lambda a: Spy(chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the cup")])]
    chat.collects = [ready("remove the cup")]
    chat.turn("remove the cup")
    assert seen[0].endswith("[From user via orchestrator | task 1.T1 | to image-editor]")


def test_drawn_boxes_answer_the_layering_question_without_navigation(chat):
    class LayerAsk(AskingAgent):
        def invoke(self, inputs, context=None, config=None):
            from langgraph.types import Command
            if isinstance(inputs, Command):
                return super().invoke(inputs, context, config)
            self.log.append((self.agent_id, "asked"))
            return {"messages": inputs["messages"], "__interrupt__": [SimpleNamespace(value={
                "question": "Which elements?", "ask_kind": "layering_targets",
                "accepts": ["object_names", "bboxes"]})]}
    chat.cards["layering-agent"] = SimpleNamespace(id="layering-agent")
    chat.agent = lambda a: LayerAsk(chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "layering-agent", "make layers", "layers")])]
    chat.collects = [ready("make layers")]
    chat.turn("make layers")
    assert chat.events[-1]["ask_kind"] == "layering_targets"
    assert chat.events[-1]["extra"]["accepts"] == ["object_names", "bboxes"]
    chat.turn("", bboxes=[{"bbox": [1, 2, 3, 4], "label": "object_1"}])   # no route consumed
    assert chat.ran[-1] == ("layering-agent", "resumed: (drew 1 box(es) on the image)")
    assert chat.session.manual_bboxes == [{"bbox": [1, 2, 3, 4], "label": "object_1"}]


def test_suggestion_loop_ends_with_the_layering_offer(flow):
    flow.cards["layering-agent"] = SimpleNamespace(id="layering-agent")
    offer = flow._new_task("", "Want more?", "other", {})
    offer.kind = "more_suggestions"
    flow._resolve_offer(offer, "no")
    layering_offer = flow.tasks[flow.pending_id]
    assert layering_offer.kind == "layering_offer"
    flow.collects = [ready("Separate the current design into layers")]
    flow._resolve_offer(layering_offer, "yes")
    flow._drain()
    assert flow.ran[-1][0] == "layering-agent"


def test_several_layering_tasks_in_one_message_run_once(chat):
    chat.cards["layering-agent"] = SimpleNamespace(id="layering-agent")
    chat.routes = [Route("new_task", tasks=[
        task("T1", "layering-agent", "separate the headline", "extracted_element"),
        task("T2", "layering-agent", "separate the server racks", "extracted_element")])]
    chat.collects = [ready("separate the headline; separate the server racks")]
    chat.turn("separate the headline and the server racks into layers")
    assert [a for a, _ in chat.ran] == ["layering-agent"]
    assert "2.T2" not in chat.tasks and "1.T2" not in chat.tasks


def test_edit_with_nothing_selected_uses_the_latest_graphic(chat):
    chat.session.versions.clear()
    chat.session.fallback_image = "https://example.com/latest.webp"
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "make it brighter")])]
    chat.collects = [ready("make it brighter")]
    chat.turn("make it brighter")
    assert chat.session.current.url == "https://example.com/latest.webp"
    assert chat.ran == [("image-editor", "make it brighter")]


def test_generation_with_nothing_selected_leaves_the_latest_graphic_alone(chat):
    chat.session.versions.clear()
    chat.session.fallback_image = "https://example.com/latest.webp"
    chat.routes = [Route("new_task", tasks=[task("T1", "image-generator", "diwali post", "new_image")])]
    chat.collects = [ready("diwali post")]
    chat.turn("make a diwali post")
    assert chat.session.current is None


def test_retry_reply_with_more_words_is_read_not_pattern_matched(chat, monkeypatch):
    attempts = []
    chat.agent = lambda a: FailingAgent(attempts, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-generator", "iron man suit sale", "new_image")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x",
                         "input": "yes try again but with a different prompt I'll give you"})]
    chat.collects = [ready("iron man suit sale")]
    monkeypatch.setattr(chat_mod, "vision_json", lambda prompt, images: (
        {"decision": "will_give_details", "change": ""}, 0.0, 0.0))
    chat.turn("make an iron man suit sale post")
    chat.turn("yes try again but with a different prompt I'll give you")
    assert len(attempts) == 1                                   # not retried blindly
    assert chat.events[-1]["text"] == chat_mod.RETRY_NEW_DETAILS and chat.pending_id is None


def test_retry_with_a_change_reruns_with_it(chat, monkeypatch):
    attempts = []
    chat.agent = lambda a: (FailingAgent if not attempts else FakeAgent)(attempts, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-generator", "suit sale", "new_image")]),
                   Route("input_for_pending_task", pending_input={"task_id": "x",
                         "input": "yes, but make it a red and gold armored suit"})]
    chat.collects = [ready("suit sale"), ready("suit sale, red and gold armored suit")]
    monkeypatch.setattr(chat_mod, "vision_json", lambda prompt, images: (
        {"decision": "yes_with_change", "change": "a red and gold armored suit"}, 0.0, 0.0))
    chat.turn("suit sale")
    chat.turn("yes, but make it a red and gold armored suit")
    assert "red and gold armored suit" in chat.tasks["1.T1"].collect.goal
    assert len(attempts) == 2 and chat.tasks["1.T1"].status == "done"


def test_safety_refusal_is_explained_without_a_retry_offer(chat):
    class Refused(FakeAgent):
        def invoke(self, inputs, context=None, config=None):
            context.record("generate_catalog_design", ok=False, detail="BLOCKED_BY_SAFETY",
                           blocked="safety")
            return super().invoke(inputs, context, config)
    chat.agent = lambda a: Refused(chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-generator", "iron man suit", "new_image")])]
    chat.collects = [ready("iron man suit")]
    chat.turn("make an iron man suit post")
    assert chat.tasks["1.T1"].status == "failed" and chat.pending_id is None
    assert not any(e["type"] == "ask" for e in chat.events)


def test_safety_refusal_is_detected_from_the_render_attempts():
    from graphic_designer.agents.generation_tools import _safety_block
    attempts = [{"model": "gpt-image-2.5-flare", "ok": False, "detail": "BadRequestError: Error code: "
                 "400 - {'error': {'message': 'Your request was rejected by the safety system."},
                {"model": "gemini-3.1-flash-image", "ok": False, "detail": "no image part in response"}]
    assert _safety_block({"v33_render_attempts": attempts})
    assert not _safety_block({"v33_render_attempts": [{"detail": "TimeoutError"}]})


def test_answer_plus_new_task_finishes_the_current_one_first(chat):
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "change the colour")]),
                   Route("input_for_pending_task", pending_input={"task_id": "1.T1", "input": "blue"},
                         tasks=[task("T1", "image-generator", "an image with a dog", "new_image")])]
    chat.collects = [ask("Which colour?"), ready("make it blue"), ready("an image with a dog")]
    chat.turn("change the colour")
    chat.turn("blue, and after this generate another image with a dog")
    assert [a for a, _ in chat.ran] == ["image-editor", "image-generator"]   # current first
    assert any(e.get("text", "").startswith("Now on to your next request") for e in chat.events)


def test_new_task_starts_once_the_design_is_done(flow):
    flow.defer_auto_critique = True
    flow.routes = [Route("new_task", tasks=[task("T1", "image-generator", "suit sale", "new_image")]),
                   Route("input_for_pending_task", pending_input={"task_id": "1.T1", "input": "70%"},
                         tasks=[task("T1", "image-generator", "a dog image", "new_image")])]
    flow.collects = [ask("What is the offer?"), ready("suit sale 70%"), ready("a dog image")]
    flow.turn("suit sale")
    flow.turn("70% and after this make a dog image")
    # the design is done (critique waits for the frontend's go-ahead): the dog starts now
    assert [g for _, g in flow.ran] == ["suit sale 70%", "a dog image"]


def test_new_task_waits_while_a_suggestion_question_is_open(flow, monkeypatch):
    flow.routes = [Route("new_task", tasks=[task("T1", "image-generator", "suit sale", "new_image")])]
    flow.collects = [ready("suit sale"), ready("review")]
    flow.turn("suit sale")                                        # critique ran, cards open
    assert flow.tasks[flow.pending_id].kind == "suggestions"
    later = flow._add([task("T9", "image-generator", "a dog image", "new_image")])
    flow.tasks[later[0]].payload["after"] = "1.T1"
    flow.queue.extend(later)
    flow._drain()
    assert flow.ran[-1][0] == "critic-agent"                      # dog still waiting


def test_waiting_new_task_runs_when_the_flow_ends(flow, monkeypatch):
    test_new_task_waits_while_a_suggestion_question_is_open(flow, monkeypatch)
    monkeypatch.setattr(chat_mod, "vision_json", lambda prompt, images: ({"apply": []}, 0.0, 0.0))
    flow.collects = [ready("a dog image"), ready("review the dog image")]   # dog, then its critique
    flow.routes = [Route("input_for_pending_task", pending_input={"task_id": "x", "input": "none"}),
                   Route("input_for_pending_task", pending_input={"task_id": "x", "input": "no"})]
    flow.turn("none")                                             # -> layering offer
    assert flow.tasks[flow.pending_id].kind == "layering_offer"
    flow.turn("no")                                               # flow over -> dog runs
    assert ("image-generator", "a dog image") in flow.ran


def test_agent_question_is_a_structured_ask_and_a_button_click_answers_it(chat):
    chat.agent = lambda a: AskingAgent(chat.ran, a)
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "remove the cup")])]
    chat.collects = [ready("remove the cup")]
    chat.turn("remove the cup")
    spec = chat.events[-1]["ask_spec"]
    assert spec["schema"] == "kognie.ask.v1" and spec["kind"] == "single_choice"
    assert spec["asked_by"] == "image-editor" and spec["task_id"] == "1.T1"
    assert [o["label"] for o in spec["options"]] == ["Left", "Right"]
    chat.turn("", ask_reply={"id": spec["id"], "option_ids": ["left"]})   # no route consumed
    assert chat.ran[-1] == ("image-editor", "resumed: Left")


def test_bot_offers_are_structured_too(chat):
    chat.routes = [Route("new_task", tasks=[task("T1", "image-editor", "change the colour")]),
                   Route("new_task", tasks=[task("T1", "chitchat-handler", "hi", "conversation")])]
    chat.collects = [ask("Which colour?"), ready("hi")]
    chat.turn("change the colour")
    chat.turn("hi")
    spec = chat.events[-1]["ask_spec"]
    assert spec["purpose"] == "resume" and spec["asked_by"] == "orchestrator"
    assert [o["id"] for o in spec["options"]] == ["continue", "drop_it"]
    assert "- Continue" not in chat.events[-1]["question"]
    chat.turn("", ask_reply={"id": spec["id"], "option_ids": ["drop_it"]})
    assert chat.tasks["1.T1"].status == "cancelled"
