"""End-to-end flows through the orchestrator graph, with a scripted model."""
from __future__ import annotations

from PIL import Image

from conftest import (assign, assignments, intent, interpretation, plans, ready, ready_all,
                      tools)
from image_agents.schemas import AssetRef


def test_multi_target_input_runs_as_a_chain_in_one_call_per_step(assistant, llm, backend,
                                                                  banner):
    llm.add("interpret", interpretation(
        intent("t1", "Enlarge the headline in a1 v1."),
        intent("t2", "Change the background of a1 to navy blue.", depends=["t1"]),
        intent("t3", "Round the corners of a1.", depends=["t2"]),
        intent("t4", "Make the CTA button in a1 orange.", depends=["t3"]),
    ))
    llm.add("select", assignments(*[(f"T{i}", "image-editor") for i in range(1, 5)]))
    llm.add("readiness", ready_all("T1", "T2", "T3", "T4", target="x", change="y"))
    llm.add("editor_plan", plans(
        ("T1", tools(("adjust", {"brightness": 1.1}), amount="10% larger")),
        ("T2", tools(("generative_edit", {"instruction": "navy blue background"}))),
        ("T3", tools(("round_corners", {"radius": 20}))),
        ("T4", tools(("recolor_region", {"box": [10, 10, 60, 40], "color": "orange"})))))

    reply = assistant.send("s1", "Make the headline bigger, change the background to navy "
                                 "blue, round the corners and make the CTA button orange.")

    # Four tasks, but one model call per step.
    assert [step for step, _ in llm.calls] == ["interpret", "select", "readiness", "editor_plan"]
    tasks = assistant.tasks("s1")
    assert [t["status"] for t in tasks.values()] == ["completed"] * 4
    # Each task worked on the result of the one before it.
    assert reply["assets"] == [{"asset_id": "a1", "version": 5}]
    assert [assistant.store.get(AssetRef(asset_id="a1", version=v)).tool
            for v in range(2, 6)] == ["adjust", "generative_edit", "round_corners",
                                      "recolor_region"]
    assert backend.instructions == ["navy blue background"]
    # One generative step in the plan, so the quality gate ran once, on the final version.
    assert "evaluation" in assistant.store.get(AssetRef(asset_id="a1", version=5)).reports
    assert "evaluation" not in assistant.store.get(AssetRef(asset_id="a1", version=3)).reports
    assert reply["request"] is None and not llm.remaining()


def test_batch_size_is_capped(llm, tmp_path):
    from image_agents import Assistant, Config, Limits
    from image_agents.agents import ImageEditorAgent

    assistant = Assistant(llm, tmp_path, agents=[ImageEditorAgent()],
                          config=Config(limits=Limits(max_batch_size=2)))
    assistant.add_image("s1", Image.new("RGB", (100, 100)), "banner")
    llm.add("interpret", interpretation(
        intent("t1", "Make a1 warmer."),
        intent("t2", "Make a1 brighter.", depends=["t1"]),
        intent("t3", "Round the corners of a1.", depends=["t2"])))
    llm.add("select", assignments(("T1", "image-editor"), ("T2", "image-editor"),
                                  ("T3", "image-editor")))
    llm.add("readiness", ready_all("T1", "T2", change="x"), ready_all("T3", change="x"))
    llm.add("editor_plan",
            plans(("T1", tools(("adjust", {"warmth": 0.4}))),
                  ("T2", tools(("adjust", {"brightness": 1.1})))),
            plans(("T3", tools(("round_corners", {"radius": 8})))))
    assistant.send("s1", "warmer, brighter, rounded corners")
    assert [s for s, _ in llm.calls].count("editor_plan") == 2
    assert all(t["status"] == "completed" for t in assistant.tasks("s1").values())


def test_batched_plan_that_misses_a_task_falls_back_to_one_at_a_time(assistant, llm, banner):
    llm.add("interpret", interpretation(
        intent("t1", "Make a1 warmer."),
        intent("t2", "Round the corners of a1.", depends=["t1"])))
    llm.add("select", assignments(("T1", "image-editor"), ("T2", "image-editor")))
    llm.add("readiness", ready_all("T1", "T2", change="x"))
    llm.add("editor_plan",
            {"plans": [{"task_id": "T9", "action": "run_tools", "calls": []}]},  # unusable
            tools(("adjust", {"warmth": 0.4})),                                  # T1 alone
            tools(("round_corners", {"radius": 8})))                             # T2 alone
    reply = assistant.send("s1", "warmer, and round the corners")
    assert [s for s, _ in llm.calls].count("editor_plan") == 3
    assert all(t["status"] == "completed" for t in assistant.tasks("s1").values())
    assert reply["assets"] == [{"asset_id": "a1", "version": 3}]


def test_batched_selection_that_misses_a_task_asks_again_for_that_task(assistant, llm, banner):
    llm.add("interpret", interpretation(
        intent("t1", "Make a1 warmer."),
        intent("t2", "Suggest how to improve a1.", result="design_advice", depends=["t1"])))
    llm.add("select", assignments(("T1", "image-editor")),           # T2 left out
            assign("critic-agent"))                                   # asked again, alone
    llm.add("readiness", ready(change="warmer"))
    llm.add("editor_plan", tools(("adjust", {"warmth": 0.4})))
    assistant.send("s1", "make it warmer and tell me how to improve it")
    tasks = assistant.tasks("s1")
    assert tasks["T2"]["agent_id"] == "critic-agent"
    assert [t["status"] for t in tasks.values()] == ["completed", "completed"]


def test_gap_found_for_a_later_task_is_asked_when_its_turn_comes(assistant, llm, banner):
    llm.add("interpret", interpretation(
        intent("t1", "Make a1 warmer."),
        intent("t2", "Change the colour of the car in a1.", depends=["t1"])))
    llm.add("select", assignments(("T1", "image-editor"), ("T2", "image-editor")))
    llm.add("readiness", {"results": [
        {**ready(change="warmer"), "task_id": "T1"},
        {"task_id": "T2", "resolved": {"target": "the car"}, "defaults_used": [],
         "missing": [{"name": "new_value", "request": "Which colour should the car be?",
                      "options": []}]}]})
    llm.add("editor_plan", tools(("adjust", {"warmth": 0.4})))       # T1 runs alone
    reply = assistant.send("s1", "make it warmer and change the car's colour")
    tasks = assistant.tasks("s1")
    assert tasks["T1"]["status"] == "completed" and tasks["T2"]["status"] == "pending"
    assert reply["request"]["text"] == "Which colour should the car be?"
    assert reply["assets"] == [{"asset_id": "a1", "version": 2}]
    assert [s for s, _ in llm.calls].count("readiness") == 1         # no second check


def test_pending_task_supporting_task_then_input(assistant, llm, banner):
    # Turn 1: no colour given, so the readiness check asks before the editor starts.
    llm.add("interpret", interpretation(intent("t1", "Change the colour of the car in a1 v1.")))
    llm.add("select", assign("image-editor"))
    llm.add("readiness", {"resolved": {"target": "the car"}, "defaults_used": [], "missing": [
        {"name": "new_value", "request": "Which colour should the car be?",
         "options": ["suggest some", "you choose"]}]})
    reply = assistant.send("s1", "Change the colour of the car")
    assert reply["request"]["text"] == "Which colour should the car be?"
    assert assistant.tasks("s1")["T1"]["status"] == "pending"

    # Turn 2: the user asks for help deciding. T1 stays pending.
    def supporting(prompt):
        assert '"id": "T1"' in prompt                 # the pending task is shown to the model
        return interpretation(
            intent("t1", "Suggest colours for the car in a1 v1.", result="edit_options",
                   supports="T1"), kind="supporting_task")
    llm.add("interpret", supporting)
    llm.add("select", assign("suggestion-agent"))
    reply = assistant.send("s1", "What colours would work?")
    tasks = assistant.tasks("s1")
    assert tasks["T1"]["status"] == "pending" and tasks["T2"]["status"] == "completed"
    assert "option two" in reply["text"]

    # Turn 3: "the second one" is resolved against the suggestion task's results.
    def answer(prompt):
        assert "[stub] option two" in prompt          # supporting results reach the prompt
        return interpretation(kind="input_for_pending_task", pending_task_input={
            "task_id": "T1", "input": "[stub] option two"})
    llm.add("interpret", answer)
    llm.add("readiness", lambda p: (ready(target="the car", new_value="[stub] option two")
                                    if "[stub] option two" in p else {}))
    llm.add("editor_plan", tools(("recolor_region", {"box": [0, 0, 50, 50], "color": "teal"})))
    reply = assistant.send("s1", "the second one")
    assert assistant.tasks("s1")["T1"]["status"] == "completed"
    assert reply["assets"] == [{"asset_id": "a1", "version": 2}]
    assert not llm.remaining()


def test_contradiction_holds_the_whole_input(assistant, llm, banner):
    llm.add("interpret", interpretation(
        intent("t1", "Make a1 minimalist while adding five decorations."),   # must be dropped
        conflicts=[{"between": ["make it minimalist", "add more elements"],
                    "kind": "contradiction", "explanation": "", "resolutions": []}],
        clarification={"request": "Minimalist and five added elements pull apart. Which?",
                       "options": ["Add them in a minimal style", "Add only one or two"],
                       "why": ""}))
    reply = assistant.send("s1", "Make it minimalist and add confetti, balloons, stars...")
    assert reply["request"]["options"] == ["Add them in a minimal style", "Add only one or two"]
    assert assistant.tasks("s1") == {}


def test_manual_step_parks_dependents_until_the_user_acts(assistant, llm, banner):
    llm.add("interpret", interpretation(
        intent("t1", "Open the crop tool on a1 v1 so the user can crop it by hand.", by="user"),
        intent("t2", "Round the corners of a1.", depends=["t1"])))

    def per_task_candidates(prompt):
        # A user-performed task may only go to an agent that opens an interface.
        assert '"candidate_agents": [\n      "manual-ops"\n    ]' in prompt
        return assignments(("T1", "manual-ops"), ("T2", "image-editor"))
    llm.add("select", per_task_candidates)
    reply = assistant.send("s1", "Let me crop it, then round the corners")

    action = reply["client_actions"][0]
    assert action["interface"] == "crop_tool" and action["task_id"] == "T1"
    tasks = assistant.tasks("s1")
    assert tasks["T1"]["status"] == "completed"       # the interface is open; nothing waits
    assert tasks["T2"]["status"] == "parked" and tasks["T2"]["waiting_on"] == "ui:T1"

    # The user crops. The event is not interpreted; T2 resumes on the user's version.
    llm.add("readiness", ready(target="image", change="round corners"))
    llm.add("editor_plan", tools(("round_corners", {"radius": 12})))
    reply = assistant.ui_event("s1", "action_completed", task_id="T1",
                               image=Image.new("RGB", (300, 300), "#888888"))
    assert assistant.store.get(AssetRef(asset_id="a1", version=2)).made_by == "user"
    assert assistant.store.get(AssetRef(asset_id="a1", version=3)).tool == "round_corners"
    assert assistant.store.open(AssetRef(asset_id="a1", version=3)).size == (300, 300)
    assert reply["assets"] == [{"asset_id": "a1", "version": 3}]
    assert [c[0] for c in llm.calls].count("interpret") == 1


def test_account_actions_always_go_to_the_user(assistant, llm):
    # The model wrongly says the system performs it; code overrides that.
    llm.add("interpret", interpretation(
        intent("t1", "Open the screen for adding money to the user's account.",
               result="account_action", by="system", constraints=["amount: 20 pounds"],
               assets=())))
    llm.add("select", lambda p: assign("manual-ops") if '"id": "image-editor"' not in p else {})
    reply = assistant.send("s1", "add 20 pounds to my account")
    action = reply["client_actions"][0]
    assert action["interface"] == "add_funds" and action["prefill"] == ["amount: 20 pounds"]
    assert action["asset"] is None


def test_agent_requests_assistance_and_resumes(assistant, llm, banner):
    llm.add("interpret", interpretation(intent("t1", "Move the logo in a1 v1 to the top right.")))
    llm.add("select", assign("image-editor"))
    llm.add("readiness", ready(target="logo", change="move", new_value="top right"))
    llm.add("editor_plan", {"action": "assistance_required",
                            "need": "The logo in a1 v1 separated as a layer.",
                            "acceptance": "A layer with a mask covering only the logo."})
    # The need is interpreted and an agent selected exactly as for user input.
    llm.add("interpret", lambda p: interpretation(
        intent("t1", "Extract the logo in a1 v1 as a layer.", result="extracted_element",
               by="user"))                                   # code forces "system" for agents
        if "<source>image-editor</source>" in p else {})
    llm.add("select", lambda p: assign("layering-agent")
            if '"id": "image-editor"' not in p else {})      # the requester is excluded
    llm.add("editor_plan", lambda p: tools(("crop", {"aspect_ratio": "1:1"}))
            if "Separated the element as a layer" in p else {})

    reply = assistant.send("s1", "Move the logo to the top right")
    tasks = assistant.tasks("s1")
    assert tasks["T2"]["parent_id"] == "T1" and tasks["T2"]["performed_by"] == "system"
    assert tasks["T1"]["status"] == tasks["T2"]["status"] == "completed"
    assert reply["assets"] == [{"asset_id": "a1", "version": 2}]
    assert "layer" not in reply["text"].lower()              # only the parent speaks to the user
    assert not llm.remaining()


def test_user_moves_on_then_parked_request_returns(assistant, llm, banner):
    llm.add("interpret", interpretation(intent("t1", "Change the colour of the car in a1 v1.")))
    llm.add("select", assign("image-editor"))
    llm.add("readiness", {"resolved": {}, "defaults_used": [], "missing": [
        {"name": "new_value", "request": "Which colour?", "options": []}]})
    assistant.send("s1", "Change the colour of the car")

    llm.add("interpret", interpretation(
        intent("t1", "Respond to the user's greeting.", result="conversation", assets=())))
    llm.add("select", assign("chitchat-handler"))
    reply = assistant.send("s1", "hello by the way")
    # T1 was parked while the new task ran, then offered again.
    assert reply["request"]["text"].startswith("Coming back to your earlier request.")
    assert assistant.tasks("s1")["T1"]["status"] == "pending"

    llm.add("interpret", interpretation(kind="cancellation", affected_task_id="T1"))
    reply = assistant.send("s1", "never mind")
    assert assistant.tasks("s1")["T1"]["status"] == "cancelled" and reply["request"] is None


def test_no_design_in_session_is_caught_before_dispatch(assistant, llm):
    llm.add("interpret", interpretation(intent("t1", "Enlarge the logo.", assets=())))
    llm.add("select", assign("image-editor"))
    reply = assistant.send("s1", "make the logo bigger")
    assert "Which design" in reply["request"]["text"]
    assert [c[0] for c in llm.calls] == ["interpret", "select"]   # no editor call was made


def test_amendment_replaces_the_goal_of_an_unfinished_task(assistant, llm, banner):
    llm.add("interpret", interpretation(intent("t1", "Change the colour of the car in a1 v1.")))
    llm.add("select", assign("image-editor"))
    llm.add("readiness", {"resolved": {}, "defaults_used": [], "missing": [
        {"name": "new_value", "request": "Which colour?", "options": []}]})
    assistant.send("s1", "Change the colour of the car")

    llm.add("interpret", interpretation(
        intent("t1", "Make the car in a1 v1 black and white.", constraints=["black and white"]),
        kind="amendment", affected_task_id="T1"))
    llm.add("select", assign("image-editor"))
    llm.add("readiness", ready(target="car", change="black and white"))
    llm.add("editor_plan", tools(("grayscale", {})))
    reply = assistant.send("s1", "actually just make it black and white")
    tasks = assistant.tasks("s1")
    assert list(tasks) == ["T1"] and tasks["T1"]["status"] == "completed"
    assert tasks["T1"]["goal"] == "Make the car in a1 v1 black and white."
    assert reply["request"] is None and reply["assets"] == [{"asset_id": "a1", "version": 2}]


def test_external_agent_needs_policy_and_consent(llm, tmp_path):
    from image_agents import Assistant, Card, Config, Policy, Registry
    from image_agents.agents import ExternalAgent

    sent = []

    def transport(request):
        sent.append(request)
        return {"state": "completed", "text": "Stitch file ready."}

    registry = Registry.load()
    registry.add(Card(id="stitch-agent", name="StitchWorks", location="external",
                      provider="StitchWorks Ltd", description="Converts designs to stitch files.",
                      behaviour={"readiness_model": False}))
    blocked = Assistant(llm, tmp_path / "a", registry=registry)
    assert "stitch-agent" not in {c.id for c in registry.candidates(
        excluded=[], user_performed=False, policy=blocked.config.policy)}

    assistant = Assistant(llm, tmp_path / "b", registry=registry,
                          agents=[ExternalAgent("stitch-agent", transport, share=["assets"])],
                          config=Config(policy=Policy(allow_external_agents=True)))
    assistant.add_image("s1", Image.new("RGB", (50, 50)), "logo")
    llm.add("interpret", interpretation(
        intent("t1", "Convert a1 v1 to an embroidery stitch file.", result="other",
               constraints=["secret note"])))
    llm.add("select", {**assign("stitch-agent"), "location": "external"})
    reply = assistant.send("s1", "turn this into a stitch file")
    assert "outside service run by StitchWorks Ltd" in reply["request"]["text"] and not sent

    llm.add("interpret", interpretation(kind="input_for_pending_task", pending_task_input={
        "task_id": "T1", "input": "Yes, share it"}))
    reply = assistant.send("s1", "yes")
    assert reply["text"] == "Stitch file ready."
    assert sent[0]["assets"][0]["asset_id"] == "a1" and "constraints" not in sent[0]


def test_subgraph_adapter_wraps_an_existing_langgraph_graph(llm, tmp_path):
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from image_agents import Assistant
    from image_agents.agents import SubgraphAdapter
    from image_agents.schemas import Outcome

    class S(TypedDict, total=False):
        brief: str
        answer: str

    sub = StateGraph(S)
    sub.add_node("work", lambda s: {"answer": s["brief"].upper()})
    sub.add_edge(START, "work")
    sub.add_edge("work", END)
    adapter = SubgraphAdapter(
        "question-handler", sub.compile(),
        to_state=lambda env, ctx: {"brief": env.goal},
        from_state=lambda final, env, ctx: Outcome(kind="completed", summary=final["answer"]))

    assistant = Assistant(llm, tmp_path, agents=[adapter])
    llm.add("interpret", interpretation(
        intent("t1", "explain kerning", result="information", assets=())))
    llm.add("select", assign("question-handler"))
    assert assistant.send("s1", "what is kerning?")["text"] == "EXPLAIN KERNING"
