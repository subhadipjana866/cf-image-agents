"""Offline checks of the navigation wiring, with scripted model replies.

    .venv/bin/python -m pytest agent_navigation/test_navigator.py -q
"""
from __future__ import annotations

from image_agents.llm import ScriptedLLM

from agent_navigation.navigator import CARDS_DIR, Navigator
from image_agents.registry import Registry


def _intent(tid, goal, result="changed_image", by="system", depends=()):
    return {"id": tid, "goal": goal, "desired_result": result, "performed_by": by,
            "target_assets": [{"asset_id": "a1", "version": 1}], "constraints": [],
            "assumptions": [], "depends_on": list(depends), "supports_task_id": None}


def _interp(*tasks, kind="new_task", **extra):
    return {"analysis": "", "input_type": kind, "pending_task_input": None,
            "affected_task_id": None, "requested_changes": [], "conflicts": [],
            "tasks": list(tasks), "clarification": None, **extra}


def _assign(tid, agent):
    return {"task_id": tid, "analysis": "", "decision": "assign", "agent_id": agent,
            "location": "internal", "confidence": "high"}


def test_all_cards_load_with_tools():
    cards = Registry.load(CARDS_DIR).all()
    assert len(cards) == 11
    for card in cards:
        if card.id != "custom-agent":            # a direct LLM answer, deliberately no tools
            assert card.operational.get("tools"), card.id
        assert "operational" not in card.prompt_view()   # tools never reach the selection prompt


def test_chain_routes_each_task_and_user_task_only_sees_interface_agents():
    llm = ScriptedLLM()
    llm.add("interpret", _interp(_intent("t1", "make background navy"),
                                 _intent("t2", "open crop tool", by="user", depends=["t1"])))
    llm.add("select", {"assignments": [_assign("T1", "image-editor"), _assign("T2", "manual-ops")]})
    route = Navigator(llm).route("navy background, then let me crop")
    assert route.agents == ["image-editor", "manual-ops"]
    assert route.tasks[1].depends_on == ["T1"]
    select_prompt = llm.calls[1][1]
    assert '"candidate_agents": [\n      "manual-ops"\n    ]' in select_prompt


def test_account_action_is_forced_to_the_user():
    llm = ScriptedLLM()
    llm.add("interpret", _interp(_intent("t1", "add funds", result="account_action")))
    llm.add("select", {"assignments": [_assign("T1", "manual-ops")]})
    route = Navigator(llm).route("add 20 dollars")
    assert route.tasks[0].performed_by == "user"


def test_agent_outside_candidates_is_rejected():
    llm = ScriptedLLM()
    llm.add("interpret", _interp(_intent("t1", "open crop tool", by="user")))
    llm.add("select", {"assignments": [_assign("T1", "image-editor")]})
    assert Navigator(llm).route("let me crop it").agents == ["none"]


def test_disabled_agent_is_not_offered():
    llm = ScriptedLLM()
    llm.add("interpret", _interp(_intent("t1", "fix the hands")))
    llm.add("select", {"assignments": [_assign("T1", "slop-terminator")]})
    route = Navigator(llm, disabled=("slop-terminator",)).route("fix the hands")
    assert route.agents == ["none"]
    assert "slop-terminator" not in llm.calls[1][1]


def test_pending_answer_and_clarification_skip_selection():
    llm = ScriptedLLM()
    llm.add("interpret", _interp(kind="input_for_pending_task",
                                 pending_task_input={"task_id": "k7", "input": "cool blue dusk"}))
    route = Navigator(llm).route("the second", pending_task={
        "id": "k7", "goal": "recolour car", "requested": "which colour?",
        "options": ["warm", "cool blue dusk"], "agent_id": "image-editor"})
    assert route.agents == ["pending"]
    assert route.pending_input == {"task_id": "k7", "input": "cool blue dusk",
                                   "agent_id": "image-editor"}
    assert "agent_id" not in llm.calls[0][1].split("<pending_task>")[1].split("</pending_task>")[0]

    llm = ScriptedLLM()
    llm.add("interpret", _interp(kind="new_task", clarification={"request": "Which?", "options": []}))
    assert Navigator(llm).route("make it red and blue").agents == ["clarify"]
    assert [step for step, _ in llm.calls] == ["interpret"]

