"""Shared test helpers. All tests run offline against a scripted model."""
from __future__ import annotations

import pytest
from PIL import Image

from image_agents import Assistant, ScriptedLLM
from image_agents.agents import ImageEditorAgent


class FakeGenerativeBackend:
    """Stands in for an image-editing model: returns the image unchanged."""

    def __init__(self):
        self.instructions = []

    def edit(self, image, instruction, box=None):
        self.instructions.append(instruction)
        return image.copy()


@pytest.fixture
def llm():
    return ScriptedLLM()


@pytest.fixture
def backend():
    return FakeGenerativeBackend()


@pytest.fixture
def assistant(llm, backend, tmp_path):
    return Assistant(llm, tmp_path, agents=[ImageEditorAgent(generative_backend=backend)])


@pytest.fixture
def banner(assistant):
    """A design already in the session, in focus as a1 v1."""
    return assistant.add_image("s1", Image.new("RGB", (400, 300), "#888888"), "product banner")


# ------------------------------------------------------------ scripted replies
def intent(tid, goal, *, result="changed_image", by="system", depends=(), supports=None,
           constraints=(), assets=(("a1", 1),)):
    return {
        "id": tid, "goal": goal, "desired_result": result, "performed_by": by,
        "target_assets": [{"asset_id": a, "version": v} for a, v in assets],
        "constraints": list(constraints), "assumptions": [], "depends_on": list(depends),
        "supports_task_id": supports,
    }


def interpretation(*tasks, kind="new_task", **extra):
    return {"analysis": "", "input_type": kind, "pending_task_input": None,
            "affected_task_id": None, "requested_changes": [], "conflicts": [],
            "tasks": list(tasks), "clarification": None, **extra}


def assign(agent_id, confidence="high"):
    return {"analysis": "", "decision": "assign", "agent_id": agent_id, "location": "internal",
            "confidence": confidence, "alternates": [], "missing_inputs": [],
            "capability_gap": None, "external_handoff": None}


def ready(**resolved):
    return {"analysis": "", "resolved": resolved, "defaults_used": [], "missing": []}


def tools(*calls, **chosen):
    return {"action": "run_tools", "chosen_values": chosen,
            "calls": [{"tool": name, "args": args} for name, args in calls]}


# -------------------------------------------------- batched replies (one call, many tasks)
def assignments(*pairs):
    """pairs: (task_id, agent_id)"""
    return {"assignments": [{**assign(agent), "task_id": tid} for tid, agent in pairs]}


def ready_all(*task_ids, **resolved):
    return {"results": [{**ready(**resolved), "task_id": tid} for tid in task_ids]}


def plans(*items):
    """items: (task_id, plan dict)"""
    return {"plans": [{**plan, "task_id": tid} for tid, plan in items]}
