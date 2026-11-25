"""Offline walk-through: one multi-target request, then a manual crop.

Runs without any model access. The replies a model would give are scripted
here so you can watch the orchestrator's behaviour. To use a real model,
replace ScriptedLLM with ChatModelLLM (see the README).

    python examples/demo.py
"""
import tempfile

from PIL import Image

from image_agents import Assistant, ScriptedLLM


def task(tid, goal, by="system", depends=(), version=1):
    return {"id": tid, "goal": goal, "desired_result": "changed_image", "performed_by": by,
            "target_assets": [{"asset_id": "a1", "version": version}], "constraints": [],
            "assumptions": [], "depends_on": list(depends), "supports_task_id": None}


def assign(agent):
    return {"decision": "assign", "agent_id": agent, "location": "internal",
            "confidence": "high"}


llm = ScriptedLLM()
assistant = Assistant(llm, tempfile.mkdtemp())
assistant.add_image("demo", Image.new("RGB", (1200, 628), "#b8c4d0"), "coffee brand banner")

# --- Turn 1: two changes, two targets, so two chained tasks ------------------
llm.add("interpret", {"input_type": "new_task", "tasks": [
    task("t1", "Make a1 v1 warmer and slightly brighter."),
    task("t2", "Round the corners of a1 to 24 pixels.", depends=["t1"])]})
# Two chained tasks for the same agent: one call per step covers both.
llm.add("select", {"assignments": [{**assign("image-editor"), "task_id": "T1"},
                                   {**assign("image-editor"), "task_id": "T2"}]})
llm.add("readiness", {"results": [
    {"task_id": "T1", "resolved": {"target": "whole image", "change": "warmer, brighter"}},
    {"task_id": "T2", "resolved": {"target": "whole image", "change": "round corners",
                                   "new_value": "24 px"}}]})
llm.add("editor_plan", {"plans": [
    {"task_id": "T1", "action": "run_tools", "chosen_values": {"brightness": "+8%"},
     "calls": [{"tool": "adjust", "args": {"warmth": 0.5, "brightness": 1.08}}]},
    {"task_id": "T2", "action": "run_tools",
     "calls": [{"tool": "round_corners", "args": {"radius": 24}}]}]})

print("USER: Make it warmer and a bit brighter, and round the corners to 24 pixels")
reply = assistant.send("demo", "Make it warmer and a bit brighter, and round the corners to 24 pixels")
print("ASSISTANT:", reply["text"].replace("\n", "\n           "))
print("   assets:", reply["assets"])
print("   model calls:", [step for step, _ in llm.calls], "\n")

# --- Turn 2: the user wants to do it by hand -------------------------------
llm.add("interpret", {"input_type": "new_task", "tasks": [
    task("t1", "Open the crop tool on a1 v3 so the user can crop it by hand.", by="user",
         version=3)]})
llm.add("select", assign("manual-ops"))

print("USER: let me crop it myself")
reply = assistant.send("demo", "let me crop it myself")
print("ASSISTANT:", reply["text"])
print("   client action:", reply["client_actions"][0], "\n")

# --- The client reports that the user cropped the image ------------------
assistant.ui_event("demo", "action_completed", task_id=reply["client_actions"][0]["task_id"],
                   image=Image.new("RGB", (628, 628), "#b8c4d0"))
print("UI EVENT: action_completed")
for t in assistant.tasks("demo").values():
    print(f"   {t['id']}  {t['status']:<10} {t['agent_id']:<13} {t['goal']}")
print("   versions of a1:", [(v.ref.version, v.made_by, v.tool or v.method)
                             for v in assistant.store._assets["a1"].versions])
