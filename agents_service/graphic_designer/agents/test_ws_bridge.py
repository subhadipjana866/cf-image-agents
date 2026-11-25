"""Offline checks of the websocket frame mapping (no models, no DB).

    .venv/bin/python -m pytest graphic_designer/agents/test_ws_bridge.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from graphic_designer.agents import ws_bridge
from graphic_designer.agents.session import DesignSession


def chat_with(events, session=None):
    s = session or DesignSession()
    return SimpleNamespace(session=s, events=events, on_progress=None)


def test_generation_result_frame_carries_messages_cost_and_text_layers():
    s = DesignSession()
    v = s.add_version("https://x/1.webp", made_by="image-generator", tool="generate_catalog_design",
                      instruction="diwali", model="gpt-image-2.5-flare",
                      meta={"text_layers": {"layers": {"elements": []}, "notices": ["font swapped"]}})
    events = [{"type": "progress", "agent_id": "image-generator"},
              {"type": "result", "agent_id": "image-generator", "reply": "Here it is.",
               "goal": "Diwali post", "new_versions": [v], "previous": None, "cost": 0.2,
               "payload": {}},
              {"type": "message", "text": "Your graphic has been generated. Now our Critic..."}]
    frames = ws_bridge._frames(chat_with(events, s), turn_cost=0.25)
    assert frames[0]["type"] == "generation_started" and frames[0]["_spinner"]
    gen = frames[1]
    assert gen["type"] == "generation_completed" and gen["current_image_url"] == "https://x/1.webp"
    assert gen["text_layers"] == {"elements": []} and gen["typography_notices"] == ["font swapped"]
    assert gen["show_cost"] is True
    # the message after the result is folded into it (as today's generation_completed question)
    assert frames[-1] is gen and "Critic" in gen["question"]
    assert abs(sum(f["cost"] for f in frames) - 0.25) < 1e-9


def test_ask_frames_match_todays_shapes():
    s = DesignSession()
    s.start_with_image("https://x/a.webp")
    form = ws_bridge._frames(chat_with([{"type": "ask", "kind": "task", "ask_kind": "form",
                                         "question": "Details?", "options": [], "fields": [
        {"field_id": "discount_amount", "label": "Discount", "question": "What offer?"}]}], s), 0)
    assert form[-1]["fields"][0]["field_id"] == "discount_amount" and form[-1]["skippable"]
    pick = ws_bridge._frames(chat_with([{"type": "ask", "kind": "task", "ask_kind": "select_image",
                                         "question": "Select one", "options": [], "fields": []}], s), 0)
    assert pick[-1]["kind"] == "reselect_image"
    cards = ws_bridge._frames(chat_with([{"type": "ask", "kind": "suggestions", "ask_kind": "text",
                                          "question": "Choose", "options": [], "fields": [],
                                          "suggestions": [{"n": 1, "text": "Bigger headline",
                                                           "type": "EDIT", "instruction": "x"}]}], s), 0)
    item = cards[-1]["suggestions"][0]
    assert item["s_id"] == 1 and item["status"] == "pending" and item["user_friendly_text"]


def test_manual_ops_becomes_manual_edit_frame_with_reply():
    s = DesignSession()
    s.start_with_image("https://x/a.webp")
    events = [{"type": "client_action", "action": {"interface": "canvas_control",
                                                    "prefill": {"control": "brightness"}}},
              {"type": "result", "agent_id": "manual-ops", "reply": "I opened brightness.",
               "goal": "g", "new_versions": [], "previous": None, "cost": 0.0, "payload": {}}]
    frames = ws_bridge._frames(chat_with(events, s), 0)
    assert frames[-1]["type"] == "manual_edit" and frames[-1]["manual_op"] == "brightness"
    assert frames[-1]["question"] == "I opened brightness."


def test_edit_kinds():
    s = DesignSession()
    a = s.start_with_image("https://x/a.webp")
    b = s.add_version("https://x/b.webp", made_by="image-editor", tool="resize_for_platforms",
                      instruction="story")
    ev = {"type": "result", "agent_id": "image-editor", "reply": "Done", "goal": "story",
          "new_versions": [b], "previous": a, "cost": 0.06, "payload": {}}
    assert ws_bridge._frames(chat_with([ev], s), 0.06)[0]["type"] == "expand_successful"
    ev["new_versions"][0].tool = "erase_object"
    ev["payload"] = {"from_suggestions": True}
    assert ws_bridge._frames(chat_with([ev], s), 0.06)[0]["type"] == "suggestions_executed"


def test_frames_carry_sender_task_recipient_and_folded_messages():
    s = DesignSession()
    s.start_with_image("https://x/a.webp")
    events = [{"type": "message", "text": "Using version 1.", "sender": "input-collector",
               "task_id": "1.T1", "recipient": "user"},
              {"type": "ask", "kind": "task", "ask_kind": "text", "question": "Which cup?",
               "options": [], "fields": [], "sender": "image-editor", "task_id": "1.T1",
               "recipient": "user"}]
    frame = ws_bridge._frames(chat_with(events, s), 0)[-1]
    assert (frame["sender"], frame["task_id"], frame["recipient"]) == ("image-editor", "1.T1", "user")
    assert [m["sender"] for m in frame["messages"]] == ["input-collector", "image-editor"]
    assert frame["messages"][-1]["text"] == "Which cup?"


def test_no_canvas_selection_does_not_load_the_latest_graphic():
    s = DesignSession()
    picked = ws_bridge._selected_image(s, {}, {"session_last_graphic": {"image_url": "https://x/old.webp"}})
    assert picked is None and s.current is None
    assert s.canvas_selected is False and s.fallback_image == "https://x/old.webp"
    picked = ws_bridge._selected_image(s, {"current_image_url": "https://x/a.webp"}, {})
    assert picked == "https://x/a.webp" and s.canvas_selected is True


def test_structured_ask_frame_and_reply_text():
    from graphic_designer.agents import asks
    s = DesignSession()
    spec = asks.build(ask_id="q3", task_id="1.T1", asked_by="image-editor", purpose="agent_question",
                      prompt="Pick a size and type the new headline",
                      options=["Small", "Large"], fields=[{"id": "headline", "label": "Headline"}])
    assert spec["kind"] == "form" and spec["fields"][0]["type"] == "text"
    frame = ws_bridge._frames(chat_with([{"type": "ask", "kind": "task", "ask_kind": "text",
                                          "question": spec["prompt"], "options": ["Small", "Large"],
                                          "fields": [], "ask_spec": spec}], s), 0)[-1]
    assert frame["type"] == "text_input" and frame["ask"]["id"] == "q3"
    assert "- Small" not in frame["question"]
    text = asks.reply_text(spec, {"id": "q3", "option_ids": ["large"],
                                  "values": {"headline": "Big Sale"}}, "thanks")
    assert text == "Large; Headline: Big Sale; thanks"
    yes_no = asks.build(ask_id="q4", task_id=None, asked_by="orchestrator", purpose="retry",
                        prompt="Try again?", options=["Yes", "No"])
    assert yes_no["kind"] == "confirm"
