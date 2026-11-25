"""Offline checks of the layering agent's wiring (the layering pipeline itself is not run).

    .venv/bin/python -m pytest graphic_designer/agents/test_layering.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.types import Command

from graphic_designer.agents import layering, runner, ws_bridge
from graphic_designer.agents.runner import ChainResult
from graphic_designer.agents.session import DesignSession

LAYERS = ["https://s3/run/layer_000_background.png", "https://s3/run/layer_001_chair.png"]


@pytest.fixture
def session():
    s = DesignSession()
    s.start_with_image("https://example.com/room.webp")
    return s


def call(session, request):
    return layering.separate_layers.func(runtime=SimpleNamespace(context=session), request=request)


def test_named_targets_run_the_subgraph_from_bbox_generator(session, monkeypatch):
    from graphic_designer import nodes
    runs = []
    monkeypatch.setattr(nodes, "_layeringTargetsFromPrompt",
                        lambda state: (["chair"], ["SALE"], False, 0.001))

    def fake_chain(start, state):
        runs.append((start, state))
        return ChainResult("done", {**state, "segmentation_result": LAYERS,
                                    "bboxes": [{"bbox": [1, 2, 3, 4]}], "cost": 0.2},
                           ["bboxGenerator", "segmentGenerator", "pasteBackTexts"], "layeringDone")
    monkeypatch.setattr(layering, "run_chain", fake_chain)
    reply = call(session, "the chair and the SALE text")
    start, state = runs[0]
    assert start == "bboxGenerator"
    assert state["requested_objects"] == ["chair"] and state["requested_texts"] == ["SALE"]
    assert state["layering_targets_resolved"] and state["layering_mode"] == "auto"
    assert state["current_image_url"] == "https://example.com/room.webp"
    assert reply.startswith("OK: made 2 layer(s)")
    assert session.layers[-1]["segmentation_result"] == LAYERS and session.layers[-1]["version"] == 1


def test_drawn_boxes_run_manual_layering_and_are_used_once(session, monkeypatch):
    runs = []
    monkeypatch.setattr(layering, "run_chain", lambda start, state: runs.append((start, state))
                        or ChainResult("done", {**state, "segmentation_result": LAYERS}, [], ""))
    session.manual_bboxes = [{"bbox": [10, 10, 50, 50], "label": "object_1"}]
    call(session, "the marked regions")
    start, state = runs[0]
    assert start == "manualLayering" and state["layering_mode"] == "manual"
    assert state["manual_bboxes"] == [{"bbox": [10, 10, 50, 50], "label": "object_1"}]
    assert session.manual_bboxes == []


def test_no_targets_asks_and_nothing_found_is_reported(session, monkeypatch):
    from graphic_designer import nodes
    monkeypatch.setattr(nodes, "_layeringTargetsFromPrompt", lambda state: ([], [], False, 0.0))
    assert call(session, "make layers").startswith("NEEDS_TARGETS")
    monkeypatch.setattr(nodes, "_layeringTargetsFromPrompt", lambda state: ([], [], True, 0.0))
    monkeypatch.setattr(layering, "run_chain", lambda start, state: ChainResult(
        "blocked", state, ["bboxGenerator"], "layDoNotLayer"))
    assert call(session, "everything").startswith("NOTHING_FOUND") and session.layers == []


def test_runner_follows_the_subgraph_static_edges(monkeypatch):
    seen = []
    fake = {
        "bboxGenerator": lambda st: seen.append("bbox") or Command(goto="segmentGenerator",
                                                                   update={"bboxes": [1]}),
        "segmentGenerator": lambda st: seen.append("seg") or {**st, "segmentation_result": LAYERS},
        "pasteBackTexts": lambda st: seen.append("paste") or st,
    }
    monkeypatch.setattr(runner, "_nodes", lambda: fake)
    out = runner.run_chain("bboxGenerator", {"cost": 0.0})
    assert out.outcome == "done" and seen == ["bbox", "seg", "paste"]
    assert out.state["segmentation_result"] == LAYERS


def test_layering_frames_match_the_old_graph(session):
    ask = {"type": "ask", "kind": "task", "ask_kind": "layering_targets", "question": "Which?",
           "options": [], "fields": [], "sender": "layering-agent", "task_id": "1.T1",
           "extra": {"accepts": ["object_names", "bboxes"], "auto_text": False, "note": "n"}}
    frame = ws_bridge._frames(SimpleNamespace(session=session, events=[ask], on_progress=None), 0)[-1]
    assert frame["type"] == "layering_started" and frame["accepts"] == ["object_names", "bboxes"]
    made = {"image_url": "https://example.com/room.webp", "segmentation_result": LAYERS,
            "bboxes": [], "extracted_texts": [{"text": "SALE"}], "version": 1}
    result = {"type": "result", "agent_id": "layering-agent", "reply": "Done", "goal": "layers",
              "new_versions": [], "new_layers": [made], "previous": session.current,
              "cost": 0.4, "payload": {}, "sender": "layering-agent", "task_id": "1.T1"}
    frame = ws_bridge._frames(SimpleNamespace(session=session, events=[result], on_progress=None),
                              0.4)[-1]
    assert frame["type"] == "layering_executed" and frame["segmentation_result"] == LAYERS
    assert frame["extracted_texts"] == [{"text": "SALE"}] and frame["show_cost"] is True


def test_drawn_boxes_are_normalised_like_main_py():
    boxes = ws_bridge._drawn_boxes({"boxes": [{"x1": 1, "y1": 2, "x2": 30, "y2": 40, "label": "cup"}]})
    assert boxes == [{"bbox": [1, 2, 30, 40], "label": "cup"}]
    assert ws_bridge._drawn_boxes({}) == []
