"""Offline collector checks (model replies scripted).

    .venv/bin/python -m pytest graphic_designer/agents/test_collector.py -q
"""
from __future__ import annotations

from graphic_designer.agents import collector
from graphic_designer.agents.collector import CollectTask, collect
from graphic_designer.agents.session import Brief, DesignSession
from image_agents.registry import Registry

CARD = {c.id: c for c in Registry.load("agent_navigation/cards").all()}["image-generator"]


def test_model_invented_field_answers_never_skip_the_required_fields(monkeypatch):
    """Regression: the collector model filled the catalog fields itself ("the agent can
    write them"), so the user was never asked for the offer, headline or deadline."""
    replies = iter([
        ({"status": "ready", "checks": ["fields can be written by the agent"], "goal": "flash sale post",
          "brief": {"new_design": True, "design_goal": "Flash Sale",
                    "platform": "Instagram Square Post", "prompt": "flash sale on headphones",
                    "field_answers": {"discount_amount": "agent will write", "sale_headline": "TBD",
                                      "brand_name": "generic", "urgency": "limited time"}}}, 0.0, 0.0),
        ({}, 0.0, 0.0),   # _fill_fields: the user said none of it
    ])
    monkeypatch.setattr(collector, "vision_json", lambda prompt, images: next(replies))
    s = DesignSession()
    s.brief = Brief(prompt="", design_goal="Flash Sale", platform="Instagram Square Post")
    out = collect(CollectTask(id="T1", agent_id="image-generator", goal="flash sale post",
                              desired_result="new_image"), s, CARD)
    assert out.status == "ask" and out.ask_kind == "form"
    asked = {f["field_id"] for f in out.fields}
    assert {"discount_amount", "urgency", "brand_name"} <= asked
    assert "sale_headline" not in asked        # the copywriter writes the headline
    assert s.brief.field_answers == {}


def test_generator_does_not_see_the_image_when_nothing_is_selected(monkeypatch):
    from types import SimpleNamespace
    from graphic_designer.agents import collector
    from graphic_designer.agents.collector import CollectTask
    from graphic_designer.agents.session import DesignSession
    seen = {}

    def fake_vision(prompt, images):
        seen["images"], seen["prompt"] = images, prompt
        return {"status": "ready", "goal": "x"}, 0.0, 0.0
    monkeypatch.setattr(collector, "vision_json", fake_vision)
    monkeypatch.setattr(collector, "_generator_fields", lambda task, s, out: out)
    s = DesignSession()
    s.start_with_image("https://x/server-room.webp")
    s.canvas_selected = False
    card = SimpleNamespace(id="image-generator", name="Image Generator", description="d",
                           required_inputs=[], prompt_view=lambda: {})
    monkeypatch.setattr(collector, "_card_text", lambda card: ("d", "(none)"))
    collector.collect(CollectTask(id="1.T1", agent_id="image-generator", goal="flash sale post",
                                  desired_result="new_image"), s, card)
    assert seen["images"] == [] and "No image is selected on the canvas." in seen["prompt"]
