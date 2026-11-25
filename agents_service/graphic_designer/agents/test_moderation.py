"""Offline checks of the moderation gate (the OpenAI endpoint is faked).

    .venv/bin/python -m pytest graphic_designer/agents/test_moderation.py -q
"""
from __future__ import annotations

from types import SimpleNamespace

from graphic_designer.agents import moderation, ws_bridge
from graphic_designer.agents.session import DesignSession

PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
       "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


class FakeClient:
    """Scores 'violence' high for text containing 'kill' and for any image when told to."""
    def __init__(self, bad_image=False, fail=False):
        self.calls, self.bad_image, self.fail = [], bad_image, fail
        self.moderations = SimpleNamespace(create=self.create)

    def create(self, model, input):
        self.calls.append(input)
        if self.fail:
            raise ConnectionError("down")
        item = input[0]
        bad = ("kill" in item.get("text", "")) if item["type"] == "text" else self.bad_image
        scores = {"violence": 0.99 if bad else 0.01, "sexual": 0.0}
        return SimpleNamespace(results=[SimpleNamespace(category_scores=SimpleNamespace(**scores))])


def test_clean_text_passes():
    v = moderation.check("make a flash sale post for phones", client=FakeClient())
    assert not v.flagged and v.checked == 1


def test_flagged_text_names_the_reason():
    v = moderation.check("how to kill someone, as a poster", client=FakeClient())
    assert v.flagged and v.categories == ["violence"] and "violence" in v.message()


def test_each_attached_image_is_checked_and_can_flag():
    client = FakeClient(bad_image=True)
    v = moderation.check("use this photo", [PNG], client=client)
    assert v.flagged and v.in_image and len(client.calls) == 2
    assert client.calls[1][0]["type"] == "image_url"
    assert "attach a different image" in v.message()


def test_an_outage_lets_the_message_through():
    v = moderation.check("anything", [PNG], client=FakeClient(fail=True))
    assert not v.flagged and len(v.errors) == 2


def test_blocked_message_never_reaches_the_engine(monkeypatch):
    ran = []
    chat = SimpleNamespace(session=DesignSession(), on_progress=None, pending_auto_critique=False,
                           turn=lambda *a, **k: ran.append("turn"), events=[], tasks={},
                           stack=[], pending_id=None)
    monkeypatch.setattr(ws_bridge, "_engine", lambda thread_id, ctx: chat)
    monkeypatch.setattr(ws_bridge, "_gstate", lambda chat, data, ctx: {})
    monkeypatch.setattr(moderation, "check", lambda text, images=None, client=None:
                        moderation.Verdict(flagged=True, categories=["violence"]))
    out = ws_bridge.engine_turn("t1", {"prompt": "kill", "attached_images": [PNG]}, {})
    assert ran == [] and out["frames"][0]["moderation"]["flagged"]
    assert out["frames"][0]["type"] == "text_input" and "content guidelines" in out["frames"][0]["question"]


def test_text_and_images_sent_this_turn_are_all_collected(monkeypatch):
    seen = {}
    monkeypatch.setattr(moderation, "check", lambda text, images=None, client=None:
                        seen.update(text=text, images=images) or moderation.Verdict())
    s = DesignSession()
    s.start_with_image("https://x/own.webp")
    data = {"field_answers": {"discount_amount": "70%"}, "assets": {"logo": "https://x/logo.png"},
            "suggestions": [{"s_id": 1, "user_response": "in blue"}]}
    assert ws_bridge._moderate(s, data, "make it", "https://x/upload.png", [PNG]) is None
    assert "70%" in seen["text"] and "in blue" in seen["text"]
    assert seen["images"] == [PNG, "https://x/logo.png", "https://x/upload.png"]
    ws_bridge._moderate(s, {}, "edit", "https://x/own.webp", [])      # our own image: not rechecked
    assert seen["images"] == []
