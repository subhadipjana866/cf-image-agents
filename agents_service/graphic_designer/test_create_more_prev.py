"""Regression: create-more on an image that has no stored v33 result.

``_v32_prev_from_image`` rebuilds ``prev`` from the image (its XMP, or one vision
call). It used to store ``subject`` as a plain string, and v33's
``create_more()`` passes ``prev["subject"]`` to ``antislop.in_frame()``, which
calls ``.get()`` on it:  AttributeError: 'str' object has no attribute 'get'.

    .venv/bin/python -m pytest graphic_designer/test_create_more_prev.py -q
"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from graphic_designer import nodes
from graphic_designer.image_metadata import tagged
from graphic_designer.v33_bridge import v33


def _data_uri(data: bytes) -> str:
    return "data:image/webp;base64," + base64.b64encode(data).decode()


def _webp(size=(1080, 1350)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, "#884422").save(buf, "WEBP")
    return buf.getvalue()


def _assert_create_more_ready(prev: dict) -> None:
    assert isinstance(prev["subject"], dict) and prev["subject"]["subject"]
    import antislop   # v33 module, on sys.path via v33_bridge
    # The two calls create_more() makes with prev's ART DIRECTOR and SUBJECT.
    assert antislop.in_frame(prev.get("art_director"), prev.get("subject")) is None
    assert antislop.body_parts(None, "a saree stack", prev.get("art_director"),
                               prev.get("subject")) == []


def test_prev_from_embedded_xmp_has_dict_subject():
    goal, platform = "Festive / Seasonal Sale", "Instagram Portrait Post"
    state = {"design_goal": goal, "platform_canonical": platform, "user_id": "u", "session_id": "s",
             "prompt": "Diwali sale for Kavya Sarees"}
    result = {"locked_text": {"headline": "Drape the Festive Light"}}
    data = tagged(BytesIO(_webp()), state, "generated", "gpt-image-2.5-flare", result).getvalue()

    prev, lineage, cost = nodes._v32_prev_from_image(state, _data_uri(data))

    assert cost == 0.0 and prev["goal"] == goal
    _assert_create_more_ready(prev)


def test_prev_from_vision_reader_has_dict_subject(monkeypatch):
    read = {"design_goal": "Festive / Seasonal Sale", "platform": "Instagram Portrait Post",
            "locked_text": {"headline": "Now Open"}, "subject": "a folded silk saree stack",
            "archetype": "split", "style_signature": "warm"}
    reply = SimpleNamespace(content=json.dumps(read), usage_metadata={"total_tokens": 10})
    monkeypatch.setattr("graphic_designer.utils.invokeMultimodalWithFallback",
                        lambda *a, **k: (reply, "gemini-3-flash-preview"))

    prev, lineage, _ = nodes._v32_prev_from_image({}, _data_uri(_webp()))

    assert prev["subject"] == {"subject": "a folded silk saree stack"}
    _assert_create_more_ready(prev)
    assert v33.load_guidance(prev["goal"], prev["platform"])   # a real catalog pair
