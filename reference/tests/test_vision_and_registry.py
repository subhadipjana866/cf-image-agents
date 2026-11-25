from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_agents import Config, Registry
from image_agents.vision import (TextLine, detect_text_blocks, glyph_mask, group_lines,
                                 min_area_rect)


def line(x0, y0, x1, y1, text):
    return TextLine([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], text)


def test_lines_group_into_blocks_by_size_and_proximity():
    blocks = group_lines([
        line(50, 40, 400, 90, "SUMMER"),          # two-line headline
        line(50, 100, 380, 150, "SALE"),
        line(50, 230, 300, 250, "Up to half off"),  # smaller body copy, further down
        line(50, 256, 280, 276, "this weekend only"),
        line(500, 400, 600, 440, "Shop now"),     # button label, elsewhere
    ])
    assert [b.text for b in blocks] == ["SUMMER\nSALE", "Up to half off\nthis weekend only",
                                        "Shop now"]
    assert blocks[0].id == "tb1" and blocks[0].angle == 0
    xs = [p[0] for p in blocks[0].quad]
    ys = [p[1] for p in blocks[0].quad]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx((50, 40, 400, 150))


def test_min_area_rect_recovers_rotation():
    theta = math.radians(20)
    c, s = math.cos(theta), math.sin(theta)
    corners = [(0, 0), (200, 0), (200, 40), (0, 40)]
    rotated = [(100 + x * c - y * s, 50 + x * s + y * c) for x, y in corners]
    quad, angle = min_area_rect(rotated)
    assert angle == pytest.approx(20, abs=0.01)
    side_lengths = sorted(math.dist(quad[i], quad[(i + 1) % 4]) for i in range(4))
    assert side_lengths == pytest.approx([40, 40, 200, 200], abs=0.01)


def test_glyph_mask_covers_letters_and_a_margin_only():
    img = Image.new("RGB", (200, 100), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 40, 99, 59), fill=(10, 10, 10))       # a "letter" inside the block
    draw.rectangle((150, 10, 170, 30), fill=(10, 10, 10))     # dark pixels outside the block
    mask = glyph_mask(img, [(40, 30), (130, 30), (130, 70), (40, 70)], dilate_px=3)
    assert mask.shape == (100, 200) and mask.dtype == np.uint8
    assert mask[50, 80] == 255                                # the letter
    assert mask[38, 80] == 255 and mask[36, 80] == 0          # dilated by 3 px, no further
    assert mask[20, 160] == 0                                 # outside the block is untouched
    assert mask[50, 120] == 0                                 # background inside the block


def test_registry_loads_ten_cards_and_hides_operational_fields():
    registry = Registry.load()
    assert len(registry.all()) == 10
    view = registry.get("image-editor").prompt_view()
    assert "operational" not in view and "not_for" in view
    policy = Config().policy
    system = {c.id for c in registry.candidates(excluded=[], user_performed=False, policy=policy)}
    user = {c.id for c in registry.candidates(excluded=[], user_performed=True, policy=policy)}
    assert user == {"manual-ops"} and "manual-ops" not in system and len(system) == 9
    assert "image-editor" not in {c.id for c in registry.candidates(
        excluded=["image-editor"], user_performed=False, policy=policy)}


def test_detect_text_blocks_accepts_any_ocr_engine():
    def fake_engine(image):                       # the shape every detector returns
        yield [(50, 40), (400, 40), (400, 90), (50, 90)], "SUMMER", 0.97
        yield [(50, 100), (380, 100), (380, 150), (50, 150)], "SALE", 0.91
    blocks = detect_text_blocks(Image.new("RGB", (500, 200)), fake_engine)
    assert len(blocks) == 1 and blocks[0]["text"] == "SUMMER\nSALE"
    assert blocks[0]["confidence"] == 0.91 and blocks[0]["line_count"] == 2
