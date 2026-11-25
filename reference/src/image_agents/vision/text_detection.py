"""Locating text in a flat image.

Pipeline, as in the architecture document:

1. detect_lines      a text detection and recognition engine returns a
                     quadrilateral and a string per line
2. group_lines       lines are clustered into text blocks
3. min_area_rect     each block gets a rotated bounding rectangle
4. glyph_mask        the letter pixels inside a block, dilated, for erasing

Steps 2 to 4 are plain Python and NumPy. Step 1 needs an OCR engine, which
you choose: pass any callable that takes an image and returns
(quad, text, confidence) tuples. `tesseract_detector` adapts Tesseract; cloud
OCR services and other engines fit the same shape.

Run `detect_text_blocks` once when a version is created and store the result
with `AssetStore.set_text_blocks(ref, blocks, source="detected")`. Positions
are in pixels of that version; a crop or resize invalidates them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

Point = tuple[float, float]
# An OCR engine: image in, (quad, text, confidence) per line out.
Detector = Callable[[Any], Iterable[tuple[list, str, float]]]


@dataclass
class TextLine:
    quad: list[Point]            # four corners, clockwise from top-left
    text: str
    confidence: float = 1.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs, ys = [p[0] for p in self.quad], [p[1] for p in self.quad]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def height(self) -> float:
        # Mean length of the two side edges, so rotation does not inflate it.
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.quad
        return (math.hypot(x3 - x0, y3 - y0) + math.hypot(x2 - x1, y2 - y1)) / 2


@dataclass
class TextBlock:
    id: str
    text: str
    quad: list[Point]            # minimum-area rotated rectangle
    angle: float                 # degrees, counter-clockwise
    confidence: float
    lines: list[TextLine] = field(default_factory=list)

    def to_record(self) -> dict:
        """The form kept in the asset store."""
        return {"id": self.id, "text": self.text, "quad": [list(p) for p in self.quad],
                "angle": self.angle, "confidence": self.confidence,
                "line_count": len(self.lines)}


# ------------------------------------------------------------------- 1. detection
def detect_lines(image: Any, detector: Detector) -> list[TextLine]:
    """Detect and read text lines with the OCR engine you supply.

    Run at full resolution: downscaling loses small text.
    """
    return [TextLine([(float(x), float(y)) for x, y in quad], text, float(conf))
            for quad, text, conf in detector(image)]


def tesseract_detector(lang: str = "eng", min_confidence: float = 30.0) -> Detector:
    """A detector backed by Tesseract. Needs the `ocr` extra and the tesseract binary.

    Tesseract returns axis-aligned word boxes; words are merged into lines
    here. It is built for documents and is weaker on stylised or rotated
    text over images, so prefer a scene-text engine for generated designs.
    """
    import pytesseract

    def detect(image: Any):
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        lines: dict[tuple, list[int]] = {}
        for i, word in enumerate(data["text"]):
            if word.strip() and float(data["conf"][i]) >= min_confidence:
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)
        for members in lines.values():
            x0 = min(data["left"][i] for i in members)
            y0 = min(data["top"][i] for i in members)
            x1 = max(data["left"][i] + data["width"][i] for i in members)
            y1 = max(data["top"][i] + data["height"][i] for i in members)
            text = " ".join(data["text"][i] for i in members)
            conf = min(float(data["conf"][i]) for i in members) / 100
            yield [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], text, conf

    return detect


# -------------------------------------------------------------------- 2. grouping
def group_lines(lines: list[TextLine], *, max_gap: float = 0.8,
                max_height_ratio: float = 1.5) -> list[TextBlock]:
    """Cluster lines into blocks.

    Two lines join when they are of similar height, overlap horizontally and
    the vertical gap between them is under `max_gap` line heights. A
    two-line headline becomes one block; body copy set smaller below it
    becomes another.
    """
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(lines):
        ax0, ay0, ax1, ay1 = a.bounds
        for j in range(i + 1, len(lines)):
            b = lines[j]
            bx0, by0, bx1, by1 = b.bounds
            tall, short = max(a.height, b.height), min(a.height, b.height)
            if short <= 0 or tall / short > max_height_ratio:
                continue
            if min(ax1, bx1) - max(ax0, bx0) <= 0:           # no horizontal overlap
                continue
            gap = max(ay0, by0) - min(ay1, by1)              # negative when they overlap
            if gap < max_gap * short:
                parent[find(i)] = find(j)

    groups: dict[int, list[TextLine]] = {}
    for i, line in enumerate(lines):
        groups.setdefault(find(i), []).append(line)

    blocks = []
    ordered = sorted(groups.values(), key=lambda g: min(l.bounds[1] for l in g))
    for n, members in enumerate(ordered, start=1):
        members.sort(key=lambda l: l.bounds[1])
        quad, angle = min_area_rect([p for l in members for p in l.quad])
        blocks.append(TextBlock(
            id=f"tb{n}", text="\n".join(l.text for l in members), quad=quad, angle=angle,
            confidence=min(l.confidence for l in members), lines=members))
    return blocks


# ---------------------------------------------------------- 3. rotated rectangle
def _convex_hull(points: list[Point]) -> list[Point]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def min_area_rect(points: list[Point]) -> tuple[list[Point], float]:
    """Smallest rotated rectangle around the points: (four corners, angle in degrees)."""
    hull = np.array(_convex_hull(points), dtype=float)
    if len(hull) < 3:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))], 0.0
    best = None
    for i in range(len(hull)):                    # one side of the rectangle lies on a hull edge
        edge = hull[(i + 1) % len(hull)] - hull[i]
        theta = math.atan2(edge[1], edge[0])
        c, s = math.cos(-theta), math.sin(-theta)
        rot = hull @ np.array([[c, s], [-s, c]])
        (x0, y0), (x1, y1) = rot.min(axis=0), rot.max(axis=0)
        area = (x1 - x0) * (y1 - y0)
        if best is None or area < best[0] - 1e-9:
            best = (area, theta, (x0, y0, x1, y1))
    _, theta, (x0, y0, x1, y1) = best
    c, s = math.cos(theta), math.sin(theta)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    quad = [(x * c - y * s, x * s + y * c) for x, y in corners]
    angle = math.degrees(theta)
    angle = (angle + 45) % 90 - 45                # report the small tilt, not the edge direction
    return quad, round(angle, 3)


# ------------------------------------------------------------------ 4. glyph mask
def glyph_mask(image: Any, quad: list[Point], *, threshold: float = 40.0,
               dilate_px: int = 3) -> np.ndarray:
    """Mask of the letter pixels inside a block (uint8, 255 = glyph).

    The block's background colour is taken as the median colour inside the
    quadrilateral; pixels far from it are glyphs. The mask is then dilated so
    that shadows, glows and outlines around the letters are erased with them.
    Erasing by this mask, instead of by the box, keeps the background between
    and around the letters.
    """
    rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image,
                     dtype=np.float32)
    h, w = rgb.shape[:2]

    # Pixels inside the convex quadrilateral: on the same side of all four edges.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    sides = []
    for i in range(4):
        (x0, y0), (x1, y1) = quad[i], quad[(i + 1) % 4]
        sides.append((x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0))
    sides = np.stack(sides)
    inside = (sides >= 0).all(axis=0) | (sides <= 0).all(axis=0)
    if not inside.any():
        return np.zeros((h, w), dtype=np.uint8)

    background = np.median(rgb[inside], axis=0)
    distance = np.linalg.norm(rgb - background, axis=-1)
    glyphs = inside & (distance > threshold)

    if dilate_px > 0:                             # square dilation, one axis at a time
        for axis in (0, 1):
            padded = np.pad(glyphs, [(dilate_px, dilate_px) if a == axis else (0, 0)
                                     for a in (0, 1)])
            size = glyphs.shape[axis]
            glyphs = np.any([np.take(padded, range(k, k + size), axis=axis)
                             for k in range(2 * dilate_px + 1)], axis=0)
    return glyphs.astype(np.uint8) * 255


def detect_text_blocks(image: Any, detector: Detector) -> list[dict]:
    """Steps 1 to 3 in one call, returning records ready for the asset store."""
    return [b.to_record() for b in group_lines(detect_lines(image, detector))]
