"""
typography.py
----------------------------------------------------------------------
Real fonts, accessible text, and editable-text metadata for every graphic.

WHY THIS EXISTS
    Image models draw text that LOOKS like type but belongs to no font, so it
    can never be edited or reproduced. Experiments (experiments/font_adherence,
    Oct-Nov 2026) established:
      * naming a font in the prompt does not work (0/4 lines on GPT Image 2.5);
      * attaching an image of the font's alphabet does (24/28 lines);
      * handing the model a TYPESET LAYOUT IMAGE -- the copy already set in the
        real fonts at exact positions -- keeps the text where it was planned,
        so the metadata can be produced BEFORE generation;
      * on non-2:3 canvases the model sometimes moves a block 10-70 px, so a
        ~0.1 s registration step confirms or corrects each element afterwards.

WHAT IT ADDS TO THE PIPELINE
    TYPESET (after the three specialists, before the director)
        one reasoning call sets the exact copy, boxes, sizes and colours;
        deterministic code then enforces the product rules below, typesets
        with the real font files, and draws the layout image.
    RENDER
        layout image + font samples (or the user's own font image) go to the
        renderer ahead of the client's references.
    TEXT QA (after render)
        registration per element -> confirmed | corrected | measured/unverified;
        WCAG contrast against the artwork actually painted behind the text.
    OUTPUT
        result["text_layers"]  (schema kognie.text_layers.v2) -- everything an
        HTML canvas needs to recreate each text element as an editable object.

PRODUCT RULES (2026-11-17)
    * Text is optional; zero elements is valid.
    * Fonts: default 2 per graphic, max 3; the title ALWAYS has its own font.
    * Named font supported  -> used (its sample image is the font reference).
      Named font unsupported -> user notice; closest supported style is used.
    * User's own image as the font source (reference role "font") -> it is the
      ONLY font reference; metadata records the nearest Kognie font (approximate).
    * WCAG 2.x AA contrast at display size; platform-specific minimum sizes.

Dependencies: Pillow, numpy, opencv-python-headless. typography_ocr.py
(EasyOCR) is optional and only used for user-font-image matching and for
measuring elements that registration cannot place.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("KOGNIE_TYPOGRAPHY_ASSETS", HERE / "typography_assets"))
ENABLED = os.environ.get("KOGNIE_TYPOGRAPHY", "on").lower() not in ("0", "off", "false", "no")

# ---------------------------------------------------------------------------
# Font library. The sample image's name is the font's real name.
# ---------------------------------------------------------------------------

FONTS = {
    "montserrat_light": {"family": "Montserrat", "weight": 300, "generic": "sans-serif",
                         "file": "fonts/montserrat/Montserrat[wght].ttf",
                         "category": "geometric sans", "readability": "text",
                         "aliases": ["montserrat", "montserrat light", "monstreat"]},
    "comfortaa_medium": {"family": "Comfortaa", "weight": 500, "generic": "sans-serif",
                         "file": "fonts/comfortaa/Comfortaa[wght].ttf",
                         "category": "rounded sans", "readability": "text",
                         "aliases": ["comfortaa", "comforta", "comfortaa medium"]},
    "lexend": {"family": "Lexend", "weight": 400, "generic": "sans-serif",
               "file": "fonts/lexend/Lexend[wght].ttf",
               "category": "readability sans", "readability": "text", "aliases": ["lexend"]},
    "courier_new": {"family": "Courier New", "weight": 400, "generic": "monospace",
                    "file": "fonts/courier_new/cour.ttf",   # not redistributable: system lookup below
                    "system": ["C:/Windows/Fonts/cour.ttf", "/Library/Fonts/Courier New.ttf",
                               "/System/Library/Fonts/Supplemental/Courier New.ttf",
                               "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
                               "/usr/share/fonts/truetype/msttcorefonts/cour.ttf"],
                    "category": "monospace slab", "readability": "text",
                    "aliases": ["courier", "courier new", "typewriter"]},
    "amatic_sc": {"family": "Amatic SC", "weight": 400, "generic": "cursive", "caps_only": True,
                  "file": "fonts/amatic_sc/AmaticSC-Regular.ttf",
                  "category": "condensed hand-drawn caps", "readability": "display",
                  "aliases": ["amatic", "amatic sc"]},
    "pacifico": {"family": "Pacifico", "weight": 400, "generic": "cursive",
                 "file": "fonts/pacifico/Pacifico-Regular.ttf",
                 "category": "brush script", "readability": "display", "aliases": ["pacifico", "pasifico"]},
    "caveat": {"family": "Caveat", "weight": 400, "generic": "cursive",
               "file": "fonts/caveat/Caveat[wght].ttf",
               "category": "handwriting", "readability": "display", "aliases": ["caveat"]},
}
DEFAULT_TEXT_FONT = "lexend"
DISPLAY_FONT_MAX_WORDS = 12

# Characters every sample sheet shows. The renderer copies letterforms from the
# sample, so a character the sheet lacks has no reference glyph and gets invented.
SHEET_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.!, \n")
TOLERATED = set("?'-:&%\"")          # common: allowed, but flagged
REPLACE = {"\u00b7": ",", "\u2022": ",", "|": ",", "\u2013": "-", "\u2014": "-", "\u2794": "", "\u2192": "",
           "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "/": " ", "\u2026": "."}


def font_path(key: str) -> Path:
    f = FONTS[key]
    p = ASSETS / f["file"]
    if p.exists():
        return p
    for s in f.get("system", []):
        if Path(s).exists():
            return Path(s)
    raise FileNotFoundError(
        f"font file for {f['family']} not found (looked in {p} and system font folders). "
        f"Install it or set KOGNIE_TYPOGRAPHY_ASSETS.")


def sample_path(key: str) -> Path:
    return ASSETS / "samples" / f"{key}.png"


_fcache: dict = {}


def load_font(key: str, size: float, weight: int | None = None) -> ImageFont.FreeTypeFont:
    size = max(4, int(round(size)))
    weight = weight or FONTS[key]["weight"]
    ck = (key, size, weight)
    if ck not in _fcache:
        f = ImageFont.truetype(str(font_path(key)), size)
        try:
            vals = []
            for ax in f.get_variation_axes():
                name = ax.get("name", b"")
                name = name.decode() if isinstance(name, bytes) else str(name)
                vals.append(max(ax["minimum"], min(ax["maximum"], weight))
                            if name.lower().startswith("weight") else ax["default"])
            f.set_variation_by_axes(vals)
        except Exception:
            pass  # static font
        _fcache[ck] = f
    return _fcache[ck]


def ink_bounds(text: str, f) -> tuple | None:
    """Ink bbox of `text` relative to its (left, baseline) origin."""
    l, t, r, b = f.getbbox(text, anchor="ls")
    pad = 8
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls")
    a = np.array(im) > 96
    ys, xs = np.where(a)
    if not len(ys):
        return None
    ox, oy = pad - l, pad - t
    return int(xs.min() - ox), int(ys.min() - oy), int(xs.max() + 1 - ox), int(ys.max() + 1 - oy)


_mcache: dict = {}


def font_metrics(key: str) -> dict:
    """Per-em metrics measured from the real file (what CSS/canvas need)."""
    if key not in _mcache:
        S = 1000
        f = load_font(key, S)
        asc, desc = f.getmetrics()
        cap, xh = ink_bounds("H", f), ink_bounds("x", f)
        sample = "The quick brown fox jumps over the lazy dog"
        _mcache[key] = {"units": "em", "ascent": round(asc / S, 4), "descent": round(desc / S, 4),
                        "cap_height": round(-cap[1] / S, 4), "x_height": round(-xh[1] / S, 4),
                        "avg_char_width": round(f.getlength(sample) / len(sample) / S, 4)}
    return _mcache[key]


def font_block(key: str) -> dict:
    f = FONTS[key]
    return {"key": key, "family": f["family"], "weight": f["weight"], "style": "normal",
            "css_font_family": f"'{f['family']}', {f['generic']}", "sample_image": f"{key}.png",
            "caps_only": bool(f.get("caps_only")), "metrics": font_metrics(key)}


def resolve_font_name(name: str) -> str | None:
    n = re.sub(r"[^a-z ]", "", str(name).lower()).strip()
    if not n:
        return None
    for key, f in FONTS.items():
        if n in FONTS or n == key:
            return n
        for a in f["aliases"] + [f["family"].lower()]:
            if n == a or difflib.SequenceMatcher(None, n, a).ratio() >= 0.88:
                return key
    return None


def library_block() -> str:
    rows = []
    for k, f in FONTS.items():
        rows.append(f'  - {k}: {f["family"]} {f["weight"]} -- {f["category"]}, readability "{f["readability"]}"'
                    + (", capitals only" if f.get("caps_only") else ""))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Platforms: where and how big each graphic is actually seen
#   display_w  typical rendered width in CSS reference px
#   min_css    minimum size per role at that width (CSS px)
#   safe       fractions kept free of text (top, right, bottom, left)
# ---------------------------------------------------------------------------

BASE_MIN = {"title": 20, "subheading": 15, "body": 12, "cta": 14, "caption": 11}
PLATFORMS = {
    "Instagram Square Post": {"display_w": 390, "safe": (0.05, 0.05, 0.05, 0.05)},
    "Instagram Portrait Post": {"display_w": 390, "safe": (0.05, 0.05, 0.05, 0.05)},
    "Instagram Story": {"display_w": 390, "safe": (0.14, 0.06, 0.20, 0.06)},
    "Facebook Feed Landscape": {"display_w": 390, "safe": (0.05, 0.04, 0.05, 0.04)},
    "Pinterest Pin": {"display_w": 236, "safe": (0.05, 0.06, 0.10, 0.06)},
    "LinkedIn Feed Post": {"display_w": 390, "safe": (0.05, 0.05, 0.05, 0.05)},
    "YouTube Thumbnail": {"display_w": 246, "safe": (0.05, 0.05, 0.14, 0.05), "max_words": 6},
    "Presentation Slide": {"display_w": 960, "safe": (0.07, 0.07, 0.10, 0.07),
                           "min_css": {"title": 48, "subheading": 36, "body": 32, "cta": 32, "caption": 24}},
    "Digital Signage Portrait": {"display_w": 405, "safe": (0.08, 0.08, 0.08, 0.08),
                                 "min_css": {"title": 28, "subheading": 20, "body": 16, "cta": 18, "caption": 14}},
    "X (Twitter) In-Stream Post": {"display_w": 360, "safe": (0.05, 0.04, 0.05, 0.04)},
}
DEFAULT_PLATFORM = {"display_w": 390, "safe": (0.06, 0.06, 0.06, 0.06)}

ROLE_ALIASES = {"headline": "title", "heading": "title", "title": "title", "primary": "title",
                "subtitle": "subheading", "subheading": "subheading", "subhead": "subheading",
                "secondary": "subheading", "tagline": "subheading", "offer": "subheading",
                "body": "body", "copy": "body", "tertiary": "body", "details": "body",
                "cta": "cta", "button": "cta", "action": "cta",
                "caption": "caption", "footer": "caption", "legal": "caption", "attribution": "caption"}


def role_of(r: str) -> str:
    return ROLE_ALIASES.get(str(r).lower().strip(), "body")


def platform_of(name: str) -> dict:
    return PLATFORMS.get(name, DEFAULT_PLATFORM)


def min_font_px(platform: str, W: int, role: str, key: str, text: str) -> float:
    """Minimum font-size on a W-px-wide canvas, corrected for x-height (cap
    height for all-caps) and light weights."""
    p = platform_of(platform)
    css = p.get("min_css", BASE_MIN)[role]
    scale = p["display_w"] / W
    m = font_metrics(key)
    caps = text.upper() == text or FONTS[key].get("caps_only")
    eff = (m["cap_height"] / 0.70) if caps else (m["x_height"] / 0.50)
    return css / scale / max(0.5, min(eff, 1.3)) * (1.25 if FONTS[key]["weight"] < 400 else 1.0)


# ---------------------------------------------------------------------------
# WCAG contrast
# ---------------------------------------------------------------------------

def hex2rgb(h: str):
    h = str(h).lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def rgb2hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(np.clip(round(float(v)), 0, 255)) for v in rgb)


def luminance(rgb) -> float:
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(a, b) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def is_large_text(font_px: float, weight: int, scale: float) -> bool:
    css = font_px * scale
    return css >= 24 or (weight >= 700 and css >= 18.66)


def fix_contrast(text_hex: str, bg_hex: str, target: float) -> tuple[str, float, bool]:
    """Keep hue/chroma, move CIELAB lightness the shorter way to the target ratio."""
    bg = hex2rgb(bg_hex)
    if contrast(hex2rgb(text_hex), bg) >= target:
        return text_hex, contrast(hex2rgb(text_hex), bg), False
    lab = cv2.cvtColor(np.uint8([[hex2rgb(text_hex)]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    best = None
    for direction in (+1, -1):
        L = lab[0]
        while 0 <= L <= 255:
            L += direction * 2
            rgb = cv2.cvtColor(np.uint8([[[np.clip(L, 0, 255), lab[1], lab[2]]]]), cv2.COLOR_LAB2RGB)[0, 0]
            if contrast(rgb, bg) >= target:
                if best is None or abs(L - lab[0]) < best[0]:
                    best = (abs(L - lab[0]), rgb2hex(rgb))
                break
    if best is None:
        bw = "#000000" if contrast([0, 0, 0], bg) >= contrast([255, 255, 255], bg) else "#ffffff"
        return bw, contrast(hex2rgb(bw), bg), True
    return best[1], contrast(hex2rgb(best[1]), bg), True


def sanitize(text: str) -> tuple[str, list[str], list[str]]:
    out, replaced, tolerated = [], [], []
    for ch in str(text):
        if ch in SHEET_CHARS:
            out.append(ch)
        elif ch in TOLERATED:
            out.append(ch)
            tolerated.append(ch)
        else:
            rep = REPLACE.get(ch, "")
            out.append(rep)
            replaced.append(ch)
            if rep and rep in TOLERATED:
                tolerated.append(rep)
    t = re.sub(r" {2,}", " ", "".join(out)).strip()
    t = re.sub(r"\s+,", ",", t)
    return t, replaced, sorted(set(tolerated))


# ---------------------------------------------------------------------------
# Typesetting (deterministic, real font files)
# ---------------------------------------------------------------------------

def apply_transform(text: str, transform: str) -> str:
    return text.upper() if transform == "uppercase" else text


def line_width(s: str, f, ls_px: float) -> float:
    return f.getlength(s) + ls_px * max(0, len(s) - 1)


def wrap(text: str, f, ls_px: float, max_w: float) -> list[str]:
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split():
            trial = f"{cur} {w}".strip()
            if cur and line_width(trial, f, ls_px) > max_w:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
    return lines or [""]


def typeset(key: str, text: str, spec: dict, floor: float = 0.0) -> dict:
    """Exact geometry: shrink-to-fit (never below `floor`), word-wrap, per-line
    origin x and baseline y, ink boxes."""
    text = apply_transform(text, spec.get("text_transform", "none"))
    bx = dict(spec["box"])
    size = max(float(spec["font_size_px"]), floor)
    lh = float(spec.get("line_height") or 1.2)
    max_lines = int(spec.get("max_lines") or 3)
    lsem = float(spec.get("letter_spacing_em") or 0)

    def fits(sz, box):
        f = load_font(key, sz)
        ls = lsem * sz
        lines = wrap(text, f, ls, box["w"])
        asc, desc = f.getmetrics()
        block_h = asc + desc + (len(lines) - 1) * lh * sz
        ok = len(lines) <= max_lines and block_h <= box["h"] and all(line_width(s, f, ls) <= box["w"] for s in lines)
        return ok, lines, block_h

    ok, lines, block_h = fits(size, bx)
    while not ok and size * 0.96 >= floor:
        size *= 0.96
        ok, lines, block_h = fits(size, bx)
    overflow = False
    if not ok:  # cannot shrink further without going illegible: grow the box instead
        overflow = True
        f = load_font(key, size)
        ls = lsem * size
        widest = max(line_width(w, f, ls) for w in text.split()) + 4 if text.split() else 0
        if widest > bx["w"]:
            cx = bx["x"] + bx["w"] / 2
            bx["w"] = int(widest)
            bx["x"] = int(cx - widest / 2)
        lines = wrap(text, f, ls, bx["w"])
        asc, desc = f.getmetrics()
        block_h = asc + desc + (len(lines) - 1) * lh * size
        bx["h"] = int(block_h) + 2
    f = load_font(key, size)
    ls = lsem * size
    asc, desc = f.getmetrics()
    top = bx["y"] + (bx["h"] - block_h) / 2
    align = spec.get("align", "left")
    out_lines = []
    for i, s in enumerate(lines):
        w = line_width(s, f, ls)
        x = {"left": bx["x"], "center": bx["x"] + (bx["w"] - w) / 2, "right": bx["x"] + bx["w"] - w}.get(align, bx["x"])
        base = top + asc + i * lh * size
        ib = ink_bounds(s, f) if s else None
        out_lines.append({"text": s, "x": round(x, 1), "baseline_y": round(base, 1), "advance_width": round(w, 1),
                          "ink_box": [round(x + ib[0], 1), round(base + ib[1], 1),
                                      round(x + ib[2] + ls * max(0, len(s) - 1), 1), round(base + ib[3], 1)]
                          if ib else None})
    return {"font_size_px": round(size, 2), "letter_spacing_px": round(ls, 2), "line_height": lh,
            "align": align, "text_transform": spec.get("text_transform", "none"),
            "color_hex": spec["color_hex"], "rotation_deg": float(spec.get("rotation_deg") or 0),
            "box": bx, "lines": out_lines, "overflow": overflow, "min_font_px": round(floor, 1),
            "legibility_met": size >= floor - 0.5}


def element_ink_box(p: dict):
    bs = [l["ink_box"] for l in p["lines"] if l.get("ink_box")]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def draw_text(d: ImageDraw.ImageDraw, key: str, weight: int, p: dict, fill):
    f = load_font(key, p["font_size_px"], weight)
    ls = p["letter_spacing_px"]
    for ln in p["lines"]:
        if abs(ls) > 0.05:
            x = ln["x"]
            for ch in ln["text"]:
                d.text((x, ln["baseline_y"]), ch, font=f, fill=fill, anchor="ls")
                x += f.getlength(ch) + ls
        else:
            d.text((ln["x"], ln["baseline_y"]), ln["text"], font=f, fill=fill, anchor="ls")


def render_layout(elements: list, W: int, H: int, bg_hex: str, out: Path, glyphs: bool = True):
    """The typeset layout handed to the renderer as image 1. glyphs=False draws
    placement bars only (user-font-image mode: no Kognie letterforms may leak)."""
    im = Image.new("RGB", (W, H), bg_hex)
    d = ImageDraw.Draw(im)
    for e in elements:
        p = e["planned"]
        if glyphs:
            draw_text(d, e["font"]["key"], e["font"]["weight"], p, p["color_hex"])
        else:
            for ln in p["lines"]:
                if ln.get("ink_box"):
                    d.rounded_rectangle(ln["ink_box"], radius=6, outline=p["color_hex"], width=3)
    im.save(out)


# ---------------------------------------------------------------------------
# TYPESET node: one reasoning call, then deterministic enforcement
# ---------------------------------------------------------------------------

TYPESET_PROMPT = """You are the typesetter. The copy, the type system and the composition have been decided by
other specialists. Your job is to set the EXACT text of this graphic in real fonts: every element's final
string, role, font, frame, size and colour. Deterministic code will typeset what you return with the real
font files, so be precise.

DESIGN GOAL: {goal}     PLATFORM: {platform}     CANVAS: {W} x {H} px (origin top-left, y grows downward)
Keep text out of these margins (fractions of the canvas): top {s0}, right {s1}, bottom {s2}, left {s3}.
{max_words}
TEXT THAT MUST APPEAR (verbatim; each exactly once):
{locked}

ALSO STATED BY THE CLIENT (place unless it would genuinely harm the graphic):
{other_facts}

TYPOGRAPHER'S DECISIONS (hierarchy, CTA decision, fonts per role):
{typo}

COMPOSITOR'S PLACEMENT (where the subject and the text masses sit):
{comp}

ART DIRECTOR'S PALETTE (use it for text colours and to know what sits behind text):
{palette}

FONTS AVAILABLE (keys):
{library}
{font_rules}

COPY RULES
- Text is optional. If nothing above must appear, return "elements": [].
- Every locked string appears EXACTLY ONCE, verbatim. You may set instruction words around a code
  ("USE CODE HARVEST30 AT CHECKOUT") -- the locked string stays intact -- and the code is the dominant
  part of that unit. A quotation someone SAID carries visible opening and closing quotation marks.
- Never invent text: no taglines, no slogans, no facts that are not listed above.
- Use only letters, digits, spaces and . , ! -- avoid other symbols unless a locked string contains them.

LEGIBILITY AND ACCESSIBILITY
- Minimum font_size_px on this canvas, per role, for the fonts chosen:
{minimums}
- Text colour must meet WCAG AA against the colour directly behind it: 4.5:1, or 3:1 for large text.
  Give that colour as backdrop_hex, and plan the composition so the area behind text is calm.
- One clear primary element (roughly 3-5x the body size); boxes inside the margins and never overlapping;
  keep the subject's area (per the compositor) free of text.

Return ONLY strict JSON:
{{"background_hex": "<dominant colour behind the text areas>",
  "elements": [{{"id": "<short id>", "role": "title|subheading|body|cta|caption", "text": "<final string>",
      "font": "<key>", "box": {{"x": int, "y": int, "w": int, "h": int}}, "align": "left|center|right",
      "font_size_px": number, "letter_spacing_em": number, "line_height": number, "max_lines": int,
      "text_transform": "none|uppercase", "color_hex": "#rrggbb", "backdrop_hex": "#rrggbb"}}],
  "notes": "<one sentence>"}}"""


def font_rules_block(font_requests: list, user_font: dict | None, typo_fonts: dict) -> str:
    rows = ["FONT RULES",
            "- Use 2 fonts by default, at most 3. The title ALWAYS has its own font; no other element shares it.",
            "  Subheading, body, cta and caption may share one font with different size/weight/case.",
            f"- Decorative fonts (readability \"display\") only for titles or short lines (<= {DISPLAY_FONT_MAX_WORDS} words)."]
    if typo_fonts:
        rows.append(f"- The typographer chose: {json.dumps(typo_fonts)}. Keep these unless a rule above forbids it.")
    for r in font_requests or []:
        k = resolve_font_name(r.get("requested", ""))
        if k:
            rows.append(f"- The client asked for {FONTS[k]['family']} ({k}) for: {r.get('applies_to') or 'the title'}. Use it.")
    if user_font:
        nf = user_font.get("nearest")
        rows.append("- The client supplied their own image as the FONT SOURCE: one typeface for all text"
                    + (f"; use the key {nf} for metrics." if nf else "; pick the closest key for metrics."))
    return "\n".join(rows)


def minimums_block(platform: str, W: int, keys: list) -> str:
    keys = keys or list(FONTS)
    return "\n".join(f"  {role}: " + ", ".join(f"{k} {min_font_px(platform, W, role, k, 'Sample'):.0f}" for k in keys)
                     for role in BASE_MIN)


def typeset_prompt(*, goal, platform, W, H, locked: str, other_facts: str, typo: dict, comp: dict,
                   art: dict, font_requests: list, user_font: dict | None) -> str:
    p = platform_of(platform)
    typo_fonts = (typo or {}).get("fonts") or {}
    keys = sorted({v for v in typo_fonts.values() if v in FONTS}) or list(FONTS)
    palette = {k: (art or {}).get(k) for k in ("palette", "colour", "color", "palette_and_values", "light")
               if (art or {}).get(k)}
    return TYPESET_PROMPT.format(
        goal=goal, platform=platform, W=W, H=H, s0=p["safe"][0], s1=p["safe"][1], s2=p["safe"][2], s3=p["safe"][3],
        max_words=(f"At most {p['max_words']} words of text in total on this platform.\n" if p.get("max_words") else ""),
        locked=locked, other_facts=other_facts,
        typo=json.dumps({k: v for k, v in (typo or {}).items() if k != "_failed"}, ensure_ascii=False)[:3000],
        comp=json.dumps({k: v for k, v in (comp or {}).items() if k != "_failed"}, ensure_ascii=False)[:3000],
        palette=json.dumps(palette, ensure_ascii=False)[:1500] or "(not specified)",
        library=library_block(), font_rules=font_rules_block(font_requests, user_font, typo_fonts),
        minimums=minimums_block(platform, W, keys))


def enforce(plan: dict, *, platform: str, W: int, H: int, font_requests: list,
            user_font: dict | None) -> dict:
    """Deterministic product rules on the typesetter's plan. Every change is logged."""
    p = platform_of(platform)
    scale = p["display_w"] / W
    log, notices = [], []
    els = [e for e in (plan.get("elements") or []) if str(e.get("text", "")).strip()]

    for i, e in enumerate(els):
        e["id"] = str(e.get("id") or f"text{i + 1}")
        e["role"] = role_of(e.get("role", "body"))
        clean, replaced, tolerated = sanitize(e["text"])
        if replaced:
            log.append(f"{e['id']}: characters without a reference glyph {replaced!r} replaced")
        e["text"] = clean
        e["chars_without_reference_glyph"] = tolerated
        if e.get("font") not in FONTS:
            k = resolve_font_name(e.get("font", "")) or DEFAULT_TEXT_FONT
            log.append(f"{e['id']}: unknown font '{e.get('font')}' -> {k}")
            e["font"] = k
        for k, d in (("align", "left"), ("line_height", 1.2), ("max_lines", 3), ("letter_spacing_em", 0),
                     ("text_transform", "none"), ("color_hex", "#000000")):
            e.setdefault(k, d)
        e.setdefault("box", {"x": int(W * 0.08), "y": int(H * 0.08), "w": int(W * 0.84), "h": int(H * 0.15)})

    # 1. named fonts: the alias table decides supported/unsupported, not the LLM
    unsupported = []
    for r in font_requests or []:
        req = str(r.get("requested", "")).strip()
        if not req or re.search(r"\b(image|photo|picture|reference|attached|uploaded)\b", req, re.I):
            continue
        key = resolve_font_name(req)
        applies = str(r.get("applies_to") or "title").lower()
        targets = [e for e in els if applies in ("all", "everything", "all text") or role_of(applies) == e["role"]]
        if key:
            for e in targets or [x for x in els if x["role"] == "title"]:
                if e["font"] != key:
                    log.append(f"{e['id']}: client asked for {FONTS[key]['family']} -> applied")
                    e["font"] = key
        else:
            unsupported.append(req)

    # 2. user font image: one face for everything, metrics from the nearest Kognie font
    if user_font:
        nf = user_font.get("nearest") or (els[0]["font"] if els else DEFAULT_TEXT_FONT)
        for e in els:
            e["font"] = nf
        if user_font.get("nearest") and (user_font.get("confidence") or 0) < 0.5:
            notices.append(f"We used the lettering style from your image. For editable text we matched it to "
                           f"{FONTS[nf]['family']}, the closest font on Kognie; it may look slightly different "
                           f"when you edit the text.")
        log.append(f"user font image -> metadata font {nf} (approximate, confidence {user_font.get('confidence')})")
    else:
        # 3. decorative fonts are not for long copy
        for e in els:
            words = len(e["text"].split())
            if FONTS[e["font"]]["readability"] == "display" and words > DISPLAY_FONT_MAX_WORDS:
                text_fonts = [x["font"] for x in els if FONTS[x["font"]]["readability"] == "text" and x["role"] != "title"]
                new = text_fonts[0] if text_fonts else DEFAULT_TEXT_FONT
                log.append(f"{e['id']}: {words} words in decorative {e['font']} -> {new} (readability)")
                e["font"] = new
        # 4. title keeps its own font; at most 3 families
        titles = [e for e in els if e["role"] == "title"]
        if titles:
            tf = titles[0]["font"]
            for e in els:
                if e["role"] != "title" and e["font"] == tf:
                    others = [x["font"] for x in els if x["role"] != "title" and x["font"] != tf]
                    new = others[0] if others else (DEFAULT_TEXT_FONT if tf != DEFAULT_TEXT_FONT else "montserrat_light")
                    log.append(f"{e['id']}: shared the title font {tf} -> {new} (title keeps its own font)")
                    e["font"] = new
        fams = list(dict.fromkeys(e["font"] for e in els))
        while len(fams) > 3:
            drop, keep = fams.pop(), fams[1]
            for e in els:
                if e["font"] == drop:
                    e["font"] = keep
            log.append(f"more than 3 fonts: {drop} -> {keep}")

    # notices for unsupported requests name the FINAL fonts, after every rule above
    fams = list(dict.fromkeys(FONTS[e["font"]]["family"] for e in els))
    for req in unsupported:
        notices.append(f'"{req}" isn\'t available on Kognie yet. We used the closest supported style '
                       f'instead: {" and ".join(fams) or "a default font"}.')

    # 5. caps-only fonts, legibility floor, WCAG AA, typesetting
    words_total = 0
    for e in els:
        if FONTS[e["font"]].get("caps_only"):
            e["text_transform"] = "uppercase"
        words_total += len(e["text"].split())
        floor = min_font_px(platform, W, e["role"], e["font"], apply_transform(e["text"], e["text_transform"]))
        if float(e.get("font_size_px") or 0) < floor:
            log.append(f"{e['id']}: {e.get('font_size_px')}px below the {platform} minimum for {e['role']} "
                       f"({floor:.0f}px) -> raised")
            e["font_size_px"] = round(floor, 1)
        weight = FONTS[e["font"]]["weight"]
        target = 3.0 if is_large_text(float(e["font_size_px"]), weight, scale) else 4.5
        bd = e.get("backdrop_hex") or plan.get("background_hex") or "#808080"
        new, ratio, changed = fix_contrast(e["color_hex"], bd, target)
        e["contrast_plan"] = {"backdrop_hex": bd, "ratio": round(ratio, 2), "target": target,
                              "level": "AA large" if target == 3.0 else "AA", "original_color": e["color_hex"]}
        if changed:
            log.append(f"{e['id']}: colour {e['color_hex']} on {bd} below {target}:1 -> {new} ({ratio:.2f}:1)")
            e["color_hex"] = new
        e["planned"] = typeset(e["font"], e["text"], e, floor)
        if e["planned"]["overflow"]:
            log.append(f"{e['id']}: copy needs more room than its frame at a legible size -> frame grown")
    if p.get("max_words") and words_total > p["max_words"]:
        log.append(f"{words_total} words exceed the {platform} budget of {p['max_words']} -- review copy")

    # 6. hierarchy and overlap warnings
    rank = {"title": 0, "subheading": 1, "cta": 2, "body": 2, "caption": 3}
    for a in els:
        for b in els:
            if rank[a["role"]] < rank[b["role"]] and a["planned"]["font_size_px"] < b["planned"]["font_size_px"]:
                log.append(f"hierarchy: {a['id']} ({a['role']}) is smaller than {b['id']} ({b['role']})")
    # 7. overlaps: push the lower block down (a flag nobody acts on still ships a broken graphic)
    bottom_limit = H * (1 - p["safe"][2])
    for _ in range(len(els) * 2):
        moved = False
        order = sorted(els, key=lambda e: (element_ink_box(e["planned"]) or [0, 0, 0, 0])[1])
        for i, upper in enumerate(order):
            a = element_ink_box(upper["planned"])
            for lower in order[i + 1:]:
                b = element_ink_box(lower["planned"])
                if not a or not b:
                    continue
                if min(a[2], b[2]) > max(a[0], b[0]) and b[1] < a[3] + 0.25 * lower["planned"]["font_size_px"]:
                    dy = a[3] + 0.25 * lower["planned"]["font_size_px"] - b[1]
                    lower["planned"] = transform_planned(lower["planned"], 0, 0, 1.0, 0, dy)
                    log.append(f"overlap: {lower['id']} moved down {dy:.0f}px to clear {upper['id']}")
                    moved = True
        if not moved:
            break
    for e in els:
        b = element_ink_box(e["planned"])
        if b and b[3] > bottom_limit:
            log.append(f"layout: {e['id']} now extends into the bottom margin -- review layout")

    elements = [{"id": e["id"], "role": e["role"], "text": e["text"], "font": font_block(e["font"]),
                 "planned": e["planned"], "contrast_plan": e["contrast_plan"],
                 "chars_without_reference_glyph": e["chars_without_reference_glyph"]} for e in els]
    return {"canvas": {"width": W, "height": H}, "background_hex": plan.get("background_hex") or "#808080",
            "display_scale": round(scale, 4), "elements": elements,
            "fonts_used": list(dict.fromkeys(e["font"]["key"] for e in elements)),
            "policy_log": log, "user_notices": notices, "notes": plan.get("notes")}


# ---------------------------------------------------------------------------
# Prompt blocks for the director and the renderer
# ---------------------------------------------------------------------------

def _pct(v, t):
    return f"{100 * v / t:.0f}%"


def director_note(ts: dict) -> str:
    """Tells the director the typography is already set. Empty when there is no text."""
    if not ts or not ts["elements"]:
        return ""
    W, H = ts["canvas"]["width"], ts["canvas"]["height"]
    rows = []
    for e in ts["elements"]:
        b = element_ink_box(e["planned"]) or [0, 0, 0, 0]
        rows.append(f'  - {e["role"].upper()} "{e["text"]}": {_pct(b[0], W)}-{_pct(b[2], W)} across, '
                    f'{_pct(b[1], H)}-{_pct(b[3], H)} down, colour {e["planned"]["color_hex"]} '
                    f'on a backdrop near {e["contrast_plan"]["backdrop_hex"]}')
    return ("TYPOGRAPHY IS ALREADY SET -- this overrides every instruction below about writing text, typefaces "
            "or text placement into the prompt.\nAll text in this graphic has been typeset in advance with real "
            "fonts and goes to the renderer as a finished typographic layout (attached image 1). The text "
            "blocks and where they sit:\n" + "\n".join(rows) +
            "\nIn final_prompt: do NOT restate any text string, do NOT name or describe typefaces and do NOT "
            "describe text placement. Instead include this sentence: 'All typography is supplied, already set, "
            "in the attached typographic layout (image 1): reproduce it exactly -- same words, letterforms, "
            "sizes, colours, positions and line breaks -- and add no other text anywhere.' Compose the artwork "
            "AROUND those blocks: keep the areas behind them calm and close to the backdrop colours listed so the "
            "text keeps its contrast, and place the subject where it does not collide with them. Use the words "
            "you would have spent on text for the subject, light and materials. Every other requirement stands.")


def render_header(ts: dict, font_ref_names: list, n_client_refs: int, user_font_mode: bool) -> str:
    """Prepended to the renderer prompt: what each attached image is."""
    if not ts or not ts["elements"]:
        return ""
    k = len(font_ref_names)
    rows = ["ATTACHED IMAGES, IN ORDER:"]
    if user_font_mode:
        rows.append("  - Image 1 = TEXT PLACEMENT: each outlined bar marks where one line of text goes and how "
                    "tall it is. Write each line listed below inside its bar, filling its height, in the listed "
                    "colour; the bar outlines themselves are NOT drawn.")
    else:
        rows.append("  - Image 1 = the FINAL TYPOGRAPHY, already set in the correct typefaces, sizes, colours, "
                    "positions and line breaks. Keep every piece of text EXACTLY as it is there; never move, "
                    "resize, restyle, re-wrap, duplicate or add text.")
    for i, name in enumerate(font_ref_names):
        rows.append(f"  - Image {i + 2} = font reference: {name} (letterforms only; never show this sheet)")
    if n_client_refs:
        rows.append(f"  - Images {k + 2}-{k + 1 + n_client_refs} = the client's own references (logo, hero, "
                    f"product, layout or style), in the order they are described below.")
    if user_font_mode:
        rows.append("TEXT (spelled exactly as written, line breaks as shown, letterforms copied from the client's "
                    "font reference):")
        for e in ts["elements"]:
            rows.append(f'  - {e["role"].upper()}: ' + " / ".join(f'"{l["text"]}"' for l in e["planned"]["lines"])
                        + f', colour {e["planned"]["color_hex"]}')
    rows.append("Copy letterforms exactly from the font references; no outlines, textures or effects on letters; "
                "keep stroke weight as referenced.")
    return "\n".join(rows) + "\n\n"


# ---------------------------------------------------------------------------
# TEXT QA: registration + contrast after render
# ---------------------------------------------------------------------------

def element_mask(e: dict, W: int, H: int) -> np.ndarray:
    layer = Image.new("L", (W, H), 0)
    draw_text(ImageDraw.Draw(layer), e["font"]["key"], e["font"]["weight"], e["planned"], 255)
    return np.array(layer) > 127


def register(img: np.ndarray, mask: np.ndarray, color_hex: str, window: int = 120,
             scales=tuple(np.round(np.arange(0.85, 1.151, 0.03), 2))) -> dict:
    """Where the planned text actually landed: shift (dx, dy) of its top-left and
    scale about that point, by normalised cross-correlation on a text-colour map."""
    ys, xs = np.where(mask)
    if not len(ys):
        return {"dx": 0, "dy": 0, "scale": 1.0, "score": 0.0, "anchor": [0, 0]}
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    H, W = mask.shape
    smax = max(scales)
    wx0, wy0 = max(0, x0 - window), max(0, y0 - window)
    wx1, wy1 = min(W, int(x0 + (x1 - x0) * smax) + window), min(H, int(y0 + (y1 - y0) * smax) + window)
    win = img[wy0:wy1, wx0:wx1].astype(float)
    bg = np.median(win.reshape(-1, 3), 0)
    d_text = np.linalg.norm(win - np.array(hex2rgb(color_hex), float), axis=2)
    d_bg = np.linalg.norm(win - bg, axis=2)
    like = (d_bg / (d_text + d_bg + 1e-6)).astype(np.float32)
    base = mask[y0:y1, x0:x1].astype(np.float32)
    best = None
    for s in scales:
        tw, th = max(2, int(round(base.shape[1] * s))), max(2, int(round(base.shape[0] * s)))
        if th > like.shape[0] or tw > like.shape[1]:
            continue
        tmpl = base if s == 1.0 else cv2.resize(base, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(like, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if best is None or score > best[0]:
            best = (score, s, loc)
    if best is None:
        return {"dx": 0, "dy": 0, "scale": 1.0, "score": 0.0, "anchor": [int(x0), int(y0)]}
    score, s, loc = best
    return {"dx": int(wx0 + loc[0] - x0), "dy": int(wy0 + loc[1] - y0), "scale": round(float(s), 3),
            "score": round(float(score), 3), "anchor": [int(x0), int(y0)]}


def verdict(r: dict) -> str:
    if r["score"] < 0.6:
        return "not_found"
    if max(abs(r["dx"]), abs(r["dy"])) <= 2 and abs(r["scale"] - 1) <= 0.02:
        return "confirmed"
    return "corrected"


def transform_planned(p: dict, ax: float, ay: float, s: float, dx: float, dy: float) -> dict:
    """Apply new = anchor + (old - anchor) * s + shift to every coordinate."""
    tx = lambda x: round(ax + (x - ax) * s + dx, 1)
    ty = lambda y: round(ay + (y - ay) * s + dy, 1)
    q = json.loads(json.dumps(p))
    q["font_size_px"] = round(p["font_size_px"] * s, 2)
    q["letter_spacing_px"] = round(p["letter_spacing_px"] * s, 2)
    b = p["box"]
    q["box"] = {"x": tx(b["x"]), "y": ty(b["y"]), "w": round(b["w"] * s, 1), "h": round(b["h"] * s, 1)}
    for ln, lq in zip(p["lines"], q["lines"]):
        lq["x"], lq["baseline_y"] = tx(ln["x"]), ty(ln["baseline_y"])
        lq["advance_width"] = round(ln["advance_width"] * s, 1)
        if ln.get("ink_box"):
            ib = ln["ink_box"]
            lq["ink_box"] = [tx(ib[0]), ty(ib[1]), tx(ib[2]), ty(ib[3])]
    return q


def scale_typesetting(ts: dict, sx: float, sy: float) -> dict:
    """The renderer returned a different pixel size (fallback renderer): rescale."""
    if abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3:
        return ts
    out = json.loads(json.dumps(ts))
    s = (sx + sy) / 2
    for e in out["elements"]:
        e["planned"] = transform_planned(e["planned"], 0, 0, s, 0, 0)
    out["canvas"] = {"width": int(round(ts["canvas"]["width"] * sx)), "height": int(round(ts["canvas"]["height"] * sy))}
    return out


def post_contrast(img: np.ndarray, mask: np.ndarray, color_hex: str, target: float) -> dict:
    """Editable-text colour vs the artwork actually painted around the glyphs (worst case)."""
    ys = np.where(mask)[0]
    if not len(ys):
        return {"ratio": None, "pass": None}
    k = max(3, int(0.15 * (np.ptp(ys) + 1)))
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~mask
    px = img[ring]
    if len(px) < 20:
        return {"ratio": None, "pass": None}
    px = px[:: max(1, len(px) // 4000)]
    lum = np.array([luminance(p) for p in px])
    tl = luminance(hex2rgb(color_hex))
    worst = np.percentile(lum, 10) if tl > np.median(lum) else np.percentile(lum, 90)
    r = (max(tl, worst) + 0.05) / (min(tl, worst) + 0.05)
    return {"ratio": round(float(r), 2), "target": target, "pass": bool(r >= target),
            "backdrop_median_hex": rgb2hex(np.median(px, 0))}


def text_qa(image_path: Path, ts: dict, user_font_mode: bool, ocr=None) -> dict:
    """Confirm/correct every element's metadata against the rendered pixels.
    `ocr` is typography_ocr (optional) for elements registration cannot place."""
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    ts = scale_typesetting(ts, W / ts["canvas"]["width"], H / ts["canvas"]["height"])
    out = []
    for e in ts["elements"]:
        p = e["planned"]
        if user_font_mode:
            reg, v = None, "not_found"
        else:
            reg = register(img, element_mask(e, W, H), p["color_hex"])
            v = verdict(reg)
        final, source = p, "planned"
        if v == "corrected":
            final = transform_planned(p, reg["anchor"][0], reg["anchor"][1], reg["scale"], reg["dx"], reg["dy"])
            source = "planned+registration"
        elif v == "not_found":
            m = ocr.measure_element(img, e) if ocr else None
            if m:
                final, source, v = m, "measured", "measured"
            else:
                source, v = "planned (unverified)", "unverified"
        target = e["contrast_plan"]["target"]
        el2 = {**e, "planned": final}
        pc = post_contrast(img, element_mask(el2, W, H), final["color_hex"], target)
        # legibility as rendered: the model can shrink text below the platform floor
        floor = p.get("min_font_px") or 0
        leg = {"font_size_px": final["font_size_px"], "min_font_px": floor,
               "pass": bool(final["font_size_px"] >= 0.95 * floor)}
        out.append({**e, "final": final, "qa": {"status": v, "source": source, "registration": reg,
                                                 "contrast_after": pc, "legibility_after": leg}})
    return {**ts, "canvas": {"width": W, "height": H}, "elements": out}


def text_layers(ts: dict, image_name: str) -> dict:
    """kognie.text_layers.v2 -- what an HTML canvas needs to recreate each element."""
    W, H = ts["canvas"]["width"], ts["canvas"]["height"]
    els = []
    for e in ts["elements"]:
        f = e.get("final") or e["planned"]
        b = element_ink_box(f) or [f["box"]["x"], f["box"]["y"], f["box"]["x"] + f["box"]["w"], f["box"]["y"] + f["box"]["h"]]
        els.append({
            "id": e["id"], "role": e["role"], "text": e["text"], "text_transform": f["text_transform"],
            "font_family": e["font"]["family"], "css_font_family": e["font"]["css_font_family"],
            "font_weight": e["font"]["weight"], "font_style": "normal", "font_key": e["font"]["key"],
            "font_metrics": e["font"]["metrics"],
            "font_size_px": f["font_size_px"], "letter_spacing_px": f["letter_spacing_px"],
            "line_height": f["line_height"], "align": f["align"], "color_hex": f["color_hex"],
            "rotation_deg": f.get("rotation_deg", 0),
            "lines": [{"text": l["text"], "x": l["x"], "baseline_y": l["baseline_y"],
                       "advance_width": l.get("advance_width"), "ink_box": l.get("ink_box")} for l in f["lines"]],
            "box": {"x": b[0], "y": b[1], "w": round(b[2] - b[0], 1), "h": round(b[3] - b[1], 1)},
            "box_norm": {"x": round(b[0] / W, 4), "y": round(b[1] / H, 4),
                         "w": round((b[2] - b[0]) / W, 4), "h": round((b[3] - b[1]) / H, 4)},
            "contrast": {"plan": e.get("contrast_plan"), "after_render": (e.get("qa") or {}).get("contrast_after")},
            "qa": {k: v for k, v in (e.get("qa") or {}).items() if k != "contrast_after"},
            "approximate_font": bool(ts.get("user_font")),
        })
    doc = {"schema": "kognie.text_layers.v2", "image": image_name,
            "canvas": {"width": W, "height": H, "units": "px", "origin": "top-left"},
            "coordinates": ("Per line: x = left origin, baseline_y = alphabetic baseline "
                            "(canvas fillText(text, x, baseline_y) at font_size_px with letter_spacing_px). "
                            "ink_box = [x0, y0, x1, y1] of visible ink. box_norm is 0-1."),
            "fonts_used": [FONTS[k]["family"] for k in ts.get("fonts_used", [])],
            "user_notices": ts.get("user_notices", []), "policy_log": ts.get("policy_log", []),
            "elements": els}
    # plain JSON types only: callers json.dumps this without a custom encoder
    return json.loads(dumps(doc))


def _native(o):
    return o.item() if hasattr(o, "item") else str(o)


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_native)
