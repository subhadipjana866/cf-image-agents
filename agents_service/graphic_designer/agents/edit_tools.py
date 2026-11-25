"""Tools of the image-edit agent.

Three kinds, all working on the session's latest image unless `version` is given,
and each producing a new version:

* Generative edits. One tool per kind of change, filling today's edit template
  (``config.editPrompts``) from the tool's arguments, then running today's
  executor chain unchanged: aspectRatioHandler -> gpt-image -> nano-banana-pro
  -> nano-banana -> flux-kontext. ``apply_multiple_edits`` sends several
  changes in one render.
* Exact edits (Pillow, no AI, free): crop, resize, rotate, tone, borders...
* Canvas, quality and history: platform resizes, upscale, logo paste, undo,
  export, image info.

Nothing here asks the user; failures come back as FAILED for the agent to explain.
"""
from __future__ import annotations

import base64
import math
import re
import time
from io import BytesIO
from typing import Literal, Optional

import pystache
from langchain.tools import ToolRuntime, tool
from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageOps

from image_agents.agents.editor import (_adjust, _crop, _grayscale, _recolor_region,
                                        _resize, _round_corners)

from ..config import editPrompts
from .common import base_state, describe, logo_data_uri, meta_state, safe_tool
from .images import encode, load_bytes, load_image, to_data_uri, upload
from .runner import run_chain
from .session import DesignSession, Version

AGENT_ID = "image-editor"


# ------------------------------------------------------------------ shared plumbing
def _save(session: DesignSession, img: Image.Image, *, tool_name: str, instruction: str,
          base: Version, started: float, cost: float = 0.0, model: Optional[str] = None,
          fmt: Optional[str] = None, kind: str = "edited") -> str:
    data, ext = encode(img, fmt)
    url = upload(data, ext, user_id=session.user_id, session_id=session.session_id,
                 meta_state=meta_state(session, base), kind=kind, model=model or "")
    v = session.add_version(url, made_by=AGENT_ID, tool=tool_name, instruction=instruction,
                            parent=base.number, model=model,
                            meta={"size": list(img.size)})
    session.record(tool_name, ok=True, detail=instruction, cost=cost, started=started,
                   version=v.number)
    return f"OK: {describe(v)} | size {img.size[0]}x{img.size[1]} | ${cost:.4f}"


def _exact(runtime: ToolRuntime[DesignSession], version: Optional[int], tool_name: str,
           instruction: str, fn) -> str:
    session = runtime.context
    started = time.monotonic()
    base = session.get(version)
    return _save(session, fn(load_image(base.url)), tool_name=tool_name,
                 instruction=instruction, base=base, started=started)


def _generative(runtime: ToolRuntime[DesignSession], *, tool_name: str, prompt: str,
                version: Optional[int], use_logo: bool = False,
                references: Optional[list[str]] = None, style_switch: bool = False) -> str:
    """Run today's edit executor chain on one version with a ready prompt."""
    session = runtime.context
    started = time.monotonic()
    base = session.get(version)
    # The nodes fetch http(s) URLs and pass data URIs through; anything else
    # (a local path, file://) must arrive as a data URI.
    source = base.url if base.url.startswith(("http://", "https://", "data:")) else to_data_uri(base.url)
    state = base_state(session) | {
        "current_image_url": source,
        "prompt": prompt,
        "edit_style_switch": style_switch,
        "edit_reference_images": [to_data_uri(r) for r in references or []],
    }
    if use_logo:
        logo = logo_data_uri(session)
        if not logo:
            return "FAILED: no logo file is available in this session (brief or brand kit)."
        state.update(edit_logo_mode="provided", logo_asset=logo)
    chain = run_chain("aspectRatioHandler", state)
    cost = float(chain.state.get("cost") or 0.0)
    if chain.outcome != "done":
        reply = (f"FAILED: all four edit models failed (gpt-image, nano-banana-pro, nano-banana, "
                 f"flux). The image is unchanged at version {base.number}.")
        session.record(tool_name, ok=False, detail=reply, cost=cost, started=started,
                       path=chain.path)
        return reply
    url = chain.state["current_image_url"]
    model = (chain.state.get("image_models") or {}).get(url) or chain.path[-1]
    v = session.add_version(url, made_by=AGENT_ID, tool=tool_name, instruction=prompt,
                            parent=base.number, model=model)
    session.record(tool_name, ok=True, detail=prompt[:200], cost=cost, started=started,
                   version=v.number, path=chain.path)
    return f"OK: {describe(v)} | ${cost:.4f}"


def _template(class_id: str, fields: dict) -> str:
    return pystache.render(editPrompts[class_id], fields)


# --------------------------------------------------------------- generative edits
@tool
@safe_tool
def insert_object(runtime: ToolRuntime[DesignSession], new_object: str,
                  placement: Optional[str] = None, appearance: Optional[str] = None,
                  scale: Optional[str] = None, version: Optional[int] = None) -> str:
    """Add a new object that is not in the image yet (a plant, a hat, a coffee cup, confetti).

    When to use: "add", "put", "insert", "place a ... on/next to ...".
    Not for: swapping an existing object for another (replace_object), copying
    one already in the image (duplicate_object), adding text (edit_text_or_graphic)
    or adding the brand logo file (place_logo).
    """
    prompt = _template("1A", {"target": new_object, "attributes": {
        "placement": placement, "appearance": appearance, "scale": scale}})
    return _generative(runtime, tool_name="insert_object", prompt=prompt, version=version)


@tool
@safe_tool
def replace_object(runtime: ToolRuntime[DesignSession], existing_object: str, new_object: str,
                   appearance: Optional[str] = None, version: Optional[int] = None) -> str:
    """Swap an object in the image for a different object, in the same place and scale.

    When to use: "replace the cup with a glass", "make the dog a cat", "swap X for Y".
    Not for: changing only the colour or material of the same object
    (change_object_attribute), or the background (replace_background).
    """
    prompt = _template("1A", {"replaces": existing_object, "target": new_object,
                              "attributes": {"appearance": appearance}})
    return _generative(runtime, tool_name="replace_object", prompt=prompt, version=version)


@tool
@safe_tool
def duplicate_object(runtime: ToolRuntime[DesignSession], object_to_copy: str,
                     placement: Optional[str] = None, version: Optional[int] = None) -> str:
    """Duplicate an object already in the image and add the copy elsewhere.

    When to use: "add another one of these", "clone the balloon", "two of the bottle".
    """
    prompt = _template("1A", {"source": object_to_copy, "target": object_to_copy,
                              "attributes": {"placement": placement}})
    return _generative(runtime, tool_name="duplicate_object", prompt=prompt, version=version)


@tool
@safe_tool
def transform_object(runtime: ToolRuntime[DesignSession], target: str, action: str,
                     direction: Optional[str] = None, distance: Optional[str] = None,
                     angle: Optional[str] = None, scale_change: Optional[str] = None,
                     version: Optional[int] = None) -> str:
    """Move, rotate, flip or resize ONE element inside the design, keeping how it looks.

    When to use: "move the logo to the top right", "make the product bigger",
    "rotate the badge 15 degrees", "flip the person to face left".
    `action` is move | rotate | flip | resize. Not for rotating or flipping the
    WHOLE image (rotate_image, flip_image) or resizing the whole canvas.
    """
    prompt = _template("1B", {"target": target, "action": action, "parameters": {
        "direction": direction, "distance": distance or "a noticeable amount",
        "angle": angle, "scale_change": scale_change}})
    return _generative(runtime, tool_name="transform_object", prompt=prompt, version=version)


@tool
@safe_tool
def erase_object(runtime: ToolRuntime[DesignSession], target: str,
                 version: Optional[int] = None) -> str:
    """Remove an object, person, watermark or blemish and fill the gap seamlessly.

    When to use: "remove", "delete", "erase", "get rid of". Include where it is
    in `target` when there are several ("the cup on the left").
    Not for removing text only (edit_text_or_graphic with action remove), or
    cutting the subject out onto transparency (not an edit: the layering agent
    does that).
    """
    return _generative(runtime, tool_name="erase_object",
                       prompt=_template("2", {"target": target}), version=version)


@tool
@safe_tool
def change_object_attribute(runtime: ToolRuntime[DesignSession], target: str,
                            attribute_type: str, new_value: str,
                            version: Optional[int] = None) -> str:
    """Change the colour, material, texture, pattern or finish of one object; its shape stays.

    When to use: "make the button orange", "car in matte black", "wooden table
    instead of glass", "gold frame". `attribute_type` is colour | material |
    texture | pattern | finish.
    Not for the colour of the WHOLE image (adjust_tone, change_mood), text colour
    (edit_text_or_graphic) or a flat colour box when you know its pixels (recolor_region).
    """
    prompt = _template("3", {"target": target, "attribute_type": attribute_type,
                             "new_value": new_value})
    return _generative(runtime, tool_name="change_object_attribute", prompt=prompt,
                       version=version)


@tool
@safe_tool
def modify_subject(runtime: ToolRuntime[DesignSession], target: str, aspect: str,
                   details: str, version: Optional[int] = None) -> str:
    """Change a person's or animal's pose, expression, outfit, hairstyle, age or gesture.

    When to use: "make her smile", "put him in a suit", "make the dog sit".
    `aspect` is pose | expression | outfit | hair | age | gesture.
    Not for malformed hands or faces left by AI generation (artifact repair is not
    available yet), or swapping the subject for a different one (replace_object).
    """
    prompt = _template("4", {"target": target, "action": aspect, "attributes": details})
    return _generative(runtime, tool_name="modify_subject", prompt=prompt, version=version)


@tool
@safe_tool
def replace_background(runtime: ToolRuntime[DesignSession], new_background: str,
                       version: Optional[int] = None) -> str:
    """Replace the whole background or scene behind the subject (colour, place, sky, weather).

    When to use: "background navy blue", "put it on a beach", "sunset sky",
    "marble countertop behind the product".
    Not for a transparent background (the layering agent cuts subjects out), or
    the overall lighting mood with the same scene (change_mood).
    """
    return _generative(runtime, tool_name="replace_background",
                       prompt=_template("5", {"attributes": new_background}), version=version)


@tool
@safe_tool
def apply_art_style(runtime: ToolRuntime[DesignSession], style: str,
                    version: Optional[int] = None) -> str:
    """Re-render the whole design in another visual style, keeping its content.

    When to use: "make it Bauhaus", "watercolour", "anime", "retro", "minimal",
    "black-and-white sketch". Preset names from the style library (minimal, swiss,
    editorial, luxury, bold-typographic, bauhaus, brutalist, neo-brutalism,
    memphis, retro, risograph, collage, organic, hand-drawn, flat-illustration,
    soft-3d, photographic, gradient, dark-neon, pop-art, art-deco) get the full
    preset direction. Not for a plain black-and-white filter (convert_to_black_and_white).
    """
    from .. import style_presets
    session = runtime.context
    if style_presets.find_preset(style):
        prompt = style_presets.restyle_prompt(style, brand_kit=session.brand_dna,
                                              request=f"Restyle this design as {style}")
    else:
        prompt = _template("6", {"attributes": style})
    return _generative(runtime, tool_name="apply_art_style", prompt=prompt, version=version,
                       style_switch=True)


@tool
@safe_tool
def edit_text_or_graphic(runtime: ToolRuntime[DesignSession], action: str,
                         text: Optional[str] = None, font: Optional[str] = None,
                         color: Optional[str] = None, size: Optional[str] = None,
                         placement: Optional[str] = None, version: Optional[int] = None) -> str:
    """Change, add, remove, restyle or fix text or simple graphics in the design.

    When to use: "change the headline to 'Now Open'", "fix the spelling of
    Resturant", "make the date bold and white", "remove the watermark text",
    "add 'Free entry' under the date". `action` is change | add | remove |
    restyle | fix. Put the exact new wording in `text`, and which text block it
    is in `placement` (e.g. "the headline", "under the date").
    Not for writing new wording when the user gave none: ask them or suggest options first.
    """
    prompt = _template("7", {"action": action, "attributes": {
        "text": text, "font": font, "color": color, "size": size, "placement": placement}})
    return _generative(runtime, tool_name="edit_text_or_graphic", prompt=prompt, version=version)


@tool
@safe_tool
def reframe_with_ai(runtime: ToolRuntime[DesignSession], action: str, parameters: str,
                    focus: Optional[str] = None, version: Optional[int] = None) -> str:
    """Zoom in or out, or extend the scene beyond its edges, generating new content as needed.

    When to use: "zoom out to show more of the room", "extend the left side",
    "zoom in on the product". `action` is zoom in | zoom out | extend.
    Not for exact crops (crop_image) or resizing for named platforms (resize_for_platforms).
    """
    prompt = _template("8", {"action": action, "target": focus, "parameters": parameters})
    return _generative(runtime, tool_name="reframe_with_ai", prompt=prompt, version=version)


@tool
@safe_tool
def enhance_quality(runtime: ToolRuntime[DesignSession], action: str,
                    area: Optional[str] = None, version: Optional[int] = None) -> str:
    """Improve technical quality with AI: denoise, deblur, restore detail, fix compression.

    When to use: "it's blurry", "remove the noise", "make it crisper", "clean up
    the pixelation". Not for a bigger resolution (upscale_image) or a simple
    sharpen filter (sharpen_image).
    """
    return _generative(runtime, tool_name="enhance_quality",
                       prompt=_template("9", {"action": action, "attributes": area}),
                       version=version)


@tool
@safe_tool
def change_mood(runtime: ToolRuntime[DesignSession], mood: str,
                version: Optional[int] = None) -> str:
    """Relight and colour-grade the whole scene for a mood or time of day.

    When to use: "make it feel cosy", "golden hour", "moodier", "more festive
    lighting", "night time". Not for exact brightness or contrast numbers (adjust_tone).
    """
    return _generative(runtime, tool_name="change_mood",
                       prompt=_template("10", {"attributes": mood}), version=version)


@tool
@safe_tool
def custom_edit(runtime: ToolRuntime[DesignSession], instruction: str,
                version: Optional[int] = None) -> str:
    """Any other visual change to the image content that no specific tool covers.

    Write `instruction` as one clear, complete edit instruction. Prefer a
    specific tool when one fits; this is the fallback.
    """
    return _generative(runtime, tool_name="custom_edit", prompt=instruction, version=version)


@tool
@safe_tool
def apply_multiple_edits(runtime: ToolRuntime[DesignSession], instructions: list[str],
                         version: Optional[int] = None) -> str:
    """Apply several AI content changes to the image in ONE render.

    When to use: two to five independent generative changes to different parts
    of the image ("headline bigger, background navy, button orange"). One render
    keeps the image cleaner than several renders in a row and costs less.
    Write each item as one complete instruction.
    Do NOT use when one change depends on the result of another, when a change
    is exact (crop, resize, rotate, tone numbers: use those tools before or
    after), or for a single change.
    """
    items = [i.strip() for i in instructions if i and i.strip()]
    if len(items) < 2:
        return "FAILED: give at least two changes; for one change use its own tool."
    prompt = ("You are an advanced image editing engine. Make ALL of the following changes to "
              "this image in a single pass. Apply each one exactly as described, keep every "
              "part of the image they do not mention unchanged, and keep the result seamless:\n"
              + "\n".join(f"{n}. {item}" for n, item in enumerate(items, 1)))
    return _generative(runtime, tool_name="apply_multiple_edits", prompt=prompt, version=version)


# ------------------------------------------------------------------- exact edits
@tool
@safe_tool
def crop_image(runtime: ToolRuntime[DesignSession], aspect_ratio: Optional[str] = None,
               box: Optional[list[int]] = None, version: Optional[int] = None) -> str:
    """Crop the whole image exactly: to an aspect ratio, centred ("1:1", "16:9"), or to a pixel box.

    `box` is [left, top, right, bottom] in pixels (check get_image_info for the size).
    Not for extending the image (reframe_with_ai) or fitting a platform's exact
    size (resize_for_platforms).
    """
    return _exact(runtime, version, "crop_image", f"crop {aspect_ratio or box}",
                  lambda img: _crop(img, box=box, aspect_ratio=aspect_ratio))


@tool
@safe_tool
def resize_image(runtime: ToolRuntime[DesignSession], width: int, height: int,
                 version: Optional[int] = None) -> str:
    """Resize the whole image to exact pixel dimensions (may stretch if the ratio changes).

    For a social platform size use resize_for_platforms instead, which never stretches.
    """
    return _exact(runtime, version, "resize_image", f"resize to {width}x{height}",
                  lambda img: _resize(img, width, height))


@tool
@safe_tool
def shrink_image(runtime: ToolRuntime[DesignSession], percent: float,
                 version: Optional[int] = None) -> str:
    """Scale the whole image down (or up) by a percentage, keeping its aspect ratio."""
    def fn(img):
        f = max(1.0, float(percent)) / 100
        return _resize(img, max(1, round(img.width * f)), max(1, round(img.height * f)))
    return _exact(runtime, version, "shrink_image", f"scale to {percent}%", fn)


@tool
@safe_tool
def rotate_image(runtime: ToolRuntime[DesignSession], degrees: float,
                 version: Optional[int] = None) -> str:
    """Rotate the WHOLE image by a number of degrees (positive = counter-clockwise).

    The canvas grows to fit and the new corners are transparent. Not for one
    element inside the design (transform_object).
    """
    return _exact(runtime, version, "rotate_image", f"rotate {degrees} degrees",
                  lambda img: img.convert("RGBA").rotate(float(degrees), expand=True,
                                                         resample=Image.BICUBIC))


@tool
@safe_tool
def flip_image(runtime: ToolRuntime[DesignSession],
               direction: Literal["horizontal", "vertical"] = "horizontal",
               version: Optional[int] = None) -> str:
    """Mirror the WHOLE image horizontally or vertically.

    Text will read backwards: warn the user, or flip only one element with
    transform_object instead.
    """
    return _exact(runtime, version, "flip_image", f"flip {direction}",
                  lambda img: ImageOps.mirror(img) if direction == "horizontal"
                  else ImageOps.flip(img))


@tool
@safe_tool
def adjust_tone(runtime: ToolRuntime[DesignSession], brightness: float = 1.0,
                contrast: float = 1.0, saturation: float = 1.0, warmth: float = 0.0,
                version: Optional[int] = None) -> str:
    """Exact tone adjustment of the whole image. Factors: 1.0 = unchanged, 1.2 = 20% more.

    `warmth` runs -1 (cooler) to 1 (warmer). When to use: "brighter by 20" ->
    brightness 1.2; "less saturated"; "a bit warmer" -> warmth 0.3; "more
    contrast". Not for mood or lighting changes that need AI (change_mood).
    """
    return _exact(runtime, version, "adjust_tone",
                  f"brightness {brightness}, contrast {contrast}, saturation {saturation}, "
                  f"warmth {warmth}",
                  lambda img: _adjust(img, brightness, contrast, saturation, warmth))


@tool
@safe_tool
def convert_to_black_and_white(runtime: ToolRuntime[DesignSession],
                               version: Optional[int] = None) -> str:
    """Convert the whole image to black and white (grayscale), exactly."""
    return _exact(runtime, version, "convert_to_black_and_white", "grayscale", _grayscale)


@tool
@safe_tool
def round_corners(runtime: ToolRuntime[DesignSession], radius: int,
                  version: Optional[int] = None) -> str:
    """Round the corners of the whole image by a radius in pixels (corners become transparent)."""
    return _exact(runtime, version, "round_corners", f"corner radius {radius}px",
                  lambda img: _round_corners(img, radius))


@tool
@safe_tool
def add_border(runtime: ToolRuntime[DesignSession], width: int, color: str = "white",
               version: Optional[int] = None) -> str:
    """Add a solid border around the outside edge of the whole image.

    `width` in pixels, `color` a CSS name or hex. The canvas grows by the border.
    """
    def fn(img):
        rgba = img.convert("RGBA")
        return ImageOps.expand(rgba, border=int(width), fill=ImageColor.getrgb(color))
    return _exact(runtime, version, "add_border", f"{width}px {color} border", fn)


@tool
@safe_tool
def set_opacity(runtime: ToolRuntime[DesignSession], percent: float,
                version: Optional[int] = None) -> str:
    """Make the whole image partly transparent: 100 = solid, 0 = invisible."""
    def fn(img):
        rgba = img.convert("RGBA")
        f = max(0.0, min(float(percent), 100.0)) / 100
        rgba.putalpha(rgba.getchannel("A").point(lambda a: round(a * f)))
        return rgba
    return _exact(runtime, version, "set_opacity", f"opacity {percent}%", fn)


@tool
@safe_tool
def sharpen_image(runtime: ToolRuntime[DesignSession], amount: float = 1.5,
                  version: Optional[int] = None) -> str:
    """Apply an exact sharpen filter to the whole image (1.0 = unchanged, 2.0 = strong).

    For real blur, noise or lost detail use enhance_quality.
    """
    def fn(img):
        alpha = img.getchannel("A") if img.mode == "RGBA" else None
        out = ImageEnhance.Sharpness(img.convert("RGB")).enhance(float(amount))
        if alpha is not None:
            out.putalpha(alpha)
        return out
    return _exact(runtime, version, "sharpen_image", f"sharpen {amount}", fn)


@tool
@safe_tool
def recolor_region(runtime: ToolRuntime[DesignSession], box: list[int], color: str,
                   version: Optional[int] = None) -> str:
    """Recolour a rectangular pixel region exactly, keeping its light and shade.

    Only when the region's pixel box is known (get_image_info gives the size).
    For a named object use change_object_attribute.
    """
    return _exact(runtime, version, "recolor_region", f"recolour {box} to {color}",
                  lambda img: _recolor_region(img, box, color))


_LOGO_ZONES = {   # x range, y range in % of the canvas
    "top-left": ([4, 26], [3, 13]), "top-center": ([38, 62], [3, 12]),
    "top-right": ([74, 96], [3, 13]), "bottom-left": ([4, 26], [87, 97]),
    "bottom-center": ([38, 62], [88, 97]), "bottom-right": ([74, 96], [87, 97]),
    "center": ([35, 65], [40, 60]),
}


@tool
@safe_tool
def place_logo(runtime: ToolRuntime[DesignSession],
               position: Literal["top-left", "top-center", "top-right", "bottom-left",
                                 "bottom-center", "bottom-right", "center"] = "top-center",
               version: Optional[int] = None) -> str:
    """Paste the real brand logo file onto the design, pixel-exact, at a position.

    Uses the logo from the brief or brand kit; never redraws it. When to use:
    "add our logo", "put the logo bottom right". Not for changing a logo already
    drawn in the image (edit_text_or_graphic or transform_object).
    """
    from ..utils import compositeLogo
    session = runtime.context
    logo = logo_data_uri(session)
    if not logo:
        return "FAILED: no logo file is available (none in the brief or brand kit)."
    started = time.monotonic()
    base = session.get(version)
    x, y = _LOGO_ZONES[position]
    out = compositeLogo(load_bytes(base.url), logo, {"x_pct": x, "y_pct": y})
    return _save(session, Image.open(BytesIO(out)), tool_name="place_logo",
                 instruction=f"logo at {position}", base=base, started=started,
                 kind="logo-composited")


# ------------------------------------------------------- canvas, quality, history
# The ten catalog platforms (layout_guidance.json canvases).
_CATALOG_CANVAS = {
    "Digital Signage Portrait": (1080, 1920), "Facebook Feed Landscape": (1080, 607),
    "Instagram Portrait Post": (1080, 1350), "Instagram Square Post": (1080, 1080),
    "Instagram Story": (1080, 1920), "LinkedIn Feed Post": (1080, 864),
    "Pinterest Pin": (1080, 1440), "Presentation Slide": (1080, 810),
    "X (Twitter) In-Stream Post": (1080, 607), "YouTube Thumbnail": (1080, 607),
}


def _target_size(target: str, img: Image.Image) -> tuple[str, int, int]:
    from ..config import PLATFORM_DIMENSIONS
    t = target.strip().lower()
    m = re.fullmatch(r"(\d+)\s*[:x]\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 64 and b > 64:                       # "1080x1350": exact pixels
            return target, a, b
        long_side = max(img.size)
        return (target, long_side, round(long_side * b / a)) if a >= b else \
               (target, round(long_side * a / b), long_side)
    if t in PLATFORM_DIMENSIONS:
        d = PLATFORM_DIMENSIONS[t]
        return target, d["width"], d["height"]
    for name, (w, h) in _CATALOG_CANVAS.items():
        if name.lower() == t:
            return name, w, h
    raise ValueError(f"unknown platform or size '{target}'. Use a platform name such as "
                     f"'instagram story', a ratio like '9:16', or pixels like '1080x1920'.")


@tool
@safe_tool
def resize_for_platforms(runtime: ToolRuntime[DesignSession], targets: list[str],
                         version: Optional[int] = None) -> str:
    """Make versions of the design for other platforms or aspect ratios, without stretching.

    `targets`: platform names ("instagram story", "linkedin post", "youtube
    thumbnail", "facebook cover", "pinterest pin"...), ratios ("9:16") or pixels
    ("1080x1920"). Same ratio = clean resize; a new ratio extends the scene with
    AI first. Each target becomes its own new version.
    """
    from ..utils import calculateCost
    session = runtime.context
    base = session.get(version)
    data = load_bytes(base.url)
    img = Image.open(BytesIO(data))
    lines = []
    for target in targets:
        started = time.monotonic()
        try:
            name, tw, th = _target_size(target, img)
            cost = 0.0
            if abs(img.width / img.height - tw / th) < 0.01:
                out = img.resize((tw, th), Image.LANCZOS)
            else:
                from ..nodes import gptExpandImage
                g = math.gcd(tw, th)
                expanded = gptExpandImage(data, f"{tw // g}:{th // g}")
                out = Image.open(BytesIO(expanded)).resize((tw, th), Image.LANCZOS)
                cost = calculateCost("gpt-expand", 1000)
            lines.append(_save(session, out, tool_name="resize_for_platforms",
                               instruction=f"{name} ({tw}x{th}) from v{base.number}", base=base,
                               started=started, cost=cost, kind="expanded"))
        except Exception as err:  # noqa: BLE001 - one target failing must not hide the others
            lines.append(f"FAILED for '{target}': {err}")
            session.record("resize_for_platforms", ok=False, detail=f"{target}: {err}",
                           started=started)
    return "\n".join(lines)


@tool
@safe_tool
def upscale_image(runtime: ToolRuntime[DesignSession], scale: float = 2.0,
                  version: Optional[int] = None) -> str:
    """Increase the resolution of the whole image (2x or 4x) for print or large screens."""
    from ..services import upscaleImage
    from ..utils import calculateCost
    session = runtime.context
    started = time.monotonic()
    base = session.get(version)
    b64 = base64.b64encode(load_bytes(base.url)).decode()
    out = upscaleImage(b64, scale=float(scale))
    try:
        img = Image.open(BytesIO(base64.b64decode(out)))
        img.load()
    except Exception:
        return f"FAILED: the upscaler did not return an image ({str(out)[:200]})"
    return _save(session, img, tool_name="upscale_image", instruction=f"upscale x{scale}",
                 base=base, started=started, cost=calculateCost("upscale-fast", 1000))


@tool
@safe_tool
def get_image_info(runtime: ToolRuntime[DesignSession], version: Optional[int] = None) -> str:
    """Size, format and transparency of a version, plus the list of all versions. Free.

    Use before exact pixel edits, or when the user asks about the image or its history.
    """
    session = runtime.context
    v = session.get(version)
    img = load_image(v.url)
    from .images import has_transparency
    history = "\n".join(f"- {describe(x)} <- {x.instruction[:80]!r}" for x in session.versions)
    return (f"version {v.number}: {img.width}x{img.height}, {img.format}, "
            f"transparent={has_transparency(img)}\nVersions:\n{history}")


@tool
@safe_tool
def restore_version(runtime: ToolRuntime[DesignSession], version: int) -> str:
    """Undo: bring back an earlier version as the newest version (nothing is deleted).

    When to use: "undo", "go back", "I liked the previous one better", "use the first one".
    """
    session = runtime.context
    started = time.monotonic()
    old = session.get(version)
    v = session.add_version(old.url, made_by=AGENT_ID, tool="restore_version",
                            instruction=f"restored version {old.number}", parent=old.number,
                            model=old.model, meta=dict(old.meta))
    session.record("restore_version", ok=True, detail=f"v{old.number} -> v{v.number}",
                   started=started, version=v.number)
    return f"OK: {describe(v)} (same image as version {old.number})"


@tool
@safe_tool
def export_image(runtime: ToolRuntime[DesignSession], format: Literal["png", "jpg", "webp"],
                 version: Optional[int] = None) -> str:
    """Give the user a download link for a version in another file format.

    PNG keeps transparency; JPG has none. Does not change the design or add a version.
    """
    session = runtime.context
    started = time.monotonic()
    v = session.get(version)
    img = load_image(v.url)
    data, ext = encode(img, format)
    url = upload(data, ext, user_id=session.user_id, session_id=session.session_id,
                 meta_state=meta_state(session, v), kind="edited")
    session.record("export_image", ok=True, detail=f"v{v.number} as {ext}", started=started)
    from .images import has_transparency
    note = " (no transparency in this image)" if ext == "png" and not has_transparency(img) else ""
    return f"OK: version {v.number} as {ext.upper()}{note}: {url}"


EDIT_TOOLS = [
    # generative
    insert_object, replace_object, duplicate_object, transform_object, erase_object,
    change_object_attribute, modify_subject, replace_background, apply_art_style,
    edit_text_or_graphic, reframe_with_ai, enhance_quality, change_mood, custom_edit,
    apply_multiple_edits,
    # exact
    crop_image, resize_image, shrink_image, rotate_image, flip_image, adjust_tone,
    convert_to_black_and_white, round_corners, add_border, set_opacity, sharpen_image,
    recolor_region, place_logo,
    # canvas, quality, history
    resize_for_platforms, upscale_image, get_image_info, restore_version, export_image,
]
