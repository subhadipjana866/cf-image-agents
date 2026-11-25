"""The project's art style, applied FLAT to a v3.3 generation.

One style per generation. ``state["style"]`` holds the user's selection -- an
id or display name from :data:`PRESETS_PATH` (``styles.json``, the 21-preset
set), or any free-text style the editor sent. Whatever it is, exactly one style
reaches a run: there is no stacking, no per-stage style and no second style
later in the same generation.

"Flat" is the injection position: the style is appended to the brief handed to
``v33.run()`` as one labelled block, before GROUND. Every reasoning stage
(GROUND, SUBJECT, TYPOGRAPHER, ART DIRECTOR, COMPOSITOR, DIRECTOR) therefore
reads the same style text, and the vendored engine stays byte-identical to its
tested source -- the raw prompt is the one free-text input ``run()`` accepts.
The two alternatives measured in ``style_experiments/`` (injecting after
SUBJECT, with or without a style-rewriting agent call) are deliberately not
used here: they need a seam v33 does not expose, and the agent costs an extra
reasoning call per generation.

The block is labelled ART DIRECTION ONLY for two reasons found in testing:
GROUND otherwise extracts style words as facts of the brief, and the renderer
otherwise prints style nouns as text on the graphic.

Precedence is part of the block, not a convention the caller has to remember:
the user's own instructions outrank everything, an attached brand's colours,
fonts, logo, product images and approved copy outrank the style, and the style
outranks the goal's default treatment (:func:`priority_block`). With a brand
attached the block also states what the style may not restyle
(:func:`brand_lock`).

:func:`apply_style` is the generation path. :func:`style_directive` is the
shorter one-line form for ``create_more()``, which re-composes from the stored
brief rather than from the raw prompt and whose prompt has a word budget.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PRESETS_PATH = Path(__file__).with_name("styles.json")

#: Every text axis a preset must carry. A preset missing one is a data error,
#: not something to paper over at generation time -- a half-filled preset reads
#: as a vague adjective list and drifts.
_REQUIRED = ("id", "name", "palette", "type", "imagery", "texture",
             "lighting", "composition", "tone")

_RULES = (
    "This is ART DIRECTION ONLY. Apply it to palette, light quality, texture, "
    "finish, materials and typographic mood. Do NOT change what the graphic "
    "depicts, its layout, or its copy: every content decision still comes from "
    "the brief above. Do not print any word from this style direction unless "
    "the brief already contains it."
)

#: What an attached brand owns outright. The style still decides every
#: treatment the brand leaves open, but a brand asset -- its palette, its
#: fonts, the supplied logo file, the supplied product photo, its approved
#: wording -- is reproduced as it is. Restyling one of those is how a styled
#: generation stops looking like the brand it was made for.
_BRAND_LOCK_HEAD = (
    "BRAND LOCK — a brand is attached to this design, so the brand's own "
    "identity outranks the style above. Keep all of it, and apply the style "
    "only to what the brand leaves open:"
)

_BRAND_COLOURS = (
    "- Colours: take the design's colours from the brand's palette. The "
    "style's palette applies only where the brand states no colour, and it "
    "never replaces a brand colour."
)

_BRAND_TYPE = (
    "- Typography: set the type in the brand's own fonts. The style sets "
    "typographic mood only -- weight, case, spacing, hierarchy -- never a "
    "different typeface."
)

_BRAND_LOGO = (
    "- Logo: reproduce the supplied logo exactly, in its own colours, "
    "proportions and lettering. Do not recolour, texture, distress, outline, "
    "crop, flatten or redraw it to match the style."
)

_BRAND_PRODUCT = (
    "- Product images: a supplied product photo is the real product. Keep its "
    "shape, colour, packaging and label text exactly. The style may treat the "
    "scene around it -- light, surface, background -- never the product itself."
)

_BRAND_TEXT = (
    "- Text: the brand name and the approved copy are reproduced word for "
    "word. The style never rewrites a line, and never renders one in a "
    "treatment that makes it hard to read."
)

_CONFLICT_RULE = (
    "Resolve every conflict in that order: follow the higher rank and drop the "
    "lower one where the two collide. Do not split the difference. If the "
    "brief asks for something this style's Avoid list forbids, do what the "
    "brief asks."
)


def _brand_facts(brand_kit) -> dict | None:
    """The attached brand's style-relevant facts, or ``None`` with no brand.

    Accepts the prepared kit ``nodes._style_brand_kit`` builds (``name``,
    ``colors``, ``fonts``, ``logo``, ``product_image``) and, for any caller
    that still hands over the raw ``state["brand_dna"]`` payload, normalizes
    that shape through the critique pipeline's ``brandNormalizer`` instead of
    reading keys that schema does not have.
    """
    if not isinstance(brand_kit, dict) or not any(brand_kit.values()):
        return None

    keys = ("name", "colors", "fonts", "logo", "product_image")
    if any(brand_kit.get(k) for k in keys):
        return {k: brand_kit.get(k) for k in keys}

    try:
        from .critiquepipeline.grounding import brandNormalizer
        brand = brandNormalizer(brand_kit)
    except Exception:
        brand = None
    if not brand:
        # A kit with no readable identity is still an attached brand: the lock
        # and its rank in the priority order stay, only the named facts go.
        return {"name": None, "colors": (), "fonts": (), "logo": False,
                "product_image": False}

    palette = (brand.get("palette") or {}).get("named") or []
    return {
        "name": brand.get("name"),
        "colors": [f"{c['name']} ({c['hex']})" for c in palette if c.get("hex")],
        "fonts": [f for f in ((brand.get("fonts") or {}).get("families") or []) if f],
        "logo": bool(brand.get("logo_url")),
        "product_image": False,
    }


def brand_lock(brand_kit) -> str:
    """The BRAND LOCK block for an attached brand, or ``""`` when none is.

    The logo and product-image rules are stated only when one is actually
    attached: told to reproduce "the supplied logo" when nothing was supplied,
    the renderer letters one of its own.
    """
    facts = _brand_facts(brand_kit)
    if facts is None:
        return ""

    colours, fonts = facts.get("colors") or (), facts.get("fonts") or ()
    colour_rule = _BRAND_COLOURS
    if colours:
        colour_rule += f" The brand's colours are {', '.join(colours)}."
    type_rule = _BRAND_TYPE
    if fonts:
        type_rule += f" The brand's fonts are {', '.join(fonts)}."

    lines = [_BRAND_LOCK_HEAD, ""]
    if facts.get("name"):
        lines.append(f"- Brand: {facts['name']}.")
    lines += [colour_rule, type_rule]
    if facts.get("logo"):
        lines.append(_BRAND_LOGO)
    if facts.get("product_image"):
        lines.append(_BRAND_PRODUCT)
    lines.append(_BRAND_TEXT)
    return "\n".join(lines)


def priority_block(has_brand: bool) -> str:
    """The ranked precedence the whole brief is resolved against.

    One ordered list rather than three separate sentences, because the case
    that matters is the three-way one: the user asked for something the brand
    and the style both argue against, and the user still wins.
    """
    ranks = ["1. The user's own instructions in the brief above. An explicit "
             "request wins even when it contradicts both the brand and this "
             "style."]
    if has_brand:
        ranks.append("2. The brand: its colours, its fonts, its approved text, "
                     "the supplied logo and the supplied product images.")
    ranks.append(f"{len(ranks) + 1}. This style direction.")
    ranks.append(f"{len(ranks) + 1}. The goal's and the platform's default "
                 f"treatment.")
    return "\n".join(["PRIORITY ORDER, highest first:", *ranks, "", _CONFLICT_RULE])


@dataclass(frozen=True)
class StylePreset:
    """One selectable style, as data rather than prose."""

    id: str
    name: str
    palette: str
    type: str
    imagery: str
    texture: str
    lighting: str
    composition: str
    tone: str
    avoid: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> "StylePreset":
        missing = [k for k in _REQUIRED if not str(data.get(k) or "").strip()]
        if missing:
            raise ValueError(f"style preset {data.get('id')!r} is missing "
                             f"required field(s): {', '.join(missing)}")
        avoid = data.get("avoid") or []
        if isinstance(avoid, str):
            avoid = [avoid]
        return cls(
            id=str(data["id"]).strip(),
            name=str(data["name"]).strip(),
            palette=str(data["palette"]).strip(),
            type=str(data["type"]).strip(),
            imagery=str(data["imagery"]).strip(),
            texture=str(data["texture"]).strip(),
            lighting=str(data["lighting"]).strip(),
            composition=str(data["composition"]).strip(),
            tone=str(data["tone"]).strip(),
            avoid=tuple(str(a).strip() for a in avoid if str(a).strip()),
        )

    def to_block(self) -> str:
        """The preset's own axes, as the body of the style block.

        Every axis goes in: a block naming only the style ("make it Swiss")
        leaves the treatment to the model, which is the same as having no
        style at all.
        """
        return "\n".join([
            f"Palette: {self.palette}",
            f"Typography: {self.type}",
            f"Imagery: {self.imagery}",
            f"Texture / finish: {self.texture}",
            f"Lighting: {self.lighting}",
            f"Composition: {self.composition}",
            f"Tone: {self.tone}",
            f"Avoid: {', '.join(self.avoid) if self.avoid else '(none stated)'}",
        ])

    def to_spec(self) -> str:
        """The style in one line, without its name: what it looks like."""
        return (f"{self.palette}; {self.texture}; {self.lighting}; "
                f"type: {self.type}")

    def to_directive(self) -> str:
        """One line, for a prompt with a word budget (``create_more()``)."""
        return f"render in the {self.name} style — {self.to_spec()}"


@lru_cache(maxsize=1)
def load_presets(path: str | Path = PRESETS_PATH) -> tuple[StylePreset, ...]:
    """Every preset in file order. Cached: the file never changes at runtime."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("styles") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source} must hold a non-empty 'styles' list")
    presets = tuple(StylePreset.from_dict(row) for row in rows)
    ids = [p.id for p in presets]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate style preset id(s): {', '.join(duplicates)}")
    return presets


def find_preset(key: str) -> StylePreset | None:
    """The preset an id or display name names, or ``None``.

    ``None`` is not an error: the editor can send a style this service has no
    preset for, and that text is still usable as a style description.
    """
    wanted = (key or "").strip().lower()
    if not wanted:
        return None
    for preset in load_presets():
        if preset.id.lower() == wanted or preset.name.lower() == wanted:
            return preset
    return None


def _normalise(text: str) -> str:
    """``"Soft 3D / Claymorphism"`` -> ``" soft 3d claymorphism "``, for word matching."""
    return " " + re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip() + " "


@lru_cache(maxsize=1)
def _search_keys() -> tuple[tuple[str, str], ...]:
    """``[(key, preset_id)]``, longest key first: every name a preset answers to.

    A preset's id ("neo-brutalism") and each segment of its display name
    ("Swiss", "International") are all names a user types. Longest first so
    "neo brutalism" is never resolved as "brutalist"'s shorter key.
    """
    keys: list[tuple[str, str]] = []
    for preset in load_presets():
        parts = [preset.id] + preset.name.split("/")
        for part in parts:
            key = _normalise(part).strip()
            if key:
                keys.append((key, preset.id))
    keys.sort(key=lambda kv: (-len(kv[0].split()), -len(kv[0])))
    return tuple(dict.fromkeys(keys))


def find_in_text(text: str) -> "StylePreset | None":
    """The preset named anywhere inside ``text``, or ``None``.

    :func:`find_preset` matches a whole field, which is what the style picker
    sends. A typed request ("change the style to bauhaus please") carries the
    name inside a sentence, so the name has to be found in it -- as a whole
    word, or "retro" matches "retrospective".
    """
    haystack = _normalise(text)
    if haystack.strip():
        for key, preset_id in _search_keys():
            if f" {key} " in haystack:
                return find_preset(preset_id)
    return None


def selected_style(state) -> str:
    """The one style selected for this design, or ``""``.

    Mirrors ``nodes._selected_style``: ``style`` is last-write-wins in state
    and holds a single value, which is what "only one style per generation"
    rests on -- there is no list to collapse here.
    """
    style = state.get("style") if hasattr(state, "get") else None
    return style.strip() if isinstance(style, str) and style.strip() else ""


def style_block(style: str, brand_kit: dict | None = None) -> str:
    """The labelled block appended to the brief, or ``""`` for no style.

    A preset contributes all nine of its axes; free text is passed through as
    the description it is. With a ``brand_kit``, the block also carries the
    BRAND LOCK rules (:func:`brand_lock`) and ranks the brand above the style,
    so a styled generation for a brand keeps the brand's colours, fonts, logo,
    product photography and approved wording and spends the style on the
    treatment around them.
    """
    style = (style or "").strip()
    if not style:
        return ""

    preset = find_preset(style)
    if preset is not None:
        head = f"STYLE DIRECTION — {preset.name} ({preset.id})"
        body = preset.to_block()
    else:
        head = f"STYLE DIRECTION — {style}"
        body = (f"Render the whole design in the {style} style: its palette, "
                f"light, texture, finish and typographic mood.")

    parts = [head, "", body, "", _RULES]
    lock = brand_lock(brand_kit)
    if lock:
        parts += ["", lock]
    parts += ["", priority_block(bool(lock))]
    return "\n".join(parts)


def apply_style(raw_prompt: str, style: str, brand_kit: dict | None = None) -> str:
    """``raw_prompt`` with the style block appended, flat.

    The brief itself is never rewritten -- the block sits after it, labelled,
    so GROUND can tell deliberate art direction from a fact the user stated.
    Returns ``raw_prompt`` unchanged when no style is selected.
    """
    block = style_block(style, brand_kit)
    if not block:
        return raw_prompt
    return f"{(raw_prompt or '').strip()}\n\n{block}"


def restyle_prompt(style: str, brand_kit: dict | None = None,
                   request: str = "") -> str:
    """The whole edit instruction for switching an existing design to ``style``.

    Used by the edit flow when the user asks for a style the preset file knows
    (``nodes.editStyleSwitch``). It is the generation block plus a restyle
    header, so one switch carries the same nine axes, the same BRAND LOCK and
    the same precedence the design was generated under -- an edit built from
    the style's name alone drifts, because the name is a label, not direction.

    ``request`` is the user's own sentence, quoted at the top and ranked above
    everything below it: a switch is often asked for with a condition attached
    ("...but keep the red headline"), and that condition is the user's, so it
    outranks both the brand and the style.
    """
    block = style_block(style, brand_kit)
    if not block:
        return ""
    preset = find_preset(style)
    label = preset.name if preset is not None else style

    parts = []
    if (request or "").strip():
        parts += [f'THE USER ASKED, VERBATIM: "{request.strip()}"', ""]
    parts += [
        f"RESTYLE THIS DESIGN — {label}",
        "",
        f"Re-render the existing image in the {label} style. This is a change "
        f"of style, not a redesign: keep every element that is already in the "
        f"image, every word of its text exactly as written, and the same "
        f"layout, hierarchy and composition. Change the visual treatment only "
        f"-- palette, type treatment, imagery handling, texture, finish and "
        f"lighting -- so the whole design reads as one {label} piece instead "
        f"of a mix of the old look and the new one.",
        "",
        block,
    ]
    return "\n".join(parts)


def style_directive(style: str) -> str:
    """The one-line form for ``create_more()``'s visual directives.

    ``create_more()`` re-composes from the stored brief, not from the raw
    prompt, so the flat block never reaches it; without this line a variation
    silently drops the style the first graphic was generated in. Its prompt is
    capped at 340-520 words, which is why this is one line and not the block.
    """
    style = (style or "").strip()
    if not style:
        return ""
    preset = find_preset(style)
    body = (preset.to_directive() if preset is not None else
            f"render in the {style} style, applied to palette, light and texture only")
    # The ranking travels with the line: a variation re-composed from stored
    # facts has no style block to read it off, and this is the one directive
    # that could otherwise be followed into recolouring a logo.
    return (f"{body} — the user's own instructions outrank this style, and so "
            f"do the brand's colours, fonts, logo, product images and approved "
            f"wording")
