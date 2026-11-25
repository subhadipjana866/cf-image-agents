"""
v5_4_logo_handler.py
----------------------------------------------------------------------
Logo decision + insertion layer. Runs inside the reasoning pipeline, right
after layout selection, before CoT fusion.

THE PROBLEM THIS FIXES
----------------------
Layout profiles reserve a `logo_placeholder` zone with handling=LEAVE_EMPTY,
and the asset manifest told the image model to "render a flat neutral
rectangle... reserved for programmatic compositing." But no compositing step
was ever built — so every graphic with a logo zone shipped with an ugly blank
rectangle that nothing ever filled. See any earlier `We're Hiring` output.

THE MODEL
---------
`logo_recommended` = the selected layout variant HAS a logo zone. When it does,
the pipeline resolves ONE of three outcomes (see resolve_logo):

  provided       -> a logo is available (user-uploaded or from the brand kit).
                    The image model is handed the logo as a reference part with
                    hard anti-morph instructions (place exactly as-is; do NOT
                    redraw/re-letter/recolor/distort). Optional
                    `logo_strict_preserve` swaps this for a deterministic Pillow
                    paste that cannot morph at all.
  omitted        -> user said "no logo". The logo zone is STRIPPED from the
                    variant so the CoT never reserves a blank rectangle and
                    rebalances the composition to fill that space.
  leave_empty    -> fallback only (should not normally happen): behaves like the
                    old blank-rectangle path. Kept so nothing silently breaks.

Brand-kit logos: if the brand profile JSON carries a logo asset (logo_url /
logo_path under brand_dna or top level), it is used by default and the
"do you want a logo?" ask is skipped.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

LOGO_ZONE_TYPE = "logo_placeholder"

# Hard anti-morph instruction reused by both the CoT routing and the image-gen
# reference part. Logos are the single worst thing for generative models to
# reproduce (text turns to gibberish), so the language is deliberately blunt.
LOGO_PRESERVE_DIRECTIVE = (
    "PLACE THE SUPPLIED BRAND LOGO EXACTLY AS PROVIDED in the reserved logo area. "
    "Reproduce it pixel-faithfully: do NOT redraw, re-letter, re-spell, restyle, "
    "recolor, add effects to, rotate, skew, crop, or change the proportions of the logo. "
    "Every letter and shape must match the supplied logo precisely. Fit it within the "
    "logo zone at its natural aspect ratio with clear padding; do not stretch it to fill."
)

LOGO_NEGATIVE_TERMS = (
    "morphed logo, distorted brand logo, misspelled logo text, gibberish logo lettering, "
    "redrawn logo, restyled logo, recolored logo, stretched logo, warped brand mark, "
    "invented logo, fake logo, altered brand mark"
)


def variant_logo_zone(variant: dict) -> dict | None:
    """Return the variant's logo_placeholder zone, or None."""
    for z in variant.get("zones", []):
        if z.get("zone_type") == LOGO_ZONE_TYPE:
            return z
    # Some profiles only flag it in asset_requirements; synthesize from that.
    req_logo = variant.get("asset_requirements", {}).get("logo", {})
    if req_logo.get("present"):
        zname = req_logo.get("zone_name")
        for z in variant.get("zones", []):
            if z.get("name") == zname:
                return z
    return None


def logo_recommended(variant: dict) -> bool:
    """True when the chosen layout has a place for a logo."""
    if variant_logo_zone(variant) is not None:
        return True
    return bool(variant.get("asset_requirements", {}).get("logo", {}).get("present"))


def brand_kit_logo(brand_profile: str | None) -> str | None:
    """Extract a logo asset path/URL from the brand profile JSON, if present."""
    if not brand_profile:
        return None
    try:
        data = json.loads(brand_profile)
    except (json.JSONDecodeError, TypeError):
        return None
    dna = data.get("brand_dna", {}) if isinstance(data, dict) else {}
    for container in (data, dna):
        if not isinstance(container, dict):
            continue
        for key in ("logo_path", "logo_url", "logo"):
            v = container.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):  # e.g. {"url": "..."} or {"path": "..."}
                for k2 in ("path", "url", "src"):
                    if isinstance(v.get(k2), str) and v[k2].strip():
                        return v[k2].strip()
    return None


def strip_logo_zone(variant: dict) -> dict:
    """
    Return a deep copy of the variant with the logo zone removed and the logo
    asset requirement cleared — so downstream CoT/description never reserve a
    blank rectangle. Used when the user declines a logo.
    """
    v = copy.deepcopy(variant)
    v["zones"] = [z for z in v.get("zones", []) if z.get("zone_type") != LOGO_ZONE_TYPE]
    if "asset_requirements" in v and "logo" in v["asset_requirements"]:
        v["asset_requirements"]["logo"] = {"present": False, "handling": "NOT_REQUIRED"}
    return v


def logo_zone_label(variant: dict) -> str:
    """Human/CoT-facing position phrase for the logo zone (for the merge directive)."""
    z = variant_logo_zone(variant)
    if not z:
        return "the reserved logo area"
    xp = z.get("x_pct", [0, 100])
    yp = z.get("y_pct", [0, 100])
    cx = (xp[0] + xp[1]) / 2
    cy = (yp[0] + yp[1]) / 2
    vert = "top" if cy < 40 else "bottom" if cy > 60 else "middle"
    horiz = "left" if cx < 40 else "right" if cx > 60 else "center"
    return f"the {vert}-{horiz} logo area"


def build_logo_directive(variant: dict) -> str:
    """The instruction handed to the CoT + image gen when a logo IS provided."""
    return (
        f"BRAND LOGO — SUPPLIED, INSERT FAITHFULLY into {logo_zone_label(variant)}. "
        + LOGO_PRESERVE_DIRECTIVE
    )
