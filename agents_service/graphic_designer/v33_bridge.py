"""sys.path bridge onto the vendored `kognie-generation-v33` package.

The directory name has hyphens, so it is not importable as a dotted path
(`graphic_designer.kognie-generation-v33` is a SyntaxError). Its own modules
also import each other as flat siblings (`import brief_store as base`,
`from brief_store import ...`), not as a package, so the fix is the same one
already used for the avenue2 CoT pipeline (see nodes.py's
`_GEN_PIPELINE_DIR` sys.path insert): put the directory on sys.path and import
it as top-level modules.

Renaming the directory, adding an `__init__.py`, and rewriting v33's own flat
imports to relative ones was considered and rejected: the package's whole
guarantee (INTEGRATION.md: "copied verbatim from the tested source ... nothing
hand-copied") is byte-identity with the tested source, and a fork would have
to be re-applied by hand on every future drop. sys.path is a zero-diff,
reversible way to host it as-is.

Collision check performed before choosing this approach: `graphic_designer/
generation_pipeline/` (the existing avenue1/avenue2 CoT package) has no
`__init__.py` and `graphic_designer/` itself is never placed on sys.path, so a
bare `import generation_pipeline` cannot resolve to it -- only to this
directory once it is inserted. Confirmed no `avenue2_generation_pipeline`
module shares a name with `generation_pipeline.py`, `brief_store.py`,
`reference_roles.py`, `create_more.py`, `copywriter.py`, `antislop.py`,
`typography.py` or `typography_ocr.py`.

v3.3 replaced v3.2 (the 2026-11-06 drop: SLOP GUARD rules, the short labelled
Director prompt, anatomy lines, the TAILOR call and the copywriter on by
default). The 2026-11-17 typography drop is merged on top of it: real fonts,
a TYPESET node before the Director, TEXT QA after the render and
``result["text_layers"]`` (see the package's TYPOGRAPHY.md). The graph node names (`v32Generation`, `v32CreateMore`,
`v32NeedsInputAsk`) and the `v32_*` state keys keep their old names on
purpose: both are persisted in checkpoints, and renaming them would strand any
session paused at or carrying them.

Import this module once, at process startup (from nodes.py, alongside the
existing avenue2 sys.path block) -- never from inside a node body. All of
v33's `ARMC3_*` env config and its 0.93MB `layout_guidance.json` are read at
import time, so importing it lazily per-request would re-read the file (slow)
and would make env overrides silently ignored on any import after the first
(the module is cached, so only the first import's env matters -- keep that
first import at boot, not on some other code path that might run later or
skip entirely).
"""

from __future__ import annotations

import sys
from pathlib import Path

_V33_DIR = str(Path(__file__).resolve().parent / "kognie-generation-v33")
if _V33_DIR not in sys.path:
    sys.path.insert(0, _V33_DIR)

import generation_pipeline as v33  # noqa: E402
import create_more as v33_more  # noqa: E402
import copywriter as v33_copy  # noqa: E402
import antislop as v33_antislop  # noqa: E402
import typography as v33_typography  # noqa: E402
from brief_store import (  # noqa: E402
    PipelineError,
    BriefNotFound,
    MissingCredential,
)

# Boot guard: fail loudly at import if some other package on sys.path shadows
# these flat module names, rather than silently serving the wrong code.
for _mod in (v33, v33_more, v33_copy, v33_antislop, v33_typography):
    _imported_from = Path(_mod.__file__).resolve().parent
    assert _imported_from == Path(_V33_DIR).resolve(), (
        f"{_mod.__name__} resolved from {_imported_from}, expected {_V33_DIR} "
        "-- another module on sys.path is shadowing the v33 package."
    )
# Courier New is not redistributable, so the drop does not ship it and loads it
# from the OS instead -- which the Linux image does not have. Left in the
# library, the TYPOGRAPHER could pick it and every such TYPESET would fail on
# the missing file (falling back to model-drawn text). Unlisted, a user who
# names it gets the "not available yet" notice and the closest supported style.
for _key in list(v33_typography.FONTS):
    try:
        v33_typography.font_path(_key)
    except FileNotFoundError:
        v33_typography.FONTS.pop(_key)

assert len(v33.combos()) == 170, (
    f"v33.combos() returned {len(v33.combos())}, expected 170 -- "
    "layout_guidance.json did not load as expected."
)

# Real catalog pair used when design_goal/platform_canonical cannot be
# resolved to one of v33's 170 combinations (e.g. resolve_catalog_selection's
# own legacy placeholder "Festive Sale", which is not itself a catalog entry
# in EITHER catalog -- confirmed against v5_4_scope_check.load_catalog(), whose
# 17 goals x 10 platforms match v33.combos() exactly). Verified present below.
V33_FALLBACK_COMBO = ("Festive / Seasonal Sale", "Instagram Square Post")
assert V33_FALLBACK_COMBO in v33.combos(), (
    f"{V33_FALLBACK_COMBO} is not in v33.combos() -- pick a real fallback pair."
)


def layout_facts(design_goal: str, platform: str, aspect_ratio: str = "") -> dict:
    """The combination's layout facts, read from v33's bundled
    ``layout_guidance.json`` instead of fetched as a wireframe layout profile.

    Replaces ``layout_store.ensure_layout_assets()`` on the generation path.
    That call resolved a layout profile plus rendered wireframe PNGs -- from
    the local store, else S3, else by generating them with an LLM and Pillow --
    and every consumer of it is now either gone (the CoT fusion that drew the
    wireframe, the per-variant matcher) or served by the guidance file, which
    is already in memory from process start and needs no network at all.

    Shaped to the subset of the old ``layout_profile`` dict the surviving
    gates actually read, so their call sites do not change:

    ``layout_variants``
        From ``composition_menu``. These are the compositions **v33 itself
        chooses between**, so the routing summary now describes the real
        options rather than a parallel wireframe set nothing renders from.
        There is no zone geometry here, by design (INTEGRATION.md §6) -- the
        old per-variant photo/logo zone rectangles have no equivalent and no
        remaining consumer except the logo paste position, which falls back to
        its own default.
    ``shared_rules.required_content_fields``
        From ``must_communicate``, in the shape ``clarify_intent()`` reads
        (it hard-subscripts ``required_level``, ``description`` and
        ``question_if_missing``, so all three are always populated).
    ``shared_rules.real_photo_required``
        From ``real_photo_preferred``.

    Returns ``{}`` for a combination v33 does not serve; every caller already
    tolerates an empty profile.
    """
    try:
        rec = v33.load_guidance(design_goal, platform)
    except BriefNotFound:
        return {}

    def _field(entry: dict, level: str) -> dict:
        return {
            "field_id": entry.get("field_id", ""),
            "required_level": level,
            "description": entry.get("label") or entry.get("field_id", ""),
            "examples": [],
            "can_infer": level != "mandatory",
            "question_if_missing": (entry.get("asks")
                                    or f"What is the {entry.get('label') or 'value'}?"),
        }

    return {
        "design_goal": rec.get("design_goal", design_goal),
        "platform": rec.get("platform", platform),
        "aspect_ratio": aspect_ratio or rec.get("aspect_ratio", ""),
        "canvas_px": rec.get("canvas_px") or [1080, 1080],
        "layout_variants": [
            {"name": c.get("shape", ""), "description": c.get("idea", ""),
             "use_case": c.get("note", "")}
            for c in (rec.get("composition_menu") or [])
        ],
        "shared_rules": {
            "required_content_fields": (
                [_field(f, "mandatory") for f in (rec.get("must_communicate") or [])]
                + [_field(f, "optional") for f in (rec.get("may_include") or [])]
            ),
            "real_photo_required": bool(rec.get("real_photo_preferred")),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# layout_guidance.json AS THE SINGLE CATALOG
#
# Goals, platforms, aspect ratios, content fields and asset asks all come out of
# this one file now. The parallel sources it replaces -- phase0_combinations.json
# for the catalog and layout_requirements.json / the S3 layout store for the
# asset manifest -- were derived twins of it (verified: identical 170 combos),
# so a combination could be servable by one and unknown to another, and an
# aspect ratio could disagree between them.
# ─────────────────────────────────────────────────────────────────────────────

def catalog() -> tuple:
    """``(goals, platforms)`` in the shape ``load_catalog()`` returns them:
    a list of goal names and a list of ``{"name", "aspect_ratio"}`` dicts."""
    goals, platforms = v33.goals(), {}
    for rec in v33.GUIDANCE.values():
        platforms.setdefault(rec["platform"], rec.get("aspect_ratio", ""))
    return goals, [{"name": n, "aspect_ratio": r} for n, r in sorted(platforms.items())]


def resolve_combo(design_goal: str, platform: str):
    """The guidance file's own spelling of this pair, or ``None`` when it does
    not serve the combination. Never substitutes a different combination: a
    design rendered against another combination's rules is a wrong answer that
    looks like a right one."""
    combos = v33.combos()
    pair = (design_goal, platform)
    if pair in combos:
        return pair
    lowered = {(g.lower(), p.lower()): (g, p) for g, p in combos}
    return lowered.get(((design_goal or "").lower(), (platform or "").lower()))


def frontend_manifest(design_goal: str, platform: str) -> dict:
    """The combination's asset/field manifest, derived from the guidance record
    instead of fetched from ``layout_requirements.json`` or the S3 layout store.

    Field shape is what the gates read: ``mandatory_fields`` and
    ``recommended_fields`` (``field_id``/``label``/``question``), and
    ``recommended_assets`` with a ``tier`` per asset.

    Tiers come from the guidance itself: a photo is asked about when the
    combination prefers a real one or any composition in its menu needs one, and
    a logo when any composition leaves space for one. Nothing is ever 'required'
    -- as in the manifest this replaces, where no combination marked either
    asset required, so neither ever interrupts the session on its own.

    :returns: ``{}`` for a combination the guidance does not serve.
    """
    try:
        rec = v33.load_guidance(design_goal, platform)
    except BriefNotFound:
        return {}

    menu = rec.get("composition_menu") or []
    wants_photo = bool(rec.get("real_photo_preferred")) or any(
        c.get("needs_photo") for c in menu)
    wants_logo = any(c.get("has_logo_space") for c in menu)

    def _fields(entries):
        return [{"field_id": f.get("field_id", ""),
                 "label": f.get("label") or f.get("field_id", ""),
                 "input_type": "text",
                 "example": "",
                 "question": (f.get("asks")
                              or f"What is the {f.get('label') or 'value'}?")}
                for f in (entries or [])]

    mandatory = _fields(rec.get("must_communicate"))
    return {
        "design_goal": rec.get("design_goal", design_goal),
        "platform": rec.get("platform", platform),
        "aspect_ratio": rec.get("aspect_ratio", ""),
        "canvas_px": rec.get("canvas_px") or [1080, 1080],
        "mandatory_fields": mandatory,
        "recommended_fields": _fields(rec.get("may_include")),
        "additional_info_placeholder": "e.g. " + ", ".join(
            f["label"].lower() for f in mandatory[:3]) if mandatory else "",
        "recommended_assets": {
            "product_imagery": {
                "tier": "recommended" if wants_photo else "optional",
                "label": "Product / Hero Image",
                "question": "Do you want to include a product or hero image?",
                "reason": ("A real product image adds impact; we'll generate one "
                           "if you skip."),
            },
            "logo": {
                "tier": "recommended" if wants_logo else "optional",
                "label": "Logo",
                "question": "Do you want to add your logo?",
                "reason": "Reinforces your brand. Optional -- you can skip it.",
            },
        },
    }
