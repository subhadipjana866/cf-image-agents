"""Grounding layer for the critique pipeline (deterministic — no LLM calls).

Responsibilities
----------------
* ``brandNormalizer``  — turn the raw brand_dna payload into a flat, prompt-safe profile
  (the raw schema does NOT match the old WITH_BRAND prompt: ``color_system.accent`` /
  ``.background`` / ``.rules`` and ``typography_system.rules`` do not exist; colors arrive
  as nested ``[[...]]`` arrays; the ``tertiary`` / ``tartiary`` typo must be merged).
* ``loadLayout``       — fetch the layout_profile.json for a goal x platform from S3;
  returns ``None`` when the combo is unsupported (→ generic mode, NOT an error).
* ``mergeConstraints`` / ``genericLimits`` — precedence rule: layout wins on limits,
  brand wins on identity, effective limit = the tighter of the two.
* ``resolveGrounding`` — the single entry point: resolves goal/platform (interrupting to
  ask the user for UPLOADS that lack them), normalizes brand, loads layout, and returns
  ``{goal, platform, layout, brand, eff, avail}``.

Import note: place this package as a sub-package alongside ``subgraphs/`` so the parent
modules resolve, e.g. ``graphic_designer/critiquepipeline/``.
"""

import re
import json
# langgraph.interrupt, config.s3_client and utils.parseJson are imported lazily inside the
# functions that need them, so this module loads without those deps (offline tests / reuse).

# The layout templates live here (see PIPELINE_CHANGES_STATUS.md: s3://kha-documents/layout_store/).
# TODO(integration): confirm bucket/prefix against your environment/config.
LAYOUT_STORE_BUCKET = "kha-documents"
LAYOUT_STORE_PREFIX = "layout_store/"

# Layout hard-limits (from shared_rules.negative_rules in the layout profiles).
_LAYOUT_MAX_COLORS = 3
_LOGO_MAX_PCT = 8
_CONTRAST_FLOOR = "7:1 (WCAG AAA per layout negative_rules)"
_SAFE_ZONE = "no critical text in the bottom 10% of the canvas"


# --------------------------------------------------------------------------- #
# brand_dna normalization
# --------------------------------------------------------------------------- #
def _flatten_colors(x):
    """Colors arrive as ``[["#..","#.."]]`` (list of lists), a flat list, or -- in the live
    ``fetch_brand_dna`` shape -- a bare ``"#hex"`` string. Flatten to a flat hex list."""
    if isinstance(x, str):
        return [x] if x.strip() else []
    out = []
    for grp in (x or []):
        if isinstance(grp, list):
            out.extend(g for g in grp if g)
        elif isinstance(grp, dict):
            if grp.get("hex"):
                out.append(grp["hex"])
        elif grp:
            out.append(grp)
    return out


def _hex_rgb(hx):
    """'#1B2A4A' / '1b2a4a' / '#abc' -> (r, g, b) floats in 0..1, or None if unparseable."""
    h = str(hx or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        return None
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# Hue buckets (degrees, upper bound exclusive) -> plain name. Coarse on purpose: the lens only
# needs a word the edit model and the user both understand ("gold", "navy"), never a hex.
_HUE_NAMES = [(12, "red"), (35, "orange"), (53, "gold"), (68, "yellow"), (160, "green"),
              (190, "teal"), (250, "blue"), (285, "purple"), (330, "magenta"), (345, "pink"),
              (361, "red")]


def _colorName(hx):
    """Deterministic plain-language name for a hex colour ("deep navy blue", "warm gold",
    "near-black", "off-white"). nano_prompts must name colours in words — the edit model cannot
    hit an exact hex — so the lenses are handed brand colours already named. None if unparseable."""
    import colorsys
    rgb = _hex_rgb(hx)
    if rgb is None:
        return None
    h, l, s = colorsys.rgb_to_hls(*rgb)
    if s < 0.12 or l < 0.06 or l > 0.96:                     # neutrals
        if l < 0.12:
            return "near-black"
        if l > 0.93:
            return "white" if s < 0.12 and l > 0.98 else "off-white"
        if s >= 0.12 and l > 0.85:
            return "cream"
        return "dark grey" if l < 0.35 else ("light grey" if l > 0.7 else "mid grey")
    deg = h * 360
    name = next(n for bound, n in _HUE_NAMES if deg < bound)
    if name == "blue" and l < 0.3:
        name = "navy blue"
    if name in ("orange", "gold") and l < 0.3:
        name = "brown"
    if name == "red" and l < 0.3:
        name = "maroon"
    if name in ("orange", "gold", "yellow") and l > 0.82:
        return "cream"
    if name == "gold":
        name = "warm gold"
    if l < 0.3 and name not in ("navy blue", "brown", "maroon"):
        prefix = "deep "
    elif l > 0.75:
        prefix = "pale "
    elif s < 0.3:
        prefix = "muted "
    else:
        prefix = ""
    if name == "navy blue" and l < 0.2:
        prefix = "deep "
    return prefix + name


def _namedPalette(groups):
    """[(role, [hex, ...]), ...] -> deduped [{name, hex, role}] (first role wins per hex)."""
    seen, out = set(), []
    for role, hexes in groups:
        for hx in hexes or []:
            key = str(hx).strip().lower()
            name = _colorName(hx)
            if not name or key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "hex": hx, "role": role})
    return out


def _has_signal(v):
    """True when a brand field carries real content (not '', {}, [], or a dict of empties)."""
    if isinstance(v, dict):
        return any(_has_signal(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_has_signal(x) for x in v)
    return bool(v)


def brandNormalizer(raw):
    """Return a flat, prompt-safe brand profile, or ``None`` when no brand is attached.

    Accepts BOTH shapes a brand can arrive in:
      * the live ``main.fetch_brand_dna`` kit that production puts in ``state['brand_dna']`` —
        flat ``tone`` / ``personality_traits`` / ``font_pairing`` / ``color_primary`` /
        ``color_secondary`` / ``color_accent`` / ``color_background``, with ranked palettes
        whose colours are keyed ``color_1..N`` (hex only);
      * the BrandDNA dataclass schema (``brand_personality`` / ``color_system`` /
        ``typography_system`` / ``constraints`` / ``confidence_scores``).
    Reading only the second shape left every production critique with an empty palette, no
    fonts and no voice even when a brand was attached — the lenses were brand-blind.
    """
    if not raw or not isinstance(raw, dict):
        return None

    cs = raw.get("color_system", {}) or {}

    # accent / background are NOT keys in the DNA schema — derive from the recommended palette.
    palettes = (raw.get("color_palettes", {}) or {}).get("ranked_palettes", []) or []
    rec = next((p for p in palettes if p.get("recommended")), palettes[0] if palettes else None)
    accent_hexes, background_hex, palette_roles = [], None, []
    if rec:
        for role, c in (rec.get("colors", {}) or {}).items():
            c = c if isinstance(c, dict) else {"hex": c}
            palette_roles.append({
                "hex": c.get("hex"), "tier": c.get("brand_tier"),
                "weight": c.get("usage_weight"), "role": role,
                "hint": c.get("usage_hint"), "oklch": c.get("oklch"),
            })
            if role == "accent" and c.get("hex"):
                accent_hexes.append(c["hex"])
            if role == "dominant":
                background_hex = c.get("hex")

    primary = _flatten_colors(cs.get("primary")) or _flatten_colors(raw.get("color_primary"))
    secondary = _flatten_colors(cs.get("secondary")) or _flatten_colors(raw.get("color_secondary"))
    neutral = _flatten_colors(cs.get("neutral"))
    tertiary = _flatten_colors(cs.get("tertiary")) + _flatten_colors(cs.get("tartiary"))
    accent = accent_hexes or _flatten_colors(cs.get("accent")) or _flatten_colors(raw.get("color_accent"))
    background = (background_hex or next(iter(_flatten_colors(cs.get("background"))), None)
                  or next(iter(_flatten_colors(raw.get("color_background"))), None))

    named = _namedPalette([
        ("primary", primary), ("secondary", secondary), ("accent", accent),
        ("background", [background] if background else []), ("neutral", neutral),
        ("tertiary", tertiary),
        ("palette", [c["hex"] for c in palette_roles if c.get("hex")]),
    ])

    personality = raw.get("brand_personality", {}) or {}
    if not _has_signal(personality):
        personality = {k: v for k, v in {
            "tone": raw.get("tone"),
            "traits": raw.get("personality_traits"),
            "context": raw.get("brand_context"),
        }.items() if _has_signal(v)}

    ts = raw.get("typography_system", {}) or {}
    typs = raw.get("typographies", []) or []
    cons = raw.get("constraints", {}) or {}
    families = ts.get("font_pairing") or raw.get("font_pairing") or []

    brand = {
        "name": raw.get("brand_name"),
        "personality": personality,
        "palette": {
            "primary": primary,
            "secondary": secondary,
            "neutral": neutral,
            "tertiary": tertiary,
            "accent": accent,                        # SYNTHESIZED when absent from the DNA
            "background": background,                 # SYNTHESIZED when absent from the DNA
            "recommended_palette": palette_roles,     # hex + tier + usage_weight + apca hint
            "named": named,                           # [{name, hex, role}] — what lenses cite
        },
        "fonts": {                                  # typography_system.rules did not exist
            "families": families,
            "headline": ts.get("headline_style"),
            "body": ts.get("body_style"),
            "roles": [{"family": t.get("family"), "role": t.get("type")}
                      for t in typs if isinstance(t, dict)],
        },
        "composition": raw.get("composition_rules", {}) or {},
        "visual_style": raw.get("visual_style", {}) or {},
        "constraints": {
            "max_colors": cons.get("max_colors"),
            "max_fonts": cons.get("max_fonts"),
            "accent_usage": cons.get("accent_usage"),
            "whitespace_usage": cons.get("whitespace_usage"),
        },
        "logo_url": (raw.get("logo") or [None])[0],
        "confidence": raw.get("confidence_scores", {}) or {},  # throttles arbiter assertiveness
    }
    # A kit that only carries asset URLs (logo, templates) has no identity to critique against;
    # report "no brand" so every brand rule is skipped rather than run against empty fields.
    identity = (named, personality, brand["fonts"]["families"], brand["fonts"]["roles"],
                brand["fonts"]["headline"], brand["visual_style"], brand["composition"])
    if not any(_has_signal(x) for x in identity):
        return None
    return brand


# --------------------------------------------------------------------------- #
# layout.json loading
# --------------------------------------------------------------------------- #
def _slug(s):
    """'Instagram Square Post' -> 'instagram_square_post' (matches the layout_store keys)."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def loadLayout(goal, platform):
    """Fetch the layout_profile for ``goal`` x ``platform`` from S3.

    Returns the parsed dict, or ``None`` if the combo is unsupported (key missing) or
    any error occurs — the pipeline then runs in generic mode.
    """
    if not goal or not platform:
        return None
    # The live store keys each combo as a *folder*:
    #   layout_store/{goal}__{platform}/layout_profile.json
    # (alongside frontend_manifest.json and wireframes/). The flat
    # `..._layout_profile.json` form this used to build never existed, so every
    # lookup 404'd into generic mode.
    key = f"{LAYOUT_STORE_PREFIX}{_slug(goal)}__{_slug(platform)}/layout_profile.json"
    try:
        from ..config import s3_client               # reuse the existing boto3 client
        obj = s3_client.get_object(Bucket=LAYOUT_STORE_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        # NoSuchKey / access / parse error → treat as unsupported → generic mode.
        return None


# --------------------------------------------------------------------------- #
# constraint precedence
# --------------------------------------------------------------------------- #
def mergeConstraints(layout, brand):
    """layout wins on limits, brand wins on identity; effective limit = the tighter one."""
    brand_max = (brand or {}).get("constraints", {}).get("max_colors") or 99
    return {
        "effective_max_colors": min(_LAYOUT_MAX_COLORS, brand_max),
        "contrast_floor": _CONTRAST_FLOOR,
        "logo_max_pct": _LOGO_MAX_PCT,
        "safe_zone": _SAFE_ZONE,
        "palette_identity": (brand or {}).get("palette"),
        "font_identity": (brand or {}).get("fonts"),
        "mood_identity": (brand or {}).get("visual_style"),
    }


def genericLimits(brand):
    """No layout.json → no hard platform limits; brand supplies identity only."""
    return {
        "effective_max_colors": (brand or {}).get("constraints", {}).get("max_colors"),
        "contrast_floor": None,
        "logo_max_pct": _LOGO_MAX_PCT if brand else None,
        "safe_zone": None,
        "palette_identity": (brand or {}).get("palette"),
        "font_identity": (brand or {}).get("fonts"),
        "mood_identity": (brand or {}).get("visual_style"),
    }


def resolveProvenance(state) -> str:
    """Is this artwork something OUR generator built against a brief, or a
    user upload?

    Shared by :func:`resolveGrounding` (decides whether goal/platform come
    from the pipeline's own catalog settings or need asking for) and
    :func:`nodes.suggestionManager` (decides whether the ``critic_intent``
    question needs asking at all -- a generated image already has a brief;
    an upload is the one case with nothing to judge against).

    Do NOT use design_goal/platform_canonical for this. They are PROJECT-launch catalog
    settings, deliberately sticky across the whole session checkpoint (see main.py: they are
    omitted from later turns precisely so they survive), and resolve_catalog_selection even
    defaults a missing one to "Festive Sale". So every upload inside a "Product Launch"
    project inherits a design_goal and would be misread as generated — which is exactly how
    an uploaded magazine cover and an uploaded earbuds poster got told to add a CTA "as
    required by the brief".

    Two image-LEVEL signals instead:
     * critiquing straight from attached_images (no current_image_url) means the bytes came
       from the user's file picker, never from our generator — a positive upload signal.
       Mirrors how suggestionCreator itself picks which images to send.
     * base_prompt is the generation spec for the image in hand.
    Absent both -> "uploaded", the SAFE direction: worst case a re-critique of a generated
    design loses its brief checks (suggestionCreator nulls base_prompt after the first pass),
    which is far better than inventing template requirements for artwork that never had one.
    ORIGIN IS STICKY. Decide once, then remember it, because none of the live signals survive
    an edit: every successful edit writes base_prompt = state['prompt'] (nodes.py 4724/4798/
    4876/4950) and a fresh current_image_url. So "upload -> critique -> accept suggestions ->
    edit -> critique again" made the SAME artwork look generated on the second pass, and the
    brief checks switched back on ("add a CTA for the Pinterest product launch" on an uploaded
    poster). base_prompt means "the prompt that last produced this image", NOT "we generated
    it from a brief" — editing an upload does not make it ours.
    v32Generation/v32CreateMore clear this marker, so a genuinely new generation re-decides.

    All of the above is the FALLBACK for frames without a graphic_id. When main.py loaded the
    frame's newimages row into ``active_graphic``, its ``source`` is the answer: it is stored
    per graphic, survives refreshes, and is not fooled by a base_prompt left over from a
    different graphic (an upload selected right after a generation).
    """
    g = state.get("active_graphic") or {}
    if g.get("source") in ("generated", "uploaded"):
        return g["source"]
    remembered = state.get("critique_provenance")
    if remembered in ("generated", "uploaded"):
        return remembered
    from_attachment = (not (state.get("current_image_url") or "").strip()
                       and bool(state.get("attached_images")))
    return "generated" if (state.get("base_prompt") and not from_attachment) else "uploaded"


# --------------------------------------------------------------------------- #
# top-level resolver
# --------------------------------------------------------------------------- #
def resolveGrounding(state):
    """Resolve every grounding input for a critique.

    * Generated images already carry ``design_goal`` / ``platform`` in state.
    * UPLOADS carry neither → use ``critique_goal`` / ``critique_platform``, parsed from the
      user's answer to suggestionManager's ``critic_intent`` question, then
      ``state['clarification']`` (frontend payload). Only if all of those are empty AND no
      ``critique_intent`` was ever captured does this ``interrupt`` to ask. (Runs in the
      node's main thread so the interrupt is valid — do NOT call this inside a
      ThreadPoolExecutor.)
    * ``critique_intent`` itself is returned in ``avail['intent']`` for the lenses, whether or
      not it mapped onto a supported goal/platform pair.

    Returns ``{goal, platform, layout, brand, eff, avail}``.
    """
    from langgraph.types import interrupt
    from ..utils import parseJson
    from ..config import CRITIQUE_UPLOAD_GROUNDING_ASK

    provenance = resolveProvenance(state)
    # What the user told suggestionManager they plan to DO with this image, and
    # the catalog pair parsed out of it. For an UPLOAD this is the only brief
    # that exists, so it outranks the project's sticky catalog settings; for a
    # generated design the pipeline's own goal/platform built the artwork and
    # stay authoritative.
    intent = (state.get("critique_intent") or "").strip()
    intent_goal = (state.get("critique_goal") or "").strip()
    intent_platform = (state.get("critique_platform") or "").strip()

    # The frame's own graphic (newimages row) carries a per-graphic goal/platform snapshot,
    # which outranks both the sticky project catalog and the session-wide critique_* fields.
    # critique_goal / critique_platform fill any gap.
    active = state.get("active_graphic")
    if active:
        goal = active.get("design_goal") or intent_goal
        platform = active.get("platform") or intent_platform
    elif provenance == "uploaded":
        goal = intent_goal or state.get("design_goal")
        platform = intent_platform or state.get("platform_canonical") or state.get("platform")
    else:
        # `platform_canonical` is the catalog name the generation pipeline writes
        # ("Instagram Square Post") and the one the layout store is keyed by;
        # `platform` is the looser free-text field. Reading only the latter left
        # every generated design in generic mode, since generation never sets it.
        goal = state.get("design_goal") or intent_goal
        platform = (state.get("platform_canonical") or state.get("platform")
                    or intent_platform)

    # Never for an active_graphic: suggestionManager's ask-once already asked about it, so
    # anything still missing means generic mode, grounded on the intent text.
    if provenance == "uploaded" and (not goal or not platform) and not active:
        clar = state.get("clarification") or {}
        goal = goal or clar.get("design_goal")
        platform = platform or clar.get("platform")
        # Only ask when NOBODY has asked yet. suggestionManager now opens every
        # general critique with "what are you planning to do with this image?"
        # (`critic_intent`), so once an intent exists the user has already
        # answered this question in their own words -- asking a second, narrower
        # version of it reads as if we did not listen. Their answer simply did
        # not name a supported goal/platform pair, which is what generic mode is
        # for: no layout rulebook, critique grounded on the intent text instead.
        if (not goal or not platform) and not intent:
            answer = interrupt({
                "type": "text_input",
                "question": CRITIQUE_UPLOAD_GROUNDING_ASK,
                "cost": state.get("cost", 0.0),
            })
            answer = parseJson(answer)
            goal = answer.get("design_goal", goal)
            platform = answer.get("platform", platform)

    layout = loadLayout(goal, platform)
    brand = brandNormalizer(state.get("brand_dna"))
    eff = mergeConstraints(layout, brand) if layout else genericLimits(brand)

    avail = {
        "base_prompt": bool(state.get("base_prompt")),
        "layout": layout is not None,
        "brand": brand is not None,
        "goal": goal,
        "platform": platform,
        "mode": "grounded" if layout is not None else "generic",
        # Grounding mode says whether we HAVE a layout spec; provenance says whether the artwork
        # was ever built against one. An upload gets a layout loaded for the goal/platform the
        # user names AFTER the fact, so "grounded + uploaded" is real and must not be read as
        # "this asset owes the template its mandatory fields" — see the lens prompts' PROVENANCE
        # rule. This is what produced "add the missing mandatory CTA" on an uploaded magazine
        # cover that was never a product launch.
        "provenance": provenance,
        # The user's own words about what this design has to do. Every lens
        # renders it, so a critique is judged against the user's brief rather
        # than generic taste.
        "intent": intent,
    }
    return {"goal": goal, "platform": platform, "layout": layout, "provenance": provenance,
            "intent": intent, "brand": brand, "eff": eff, "avail": avail}
