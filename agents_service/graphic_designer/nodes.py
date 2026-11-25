from __future__ import annotations
from .states import GraphicDesigner
from langgraph.types import Command, interrupt
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import (
    OPENAI_API_KEY, STABILITY_API_KEY, aspectRatioSet, DOCUMENT_BUCKET, s3_client, GEMINI_API_KEY,
    FLUX_API_KEY, REGION_NAME, nananoBananaProSupportedAspectRatios,
    promptCollectorMessage, bestModelNotification, suggestionHandlerMessage,
    NO_IMAGE_FOR_EDIT, GENERATING_STARTED, GENERATION_COMPLETED, EDIT_CANCELLED,
    SUGGESTION_TARGET_ASK, SUGGESTION_MORE_ASK,
    EDIT_SUCCESSFUL, EXTREME_FALLBACK, SUGGESTION_NEXT_STEP,
    SCOPE_SWITCH_GOAL_ASK, SCOPE_SWITCH_GOAL_NOTE, SCOPE_FREEFORM_GOAL_ASK,
    SCOPE_SWITCH_PLATFORM_ASK, SCOPE_SWITCH_PLATFORM_EXTRA, SCOPE_FREEFORM_PLATFORM_ASK,
    SCOPE_ABORT_ASK, MANIFEST_FIELD_FALLBACK, PHOTO_REQUIRED_DEFAULT,
    LOGO_ASK, LOGO_ASK_REASON,
    V32_NEEDS_INPUT_DEFAULT, V32_NEEDS_INPUT_REASON,
    EDITING_STARTED, EXPANSION_STARTED, EXPAND_PLATFORMS_ASK, EXPAND_SUCCESSFUL,
    CRITICISM_FALLBACK, LAYERING_OFFER, DEVICE, FLORENCE_MODEL, SAM_CHECKPOINT, SAM_CONFIG, USE_BIREFNET, BIREFNET_MODEL, STABILITY_API_KEY, USE_ERASE_ENDPOINT, DEPTH_MODEL, USE_DEPTH_REORDER, INCLUDE_BACKGROUND, EXPAND_MASKS, MAX_PER_CONCEPT, MANUAL_DETECTION, USE_TEXT_REMOVAL, TEXT_REMOVAL_LANG, TEXT_REMOVAL_GPU,
    PLATFORM_DIMENSIONS, EXPAND_PLATFORM_TO_CATALOG, GRAPHIC_METADATA_ASK,
    SLOP_REMOVAL_OFFER, NO_IMAGE_FOR_SLOP_REMOVAL,
    MANUAL_EDIT_SINGLE, MANUAL_EDIT_MULTIPLE, EDIT_WRONG_IMAGE_RESELECT,
)
from .prompts import (
    NAVIGATOR_PHASE1_PROMPT, NAVIGATOR_PHASE2_PROMPT, DECISION_HANDLER_ACCEPT_REJECT_PROMPT,
    DECISION_HANDLER_STATE_UPDATE_PROMPT, EXPAND_PLATFORM_EXTRACTOR_PROMPT, EXPAND_OUTPAINT_PROMPT,
    EDIT_INTENT_CLASSIFIER_PROMPT, SUGGESTION_REQUEST_CLASSIFIER_PROMPT,
    MANIFEST_FIELD_EXTRACTOR_PROMPT, CRITICISM_HANDLER_PROMPT,
    NAVIGATOR_MERGED_SUBTYPE_RULE, NAVIGATOR_SLOP_REMOVAL_RULE,
    LAYERING_OBJECT_ASK, LAYERING_TEXT_AUTO_NOTE, LAYERING_HANDLER_PROMPT,
    CRITIQUE_INTENT_ASK, CRITIQUE_INTENT_PARSER_PROMPT,
    EDIT_BRAND_CONFLICT_PROMPT, EDIT_BRAND_CONFLICT_WARNING,
    EDIT_EXECUTABILITY_PROMPT, EDIT_EXECUTABILITY_WARNING,
    CREATE_MORE_IMAGE_READER_PROMPT,
)
from .utils import parseJson, makeApiRequest, calculateCost, profile, invoke_with_fallback, convert_png_to_webp_base64, convert_urls_to_b64, generate_palette_reference_image, compositeLogo, detectCriticism, apiLog, requestedLayerTargets
from .cancellation import checkCancelled, RunCancelled
from .services import bananaProImageGen, nanoBanana2ImageGen, gptImageGen, upscaleImage, sugBananaEditor, sugGptImageEditor, _data_uri_to_openai_file, IMAGE_API_TIMEOUT_SECONDS
import json
from PIL import Image
import numpy as np
import imghdr
import base64
from io import BytesIO
import math
import re
from datetime import datetime
import time
import contextvars
import traceback
import pystache
from concurrent.futures import ThreadPoolExecutor
import logging

from .subgraphs.validation import subGraph as valGraph
from .subgraphs.suggestion import sgb as sugGraph
# The layering subgraph itself (graphic_designer.subgraphs.layering.lgb) is
# mounted directly as a node on the main graph (see graph.py) rather than
# invoked from inside a node function -- see layeringManager below for why.



import os, sys
import shutil
import tempfile
import requests
import settings
import uuid
from pathlib import Path
# _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if _PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _PROJECT_ROOT)

# ── Generation-pipeline (avenue2) integration imports ──────────────────────
# The pipeline package uses bare sibling imports (e.g. `from pipeline_stages import ...`)
# and expects its own directory on sys.path. avenue1 holds byte-identical duplicates;
# avenue2 is the canonical live-request path.
_GEN_PIPELINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "generation_pipeline", "avenue2_generation_pipeline",
)
if _GEN_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _GEN_PIPELINE_DIR)

from v5_4_scope_check import check_request_scope, load_catalog, platform_ratio
from .config import COT_AGENT_NAMES

# ── Generation pipeline v3.3 (new reasoning + render engine) ───────────────
from . import v33_bridge
from .generation_decision import judge_new_generation, fresh_generation_reset
from .navigator_decision import classify_request
from .slop_decision import judge_slop_offer
from .manual_edit_decision import classify_edit_mode, load_operations
from .edit_reply_decision import judge_wrong_image
from .image_metadata import tagged, read_embedded, carried
from .v33_bridge import v33, v33_more, v33_copy, PipelineError as V33PipelineError, \
    BriefNotFound as V33BriefNotFound, MissingCredential as V33MissingCredential, \
    V33_FALLBACK_COMBO
from . import style_presets


def cotProgress(node_key: str):
    """Disabled: no ``cot_progress`` frame is sent to the frontend.

    This used to fire a ``cot_progress`` interrupt carrying the agent name and
    heading from :data:`config.COT_AGENT_NAMES`. Every one of those cost a full
    interrupt/auto-resume round trip *and* re-ran its node from the top, which
    is what made the COT log look like it restarted after each pause.

    Kept as a no-op rather than deleted so the seven call sites stay put and
    re-enabling is a one-line change: restore the ``interrupt(...)`` below and
    add ``"cot_progress"`` back to the auto-continue tuple in ``main.py``.
    """
    return
    # agent, heading = COT_AGENT_NAMES.get(node_key, (node_key, ""))
    # interrupt({"type": "cot_progress", "agent": agent, "heading": heading, "cost": 0})


# gpt-image-2 only has 3 native output canvases (square / 2:3 portrait / 3:2
# landscape) -- anything meaningfully off those gets squashed onto the nearest
# one (see gptImageGen's size_map). nano-banana-2 supports a much wider set of
# native ratios via imageConfig.aspectRatio (see nananoBananaProSupportedAspectRatios),
# so route those requests there instead of letting gpt-image-2 distort them.
_GPT_IMAGE_2_RATIOS = {"1:1": 1.0, "3:2": 3 / 2, "2:3": 2 / 3}
_GPT_IMAGE_2_RATIO_TOLERANCE = 0.05


def _gpt_image_2_supports_ratio(aspect_ratio: str) -> bool:
    try:
        a, b = (float(x) for x in aspect_ratio.split(":"))
        target = a / b
    except (ValueError, AttributeError, ZeroDivisionError, TypeError):
        return True  # unparsable -- don't change existing behavior
    closest = min(_GPT_IMAGE_2_RATIOS.values(), key=lambda r: abs(r - target))
    return abs(closest - target) / target <= _GPT_IMAGE_2_RATIO_TOLERANCE


def _v32_combo(design_goal: str, platform: str):
    """Resolve a (design_goal, platform) pair against layout_guidance.json, the
    one catalog this service has.

    Returns the guidance file's own spelling of the pair, or ``None`` when it
    serves no such combination. Nothing is substituted: this used to fall back
    to ("Festive / Seasonal Sale", "Instagram Square Post"), which rendered a
    completely different brief's rules under the user's chosen goal and looked
    like a normal result. The caller routes an unresolved pair to the freeform
    renderer instead, which at least designs from the user's own words.

    ``resolve_catalog_selection`` reads the same catalog, so a pair that came
    out of it resolves here; the case that does not is its placeholder
    ``"Festive Sale"``, used when the launch payload carried no design_goal.
    """
    return v33_bridge.resolve_combo(design_goal, platform)


def _v32_raw_prompt(state: GraphicDesigner) -> str:
    """The brief text handed to v33 -- the same two pieces of information the
    retired CoT fusion received as separate ``raw_prompt``/``clarified_intent``
    kwargs of the retired CoT fusion, folded into the one string v33's
    ``run()`` takes.

    ``clarified_intent`` (set by ``intentClarifier``) is the enriched brief:
    the user's original request plus clarification answers and manifest field
    values. It is what the old CoT actually designed from.

    ``prompt`` matters too, and separately, because a *regenerate* leaves
    ``clarified_intent`` deliberately STALE (``regenerateManager`` skips
    ``intentClarifier`` on purpose -- see its docstring) and instead threads
    this turn's one-off instruction through ``prompt``
    (``last_generation_prompt`` + "ADDITIONAL INSTRUCTION..."). Using
    ``clarified_intent`` alone would silently drop every regenerate note.
    """
    clarified = (state.get("clarified_intent") or "").strip()
    raw = (state.get("prompt") or "").strip()
    if clarified and raw and raw in clarified:
        return clarified
    if clarified and raw and clarified != raw:
        # On a fresh generation `prompt` is the same request clarified_intent
        # already restates. Labelled as "additional", GROUND read it as more
        # facts and extracted the offer twice, which came back as a headline
        # and a sub-headline saying the same thing. Say plainly that a fact
        # stated in both places is one fact; a regenerate note is still read
        # as an instruction.
        return (f"{clarified}\n\nTHE USER'S OWN WORDS FOR THIS RUN (they may restate the "
                f"brief above -- a fact stated in both places is ONE fact, not two): {raw}")
    return clarified or raw


def _user_asked_cta(state: GraphicDesigner) -> bool:
    """Whether the user asked for a call to action (button or "Shop now" line).

    Read off the user's own words -- this turn's ``prompt`` (a regenerate's
    carries the saved brief plus the new note) and field answers the editor
    sent -- never off ``clarified_intent``, which is an LLM rewrite.
    """
    if v33.user_asked_cta(state.get("prompt") or ""):
        return True
    return any(v33._is_cta_key(k) and str(v).strip() and v != "skipped"
               for k, v in (state.get("field_answers") or {}).items())


def _copywriter_answers(state: GraphicDesigner, goal: str, platform: str) -> list:
    """The user's answers to this combination's field asks, as Q/A pairs.

    The copywriter reads an answer as material to write from and a suggested
    headline as a hint, which is why it gets the asks' own labels rather than
    the raw field ids: "What is the role being hired for?" tells it what the
    answer is, where ``role_title`` does not.
    """
    try:
        g = v33.load_guidance(goal, platform)
    except (V33BriefNotFound, KeyError):
        g = {}
    labels = {f["field_id"]: f.get("label") or f["field_id"]
              for f in (g.get("must_communicate") or []) + (g.get("may_include") or [])
              if f.get("field_id")}
    return [{"q": labels.get(k, k), "a": str(v).strip()}
            for k, v in (state.get("field_answers") or {}).items()
            if str(v).strip() and v != "skipped"]


def _copywriter_brand(state: GraphicDesigner) -> dict | None:
    """The brand facts the copywriter needs, or ``None`` with no brand.

    Only name and voice: the copywriter writes words, so the kit's colours and
    fonts are not its business. ``language`` sets the British/American spelling
    variant it checks its own copy against.
    """
    dna = state.get("brand_dna") if isinstance(state.get("brand_dna"), dict) else {}
    brand = {"name": (state.get("brand_name") or dna.get("brand_name") or "") or None,
             "voice": dna.get("tone") or None,
             "language": dna.get("language") or None}
    return brand if any(brand.values()) else None


def _style_brand_kit(state: GraphicDesigner) -> dict | None:
    """What the selected style must not restyle, or ``None`` with no brand.

    The style block ranks an attached brand above itself and names the assets
    it may not touch (``style_presets.brand_lock``), so it needs to know which
    of them this design actually has: the colours and fonts come off the kit,
    while the logo and the product photo are state, not kit, and a rule about
    "the supplied logo" must not be stated when none was supplied.
    """
    from .critiquepipeline.grounding import brandNormalizer

    dna = state.get("brand_dna") if isinstance(state.get("brand_dna"), dict) else {}
    brand = brandNormalizer(dna) or {}
    palette = (brand.get("palette") or {}).get("named") or []
    prefs = state.get("asset_prefs") or {}

    logo = bool(state.get("logo_asset")
                or state.get("logo_mode") == "provided"
                or (dna.get("logo") or []))
    product = bool(state.get("attached_images")
                   or (prefs.get("photo") or {}).get("status") == "provided")

    kit = {
        "name": (state.get("brand_name") or brand.get("name") or "") or None,
        "colors": [f"{c['name']} ({c['hex']})" for c in palette if c.get("hex")],
        "fonts": [f for f in ((brand.get("fonts") or {}).get("families") or []) if f],
        "logo": logo,
        # A product photo is only the brand's to protect when a brand is
        # attached at all; a bare photo on a brandless design is just an
        # attachment the style is free to treat.
        "product_image": product and bool(brand or dna),
    }
    return kit if any(kit.values()) else None


def _v32_copywriter(state: GraphicDesigner, goal: str, platform: str):
    """The COPYWRITER hook for ``v33.run()``, or ``False`` when it is off.

    It runs beside SUBJECT inside ``run()`` (neither waits for the other), so
    it costs no wall time, and its deck replaces GROUND's extracted text for
    every stage after the join. A failure inside it degrades the run to
    GROUND's text rather than failing the generation.

    It gets ``state["prompt"]`` -- the user's OWN words -- not
    ``clarified_intent``: the rewrite loses the asides, corrections and quotes
    the copywriter has to tell apart, and reads the user's suggested headline
    as settled copy. The engine itself still gets ``_v32_raw_prompt(state)``.

    Off is ``False``, never ``None``: since v3.3 ``run(copywriter=None)`` builds
    a default copywriter from ``raw_prompt`` -- here the LLM rewrite with the
    logo and style blocks appended, the wrong input for it.
    """
    if not (settings.COPYWRITER_ENABLED and v33.COPYWRITER):
        return False
    return v33_copy.make_hook(
        goal=goal, platform=platform,
        raw_prompt=(state.get("prompt") or "").strip(),
        clarifications=_copywriter_answers(state, goal, platform),
        brand=_copywriter_brand(state),
    )


@profile
def v32Generation(state: GraphicDesigner) -> Command:
    """'generation' node: runs the v3.3 pipeline (GROUND -> SUBJECT ->
    TYPOGRAPHER / ART DIRECTOR / COMPOSITOR -> DIRECTOR -> gpt-image-2.5-flare) in one
    call, replacing the old 6-stage CoT fusion + banana/gpt-image render chain.

    Every upstream gate (scopeGate, intentClarifier, manifestFieldsGate,
    photoRequiredGate, clarificationGate, imageAnalyzerLayoutSelector,
    logoDecisionGate) still runs exactly as before and hands off here through
    :func:`_afterLogoGate` -- this node owns reasoning + render only.

    A goal/platform that has no known catalog combination in EITHER system
    never reaches this node: :func:`resolve_catalog_selection` marks it
    ``is_custom_goal`` and the freeform bypass sends it straight to
    :func:`runGptImageModel` instead (see :func:`_freeformNext`).
    """
    print("[v32Generation] running the v3.3 reasoning + render pipeline")
    cost = state.get("cost", 0.0)
    user_id = state.get("user_id")
    session_id = state.get("session_id")

    design_goal, platform, aspect_ratio, _ = resolve_catalog_selection(state)
    combo = _v32_combo(design_goal, platform)
    if combo is None:
        # No layout guidance for this pair, so v33 has no brief to reason from.
        print(f"[v32Generation] ({design_goal!r}, {platform!r}) is not in "
              f"layout_guidance.json -- routing to runGptImageModel")
        return Command(goto="runGptImageModel", update={**state, "cost": cost})
    goal, platform = combo

    raw_prompt = _v32_raw_prompt(state)
    if not raw_prompt:
        # Nothing to design from at all -- fall through to the freeform
        # renderer rather than calling v33 with an empty brief.
        print("[v32Generation] no prompt text available -- routing to runGptImageModel")
        return Command(goto="runGptImageModel", update={**state, "cost": cost})

    if (state.get("logo_mode") == "provided"
            and state.get("logo_insertion_mode") == "strict_paste"):
        # v33 never sees the logo file in this mode (see
        # _v32_reference_images) -- the real logo is pasted in afterward by
        # logoCompositor, at the exact logo_zone_pct rect logoDecisionGate
        # already resolved. Naming that position (not just "leave some space
        # clean") is what keeps the paste from landing on top of something
        # v33 drew there -- a vague instruction gives it no way to avoid a
        # spot it can't see coming.
        zone_desc = _logo_zone_position_description(state.get("logo_zone_pct"))
        raw_prompt += (
            f"\n\nLOGO AREA — LEAVE CLEAN: the real brand logo is placed into "
            f"the {zone_desc} of the design afterwards, as a separate step. Do "
            f"NOT draw any logo, brand mark, lettering, box, or placeholder "
            f"there; keep that specific area clear, uncluttered background as "
            f"part of the composition."
        )
    elif state.get("logo_mode") == "omitted" and _logo_request_text(state.get("prompt", "")):
        # The brief asks for a logo and logoDecisionGate could not get a file
        # for it. v33 already refuses to print an invented wordmark on depicted
        # objects, but the words "add our logo" sitting in the brief are their
        # own instruction -- left unanswered, the render letters one. The CoT's
        # routing_instruction used to say this; it is gone with the fusion.
        raw_prompt += (
            "\n\nNO LOGO FILE: the request mentions a logo but none was "
            "supplied. Do NOT invent, draw, letter, or approximate any brand "
            "mark, wordmark, emblem, or monogram, and do not reserve a blank "
            "box for one; use that space as part of the composition."
        )

    # ── STYLE, flat: the project's one selected style as a labelled block at
    # the end of the brief, so every reasoning stage reads the same direction
    # (see style_presets). One style per generation: `style` holds a single
    # value, and nothing else here adds a second one. ─────────────────────
    style = style_presets.selected_style(state)
    if style and settings.STYLE_DIRECTION_ENABLED:
        print(f"[v32Generation] style direction: {style!r}")
        brand_kit = _style_brand_kit(state)
        if brand_kit:
            print(f"[v32Generation] brand attached: style ranks below its "
                  f"colours/fonts/logo/product images, and below the brief")
        raw_prompt = style_presets.apply_style(raw_prompt, style, brand_kit=brand_kit)

    # v33 only reads reference images from local files, so they and its
    # output PNG live in a scratch directory for this one call and are
    # deleted when the node returns. No local path outlives the node or is
    # stored in state.
    work_dir = Path(tempfile.mkdtemp(prefix="v32_gen_"))
    try:
        reference_images = _v32_reference_images(state, work_dir / "refs")
        out_dir = work_dir / "out"

        def on_event(ev: dict) -> None:
            # The only cancellation seam inside v33's ~110s blocking call --
            # checkCancelled otherwise only fires from @profile (node boundaries)
            # and apiLog(status='start'), neither of which v33 goes through
            # internally. RunCancelled is a BaseException, so it passes cleanly
            # through v33's bare `except Exception` retry handlers instead of
            # being swallowed as "this model failed, try the next one".
            checkCancelled("v32Generation")

        try:
            result = v33.run(
                raw_prompt=raw_prompt,
                goal=goal,
                platform=platform,
                reference_images=reference_images or None,
                generate_image=True,
                out_dir=out_dir,
                on_event=on_event,
                cta_requested=_user_asked_cta(state),
                copywriter=_v32_copywriter(state, goal, platform),
            )
        except RunCancelled:
            raise
        except (V33PipelineError, ValueError, RuntimeError, KeyError, OSError) as e:
            # Not a fallback-chain retry (v33 already tried its own two-model
            # fallback internally) -- a raised exception here means both v33's
            # reasoning models AND its own image-render fallback failed, or the
            # call was misconfigured. extremeFallback is the same place every
            # exhausted run*Model fallback chain lands.
            print(f"[v32Generation] v33.run() failed ({type(e).__name__}: {e}) -- extremeFallback")
            return Command(goto="extremeFallback", update={**state, "cost": cost})

        # ── Billing: v33 has no knowledge of this app's apiLog funnel, so this
        # node is the only place that can fold its spend into it. ──────────────
        reasoning_cost = 0.0
        for m in (result.get("metas") or []):
            tokens = m.get("tokens") or 0
            row_cost = calculateCost(m.get("model", ""), tokens)
            reasoning_cost += row_cost
            apiLog(provider="google", model=m.get("model", ""),
                   purpose=f"v33:{m.get('node', 'reasoning')}", tokens=tokens,
                   cost=row_cost)

        copy = result.get("copy") or {}
        if copy:
            # What to read in the logs before trusting a deck (drop README §4,
            # step 6): the checks it failed, and the wording it had to change.
            print(f"[v32Generation] copywriter: {len(copy.get('deck', {}).get('slots') or [])} slot(s), "
                  f"attempts={copy.get('attempts')}, problems={copy.get('problems') or 'none'}, "
                  f"adaptations={(copy.get('deck') or {}).get('adaptations') or 'none'}")

        guard = result.get("antislop") or {}
        if guard:
            # The SLOP GUARD's record of every conditional decision: what to read
            # when asking "why did this graphic get this prompt?".
            negatives = guard.get("negatives") or {}
            print(f"[v32Generation] slop guard: in_frame={guard.get('in_frame')}, "
                  f"body_parts={guard.get('body_parts') or []}, drawn={guard.get('drawn')}, "
                  f"exempted={sorted(guard.get('exempted') or {}) or 'none'}, "
                  f"tailored={guard.get('tailored')}, "
                  f"negatives {negatives.get('before_tailor')}->{negatives.get('final')}, "
                  f"text_restored={guard.get('text_restored') or 'none'}")

        _log_v33_typography("v32Generation", result)

        # ── needs_input: GROUND found the brief unusable. Only 4 keys come back
        # (status, question, events, metas) -- branch before reading anything else.
        if result.get("status") == "needs_input":
            return Command(goto="v32NeedsInputAsk", update={
                **state, "cost": cost + reasoning_cost, "design_goal": goal,
                "platform_canonical": platform, "aspect_ratio": aspect_ratio,
                "v32_question": result.get("question", ""),
            })

        render = result.get("render") or {}
        if not render.get("ok") or not result.get("image"):
            # v33 degrades a missing OPENAI_API_KEY to its Gemini fallback
            # renderer rather than raising -- reaching here means BOTH the
            # primary and fallback renderer failed outright.
            print(f"[v32Generation] render failed: {render.get('attempts')}")
            return Command(goto="extremeFallback", update={**state, "cost": cost + reasoning_cost,
                                                           "v33_render_attempts": render.get("attempts")})

        rendered_by = render.get("rendered_by", "gpt-image-2.5-flare")
        image_cost = calculateCost(rendered_by, 1000)
        apiLog(provider="openai" if "gpt" in rendered_by else "google",
               model=rendered_by, purpose="v33Render", tokens=1000, cost=image_cost)

        # ── Upload the local PNG v33 wrote to S3, exactly like every run*Model
        # executor (see e.g. runGptImageModel), so current_image_url stays an
        # S3 URL regardless of which pipeline produced it. ─────────────────────
        #
        # `tagged()` writes the design's own description into the file on the way
        # out (image_metadata): goal, platform, brand, model, the locked strings
        # printed on the graphic and the prompt behind them. `state` is the wrong
        # source for the catalog pair here -- v33 normalises the goal/platform it
        # was handed, and the normalised values are only written back in the
        # Command below, after this upload.
        meta_state = {**state, "design_goal": goal, "platform_canonical": platform,
                      "aspect_ratio": result.get("aspect_ratio") or aspect_ratio}
        with open(result["image"], "rb") as fh:
            image_bytes = fh.read()
        webp_buf = BytesIO()
        Image.open(BytesIO(image_bytes)).convert("RGB").save(webp_buf, format="WEBP")
        webp_buf.seek(0)
        file_name = f"{user_id}/{session_id}/image_{datetime.now().timestamp()}.webp"
        s3_client.upload_fileobj(tagged(webp_buf, meta_state, "generated", rendered_by, result), DOCUMENT_BUCKET, file_name,
                                 ExtraArgs={"ContentType": "image/webp", "ACL": "public-read"})
        file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"

        # First run() for this design becomes v32_prev -- the ONLY thing
        # create_more() can be called with (it carries ground/subject, which
        # create_more()'s own result never does). A fresh generation here always
        # starts a new lineage, whether this is the first click or a regenerate.
        # `image`, `reference_images` and `layout_image` are local paths into
        # work_dir, which is deleted on return -- they are not kept.
        # `text_layers` is: create_more() re-typesets each variation with the
        # fonts, copy and colours it records.
        v32_prev = {k: v for k, v in result.items()
                    if k not in ("events", "image", "reference_images", "layout_image")}
        v32_prev["metas"] = [{k: v for k, v in m.items() if k != "thought_summary"}
                             for m in (result.get("metas") or [])]

        return Command(goto="logoCompositor", update={
            **state,
            "current_image_url": file_url,
            "previous_image_url": state.get("current_image_url", ""),
            "base_prompt": raw_prompt,
            "fused_prompt": result.get("final_prompt", ""),
            "negative_prompt": result.get("negative_prompt", ""),
            "enhanced_prompt": raw_prompt,
            "design_goal": goal,
            "platform_canonical": platform,
            "aspect_ratio": result.get("aspect_ratio", aspect_ratio),
            "cost": cost + reasoning_cost + image_cost,
            "v32_prev": v32_prev,
            # Seeded per INTEGRATION.md §5's documented usage, not left empty:
            # create_more()'s own prompt asks the model to "name what the previous
            # version did" but is never shown the previous design at all -- the
            # ONLY prior-version signal it gets is this lineage block, so an empty
            # list reads as "no lineage exists" rather than "one entry, unknown
            # composition". The placeholder archetype "original" cannot collide
            # with a real menu label, so it doesn't spuriously exclude one.
            "v32_lineage": [{"archetype": "original", "style_signature": "as first generated"}],
            "v32_lineage_urls": [file_url],
            "image_models": _with_image_models(state, file_url, rendered_by),
            "v32_needs_input_attempts": 0,
            "v32_exhausted": False,
            "critique_provenance": None,
            "text_layers": _v33_text_layers(result, file_url),
        })
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _log_v33_typography(node: str, result: dict) -> None:
    """Log what TYPESET and TEXT QA did, from a v33 ``run()`` / ``create_more()``
    result: what to read when a graphic's text looks wrong. Silent when the
    graphic had no text (or typography is off), since neither step ran."""
    typo = result.get("typography") or {}
    layers = result.get("text_layers") or {}
    elements = layers.get("elements") or []
    if not typo and not elements:
        return
    statuses = [((e.get("qa") or {}).get("status") or "planned") for e in elements]
    contrast_fail = [e.get("id") for e in elements
                     if ((e.get("contrast") or {}).get("after_render") or {}).get("pass") is False]
    print(f"[{node}] typography: fonts={layers.get('fonts_used') or typo.get('fonts_used') or []}, "
          f"elements={len(elements)}, qa={statuses or 'none'}, "
          f"contrast_fail={contrast_fail or 'none'}, "
          f"notices={typo.get('user_notices') or layers.get('user_notices') or 'none'}, "
          f"policy={len(typo.get('policy_log') or layers.get('policy_log') or [])} fix(es)")


def _v33_text_layers(result: dict, image_url: str):
    """The ``text_layers`` state entry for the graphic just uploaded to
    ``image_url``: v33's editable-text metadata plus the font notices meant for
    the user. ``None`` when the graphic has no typeset text."""
    layers = result.get("text_layers") or {}
    if not layers.get("elements"):
        return None
    notices = (result.get("typography") or {}).get("user_notices") or layers.get("user_notices") or []
    # `image` is v33's local file name, which is gone; the S3 URL replaces it.
    return {"image_url": image_url,
            "layers": {**{k: v for k, v in layers.items() if k != "image"}, "image": image_url},
            "notices": [str(n) for n in notices if str(n).strip()]}


@profile
def v32NeedsInputAsk(state: GraphicDesigner) -> Command:
    """v33's GROUND stage found the brief unusable and returned ``needs_input``
    instead of designing. Mirrors the retired ``midCotClarificationAsk``: a
    non-skippable text question, the answer folded back into ``prompt`` so the
    same classifier/gate chain a fresh reply gets re-runs v33 with it.

    Capped at 2 rounds (``v32_needs_input_attempts``) so a brief GROUND can
    never usefully complete does not loop forever -- the 3rd attempt proceeds
    to v33 unanswered rather than asking again.
    """
    print("[v32NeedsInputAsk] v33 needs more information to design from")
    attempts = int(state.get("v32_needs_input_attempts") or 0)
    question = state.get("v32_question") or V32_NEEDS_INPUT_DEFAULT

    if attempts >= 2:
        print("[v32NeedsInputAsk] attempt ceiling hit -- proceeding to v33 unanswered")
        return Command(goto="generation", update={
            **state, "cost": 0.0, "v32_needs_input_attempts": attempts + 1})

    r = parseJson(interrupt({
        "type": "text_input",
        "skippable": False,
        "question": question,
        "reason": V32_NEEDS_INPUT_REASON,
        "cost": state.get("cost", 0.0),
    }))
    reply = str(r.get("response") or "").strip()
    return Command(goto="generation", update={
        **state,
        "cost": 0.0,
        "prompt": f"{state.get('prompt', '')}\n\n{reply}".strip() if reply else state.get("prompt", ""),
        "clarified_intent": f"{state.get('clarified_intent', '')}\n\n{reply}".strip() if reply else state.get("clarified_intent", ""),
        "v32_needs_input_attempts": attempts + 1,
        "conversation_history": state.get("conversation_history", []) + [
            {"role": "system", "content": question},
            {"role": "user", "content": reply},
        ],
    })


def _v32_prev_matches_image(state: GraphicDesigner, image_url: str) -> bool:
    """Whether ``state['v32_prev']`` is the brief behind ``image_url``.

    ``v32_lineage_urls`` lists every image the stored lineage produced. An
    image outside it (an edited result, a freeform render, a graphic from
    another design in this project) has its own content, so varying
    ``v32_prev`` would hand back a different design's text. A checkpoint from
    before URL tracking (``None``) is trusted, as it always was.
    """
    if not state.get("v32_prev"):
        return False
    urls = state.get("v32_lineage_urls")
    if urls is None or not image_url:
        return True
    return image_url in urls


def _aspect_value(ratio: str) -> float:
    try:
        a, b = (float(x) for x in str(ratio).split(":"))
        return a / b
    except (ValueError, ZeroDivisionError):
        return 1.0


def _v32_brief_from_prev(prev: dict) -> str:
    """Plain-text brief for a ``prev`` read back from an image -- stored as
    ``base_prompt`` so the critic and later edits treat the result as a
    generated design with a known brief."""
    ground = prev.get("ground") or {}
    lines = [f"{prev.get('goal', '')} for {prev.get('platform', '')}."]
    if prev.get("subject"):
        lines.append(f"Subject: {prev['subject']}")
    for k, v in (prev.get("locked_text") or {}).items():
        lines.append(f"{k.replace('_', ' ')}: {v}")
    lines += [str(f) for f in (ground.get("other_stated_facts") or [])]
    return "\n".join(lines)


def _v32_prev_from_image(state: GraphicDesigner, image_url: str):
    """Build a ``create_more()``-ready ``prev`` from a finished graphic.

    One vision call reads back the graphic's text (verbatim), subject, design
    goal, platform and current composition -- everything ``create_more()``
    needs that a stored ``v33.run()`` result would otherwise carry. The
    project's own goal/platform win when they are valid catalog values and the
    platform's aspect ratio matches the image; otherwise the reader's picks
    are used, and the platform falls back to the closest aspect ratio the goal
    supports.

    :returns: ``(prev, lineage, cost)``. ``lineage`` seeds create_more() with
        the source graphic's composition so the new version differs from it.
    """
    from .utils import invokeMultimodalWithFallback

    data = _image_bytes(image_url)
    with Image.open(BytesIO(data)) as im:
        width, height = im.size
    image_ratio = width / height if height else 1.0

    # goal -> {platform: aspect ratio}, from the one catalog v33 serves.
    by_goal: dict[str, dict[str, str]] = {}
    for rec in v33.GUIDANCE.values():
        by_goal.setdefault(rec["design_goal"], {})[rec["platform"]] = rec.get("aspect_ratio", "1:1")
    all_platforms: dict[str, str] = {}
    for plats in by_goal.values():
        all_platforms.update(plats)
    shapes = {c["shape"] for rec in v33.GUIDANCE.values()
              for c in (rec.get("composition_menu") or []) if c.get("shape")}

    def _match(value, options):
        v = str(value or "").strip().lower()
        return next((o for o in options if o.lower() == v), None)

    def _fits(platform_ratio: str) -> bool:
        return abs(_aspect_value(platform_ratio) - image_ratio) / image_ratio <= 0.05

    # The file may describe itself. Every graphic this app uploads carries its
    # own brief as XMP (see image_metadata), and the image bytes are already
    # in hand here -- so when the selected graphic is one of ours, the vision call
    # below is pure waste: it would re-read, imperfectly and for a fee, the
    # text we wrote into the file ourselves. Only a graphic carrying BOTH a
    # catalog goal and locked text qualifies; anything else (an upload, a
    # social-stripped re-upload, an older file) falls through to the reader.
    embedded = read_embedded(data)
    emb_goal = _match(embedded.get("design_goal"), by_goal)
    emb_locked = embedded.get("locked_text") if isinstance(
        embedded.get("locked_text"), dict) else {}
    if emb_goal and emb_locked:
        emb_platform = _match(embedded.get("platform"), by_goal[emb_goal]) or min(
            by_goal[emb_goal], key=lambda p: abs(_aspect_value(by_goal[emb_goal][p]) - image_ratio))
        prev = {
            "status": "ok", "goal": emb_goal, "platform": emb_platform,
            "aspect_ratio": by_goal[emb_goal][emb_platform],
            "ground": {"locked_text": emb_locked,
                       "other_stated_facts": list(embedded.get("other_facts") or []),
                       "visual_directives": [], "user_exclusions": {}, "named_people": [],
                       "redemption": {"token": "", "how": ""}},
            "locked_text": emb_locked,
            # The SUBJECT stage's output shape: create_more() hands it to
            # antislop.in_frame()/body_parts(), which read it as a dict.
            "subject": {"subject": str(embedded.get("brief") or "").strip()},
            "cta_requested": bool(embedded.get("cta_requested")),
            "reference_images": [],
            "source_image_url": image_url,
        }
        if isinstance(embedded.get("text_layers"), dict):
            # create_more() re-typesets the variation in the same fonts, copy
            # and colours; without it the variation's text is model-drawn.
            prev["text_layers"] = embedded["text_layers"]
        lineage = [{"archetype": str(embedded.get("archetype") or "").strip() or "original",
                    "style_signature": "as selected"}]
        print(f"[v32CreateMore] the selected image carries its own brief: {emb_goal!r} x "
              f"{emb_platform!r}, {len(emb_locked)} locked string(s) -- no reader call")
        return prev, lineage, 0.0

    vision_url = image_url if image_url.startswith(("http://", "https://", "data:")) else \
        "data:image/png;base64," + base64.b64encode(data).decode()
    prompt = CREATE_MORE_IMAGE_READER_PROMPT.format(
        width=width, height=height,
        goals="\n".join(f"   - {g}" for g in sorted(by_goal)),
        platforms="\n".join(f"   - {p} ({r})" for p, r in sorted(all_platforms.items())),
        imagery_terms=", ".join(sorted({s.split("|")[0] for s in shapes})),
        alignment_terms=", ".join(sorted({s.split("|")[1] for s in shapes if "|" in s})),
    )
    response, model_used = invokeMultimodalWithFallback([
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": vision_url}},
    ], temperature=0, role="classify_extract")
    cost = calculateCost(model_used, response.usage_metadata["total_tokens"])
    read = parseJson(response.content.strip())

    # Raw state values, not resolve_catalog_selection(): its placeholders
    # ("Festive Sale" / "Instagram Square Post") would override the reader.
    state_goal = state.get("design_goal")
    state_platform = state.get("platform_canonical") or state.get("platform")
    goal = _match(state_goal, by_goal) or _match(read.get("design_goal"), by_goal) \
        or v33_bridge.V33_FALLBACK_COMBO[0]
    plats = by_goal[goal]
    platform = None
    for candidate in (state_platform, read.get("platform")):
        p = _match(candidate, plats)
        if p and _fits(plats[p]):
            platform = p
            break
    if platform is None:
        platform = min(plats, key=lambda p: abs(_aspect_value(plats[p]) - image_ratio))

    locked = {str(k): str(v).strip() for k, v in (read.get("locked_text") or {}).items()
              if isinstance(v, (str, int, float)) and str(v).strip()}
    redemption = read.get("redemption") if isinstance(read.get("redemption"), dict) else {}
    ground = {
        "locked_text": locked,
        "other_stated_facts": [str(x) for x in (read.get("other_stated_facts") or []) if str(x).strip()],
        "visual_directives": [],
        "user_exclusions": {},
        "named_people": [{"name": str(n)} for n in (read.get("named_people") or []) if str(n).strip()],
        "redemption": {"token": redemption.get("token", ""), "how": redemption.get("how", "")},
    }
    prev = {
        "status": "ok", "goal": goal, "platform": platform,
        "aspect_ratio": plats[platform],
        "ground": ground, "locked_text": locked,
        "subject": {"subject": str(read.get("subject") or "").strip()},
        "reference_images": [],
        "source_image_url": image_url,
    }
    lineage = [{"archetype": str(read.get("archetype") or "").strip() or "original",
                "style_signature": str(read.get("style_signature") or "").strip() or "as selected"}]
    print(f"[v32CreateMore] read the selected image: {goal!r} x {platform!r}, "
          f"{len(locked)} locked string(s), composition {lineage[0]['archetype']!r}")
    return prev, lineage, cost


@profile
def v32CreateMore(state: GraphicDesigner) -> Command:
    """"Create more": v33's ``create_more()`` re-does only the Director stage
    (1 reasoning call vs. the 6 a fresh :func:`v32Generation` run makes),
    reusing the FIRST generation's extracted facts/subject so the content lock
    holds and guaranteeing a different composition against the whole lineage
    seen so far -- not just the immediately previous graphic.

    ``state['v32_prev']`` is always the first ``run()`` result for this
    design, never a later ``create_more()`` result (that one carries no
    ``ground``/``subject`` -- see :func:`v32Generation`'s comment on
    ``v32_prev``).

    The image varied is the selected one (``current_image_url``), else the
    last one generated in this session (``last_generated_image_url``). It is
    the only image this node takes: ``create_more()`` gets no reference
    images at all. When ``v32_prev`` is missing, or the source image is not
    one this lineage produced (see :func:`_v32_prev_matches_image`), the
    source image itself is the brief: :func:`_v32_prev_from_image` reads it
    back with one vision call and that result becomes the new ``v32_prev``.
    "Create more" never asks the user anything.
    """
    print("[v32CreateMore] re-composing from the locked brief")
    cost = state.get("cost", 0.0)
    user_id, session_id = state.get("user_id"), state.get("session_id")
    source_url = state.get("current_image_url") or state.get("last_generated_image_url") or ""

    prev = state.get("v32_prev")
    lineage = list(state.get("v32_lineage") or [])
    lineage_urls = list(state.get("v32_lineage_urls") or [])
    from_image = False
    if not _v32_prev_matches_image(state, source_url):
        if not source_url:
            # generateMoreManager is only reached with an image to vary, so
            # this is a stale/cleared checkpoint with nothing to build from.
            print("[v32CreateMore] no v32_prev and no image to read -- extremeFallback")
            return Command(goto="extremeFallback", update={**state, "cost": cost})
        try:
            prev, lineage, read_cost = _v32_prev_from_image(state, source_url)
        except RunCancelled:
            raise
        except Exception as e:
            print(f"[v32CreateMore] could not read the selected image ({type(e).__name__}: {e}) "
                  f"-- extremeFallback")
            return Command(goto="extremeFallback", update={**state, "cost": cost})
        cost += read_cost
        lineage_urls = [source_url]
        from_image = True
    elif state.get("v32_lineage_urls") is None and source_url:
        # Legacy checkpoint with no URL tracking: the selected image was
        # trusted as this lineage's, so start tracking from it.
        lineage_urls = [source_url]

    # The source image is the ONLY image "create more" takes. No reference
    # image reaches create_more(): not the first run's (older checkpoints
    # still carry their local paths), and not attached_images or asset_prefs
    # (a photo from an earlier design in the project was being pasted into
    # unrelated variations). Copy rather than mutate the checkpointed v32_prev.
    prev = {k: v for k, v in prev.items() if k != "refs"}
    prev["reference_images"] = []
    print(f"[v32CreateMore] varying {source_url!r} with no reference images")

    # The flat style block lives in the raw prompt, which create_more() never
    # reads -- it re-composes from the stored ground facts. Without this the
    # second layout of a styled design comes back unstyled. One line, not the
    # whole block: create_more()'s prompt is capped at 340-520 words.
    style = style_presets.selected_style(state)
    if style and settings.STYLE_DIRECTION_ENABLED:
        directive = style_presets.style_directive(style)
        ground = dict(prev.get("ground") or {})
        directives = [d for d in (ground.get("visual_directives") or []) if str(d).strip()]
        if directive not in directives:
            ground["visual_directives"] = directives + [directive]
            prev = {**prev, "ground": ground}
        print(f"[v32CreateMore] style direction carried over: {style!r}")

    # v33 writes its output PNG to disk; it lives in a scratch directory
    # for this one call and is deleted when the node returns.
    work_dir = Path(tempfile.mkdtemp(prefix="v33_more_"))
    try:
        out_dir = work_dir / "out"

        def on_event(ev: dict) -> None:
            checkCancelled("v32CreateMore")

        try:
            result = v33_more.create_more(
                prev, out_dir=out_dir, generate_image=True,
                on_event=on_event, lineage=lineage,
            )
        except RunCancelled:
            raise
        except (V33PipelineError, ValueError, RuntimeError, KeyError, OSError,
                AttributeError, TypeError) as e:
            # AttributeError/TypeError: a `prev` whose shape create_more() does not
            # expect (e.g. one rebuilt from an image). Fall back instead of ending the run.
            print(f"[v32CreateMore] create_more() failed ({type(e).__name__}: {e}) -- extremeFallback")
            return Command(goto="extremeFallback", update={**state, "cost": cost})

        # create_more()'s return has no per-call `metas` list the way run() does --
        # only one reasoning call happened (the Director redo) and its cost is
        # already computed, in dollars, as `cost_llm`. That figure is v33's own
        # actual-cost estimate (its RATES table, the same numbers mirrored into
        # this app's ACTUAL_RATES_PER_1K) rather than this app's customer-billed
        # GEN_PRICES rate -- reasonable here since a single ~$0.02 call isn't worth
        # reconstructing a token count this app was never given just to re-derive
        # a markup on it. With typography on, `cost_llm` also holds the TYPESET
        # call that re-sets the text for the new composition.
        reasoning_cost = float(result.get("cost_llm") or 0.0)
        apiLog(provider="google", model=v33.REASONING_MODEL, purpose="v33:create_more",
               tokens=None, cost=reasoning_cost, actual_cost=reasoning_cost)
        _log_v33_typography("v32CreateMore", result)

        render = result.get("render") or {}
        if not render.get("ok") or not result.get("image"):
            print(f"[v32CreateMore] render failed: {render.get('attempts')}")
            return Command(goto="extremeFallback", update={**state, "cost": cost + reasoning_cost,
                                                           "v33_render_attempts": render.get("attempts")})

        rendered_by = render.get("rendered_by", "gpt-image-2.5-flare")
        image_cost = calculateCost(rendered_by, 1000)
        apiLog(provider="openai" if "gpt" in rendered_by else "google",
               model=rendered_by, purpose="v33Render", tokens=1000, cost=image_cost)

        # Same embedded description as v32Generation's upload. The catalog pair
        # comes off `prev`: this variation belongs to the design that was varied,
        # whatever the session has selected since.
        meta_state = {**state, "design_goal": prev["goal"],
                      "platform_canonical": prev["platform"],
                      "aspect_ratio": result.get("aspect_ratio") or prev.get("aspect_ratio")
                                      or state.get("aspect_ratio")}
        with open(result["image"], "rb") as fh:
            image_bytes = fh.read()
        webp_buf = BytesIO()
        Image.open(BytesIO(image_bytes)).convert("RGB").save(webp_buf, format="WEBP")
        webp_buf.seek(0)
        file_name = f"{user_id}/{session_id}/image_{datetime.now().timestamp()}.webp"
        s3_client.upload_fileobj(tagged(webp_buf, meta_state, "variation", rendered_by, result), DOCUMENT_BUCKET, file_name,
                                 ExtraArgs={"ContentType": "image/webp", "ACL": "public-read"})
        file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"

        lineage.append({
            "archetype": result.get("chosen_archetype") or "",
            "style_signature": result.get("style_signature") or "",
        })

        update = {
            **state,
            "current_image_url": file_url,
            "previous_image_url": source_url,
            "fused_prompt": result.get("final_prompt", ""),
            "negative_prompt": result.get("negative_prompt", ""),
            "cost": cost + reasoning_cost + image_cost,
            # v32_prev is deliberately NOT touched on the stored-brief path -- it
            # must stay the first run() result for every later create_more() call.
            "v32_lineage": lineage,
            "v32_lineage_urls": lineage_urls + [file_url],
            "image_models": _with_image_models(state, file_url, rendered_by),
            "v32_exhausted": bool(result.get("exhausted")),
            "v32_menu_size": int(result.get("menu_size") or 0),
            "v32_used_before": list(result.get("used_before") or []),
            "critique_provenance": None,
            "text_layers": _v33_text_layers(result, file_url),
        }
        if from_image:
            # The brief read back from the selected image starts a new lineage:
            # the next "create more" on this result varies the same content.
            update.update({
                "v32_prev": prev,
                "design_goal": prev["goal"],
                "platform_canonical": prev["platform"],
                "aspect_ratio": result.get("aspect_ratio") or prev.get("aspect_ratio") or state.get("aspect_ratio"),
                "base_prompt": state.get("base_prompt") or _v32_brief_from_prev(prev),
            })
        return Command(goto="logoCompositor", update=update)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# Ordered fallback chain, now used only by runGptImageModel on the freeform /
# custom-goal bypass (_freeformNext) -- v32Generation owns its own two-model
# fallback internally (gpt-image-2.5-flare -> gemini-3.1-flash-image, see v33.run()'s
# render step) and never calls this.
#
# Trimmed to a single entry: every OTHER run*Model executor (the banana
# variants, dall-e, stable-diffusion) was retired from graph.py along with the
# per-model conditional dispatch that used to route to them. A Command(goto=
# "runNanoBanana2Model") etc. would now hit an unregistered node -- LangGraph
# drops it with "wrote to unknown channel" and the run ends with no interrupt,
# the client waiting forever on a spinner. So a freeform render failure goes
# straight to extremeFallback instead of hunting for a model with nowhere to
# route to.
_GEN_FALLBACK_CHAIN = ["gpt-image"]
_MAX_GEN_ATTEMPTS = 4
_GEN_MODEL_NODE = {
    # Must match the name the node is registered under in graph.py exactly
    # ("runGptImageModel"). A mismatch is not an error: LangGraph drops the
    # Command with "wrote to unknown channel", the run ends with no interrupt,
    # and the client waits forever on a spinner.
    "gpt-image": "runGptImageModel",
}

# The chain above identifies this executor by the bare key "gpt-image", but that
# is NOT a model id: OpenAI rejects it with 400 "The model 'gpt-image' does not
# exist", and GEN_PRICES has no entry for it either (calculateCost would raise on
# the None price). The banana executors never hit this because they hardcode
# their endpoint; gptImageGen is the only one that forwards state['model'] to the
# provider. Resolve the key to a real versioned id at the call site.
def _gptImageApiModel() -> str:
    """The concrete OpenAI image model id to send.

    Resolved from the ``image`` role rather than hardcoded, so moving to a newer
    image model is a config edit. Falls back to the id this was pinned to before
    the registry existed, because an image call must never fail on a routing
    lookup.
    """
    try:
        from .model_router import entryForRole
        return entryForRole("image")["api_id"]
    except Exception:
        return "gpt-image-2.5-flare"


# Enough for any one result frame (multi-platform expand, num_images) plus the
# few turns before it; older entries can never be asked for again.
_IMAGE_MODELS_KEPT = 50


def _with_image_models(state: GraphicDesigner, urls, model: str) -> dict:
    """``state['image_models']`` plus ``{url: model}`` for each url uploaded here.

    ``model`` must be the model that actually rendered the pixels -- after any
    fallback -- never ``state['model']``, which is only the frontend picker.
    main.py stores newimages/generations rows from this map.
    """
    models = dict(state.get("image_models") or {})
    for url in ([urls] if isinstance(urls, str) else urls):
        if url and model:
            models.pop(url, None)
            models[url] = model
    return dict(list(models.items())[-_IMAGE_MODELS_KEPT:])


# Front-of-chain consolidation. The front chain is serial, so every hop is
# latency the user waits through; merging decisions that read the same inputs
# removes a whole round trip each. Off by default until the eval confirms the
# merged call routes identically -- one merged call is also one shared failure,
# so every merge here degrades to the original path rather than to a guess.
CONSOLIDATED_FRONT_CHAIN = os.getenv("KOGNIE_CONSOLIDATED_FRONT_CHAIN", "off").lower() == "on"


def _generation_fallback(state: GraphicDesigner, failed_model: str, err: str) -> Command:
    """Route a failed generation model to the next untried banana model, or to
    extremeFallback once every model in the chain has been attempted.

    Records each attempt in ``state['tried_models']`` so the fallback can never
    loop, and keeps ``state['model']`` in step with the node we route to.
    """
    tried = list(state.get("tried_models") or [])
    if failed_model and failed_model not in tried:
        tried.append(failed_model)

    # Backstop: even if a caller passes a model outside the chain, or the chain
    # is edited into a cycle, a run can never attempt more than this.
    if len(tried) >= _MAX_GEN_ATTEMPTS:
        print(f"[generation] attempt ceiling hit after {tried}; giving up")
        return Command(goto="extremeFallback", update={**state, "tried_models": tried})

    nxt = next((m for m in _GEN_FALLBACK_CHAIN if m not in tried), None)
    if nxt is not None:
        return Command(goto=_GEN_MODEL_NODE[nxt],
                       update={**state, "model": nxt, "tried_models": tried})

    return Command(goto="extremeFallback", update={**state, "tried_models": tried})

# The stage functions the surviving gates still call, imported directly from
# the retired avenue2 pipeline's modules. Everything the 7-stage CoT fusion
# needed (v5_4_cot_multicall, v5_3_layout_profile_generator, layout_store, the
# wireframe matcher and the image analyzer) is gone with it: v33 reasons from
# layout_guidance.json and picks its own composition.
from v5_4_intent_clarifier import clarify_intent
from v5_3_intent_router import route_intent
from v5_4_fabrication_check import at_risk_missing_fields, build_fabrication_clarification_questions
# Only the two logo helpers that still have a live caller survive --
# variant_logo_zone for a stale checkpoint's zone, build_logo_directive for the
# single_call text directive.
from v5_4_logo_handler import variant_logo_zone, build_logo_directive





@profile
def promptCollector(state: GraphicDesigner) -> GraphicDesigner:
    """Collect and validate the user's prompt for image generation or editing.

    This is the entry-point node of the main graph. It validates that the required
    state fields (``model``, ``aspect_ratio``, ``prompt``) are present, appends
    the system greeting and the user's prompt to the conversation history,
    initialises the ``cost`` accumulator to ``0.0``, and sets the
    ``suggestion_type`` to ``'user_driven'``.

    :param state: The current graph state containing user and session information.
    :type state: GraphicDesigner
    :returns: The updated state with conversation history initialised and
        cost/suggestion fields set.
    :rtype: GraphicDesigner
    :raises Exception: If any of ``model``, ``aspect_ratio``, or ``prompt`` are
        missing from the state.
    """

    print("[promptCollector] collecting and validating the user prompt")
    # checks before proceeding further
    status = state["model"] and state["aspect_ratio"] and state['prompt']
    if not status:
        raise Exception("Missing required information in state before collecting prompt.")

    # Updating the state variables with user response

    state['conversation_history'] = state.get('conversation_history', []) + [{
        "role": "system",
        "content": promptCollectorMessage
    }] + [{
        "role": "user",
        "content": state['prompt']
    }]
    state['cost'] = 0.0
    state['suggestion_type'] = 'user_driven'

    
    
    return state


# This function classifies the user input 

def _last_graphic_url(state: GraphicDesigner) -> str:
    """The project's latest generated/edited graphic that main.py looked up
    for this turn (``session_last_graphic``); '' when the frame selected or
    attached an image, or the project has none."""
    return (state.get("session_last_graphic") or {}).get("image_url") or ""


def _selected_or_last_image(state: GraphicDesigner) -> str:
    """The image an image request works on: the selected one, else the
    project's latest graphic (see :func:`navigatorClassify`)."""
    return state.get("current_image_url") or _last_graphic_url(state)


@profile
def navigator(state: GraphicDesigner) -> GraphicDesigner:
    """Classify the user's input and route to the appropriate downstream node.

    Performs a two-phase LLM classification using GPT-4o:

    1. **Phase 1** — Classifies the prompt into one of four high-level
       categories: ``generation``, ``editing``, ``suggestion``, or
       ``non-image-related``.
    2. **Phase 2** *(only for non-image-related)* — Further classifies into
       ``question``, ``chit-chat``, ``decision``, or ``unclear``.

    The function is used as a conditional-edge resolver: it returns a string
    key that the ``StateGraph`` uses to select the next node.

    :param state: The current graph state with the user's prompt and
        conversation history.
    :type state: GraphicDesigner
    :returns: A routing string — one of ``'generation'``, ``'editing'``,
        ``'suggestion'``, ``'question'``, ``'chit-chat'``, ``'decision'``,
        ``'unclear'``.
    :rtype: str
    """
    
    print("[navigator] classifying user input and routing")
    user_response = state["prompt"]
    cost = state.get('cost', 0.0)

    # The frontend's Regenerate button sets this explicitly (see main.py), so
    # a button press never depends on a classifier reading the canned text it
    # sends. Checked before the LLM call: deterministic, free, and it cannot be
    # misread as an edit. Typed phrasings still reach the same route through
    # the 'regenerate' category in the classifier below.
    if state.get("is_regenerate"):
        if _selected_or_last_image(state):
            print("[navigator] explicit regenerate flag -- routing to regenerateManager")
            return "regenerate"
        # Same deterministic guard the generate_more branch uses: regenerating
        # presupposes something to regenerate.
        print("[navigator] regenerate flag with no selected image -- treating as generation")
        return "generation"

    try:
        # Decisions API first: one choice call returns the category and the
        # non-image-related subtype together. None (outage, refusal, bad
        # answer) drops to the LLM classifier below, unchanged.
        verdict = None
        if settings.NAVIGATOR_DECISION_ENABLED:
            verdict, d_cost = classify_request(
                user_response,
                state.get("current_image_url"),
                state.get("last_generated_image_url") or _last_graphic_url(state),
                state.get("conversation_history", [])[-6:],
            )
            cost += d_cost
        if verdict in ("question", "chit-chat", "unclear"):
            state["cost"] = cost
            return verdict
        if verdict:
            decision = verdict
        else:
            prompt1 = NAVIGATOR_PHASE1_PROMPT.format(
                user_response=user_response,
                current_image_url=state.get("current_image_url", "None"),
                last_generated_image_url=(state.get("last_generated_image_url")
                                          or _last_graphic_url(state) or "None"),
                chat_history=json.dumps(state.get("conversation_history", [])[-6:])
            )
            if CONSOLIDATED_FRONT_CHAIN:
                # Appended AFTER .format() above, so its braces must stay single --
                # doubling them the way the formatted template does would hand the
                # model a `{{"decision":...}}` example, which it copies verbatim and
                # parseJson then rejects (routing every merged call to unclear).
                prompt1 += NAVIGATOR_MERGED_SUBTYPE_RULE
            if settings.SLOP_REMOVAL_ENABLED:
                prompt1 += NAVIGATOR_SLOP_REMOVAL_RULE

            llm_response1, model_used = invoke_with_fallback([
                {"role": "user", "content": prompt1}
            ], temperature=0, role="classify_extract")

            cost += calculateCost(model_used, llm_response1.usage_metadata['total_tokens'])

            decision_parsed = parseJson(llm_response1.content.strip().lower())
            decision = decision_parsed.get("decision", "")


        if decision == 'generation':
            return decision
        elif decision == 'generate_more':
            # Deterministic guard: "more of the same" needs a graphic to vary --
            # the selected one, else the last generated one; with neither,
            # degrade to a normal generation.
            has_source = _selected_or_last_image(state) or state.get("last_generated_image_url")
            return decision if has_source else "generation"
        elif decision == 'regenerate':
            # Same guard, same reason. Note this route only re-runs the saved
            # brief; regenerateManager itself falls back to the cold pipeline
            # when the thread has no brief to re-run.
            return decision if _selected_or_last_image(state) else "generation"
        elif decision == 'editing':
            return decision
        elif decision == "layering":
            return decision
        elif decision == 'expand':
            return decision
        elif decision == 'slop_removal':
            # Only offered while the branch is on (navigator_decision filters
            # the choice); the guard covers the LLM fallback path too.
            if not settings.SLOP_REMOVAL_ENABLED:
                return 'editing'
            has_source = _selected_or_last_image(state) or state.get("last_generated_image_url")
            return decision if has_source else "unclear"
        elif decision == 'suggestion':
            state["suggestion_type"] = "user_driven"
            # A user-typed suggestion request is its own fresh chain -- give
            # it a full 2-ask budget rather than inheriting a count left over
            # from an earlier, unrelated critique.
            state["suggestion_more_round"] = 0
            return decision
        elif decision == 'non-image-related':

            # Consolidated path: phase 1 already returned the subtype, so the
            # second round trip is skipped entirely. Falls through to the
            # original two-call path whenever the field is missing or is not one
            # of the three valid values -- a merged call that answers badly must
            # degrade to the behavior that works, never to a wrong route.
            if CONSOLIDATED_FRONT_CHAIN:
                subtype = str(decision_parsed.get("subtype", "")).strip().lower()
                if subtype in ("question", "chit-chat", "unclear"):
                    state["cost"] = cost
                    return subtype
                print("[navigator] merged subtype missing/invalid "
                      f"({subtype!r}); falling back to the phase-2 call")

            # Perform the second phase of navigation for non-image-related queries

            prompt2 = NAVIGATOR_PHASE2_PROMPT.format(
                conversation_history=state['conversation_history'],
                user_response=user_response
            )
            
            llm_response2, model_used2 = invoke_with_fallback([
                {"role": "user", "content": prompt2}
            ], temperature=0, role="classify_extract")

            cost += calculateCost(model_used2, llm_response2.usage_metadata['total_tokens'])

            decision_parsed = parseJson(llm_response2.content.strip().lower())

            decision = decision_parsed.get("decision", "")

            state["cost"] = cost
            

            return decision


        # return state
    except Exception as e:
        # Every failure here — a provider error, a malformed reply, a response
        # shape we didn't expect — looks identical to the user: "unclear".
        # Log it, or a broken classifier silently routes the whole app to
        # unclearHandler (which is exactly what a block-list `content` did).
        print(f"[navigator] classification failed ({type(e).__name__}: {e}); routing to unclear")
        return "unclear"
    

# Where each navigator verdict goes. This is the table that used to live in
# graph.py's add_conditional_edges("navigator", ...); it moved here so the
# announce node below can route with a Command after the frontend has been told.
# questionHandler routes Kognie questions to ragbot, cost questions to the config
# price tables, and everything else to plain generation -- the same node the
# subgraphs use, so a question gets the same answer wherever it is asked.
# helperAgent (the older FAISS tool-calling agent) stays registered but unrouted.
# Routes that work on an existing image. With nothing selected they get the
# project's latest graphic (navigatorClassify). "generation" makes a new image,
# so it is not one of them.
IMAGE_SOURCE_ROUTES = frozenset({"editing", "suggestion", "layering", "expand",
                                 "generate_more", "regenerate", "slop_removal"})

NAVIGATOR_ROUTES = {
    "question": "questionHandler",
    "chit-chat": "chitChatHandler",
    "generation": "generationManager",
    "generate_more": "generateMoreManager",
    "regenerate": "regenerateManager",
    "editing": "editModeClassifier",
    "suggestion": "suggestionManager",
    "layering": "layeringManager",
    "expand": "expandManager",
    "slop_removal": "slopManager",
    "unclear": "unclearHandler",
}


@profile
def navigatorClassify(state: GraphicDesigner) -> dict:
    """Run :func:`navigator` as a real node and persist its verdict as ``request_type``.

    ``navigator`` is a conditional-edge resolver: a resolver cannot write state
    and cannot pause, so the verdict could neither be remembered nor announced.
    This node calls it on a throwaway copy (it assigns ``cost`` /
    ``suggestion_type`` onto the dict it is given -- those writes were never
    persisted from a resolver and are deliberately still not) and keeps only the
    route. The classification (the LLM calls) therefore completes and is
    checkpointed here, *before* the interrupt in :func:`requestTypeAnnounce`; the
    interrupt's resume replays only that tiny node, so the classifier is never
    re-run or re-billed.

    A verdict with no route (e.g. the phase-2 classifier answering "decision",
    which has no node) used to crash the conditional edge; it now degrades to
    ``unclear``.
    """
    route = navigator(dict(state))
    if route not in NAVIGATOR_ROUTES:
        print(f"[navigator] no route for verdict {route!r}; routing to unclear")
        route = "unclear"
    update = {"request_type": route}
    # An image request with nothing selected works on the project's latest
    # generated/edited graphic, as if the user had selected it. It becomes
    # active_graphic too, so its result rows are linked to it as their parent.
    last = state.get("session_last_graphic") or {}
    if (route in IMAGE_SOURCE_ROUTES and not state.get("current_image_url")
            and last.get("image_url")):
        print(f"[navigator] no image selected -- using the project's latest graphic "
              f"{last.get('graphic_id')} for {route}")
        update.update(current_image_url=last["image_url"], active_graphic=last)
    return update


@profile
def requestTypeAnnounce(state: GraphicDesigner) -> Command:
    """Tell the frontend what kind of request this is, then route to its handler.

    Emits a ``request_classified`` frame carrying ``request_type``. It is a
    pure notification, not a question: ``main.py`` auto-continues it (same
    treatment as the ``*_started`` spinner frames), so the user is never asked
    anything. It carries ``"cost": 0`` -- the navigator's own cost rides
    ``state['cost']`` forward to the next real interrupt like every other
    auto-continued frame.

    Kept as its own node, with nothing above the ``interrupt()`` but a dict
    read, because a node re-runs from the top on resume: any work above the
    interrupt would be repeated.
    """
    route = state.get("request_type") or "unclear"
    interrupt({"type": "request_classified", "request_type": route, "cost": 0})
    return Command(goto=NAVIGATOR_ROUTES.get(route, "unclearHandler"))


# @profile
# def decisionHandler(state: GraphicDesigner) -> Command:
#     """Handle a user's accept/reject decision on a pending system suggestion.

#     This node is invoked when the navigator classifies the user's input as a
#     ``decision``.  It uses GPT-5-mini in two steps:

#     1. Determines whether the user **accepted** or **rejected** the pending
#        suggestion (e.g., a model change recommendation).
#     2. Computes the state updates implied by the decision (e.g., switching the
#        model name) based on the ``pending_action`` and ``pending_node`` globals.

#     After processing, the pending action/node globals are cleared and a
#     :class:`Command` is returned to transition to the previously stored
#     ``pending_node``.

#     :param state: The current graph state with conversation history and
#         pending action context.
#     :type state: GraphicDesigner
#     :returns: A ``Command`` directing the graph to the ``pending_node`` with
#         the resolved state updates.
#     :rtype: Command
#     :raises Exception: If LLM invocation or JSON parsing fails.
#     """
#     try:
#         global pending_action, pending_node
#         cost = state.get('cost', 0.0)
#         prompt1 = DECISION_HANDLER_ACCEPT_REJECT_PROMPT.format(
#             conversation_history=json.dumps(state.get("conversation_history", [])[-6:]),
#             user_prompt=state['prompt']
#         )

#         response, model_used = invoke_with_fallback([
#             {
#                 "role": "user",
#                 "content": prompt1
#             }
#         ], temperature=0)

#         cost += calculateCost(model_used, response.usage_metadata['total_tokens'])

#         type_parsed = parseJson(response.content.strip().lower())

#         typed = type_parsed.get("decision", "")

#         prompt2 = DECISION_HANDLER_STATE_UPDATE_PROMPT.format(
#             pending_action=pending_action,
#             pending_node=pending_node,
#             conversation_history=json.dumps(state.get("conversation_history", [])[-6:]),
#             user_response_type=typed,
#             current_state=json.dumps({
#                 "model": state.get("model", ""),
#                 "aspect_ratio": state.get("aspect_ratio", ""),
#                 "style": state.get("style", ""),
#                 "num_images": state.get("num_images", 1),
#                 "prompt": state.get("prompt", ""),
#                 "enhanced_prompt": state.get("enhanced_prompt", "")
#             })
#         )

#         response2, model_used2 = invoke_with_fallback([
#             {
#                 "role": "user",
#                 "content": prompt2
#             }
#         ], temperature=0)
#         cost += calculateCost(model_used2, response2.usage_metadata['total_tokens'])
#         updates = parseJson(response2.content.strip().lower())
#         pending_action = None
#         goto = pending_node
#         pending_node = None

#         updates['cost'] = cost


#         return Command(goto=goto, update=updates)
#     except Exception as e:
#         return e
    

# The image_generation tool gptExpandImage pins; expandManager records it as
# the model of every outpainted graphic.
_EXPAND_IMAGE_MODEL = "gpt-image-2.5-flare"


def gptExpandImage(file_bytes, aspect_ratio):
    """Expand an image to a target aspect ratio using the OpenAI Responses API.

    Sends the image plus a "convert this image into ``W:H``" instruction to the
    Responses API with the ``image_generation`` tool, polls the response until
    it completes, and returns the raw bytes of the expanded image.

    :param file_bytes: Raw bytes of the source image.
    :param aspect_ratio: Target aspect ratio string, e.g. ``'9:16'``.
    :returns: Raw bytes of the expanded image.
    :raises Exception: If the API call fails, times out, or returns no image.
    """
    image_type = imghdr.what(None, h=file_bytes) or "png"
    image_b64 = base64.b64encode(file_bytes).decode("utf-8")

    # prompt = (
    #     f"Convert this image into a {aspect_ratio} aspect ratio. "
    #     "Reproduce the original image content with complete fidelity: every element, photo, shape, color, "
    #     "piece of text, logo, badge, and watermark that exists in the original must be preserved exactly "
    #     "as it appears — do not remove, alter, redraw, or omit anything from the original. "
    #     "Fill the newly added canvas area by naturally continuing and enriching the design: you may add "
    #     "new visual elements that match the existing style — similar shapes, cards, photos, patterns, "
    #     "textures, and background — so the result looks like a genuine, complete composition at the new "
    #     "aspect ratio rather than the original floating in empty space. "
    #     "The only restriction on the new areas: do not invent any new text, letters, words, numbers, "
    #     "captions, labels, logos, or watermarks there. New decorative or pictorial elements are welcome; "
    #     "new writing is not."
    # )

    prompt = (
    f"Convert this image into a {aspect_ratio} aspect ratio while preserving the original design "
    "with complete visual fidelity. "
    
    "Preserve every original element exactly as it appears: every photo, person, object, shape, "
    "color, texture, graphic, piece of text, logo, badge, icon, and watermark. "
    "Do not remove, alter, redraw, replace, distort, reinterpret, or omit any original content. "
    
    "Adapt the ENTIRE original composition to the new aspect ratio. "
    "The original content must be proportionally scaled and intelligently repositioned to fit the "
    "new canvas, so the overall design feels naturally composed for the new dimensions. "
    "Do not simply keep the original image at its original size and place it inside a larger empty canvas. "
    "Scale the existing visual elements, spacing, margins, proportions, and overall composition "
    "appropriately according to the new image dimensions while maintaining their relative relationships "
    "and visual hierarchy. "
    
    "If the new aspect ratio requires additional canvas space, naturally extend and enrich the composition "
    "into those areas using visual elements consistent with the original design. "
    "You may extend backgrounds, patterns, textures, gradients, shapes, decorative graphics, photos, "
    "cards, or other pictorial elements that logically continue the existing composition. "
    
    "The newly generated areas must visually blend seamlessly with the original image, as though the "
    "entire design was originally created at the requested aspect ratio. "
    
    "Do not introduce any new text, letters, words, numbers, captions, labels, logos, icons containing "
    "writing, badges, or watermarks in the newly generated areas. "
    "Existing text and typography from the original must remain unchanged and must be preserved exactly. "
    
    "Do not crop or sacrifice important original content merely to achieve the new aspect ratio. "
    "Prioritize preserving the complete original composition while intelligently scaling, repositioning, "
    "and extending it to fill the new dimensions. "
    
    "The final result should look like a professionally redesigned version of the same image for the "
    f"{aspect_ratio} format, not an original image placed onto a larger canvas."
)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    data = {
        "model": "gpt-5.4",
        "background": True,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/{image_type};base64,{image_b64}"},
                ],
            }
        ],
        "tools": [{"type": "image_generation", "model": _EXPAND_IMAGE_MODEL, "quality": "medium"}],
        "tool_choice": "required",
    }
    apiLog("openai", "gpt-5.4", "gptExpandImage", status="start", detail=f"outpaint to {aspect_ratio}")
    try:
        response = makeApiRequest('POST', "https://api.openai.com/v1/responses", headers=headers, json=data, timeout=80)
    except Exception as e:
        # Surface the API's error body — the HTTP status alone hides the actual reason.
        detail = getattr(getattr(e, 'response', None), 'text', '')
        apiLog("openai", "gpt-5.4", "gptExpandImage", status="failed", detail=str(e))
        raise Exception(f"GPT expand request failed: {e} | {detail[:500]}")
    res = response.json()
    response_id = res['id']

    while res.get('status') in ('queued', 'in_progress'):
        time.sleep(5)
        response = makeApiRequest('GET', f"https://api.openai.com/v1/responses/{response_id}", headers=headers, timeout=60)
        res = response.json()

    if res.get('status') != 'completed':
        apiLog("openai", "gpt-5.4", "gptExpandImage", status="failed",
               detail=f"status={res.get('status')}, error={res.get('error')}")
        raise Exception(f"GPT expand failed: status={res.get('status')}, error={res.get('error')}")

    usage = res.get('usage') or {}
    for item in res.get('output', []):
        if item.get('type') == 'image_generation_call' and item.get('result'):
            # Two real costs make up one call here, reported separately by
            # the API and priced against two different rates: the top-level
            # `usage` is just the text conversation turn (gpt-5.4's own rate).
            # The image_generation tool's OWN usage lives at
            # tool_usage.image_gen -- much larger, and the piece the old
            # text-only logging here used to miss entirely (~85% of the real
            # cost, confirmed against a live test call). No published rate
            # exists for this tool specifically, so it's priced against
            # gpt-image-2.5-flare's rate, the model the tool is pinned to
            # above -- see ACTUAL_RATES_PER_1K's "gpt-expand" entry for the
            # single-call fallback this degrades to if tool_usage is absent.
            apiLog("openai", "gpt-5.4", "gptExpandImage", status="ok",
                   tokens_in=usage.get('input_tokens'), tokens_out=usage.get('output_tokens'))
            image_gen_usage = (res.get('tool_usage') or {}).get('image_gen') or {}
            if image_gen_usage:
                apiLog("openai", _EXPAND_IMAGE_MODEL, "gptExpandImage", status="ok",
                       tokens_in=image_gen_usage.get('input_tokens'),
                       tokens_out=image_gen_usage.get('output_tokens'))
            return base64.b64decode(item['result'])
    apiLog("openai", "gpt-5.4", "gptExpandImage", status="failed", detail="no image in output")
    raise Exception(f"GPT expand returned no image in output: {[i.get('type') for i in res.get('output', [])]}")


# Fix aspect ratio based on the purpose

@profile
def aspectRatioHandler(state: GraphicDesigner) -> GraphicDesigner:
    """Load the current image as a base-64 data URI for the edit executors.

    Per product decision, edit operations never auto-change the image's
    aspect ratio: ``state['aspect_ratio']`` describes the project's launch
    canvas, not a target every subsequent edit should be outpainted to, and
    the edit executors (banana/gpt) are expected to return the same
    dimensions as whatever image they're given. This node therefore no
    longer outpaints toward ``state['aspect_ratio']`` -- it just base64
    encodes the current image unchanged. (Explicit aspect-ratio/platform
    resize requests still go through ``expandManager``, which is unaffected.)

    :param state: The current graph state; must contain ``current_image_url``.
    :type state: GraphicDesigner
    :returns: The updated state with ``b64_image_url`` set to the current
        image as a base-64 data URI.
    :rtype: GraphicDesigner
    """
    print("[aspectRatioHandler] loading the image for editing (no aspect-ratio change)")
    try:
        image_url = state['current_image_url']
        if _is_data_uri(image_url):
            state['b64_image_url'] = image_url
            return state
        image_file = makeApiRequest('GET', image_url, timeout=30)
        file_bytes = image_file.content
        image_type = imghdr.what(None, h=file_bytes) or "png"
        image_b64 = base64.b64encode(file_bytes).decode('utf-8')
        state['b64_image_url'] = f"data:image/{image_type};base64,{image_b64}"
        return state
    except Exception as e:
        # Must return a dict, never the raw exception -- LangGraph raises
        # InvalidUpdateError ("Expected dict, got ...") on a non-dict node
        # return, which crashes the whole graph run (and, via main.py's
        # except-clause, drops the websocket and deletes the thread). Leaving
        # b64_image_url unset lets the downstream executor chain
        # (gptImageExecutor -> bananaProExecutor -> bananaExecutor -> fluxExecutor, each with
        # its own try/except+fallback) degrade the same way it already does
        # for any other missing/failed upstream step.
        return {**state}


@profile
def generationManager(state: GraphicDesigner) -> GraphicDesigner:
    """Pass-through node that marks entry into the image generation workflow.

    This node simply logs the transition and returns the state unchanged.  It
    exists as a named entry point so that the graph can route from
    ``navigator`` → ``generationManager`` → ``bestModel``.

    :param state: The current graph state.
    :type state: GraphicDesigner
    :returns: The unmodified state.
    :rtype: GraphicDesigner
    """

    print("[generationManager] entering the image generation workflow")
    user_response = interrupt({
            "type": "generation_started",
            "question": GENERATING_STARTED,
            "current_image_url": state.get("current_image_url", ""),
            "cost": state.get('cost', 0.0)
    })

    # Track how many generation runs this project has entered — 0 before the
    # first one. Downstream logic and persistence use this to distinguish the
    # first generation from subsequent conversational ones.
    update = {**state, "generation_count": int(state.get("generation_count") or 0) + 1}

    # New design or more of the last one? The project keeps one thread, so the
    # last design's field answers, product photo and chat are all still in
    # state. A different design goal settles it; otherwise the Decisions API
    # judges the two requests. When the answer is "new", those inputs are
    # flushed and manifestFieldsGate/photoRequiredGate ask for them again.
    goal = resolve_catalog_selection(state)[0]
    prev = state.get("generation_subject") or {}
    if not prev.get("prompt"):
        print(f"[generationManager] no previous generation on record -- no new-design check "
              f"(goal={goal!r}, field_answers={sorted((state.get('field_answers') or {}).keys())})")
    else:
        if str(prev.get("design_goal") or "").strip().lower() != goal.strip().lower():
            is_new, why = True, f"design goal changed ({prev.get('design_goal')!r} -> {goal!r})"
        elif settings.GENERATION_DECISION_ENABLED:
            is_new, p_same, d_cost = judge_new_generation(
                prev, state.get("field_answers") or {}, state.get("prompt", ""), goal)
            update["cost"] = state.get("cost", 0.0) + d_cost
            why = f"Decisions API p(same)={p_same}"
        else:
            is_new, why = False, "check disabled"
        print(f"[generationManager] {'NEW design -- flushing previous inputs' if is_new else 'SAME design -- keeping previous inputs'}: {why}")
        if is_new:
            update.update(fresh_generation_reset(state))
    update["generation_subject"] = {"prompt": state.get("prompt", ""), "design_goal": goal}
    return update


@profile
def generateMoreManager(state: GraphicDesigner) -> Command:
    """"Create more": same content as the selected graphic, different layout.

    Entered when the navigator classifies "generate more"/"create more". The
    graphic to vary is the one selected on the canvas (``current_image_url``),
    or the last generated one when nothing is selected. Always routes straight
    to :func:`v32CreateMore` -- every upstream gate (scopeGate through
    logoDecisionGate) is skipped, so this path NEVER asks the user anything:

    * ``v32_prev`` is the brief behind the selected graphic -> ``create_more()``
      from it (1 reasoning call, guaranteed-different composition against the
      whole lineage).
    * otherwise (reconnect that lost the brief, an edited or freeform image, a
      graphic from another design) -> :func:`v32CreateMore` reads the selected
      graphic back with one vision call and varies that.
    """
    print("[generateMoreManager] re-composing from the selected image with a new layout")
    source_url = state.get("current_image_url") or state.get("last_generated_image_url") or ""
    interrupt({
            "type": "generation_started",
            "question": GENERATING_STARTED,
            "current_image_url": source_url,
            "cost": state.get('cost', 0.0)
    })
    cotProgress("generateMoreManager")

    gen_count = int(state.get("generation_count") or 0) + 1
    return Command(goto="v32CreateMore", update={
        **state, "generation_count": gen_count, "current_image_url": source_url})


# Canned button text: what a Regenerate click sends when the user typed
# nothing. Appending any of these to the brief would put "regenerate" into the
# design instructions, so they are treated as "no instruction" instead.
_REGEN_NOOP_PHRASES = {
    "", "regenerate", "regenerate it", "regenerate this", "regenerate image",
    "regenerate the image", "re-generate", "regen", "try again", "again",
    "redo", "redo it", "one more time", "generate again", "generate it again",
}


def _regenerate_note(state: GraphicDesigner) -> str:
    """The user's own instruction for this regenerate, or ``""``.

    A Regenerate click with nothing typed sends canned text; feeding that to
    ``route_intent`` and the CoT as part of the brief would have the fusion
    designing around the word "regenerate". Only a substantive instruction --
    "use this as the background", "make it warmer" -- is carried through.
    """
    note = (state.get("prompt") or "").strip()
    if note.lower().strip(" .!") in _REGEN_NOOP_PHRASES:
        return ""
    return note


@profile
def regenerateManager(state: GraphicDesigner) -> Command:
    """"Regenerate": the same brief and the SAME wireframe, generated again.

    Deliberately not the "generate more" path. That one exists to rotate the
    layout -- it forces an unused variant and has the image model restructure
    the previous OUTPUT image into it, so the design survives and only its
    arrangement moves. A regenerate is the opposite request: keep the
    arrangement, produce the design again, and let newly supplied inputs
    actually change it.

    What is reused: the saved brief (``last_generation_prompt``,
    ``clarified_intent``, ``layout_profile``, ``router_flags``) and the
    already-selected wireframe -- ``force_new_variant`` is deliberately NOT
    set, and ``imageAnalyzerLayoutSelector`` passes the checkpointed variant as
    ``prefer_variant_name`` whenever that flag is false, so the layout is
    pinned for free.

    What re-runs: ``route_intent`` (so the user's instruction decides whether a
    newly attached image is used directly or as a reference -- that is where
    ``detected_photo_usage`` comes from) and the full 7-stage CoT fusion, so a
    new photo and a newly picked style are folded into the brief itself rather
    than bolted onto stale prose. Roughly 9 provider calls against the 10-12 of
    a cold generation.

    What is skipped: every interactive gate -- ``scopeGate``,
    ``intentClarifier``, ``manifestFieldsGate``, ``photoRequiredGate``,
    ``clarificationGate``, and the mid-CoT ``[NEEDS INPUT]`` ask. The user
    answered all of those to get the design they are now regenerating; asking
    again is the thing that makes a Regenerate button feel broken. The hops
    that implement the skip live in :func:`intentRouter`, keyed on
    ``is_regenerate``.

    Falls back to the cold pipeline (``scopeGate``) when the checkpoint has no
    completed generation to regenerate -- a fresh thread, or a session that
    only ever ran freeform.
    """
    print("[regenerateManager] regenerating from the saved brief on the same wireframe")
    interrupt({
        "type": "generation_started",
        "question": GENERATING_STARTED,
        "current_image_url": state.get("current_image_url", ""),
        "cost": state.get("cost", 0.0),
    })
    cotProgress("regenerateManager")

    gen_count = int(state.get("generation_count") or 0) + 1
    note = _regenerate_note(state)

    layout_profile = state.get("layout_profile") or {}
    has_brief = bool(
        state.get("last_generation_prompt")
        and state.get("clarified_intent")
        and layout_profile.get("layout_variants")
        and state.get("router_flags")
    )
    if not has_brief:
        # Nothing to regenerate from. Send it through the cold pipeline rather
        # than producing something off a half-populated checkpoint; the user's
        # own words are already in `prompt`, which is what scopeGate expects.
        print("[regenerateManager] no saved brief on this thread -- running the full pipeline")
        return Command(goto="scopeGate", update={
            **state, "generation_count": gen_count, "is_regenerate": False,
        })

    base_prompt = state["last_generation_prompt"]
    # route_intent and v33 both read `prompt` as the raw request, so an
    # instruction the user typed with the regenerate has to ride there to reach
    # them. `regen_base_prompt` keeps the clean brief so the generation can
    # persist THAT as last_generation_prompt -- otherwise ten regenerates with
    # ten one-off notes would leave a brief made of contradictions.
    prompt = base_prompt
    if note:
        prompt = f"{base_prompt}\n\nADDITIONAL INSTRUCTION FOR THIS REGENERATION: {note}"
        print(f"[regenerateManager] carrying the user's instruction: {note[:120]}")

    # Same photo-reuse rule as the "generate more" path: per-turn input wipes
    # attached_images, so a photo the user gave earlier in the project has to be
    # re-attached or the regeneration silently loses it. Anything attached to
    # THIS message wins -- that is the whole point of attaching it here.
    attached = list(state.get("attached_images") or [])
    photo_mode = state.get("photo_mode") or ""
    new_refs = [r for r in (state.get("edit_reference_images") or [])
                if isinstance(r, str) and r.strip()]
    if new_refs:
        # Fresh references lead, so the layout matcher shape-matches against
        # what the user just supplied rather than the project's old photo.
        attached = new_refs + [a for a in attached if a not in new_refs]
        # 'as_is' is only the starting assumption: intentRouter runs next and
        # imageAnalyzerLayoutSelector downgrades this to 'reference' when the
        # router reads USE_AS_REFERENCE out of the user's instruction.
        photo_mode = "as_is"
    elif not attached:
        photo_pref = (state.get("asset_prefs") or {}).get("photo") or {}
        if photo_pref.get("status") == "provided" and photo_pref.get("asset"):
            attached = [photo_pref["asset"]]
            photo_mode = "as_is"

    return Command(goto="intentRouter", update={
        **state,
        "prompt": prompt,
        "regen_base_prompt": base_prompt,
        "generation_count": gen_count,
        "attached_images": attached,
        "photo_mode": photo_mode,
        # Pinned, not rotated -- the one line that separates this from
        # generateMoreManager.
        "force_new_variant": False,
        "reuse_fused_prompt": False,
        # Stays set through intentRouter so it knows to skip its gate.
        "is_regenerate": True,
    })


@profile
def expandManager(state: GraphicDesigner) -> GraphicDesigner:
    """Conversational multi-platform image expand node.

    Handles user requests such as *"convert this for Instagram Reel and Facebook"*.
    The node performs the following steps:

    1. Fires an ``expansion_started`` interrupt so the frontend can show a spinner
       (auto-continued by ``main.py``).
    2. Uses an LLM to extract which platforms the user mentioned.
    3. If no platforms are detected, fires a ``text_input`` interrupt to ask the user.
    4. Downloads ``current_image_url`` to obtain the native pixel dimensions.
    5. For each target platform, computes ``{left, right, up, down}`` padding using
       the canonical dimensions in ``PLATFORM_DIMENSIONS``.
    6. Calls the OpenAI Responses API (image generation) per platform with the target aspect ratio.
    7. Uploads the result to S3 and collects the URLs.
    8. Returns a ``Command`` that routes to ``interruptBeforeFidelitySuggestion``
       with the pipe-separated result URLs stored in ``current_image_url``.

    :param state: The current graph state; must contain ``current_image_url`` and
        ``prompt``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``interruptBeforeFidelitySuggestion`` on
        success or ``extremeFallback`` on unrecoverable error.
    :rtype: Command
    """

    print("[expandManager] expanding the image for the requested platforms")
    # --- Step 1: Interrupt to acknowledge start (auto-continued by main.py) ---
    interrupt({
        "type": "expansion_started",
        "question": EXPANSION_STARTED,
        "current_image_url": state.get("current_image_url", ""),
        "cost": state.get("cost", 0.0)
    })

    cost = state.get("cost", 0.0)
    user_prompt = state.get("prompt", "")
    current_image_url = state.get("current_image_url", "")

    # Guard: no image selected
    if not current_image_url:
        user_resp = interrupt({
            "type": "text_input",
            "question": NO_IMAGE_FOR_EDIT,
            "cost": cost,
        })
        parsed = parseJson(user_resp)
        return Command(goto="navigator", update={
            **state,
            "prompt": parsed.get("response", user_prompt),
            "current_image_url": parsed.get("current_image_url", ""),
            "cost": cost,
        })

    try:
        # --- Step 2a: Explicit aspect-ratio request (e.g. "make this 2:1") ---
        # Takes priority over platform extraction so a raw ratio never gets
        # force-mapped to the closest platform by the LLM.
        custom_ratio = None
        ratio_match = re.search(r'\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b', user_prompt)
        if ratio_match:
            ra, rb = int(ratio_match.group(1)), int(ratio_match.group(2))
            if ra > 0 and rb > 0:
                g = math.gcd(ra, rb)
                custom_ratio = (ra // g, rb // g)

        target_platforms = []
        if custom_ratio is None:
            # --- Step 2b: LLM extracts target platform names from the prompt ---
            platform_list = "\n".join(f'- "{k}"' for k in PLATFORM_DIMENSIONS.keys())
            llm_prompt = EXPAND_PLATFORM_EXTRACTOR_PROMPT.format(
                user_prompt=user_prompt,
                platform_list=platform_list,
            )
            response, model_used = invoke_with_fallback(
                [{"role": "user", "content": llm_prompt}], temperature=0
            )
            cost += calculateCost(model_used, response.usage_metadata["total_tokens"])

            raw_platforms = parseJson(response.content.strip())

            # Normalise: LLM might return a list or a dict with a key
            if isinstance(raw_platforms, list):
                target_platforms = [str(p).lower().strip() for p in raw_platforms]
            elif isinstance(raw_platforms, dict):
                # e.g. {"platforms": [...]}
                target_platforms = [str(p).lower().strip() for p in next(iter(raw_platforms.values()), [])]
            else:
                target_platforms = []

            # Fallback: derive from the project's canonical platform if the LLM
            # found nothing (e.g. 'Instagram Square Post' -> 'instagram post').
            if not target_platforms:
                canonical = (state.get("platform_canonical") or "").lower().strip()
                for key in (canonical,
                            canonical.replace("square post", "post").strip(),
                            canonical.replace("portrait post", "portrait").strip(),
                            canonical.replace("feed landscape", "post").strip()):
                    if key in PLATFORM_DIMENSIONS:
                        target_platforms = [key]
                        break

        # --- Step 3: Ask the user if we still have no platforms or ratio ---
        if not target_platforms and custom_ratio is None:
            user_resp = interrupt({
                "type": "text_input",
                "question": EXPAND_PLATFORMS_ASK,
                "cost": cost,
            })
            parsed = parseJson(user_resp)
            return Command(goto="navigator", update={
                **state,
                "prompt": parsed.get("response", user_prompt),
                "cost": cost,
            })

        # --- Step 4: Determine native pixel dimensions of the source image ---
        # Primary source: dimensions sent from the frontend canvas
        # (layer.originalWidth / originalHeight resolved by the browser from the actual image file).
        # Fallback: download image and let PIL read them (accurate only if stored at native res).
        orig_width  = state.get("image_native_width", 0)
        orig_height = state.get("image_native_height", 0)

        # Always download the file bytes — the outpaint API needs them regardless.
        img_response = makeApiRequest("GET", current_image_url, timeout=30)
        file_bytes = img_response.content

        if not orig_width or not orig_height:
            # PIL fallback
            image = Image.open(BytesIO(file_bytes))
            orig_width, orig_height = image.size

        # --- Steps 5 & 6: Build target list (platforms and/or custom ratio) ---
        # Each target: (key, target_w, target_h, ratio_str)
        targets = []
        if custom_ratio:
            ra, rb = custom_ratio
            ratio = ra / rb
            if orig_width / orig_height > ratio:
                canvas_w, canvas_h = orig_width, math.ceil(orig_width / ratio)
            else:
                canvas_w, canvas_h = math.ceil(orig_height * ratio), orig_height
            targets.append((f"ratio_{ra}-{rb}", canvas_w, canvas_h, f"{ra}:{rb}"))
        for platform_key in target_platforms:
            dims = PLATFORM_DIMENSIONS.get(platform_key)
            if not dims:
                continue
            g = math.gcd(dims["width"], dims["height"])
            targets.append((platform_key, dims["width"], dims["height"], f'{dims["width"] // g}:{dims["height"] // g}'))

        result_urls = []
        # Per-URL platform, so main.py stores each expanded graphic with its own
        # catalog platform (None for a custom ratio or a non-catalog platform).
        result_graphics = []
        # Only these were rendered by a model. A plain resize is the source
        # graphic's pixels, so main.py gives it the source's model.
        outpainted_urls = []
        for platform_key, target_w, target_h, ratio_str in targets:

            target_ratio = target_w / target_h
            orig_ratio = orig_width / orig_height

            # If original ratio is wider than target ratio, we expand height to match target ratio
            if orig_ratio > target_ratio:
                final_w = orig_width
                final_h = math.ceil(orig_width / target_ratio)
            # If original ratio is taller than target ratio, we expand width to match target ratio
            else:
                final_h = orig_height
                final_w = math.ceil(orig_height * target_ratio)

            # Distribute padding symmetrically, remainder goes to right/bottom
            pad_left   = (final_w - orig_width) // 2
            pad_right  = (final_w - orig_width) - pad_left
            pad_top    = (final_h - orig_height) // 2
            pad_bottom = (final_h - orig_height) - pad_top

            if max(pad_left, pad_right, pad_top, pad_bottom) == 0:
                if orig_width == target_w and orig_height == target_h:
                    result_urls.append(current_image_url)
                    result_graphics.append({"url": current_image_url,
                                            "platform": EXPAND_PLATFORM_TO_CATALOG.get(platform_key)})
                else:
                    img = Image.open(BytesIO(file_bytes))
                    resized_image = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    output_buffer = BytesIO()
                    resized_image.save(output_buffer, format="WEBP", quality=95)
                    output_bytes = output_buffer.getvalue()

                    safe_key = platform_key.replace(" ", "_").replace("/", "-")
                    file_name = (
                        f"{state['user_id']}/{state['session_id']}/"
                        f"expand_{safe_key}_{datetime.now().timestamp():.0f}.webp"
                    )
                    s3_client.upload_fileobj(
                        tagged(BytesIO(output_bytes), state, "expanded"),
                        DOCUMENT_BUCKET,
                        file_name,
                        ExtraArgs={"ContentType": "image/webp", "ACL": "public-read"},
                    )
                    file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
                    result_urls.append(file_url)
                    result_graphics.append({"url": file_url,
                                            "platform": EXPAND_PLATFORM_TO_CATALOG.get(platform_key)})
                continue

            # Call GPT expand with the target aspect ratio (never the platform name)
            try:
                image_data = gptExpandImage(file_bytes, ratio_str)
            except Exception as e:
                # Don't abort — try remaining platforms
                continue

            # --- Step 7: Resize to target and Upload to S3 ---
            # Load and resize to exact target platform dimensions
            outpainted_image = Image.open(BytesIO(image_data))
            resized_image = outpainted_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            output_buffer = BytesIO()
            resized_image.save(output_buffer, format="WEBP", quality=95)
            output_bytes = output_buffer.getvalue()

            safe_key = platform_key.replace(" ", "_").replace("/", "-")
            file_name = (
                f"{state['user_id']}/{state['session_id']}/"
                f"expand_{safe_key}_{datetime.now().timestamp():.0f}.webp"
            )
            s3_client.upload_fileobj(
                tagged(BytesIO(output_bytes), state, "expanded"),
                DOCUMENT_BUCKET,
                file_name,
                ExtraArgs={"ContentType": "image/webp", "ACL": "public-read"},
            )
            file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
            result_urls.append(file_url)
            result_graphics.append({"url": file_url,
                                    "platform": EXPAND_PLATFORM_TO_CATALOG.get(platform_key)})
            outpainted_urls.append(file_url)
            cost += calculateCost("gpt-expand", 1000)

        if not result_urls:
            # All platforms failed outpaint
            return Command(goto="extremeFallback", update={**state, "cost": cost})

        # Pipe-separate multiple URLs (same convention used by image generation nodes)
        pipe_url = "|".join(result_urls)

        return Command(
            goto="successfulExpandHandler",
            update={
                **state,
                "current_image_url": pipe_url,
                "previous_image_url": current_image_url,
                "base_prompt": user_prompt,
                "cost": cost,
                "result_graphics": result_graphics,
                "image_models": _with_image_models(state, outpainted_urls, _EXPAND_IMAGE_MODEL),
            },
        )

    except Exception as e:
        return Command(goto="extremeFallback", update={**state, "cost": cost})


@profile
@profile
def editModeClassifier(state: GraphicDesigner) -> Command:
    """Decide whether this edit is AI work or a job for a manual control.

    The navigator's ``editing`` route lands here first. The request is checked
    against the frontend's manual controls (``manual_operations.json``, see
    :mod:`graphic_designer.manual_edit_decision`):

    * a control whose ``mode`` is ``always`` ("increase the corner radius"),
      or a ``specific`` one given an exact value ("brightness to 46%")
      -> :func:`manualEditHandler`, which hands it to the frontend;
    * several such controls in one request -> the same handler, which points
      the user at the right-side menu;
    * everything else -> :func:`editingManager`, the AI edit flow, unchanged.

    No interrupt here: the Decisions API call is checkpointed before
    :func:`manualEditHandler` pauses, so a resume never repeats it. With no
    selected graphic the classifier is skipped -- :func:`editingManager` asks
    for one (or edits the attachments), and a reply naming a graphic comes
    back through the navigator to this node.
    """
    if not settings.MANUAL_EDIT_ENABLED or not state.get("current_image_url"):
        return Command(goto="editingManager", update={"edit_manual_decision": {}})

    print("[editModeClassifier] checking the edit against the manual controls")
    decision, cost = classify_edit_mode(state.get("prompt", ""))
    update = {"edit_manual_decision": decision, "cost": state.get("cost", 0.0) + cost}
    if decision.get("mode") in ("manual", "manual_multiple"):
        return Command(goto="manualEditHandler", update=update)
    return Command(goto="editingManager", update=update)


def manualEditHandler(state: GraphicDesigner) -> Command:
    """Hand a manual-control edit to the frontend instead of running it.

    Sends a ``manual_edit`` frame. ``manual_op`` is the control's id from
    ``manual_operations.json`` (the frontend does the change itself), or null
    when the request asked for several controls -- then the message just
    points at the right-side menu and ``multiple`` is true. The image is
    never touched.

    No image work happened, so the frame carries ``"cost": 0`` and the
    classifier's spend rides ``state['cost']`` forward to the next real
    interrupt, like the pure-text handlers. The reply goes to the navigator.
    """
    decision = state.get("edit_manual_decision") or {}
    op_id = decision.get("op") if decision.get("mode") == "manual" else None
    if op_id:
        label = next((op["label"] for op in load_operations() if op["id"] == op_id),
                     op_id.replace("_", " "))
        message = MANUAL_EDIT_SINGLE.format(label=label.lower())
    else:
        message = MANUAL_EDIT_MULTIPLE
    print(f"[manualEditHandler] manual_op={op_id} ops={decision.get('ops')}")
    user_response = parseJson(interrupt({
        "type": "manual_edit",
        "question": message,
        "manual_op": op_id,
        "multiple": op_id is None,
        "prompt": state.get("prompt", ""),
        "current_image_url": state.get("current_image_url", ""),
        "cost": 0,
    }))
    reply = user_response.get("response", "")
    return Command(goto="navigator", update={
        "prompt": reply,
        "cost": state.get("cost", 0.0),
        "edit_manual_decision": {},
        "conversation_history": state.get("conversation_history", [])
            + [{"role": "assistant", "content": message},
               {"role": "user", "content": reply}],
        "current_image_url": user_response.get("current_image_url", state.get("current_image_url", "")),
        "attached_images": _reply_attachments(user_response) or state.get("attached_images", []),
        "edit_reference_images": _reply_attachments(user_response),
    })


def editingManager(state: GraphicDesigner) -> GraphicDesigner:
    """Gate-keep entry into the image editing workflow.

    Uses this turn's attached images as the edit canvas and references when no
    project graphic is selected. If neither a selected graphic nor attached
    images exist, asks the user to provide an edit target.

    If an image *is* selected, the graph proceeds to
    ``editIntentClassifier``.

    :param state: The current graph state.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``editAssetGate`` for an available
        canvas, or back to ``navigator`` after the no-image prompt.
    :rtype: Command | GraphicDesigner
    """

    print("[editingManager] entering the image editing workflow")
    if not state.get("current_image_url", "") and state.get("attached_images"):
        # An edit can start from the user's attachments when canvas has no
        # selected graphic. Let an explicit image ordinal choose the canvas
        # (e.g. "replace ... in image 2"); otherwise use first attachment as
        # canvas and pass the rest as references. Preserve order for the model.
        attachments = [src for src in state.get("attached_images") or []
                       if isinstance(src, str) and src.strip()]
        prompt = (state.get("prompt") or "").lower()
        target_index = 0
        for i, word in enumerate(("first", "second", "third", "fourth", "fifth")):
            ordinal = i + 1
            if re.search(rf"\b(?:image|picture|photo)\s*(?:#|number\s*)?{ordinal}\b", prompt) \
                    or re.search(rf"\b{word}\s+(?:image|picture|photo)\b", prompt):
                target_index = i
                break
        if target_index >= len(attachments):
            target_index = 0
        canvas = attachments[target_index]
        references = attachments[:target_index] + attachments[target_index + 1:]
        print(f"[editingManager] using attachment {target_index + 1} as edit canvas; "
              f"{len(references)} reference image(s)")
        return Command(goto="editAssetGate", update={
            **state,
            "current_image_url": canvas,
            "edit_reference_images": references,
            "attachment_edit_direct": True,
        })

    if(state.get("current_image_url","") == ""):
        user_response = interrupt({
            "type": "text_input",
            "question": NO_IMAGE_FOR_EDIT,
            "cost": state.get('cost', 0.0)
        })
        user_response = parseJson(user_response)

        conversation_history = state.get('conversation_history', []) + [{
            "role": "system",
            "content": NO_IMAGE_FOR_EDIT
        }] + [{
            "role": "user",
            "content": user_response['response']
        }]

        return Command(goto="navigator", update={**state, "prompt": user_response['response'], "cost": state.get('cost', 0.0), "current_image_url": user_response.get("current_image_url",""), "attached_images": user_response.get('attached_images', []), "conversation_history": conversation_history})

    else:
        user_response = interrupt({
            "type": "editing_started",
            "question": EDITING_STARTED,
            "cost": state.get('cost', 0.0)
        })

        return Command(goto="editAssetGate", update=state)

def _designGraphicCount(state: GraphicDesigner) -> int:
    """How many graphics this design has produced so far.

    ``v32_lineage`` carries one entry per graphic shown for the current brief
    (the original plus every "create more"), and is reset by a fresh
    :func:`v32Generation` run, so it counts graphics for THIS design rather
    than for the session. ``generation_count`` is the fallback for designs that
    never went through v33 (the freeform renderer keeps no lineage).
    """
    lineage = state.get("v32_lineage") or []
    if lineage:
        return len(lineage)
    return int(state.get("generation_count") or 0)


@profile
def suggestionManager(state: GraphicDesigner) -> Command:
    """Entry point into the suggestion workflow: split general vs specific.

    Classifies the user's request into two completely separate flows:

    * **general** — an overall design review/critique with no single named
      element ("give me suggestions to improve this"). For an UPLOAD (see
      :func:`critiquepipeline.grounding.resolveProvenance`), ASKS THE USER WHAT
      THE IMAGE IS FOR first (``critic_intent``, a real blocking question — see
      :data:`prompts.CRITIQUE_INTENT_ASK`), because a critique is a judgement
      against a brief and an upload has none. The answer is kept verbatim in
      ``critique_intent`` and parsed into ``critique_goal`` / ``critique_platform``,
      which ground every lens and let
      :func:`critiquepipeline.grounding.resolveGrounding` load the real layout
      profile. The question is asked once per image: a re-critique of the same
      upload (after edits, or after ``criticismHandler`` rejected the stack)
      reuses the stored answer. A GENERATED image already has a brief --
      ``design_goal``/``platform_canonical`` from the catalog,
      ``clarified_intent`` from ``intentClarifier`` -- so the question is
      skipped entirely and that brief is reused as ``critique_intent`` instead.
      Either way, proceeds to ``suggestionCreator``
      (:mod:`graphic_designer.critiquepipeline`) → ``suggestionHandler``.
    * **specific** — a narrow, single-element ask ("suggest a color for the
      background"). Routes straight to :func:`common_nodes.specificSuggestionCreator`,
      which answers in plain text with no suggestion stack. Never reached for
      an automatic post-generation critique (``suggestion_type == 'auto'``) or
      a post-criticism rebuild (``regenerate_suggestions``) -- both force
      general instead of classifying, since neither has an actual user-typed
      request to read a single element out of.

    :param state: The current graph state with ``prompt``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``suggestionCreator`` or
        ``specificSuggestionCreator``.
    :rtype: Command
    """
    print("[suggestionManager] entering the suggestion workflow")
    cost = state.get('cost', 0.0)

    # WHICH graphic is being critiqued, when this design has more than one.
    # "Create more" leaves several graphics on the canvas for one brief, and an
    # automatic post-generation critique used to take the latest one without
    # asking -- so a user who clicked create-more and then declined got a
    # critique of a graphic they had not chosen. Ask instead, and let the
    # canvas selection in their reply name the target.
    #
    # Only the AUTOMATIC entry asks. A typed request ("suggest improvements")
    # already carries the canvas selection with it, and criticismHandler's
    # rebuild is about the graphic just critiqued.
    if (state.get('suggestion_type') == 'auto'
            and not state.get('regenerate_suggestions')
            and _designGraphicCount(state) > 1
            and not state.get('suggestion_target_confirmed')):
        print(f"[suggestionManager] {_designGraphicCount(state)} graphics for this design "
              f"-- asking which one to critique")
        reply = parseJson(interrupt({
            "type": "text_input",
            "question": SUGGESTION_TARGET_ASK,
            "accepts": ["image"],
            "cost": cost,
            "current_image_url": state.get('current_image_url', ''),
        }))
        chosen = reply.get('current_image_url') or state.get('current_image_url', '')
        return Command(goto="suggestionManager", update={
            **state,
            "current_image_url": chosen,
            "suggestion_target_confirmed": True,
            "cost": 0.0,
            "conversation_history": state.get('conversation_history', []) + [
                {"role": "system", "content": SUGGESTION_TARGET_ASK},
                {"role": "user", "content": reply.get('response', '') or chosen},
            ],
        })

    # Two callers force the general critique, skipping the specific/general
    # classifier entirely:
    #  * criticismHandler, after the user rejected the previous suggestion
    #    set. Its remedy text ("shift the palette to jewel tones") reads as
    #    'specific' and would answer in plain prose instead of rebuilding
    #    the stack.
    #  * The automatic post-generation critique (interruptBeforeFidelitySuggestion's
    #    "yes" reply, suggestion_type == 'auto'). There is no user-typed request
    #    to classify here at all -- `prompt` still holds whatever text drove the
    #    GENERATION ("Key Benefit: ..., Button Text: TEST DRIVE NOW, ..."), and
    #    feeding a feature-listing brief through the classifier reads as naming
    #    one element ("the button text") and comes back 'specific' -- which
    #    skips the critic pipeline and every lens entirely, so the design never
    #    actually gets reviewed. An automated critique is a whole-design review
    #    by definition; it is never a narrow single-element ask.
    if state.get('regenerate_suggestions') or state.get('suggestion_type') == 'auto':
        print("[suggestionManager] forced general critique "
              f"({'post-criticism rebuild' if state.get('regenerate_suggestions') else 'automatic post-generation review'})")
    else:
        request_text = state.get('prompt', '')
        classify_prompt = SUGGESTION_REQUEST_CLASSIFIER_PROMPT.format(prompt=request_text)
        response, model_used = invoke_with_fallback([
            {"role": "user", "content": classify_prompt}
        ], temperature=0)
        cost += calculateCost(model_used, response.usage_metadata['total_tokens'])
        result = parseJson(response.content)

        # Normalised before comparing: the classifier answers 'Specific' /
        # 'SPECIFIC' / ' specific ' often enough that a bare `== 'specific'`
        # silently dropped those turns into the general critique -- the exact
        # symptom of "I asked about one element and got a whole-design review".
        kind = str(result.get('kind', '')).strip().lower()
        print(f"[suggestionManager] request classified as {kind or 'unknown'}")

        if kind == 'specific':
            return Command(goto='specificSuggestionCreator', update={
                "cost": cost,
                # Set fresh on every entry. specificSuggestionCreator reads
                # `suggestion_request or prompt` (so its own "give me other
                # ideas" loop keeps answering the original question), and the
                # field's reducer is last-write-wins with no expiry -- without
                # this, the SECOND specific request in a thread re-answered the
                # first one forever.
                "suggestion_request": request_text,
            })

    # Which artwork this critique is about, in the same terms suggestionCreator
    # picks its images: the canvas image, or the files the user attached when
    # there is no canvas image. An upload critiqued straight from
    # `attached_images` has an empty current_image_url, so keying on that alone
    # would make every upload look like "the same image" as the last one and
    # inherit its brief.
    image_now = (state.get("current_image_url", "")
                 or "|".join(state.get("attached_images") or []))

    # The frame named its graphic (main.py loaded the newimages row into
    # active_graphic), so its stored metadata is the brief, and whether we
    # already asked about it is stored too -- across refreshes, reconnects and
    # restarts, unlike the session-wide critique_* fields below. Ask at most
    # once per graphic, ever; main.py writes the answer back after the turn.
    # Frames without a graphic_id fall through to the legacy path below.
    g = state.get("active_graphic")
    if g:
        if g.get("metadata_status") == "complete" or g.get("metadata_asked"):
            print(f"[suggestionManager] graphic {g.get('graphic_id')} metadata known -- not asking")
            return Command(goto="suggestionCreator", update={
                "cost": cost,
                "regenerate_suggestions": False,
                "critique_intent": (g.get("intent") or g.get("brief")
                                    or state.get("clarified_intent") or ""),
                "critique_intent_image": image_now,
                "critique_goal": g.get("design_goal") or "",
                "critique_platform": g.get("platform") or "",
            })

        answer = interrupt({
            "type": "text_input",
            "kind": "graphic_metadata",
            "graphic_id": g["graphic_id"],
            "question": GRAPHIC_METADATA_ASK,
            # Free text only: see the goal/platform note below.
            "accepts": ["text"],
            "skippable": True,
            "current_image_url": state.get("current_image_url", ""),
            "cost": cost,
        })
        # Billed at the interrupt above, same as the critic_intent path.
        cost = 0.0

        answer = parseJson(answer)
        if not isinstance(answer, dict):
            answer = {"response": str(answer)}
        reply = "" if answer.get("skipped") else (answer.get("response") or "").strip()
        # Any design_goal/platform riding on this frame are the PROJECT's, not
        # the user's answer about this graphic (frontend contract: ignored, and
        # the editor sends them as null). Taking them would skip the parse and
        # judge the graphic against the project brief, which is what the
        # question exists to avoid. The answer is the text they typed.
        goal = platform = ""
        intent = reply

        if reply and not (goal and platform):
            size = (g["width"], g["height"]) if g.get("width") and g.get("height") else None
            parsed, parse_cost = _parseCritiqueIntent(reply, size)
            cost += parse_cost
            intent = parsed.get("intent") or reply
            goal = goal or parsed.get("design_goal", "")
            platform = platform or parsed.get("platform", "")
        goal, platform = _catalogOrEmpty(goal, platform)

        # The row's own values win (an expand child keeps its own platform).
        goal = g.get("design_goal") or goal
        platform = g.get("platform") or platform
        print(f"[suggestionManager] graphic {g['graphic_id']} metadata captured "
              f"(goal={goal or '-'}, platform={platform or '-'}, skipped={bool(answer.get('skipped'))})")

        return Command(goto="suggestionCreator", update={
            "cost": cost,
            "regenerate_suggestions": False,
            "critique_intent": intent,
            "critique_intent_image": image_now,
            "critique_goal": goal,
            "critique_platform": platform,
            "graphic_metadata_answer": {"graphic_id": g["graphic_id"], "design_goal": goal,
                                        "platform": platform, "intent": intent},
            # So a re-critique later in this same thread doesn't ask again.
            "active_graphic": {**g, "design_goal": goal or None, "platform": platform or None,
                               "intent": intent or g.get("intent"), "metadata_asked": True,
                               "metadata_status": "complete" if goal and platform else "missing"},
            "conversation_history": state.get("conversation_history", []) + [
                {"role": "system", "content": GRAPHIC_METADATA_ASK},
                {"role": "user", "content": reply},
            ],
        })

    # A GENERATED image already has a brief -- design_goal / platform_canonical
    # came from the catalog at launch, clarified_intent from intentClarifier --
    # so this question has nothing left to learn. resolveGrounding's own
    # "generated" branch ignores critique_goal/critique_platform in favor of
    # those same fields, and every lens tolerates an empty critique_intent
    # (renders "(not stated)"), so skipping the ask costs nothing downstream.
    # An UPLOAD is the one case with no brief at all -- that's what the
    # question exists for.
    from .critiquepipeline.grounding import resolveProvenance
    if resolveProvenance(state) == "generated":
        print("[suggestionManager] generated image -- brief is already known, skipping critic_intent")
        return Command(goto="suggestionCreator", update={
            "cost": cost,
            "regenerate_suggestions": False,
            "critique_intent": (state.get("critique_intent") or state.get("clarified_intent")
                                or state.get("base_prompt") or ""),
            "critique_intent_image": image_now,
            "critique_goal": state.get("critique_goal", ""),
            "critique_platform": state.get("critique_platform", ""),
        })

    # A general critique of an UPLOAD is a judgement against a brief, and
    # there is otherwise none to judge against, so ask for it. `critic_started`
    # used to sit here as a spinner ping that main.py auto-answered with
    # response:"" -- the pipeline then had to guess what the image was for,
    # and the only thing it could reach for was the sticky project catalog
    # design_goal (see grounding.resolveProvenance's note). That is how an
    # uploaded magazine cover was told to add the CTA "its" product-launch
    # brief required.
    intent = state.get("critique_intent", "")
    goal = state.get("critique_goal", "")
    platform = state.get("critique_platform", "")

    # Ask once per image. A re-critique of the SAME artwork (after edits, or
    # after criticismHandler sent us back to rebuild a rejected stack) reuses
    # the answer -- re-asking "so what is this for?" about the design we just
    # discussed reads as if we forgot. A different image is a different brief.
    # The marker records that we ASKED, not that they answered, so a user who
    # skipped the question is not asked again about the same image.
    asked_for = state.get("critique_intent_image")
    update = {
        "cost": cost,
        "regenerate_suggestions": False,
        "critique_intent": intent,
        "critique_intent_image": image_now,
        # Deliberately NOT design_goal / platform_canonical: those are the
        # project-launch catalog settings main.py persists to design_projects,
        # and overwriting them from a critique answer would retarget every
        # later generation in the session.
        "critique_goal": goal,
        "critique_platform": platform,
    }

    if asked_for is not None and asked_for == image_now:
        print("[suggestionManager] reusing the intent already captured for this image")
        return Command(goto="suggestionCreator", update=update)

    answer = interrupt({
        "type": "critic_intent",
        "question": CRITIQUE_INTENT_ASK,
        # Free text is the expected reply, but a frontend that already has the
        # user's catalog picks on screen can send them alongside and skip the parse.
        "accepts": ["text", "design_goal", "platform"],
        "current_image_url": state.get("current_image_url", ""),
        "cost": cost,
    })
    # Billed at the interrupt above; everything after it is this turn's own cost,
    # settled at suggestionHandler's terminal interrupt.
    cost = 0.0

    answer = parseJson(answer)
    if not isinstance(answer, dict):
        # A frontend that replies with the bare sentence rather than the usual
        # {"response": ...} frame. The answer IS the payload here, so take it.
        answer = {"response": str(answer)}
    reply = (answer.get("response") or "").strip()
    goal = (answer.get("design_goal") or "").strip()
    platform = (answer.get("platform") or "").strip()
    intent = reply

    if reply and not (goal and platform):
        parsed, parse_cost = _parseCritiqueIntent(reply)
        cost += parse_cost
        intent = parsed.get("intent") or reply
        goal = goal or parsed.get("design_goal", "")
        platform = platform or parsed.get("platform", "")

    print(f"[suggestionManager] intent captured (goal={goal or '-'}, platform={platform or '-'})")

    update.update({
        "cost": cost,
        "critique_intent": intent,
        "critique_goal": goal,
        "critique_platform": platform,
        "conversation_history": state.get("conversation_history", []) + [
            {"role": "system", "content": CRITIQUE_INTENT_ASK},
            {"role": "user", "content": reply},
        ],
    })
    return Command(goto="suggestionCreator", update=update)


# Words that name a platform family, and the catalog names they cover.
_PLATFORM_FAMILIES = (
    (r"\b(stor(y|ies)|reels?)\b", "instagram story"),
    (r"\b(instagram|insta|ig)\b", "instagram"),
    (r"\b(facebook|fb)\b", "facebook"),
    (r"\b(linkedin)\b", "linkedin"),
    (r"\b(pinterest|pin)\b", "pinterest"),
    (r"\b(youtube|yt|thumbnail)\b", "youtube"),
    (r"\b(twitter|tweet|x\.com)\b", "x (twitter)"),
    (r"\b(slides?|presentation|deck)\b", "presentation"),
    (r"\b(signage|screen|display board)\b", "signage"),
)


def _ratio(value):
    try:
        w, h = (float(x) for x in str(value).split(":"))
        return w / h if h else None
    except (TypeError, ValueError):
        return None


def _platformFromWords(reply, platforms, size):
    """The catalog platform a reply names by its network ("on Instagram"),
    choosing the format closest to the image's own shape when the network has
    several. "" when the reply names no network we support."""
    import re
    text_l = (reply or "").lower()
    entries = [(p.get("name"), _ratio(p.get("aspect_ratio"))) if isinstance(p, dict) else (p, None)
               for p in platforms]
    for pattern, family in _PLATFORM_FAMILIES:
        if not re.search(pattern, text_l):
            continue
        matches = [(n, r) for n, r in entries if n and family in n.lower()]
        if not matches:
            continue
        if len(matches) == 1 or not size:
            return matches[0][0]
        w, h = size
        target = (w / h) if h else 1.0
        return min(matches, key=lambda m: abs((m[1] or 1.0) - target))[0]
    return ""


def _parseCritiqueIntent(reply, size=None):
    """Turn the user's free-text plan for the image into a grounded brief.

    Returns ``({"intent", "design_goal", "platform", "audience", "must_convey"}, cost)``.
    The goal/platform are matched against the SAME catalog the generation
    pipeline uses, so a matched pair loads the real layout profile in
    ``critiquepipeline.grounding`` instead of dropping to generic mode. An
    unparseable answer is not an error: the raw sentence still grounds the
    lenses, the critique just runs generic.
    """
    try:
        from .generation_pipeline.avenue2_generation_pipeline.v5_4_scope_check import load_catalog
        goals, platforms = load_catalog()
    except Exception:
        goals, platforms = [], []

    prompt = CRITIQUE_INTENT_PARSER_PROMPT.format(
        goals="\n".join(f"- {g}" for g in goals) or "(catalog unavailable)",
        platforms="\n".join(f"- {p.get('name') if isinstance(p, dict) else p}" for p in platforms)
                  or "(catalog unavailable)",
        answer=reply,
        size=(f"{size[0]}x{size[1]} px (aspect {size[0] / size[1]:.2f})" if size and size[1] else "unknown"),
    )
    try:
        response, model_used = invoke_with_fallback(
            [{"role": "user", "content": prompt}], temperature=0, role="classify_extract")
        cost = calculateCost(model_used, response.usage_metadata['total_tokens'])
        parsed = parseJson(response.content) or {}
    except Exception as e:
        print(f"[suggestionManager] intent parse failed ({e}); falling back to the raw answer")
        return {"intent": reply, "design_goal": "",
                "platform": _platformFromWords(reply, platforms, size)}, 0.0

    # An answer that named nothing we support must not be forced onto a
    # neighbouring catalog entry -- grounding against the wrong rulebook is
    # worse than running generic.
    named = {str(g).strip().lower() for g in goals}
    plats = {str(p.get("name") if isinstance(p, dict) else p).strip().lower() for p in platforms}
    if named and str(parsed.get("design_goal", "")).strip().lower() not in named:
        parsed["design_goal"] = ""
    if plats and str(parsed.get("platform", "")).strip().lower() not in plats:
        parsed["platform"] = ""
    # The model plays safe when a network has several formats; the words plus
    # the image's own shape settle it without guessing.
    if not parsed.get("platform"):
        parsed["platform"] = _platformFromWords(reply, platforms, size)
    return parsed, cost


def _catalogOrEmpty(goal, platform):
    """Canonical catalog spelling of ``goal`` / ``platform``, or ``""`` for anything else.

    Same catalog as :func:`_parseCritiqueIntent`, matched case-insensitively. Both
    the frontend's picks and the LLM parse go through this, so newimages only ever
    stores catalog names. With the catalog unavailable nothing can be verified, so
    both come back empty rather than storing an unchecked value.
    """
    try:
        from .generation_pipeline.avenue2_generation_pipeline.v5_4_scope_check import load_catalog
        goals, platforms = load_catalog()
    except Exception:
        goals, platforms = [], []
    goal_names = {str(g).strip().lower(): str(g) for g in goals}
    plat_names = {str(p.get("name") if isinstance(p, dict) else p).strip().lower():
                  str(p.get("name") if isinstance(p, dict) else p) for p in platforms}
    return (goal_names.get(str(goal or "").strip().lower(), ""),
            plat_names.get(str(platform or "").strip().lower(), ""))




def _layeringTargetsFromPrompt(state) -> tuple:
    """What the message that entered layering already names, if anything.

    Runs the same classifier the layering subgraph uses, over the same reply
    shape, so a prompt that names its targets is read identically whether it
    arrives here or at the subgraph's own handler.

    Only ``proceed_to_layering`` answers count. Any other class means the user
    was saying something else ('skip this', 'edit it first'), which the
    subgraph's handler still owns -- reading targets out of it here would
    start a layering run nobody asked for.

    :returns: ``(requested_objects, requested_texts, detect_all, cost)`` --
        empty lists and ``False`` whenever the prompt names nothing, or the
        classifier could not be reached.
    """
    prompt_text = str(state.get("prompt") or "").strip()
    if not prompt_text:
        return [], [], False, 0.0

    try:
        response, model_used = invoke_with_fallback([
            {"role": "user", "content": LAYERING_HANDLER_PROMPT.format(
                conversation_history=json.dumps(state.get("conversation_history", [])),
                user_response=prompt_text,
            )}
        ], temperature=0, thinking_budget=0)
        cost = calculateCost(model_used, response.usage_metadata['total_tokens'])
        data = parseJson(response.content)
    except Exception as e:
        logging.getLogger('graphic_designer.layeringManager').warning(
            "Could not pre-read the layering targets (%s) — asking the user instead.", e
        )
        return [], [], False, 0.0

    if data.get("class") != "proceed_to_layering":
        return [], [], False, cost
    objects, texts, detect_all = requestedLayerTargets(data)
    return objects, texts, detect_all, cost


@profile
def layeringManager(state: GraphicDesigner) -> GraphicDesigner:
    """Entry gate into the layering workflow — asks WHAT to layer.

    The question is only asked when the message that routed here named nothing
    to layer. "layer the chair" and 'pull the "SALE" headline out' already say
    it, so those skip straight to detection with the targets
    :func:`_layeringTargetsFromPrompt` read out of them; asking again would
    turn one request into two and read as though nothing was heard.

    ``layering_started`` is no longer an auto-continued spinner ping: it is a
    real question with two answers.

    Neither answer confirms anything before SAM2: whatever detection finds
    goes straight to segmentation.

    * ``auto`` — the pipeline finds the objects itself (Gemini over the whole
      image → depth → Florence-2 → segmentation).
    * ``manual`` — the reply also carries ``bboxes``: boxes the user drew on
      the image themselves. Each one is recursed through concept generation
      and grounding over its own crop, and what that finds composes back into
      one layer per drawn box (see ``subgraphs/layering.manualLayering``).

    Anything else the user typed instead ('skip this', 'edit it first') is
    left as ``prompt`` with an empty ``layering_mode``, so the subgraph's
    ``layeringHandler`` classifies it the way it always did.

    Control then falls through the static edge to the layering subgraph
    (mounted directly as a node on ``graph``; see ``graph.py``), and from
    there to :func:`layeringResultRouter`.

    Previously this node called ``layGraph.invoke({...})`` itself. That made
    the subgraph a plain nested function call with no checkpoint of its own,
    so *any* pause inside it (an interrupt awaiting bbox confirmation, or a
    resume after an error) reran this whole node from the top -- and with it,
    a brand new ``.invoke()`` that restarted the subgraph at its own START
    node, redoing detection from scratch every round trip. Mounting the
    compiled subgraph directly as a node lets it inherit the parent's
    checkpointer, so LangGraph resumes it exactly where it paused instead.

    :param state: The current graph state.
    :type state: GraphicDesigner
    :returns: The state with ``layering_mode``/``manual_bboxes`` resolved.
    :rtype: GraphicDesigner
    """
    print("[layeringManager] entering the layering workflow")

    # Boxes the user drew on the canvas outrank everything below. main.py puts
    # them in state the moment an Isolate/Erase click carries them (see its
    # DRAWN BBOXES block), precisely so they no longer depend on being the reply
    # to the question this node would otherwise ask. Pointing at exact regions
    # is the least ambiguous instruction there is: no classifier call, no ask.
    if state.get("manual_bboxes"):
        print(f"[layeringManager] {len(state['manual_bboxes'])} drawn box(es) already in state "
              f"-- routing to manual layering, no ask")
        return {**state, "layering_mode": "manual",
                "layering_targets_resolved": False, "cost": state.get('cost', 0.0)}

    # The message that routed here may already say what to layer -- "layer the
    # chair", 'pull out the "SALE" headline'. Asking "which objects?" after
    # that reads as though the system did not hear it, so the question is only
    # put to users who named nothing.
    named_objects, named_texts, detect_all, preclassify_cost = _layeringTargetsFromPrompt(state)
    if named_objects or named_texts or detect_all:
        print(f"[layeringManager] prompt already names "
              f"{len(named_objects)} object(s), {len(named_texts)} text(s), "
              f"detect_all={detect_all} -- skipping the ask")
        return {
            **state,
            "layering_mode": "auto",
            "manual_bboxes": [],
            "requested_objects": named_objects,
            "requested_texts": named_texts,
            "layering_targets_resolved": True,
            # No interrupt happened, so nothing has billed the spend so far --
            # it rides on to the next one instead of resetting to 0.0.
            "cost": state.get('cost', 0.0) + preclassify_cost,
        }

    user_response = interrupt({
        "type": "layering_started",
        "current_image_url": state.get("current_image_url", ""),
        "question": LAYERING_OBJECT_ASK,
        # What a valid reply looks like, so the frontend can offer both a text
        # box and its box-drawing tool from the same prompt. Replaces the old
        # "auto"/"manual" options list: the mode is not a question the user
        # should have to answer, it is implied by HOW they reply.
        "accepts": ["object_names", "bboxes"],
        # When demand-driven text is on, text is not automatically extracted
        # unless requested or occluding an object.
        "auto_text": False if getattr(settings, "TEXT_DEMAND_DRIVEN", True) else True,
        "note": LAYERING_TEXT_AUTO_NOTE,
        "cost": state.get('cost', 0.0) + preclassify_cost
    })

    mode, bboxes, reply = _parse_layering_choice(user_response)
    print(f"[layeringManager] mode={mode or 'named-objects'} bboxes={len(bboxes)}")

    # The interrupt above already billed everything accumulated so far, so the
    # pending cost resets here -- otherwise the next interrupt in the layering
    # run charges for the same work a second time.
    update = {**state, "layering_mode": mode, "manual_bboxes": bboxes, "cost": 0.0,
              "layering_targets_resolved": False}
    if not mode and reply:
        # They named the objects (or said something else entirely): hand what
        # they actually said to layeringHandler instead of re-classifying the
        # stale prompt that triggered layering in the first place.
        update["prompt"] = reply
    elif mode == "manual" and reply:
        # Boxes won the routing, but the user still typed something. Keep it in
        # the history so the next turn's classifier has the context -- dropping
        # it entirely is what makes a follow-up reply look like it came out of
        # nowhere.
        update["conversation_history"] = state.get("conversation_history", []) + [
            {"role": "user", "content": reply},
        ]
    return update


# Corner-key spellings a frontend might send instead of a "bbox" list, in the
# order the detection pipeline expects (x1, y1, x2, y2).
_BBOX_KEY_SETS = (
    ("x1", "y1", "x2", "y2"),
    ("left", "top", "right", "bottom"),
    ("xmin", "ymin", "xmax", "ymax"),
)


def _normalize_bboxes(raw) -> list:
    """Coerce whatever the frontend sent into ``[{"bbox": [x1,y1,x2,y2], "label": str}]``.

    ``detection_start``'s frontend branch does ``b.get("bbox")`` on every entry,
    so a bare ``[x1, y1, x2, y2]`` list -- a perfectly reasonable thing for a
    canvas to emit -- raises AttributeError and takes the whole layering run
    down. Normalizing here means the reply parser is the only place that has to
    know which shapes are accepted.

    Malformed entries are dropped with a log line rather than passed on: one bad
    box must not cost the user the boxes they drew correctly.
    """
    if not isinstance(raw, list):
        return []

    out = []
    for i, entry in enumerate(raw):
        box = label = None

        if isinstance(entry, dict):
            label = entry.get("label") or entry.get("name") or entry.get("object")
            box = entry.get("bbox") or entry.get("box")
            if box is None:
                for keys in _BBOX_KEY_SETS:
                    if all(k in entry for k in keys):
                        box = [entry[k] for k in keys]
                        break
        elif isinstance(entry, (list, tuple)):
            box = entry

        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            print(f"[layering] dropping unusable bbox at index {i}: {str(entry)[:120]}")
            continue
        try:
            coords = [int(round(float(v))) for v in box]
        except (TypeError, ValueError):
            print(f"[layering] dropping non-numeric bbox at index {i}: {str(box)[:120]}")
            continue

        item = {"bbox": coords,
                "label": str(label).strip() if label else f"object_{len(out) + 1}"}
        # Carry a text flag through when the frontend sets one -- boxes drawn
        # over text select which detected texts stay layered.
        if isinstance(entry, dict) and entry.get("is_text_layer") is not None:
            item["is_text_layer"] = bool(entry["is_text_layer"])
        out.append(item)
    return out


def _parse_layering_choice(user_response):
    """Read the ``layering_started`` reply -- the answer to "which objects?".

    Two shapes of answer, and the reply itself tells them apart; the user is
    never asked to pick a mode:

    * **boxes drawn** (``bboxes`` present) -> ``'manual'``. Each box is
      recursed through Gemini + Florence over its own crop, and the objects
      found inside them go straight to SAM2 with no confirmation
      (``manualLayering`` -> ``detection_start(frontend_bboxes=...)``).
    * **objects named** (plain text) -> ``''``, and the text becomes the new
      ``prompt``, so the subgraph's ``layeringHandler`` extracts the named
      objects and runs the normal detection path.

    Boxes win when a reply somehow carries both: the user pointing at exact
    regions is a stronger signal than the words next to them, and it is the
    cheaper path.

    ``'auto'``/``'manual'`` mode words are still accepted, so an older frontend
    build answering the previous two-option question keeps working.

    :returns: ``(mode, bboxes, reply_text)`` where ``mode`` is ``'auto'``,
        ``'manual'`` or ``''`` (the user named objects, or said something else
        entirely).
    """
    data = user_response
    if isinstance(data, str):
        try:
            data = parseJson(data)
        except ValueError:
            data = {"response": data}
    if not isinstance(data, dict):
        data = {}

    reply = str(data.get('response') or '').strip()
    mode = str(data.get('layering_mode') or data.get('mode') or '').strip().lower()
    if mode not in ('auto', 'manual'):
        mode = reply.lower() if reply.lower() in ('auto', 'manual') else ''

    # Accept the boxes under any of the keys a frontend might use for them.
    raw_boxes = next((data[k] for k in ('bboxes', 'boxes', 'manual_bboxes')
                      if isinstance(data.get(k), list) and data[k]), None)
    bboxes = _normalize_bboxes(raw_boxes)

    # Boxes with no mode word can only mean manual -- there is nothing else
    # for the frontend to be sending them for at this point.
    if bboxes and mode != 'manual':
        if mode == 'auto':
            print("[layering] reply says 'auto' but carries boxes -- honoring the boxes")
        mode = 'manual'
    if mode == 'manual' and reply:
        print(f"[layering] boxes provided alongside text ({reply[:80]!r}) -- "
              f"the boxes are used; the text is kept in the conversation only")
    if mode != 'manual':
        bboxes = []
    return mode, bboxes, reply


@profile
def layeringResultRouter(state: GraphicDesigner) -> Command:
    """Route out of the (now-completed) layering subgraph.

    Runs once the mounted layering subgraph reaches its own END, with its
    output already merged into the live graph state (``bboxes``,
    ``segmentation_result``, ``prompt``, ``conversation_history``, etc.) --
    unlike the old ``layGraph.invoke()`` call, there's no separate
    ``subgraph_response`` to read from.

    * No bboxes / no segmentation result (skipped, cancelled, or redirected
      to a new edit/generation/suggestion) -- the subgraph's own exit node
      (``layeringSkipped`` or ``layExit``) already appended the user's real
      reply to ``conversation_history``, so this just hands ``prompt`` back to
      the navigator as-is.
    * Otherwise -- layering succeeded, so this logs the original request
      that kicked off layering and carries the layered image to ``nextStep``.

    :param state: The current graph state, post-subgraph merge.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` or ``nextStep``.
    :rtype: Command
    """
    # Keyed on the segmentation result alone, not on bboxes: a manual run can
    # legitimately select text regions only, producing layers (background +
    # texts) with an empty object-bbox list. Every no-op exit out of the
    # subgraph leaves segmentation_result None/empty, so this still catches
    # skip / cancel / redirect.
    if not state.get('segmentation_result'):
        return Command(goto="navigator", update={**state, "cost": state.get('cost', 0.0), "suggestion_type": "user_driven", "model": state.get("model", "")})
    else:
        return Command(goto="nextStep", update={**state, "cost": state.get('cost', 0.0), "bboxes": state['bboxes'], "segmentation_result": state['segmentation_result'], "extracted_texts": state.get('extracted_texts', []), "current_image_url": state.get("current_image_url", ""), "attached_images": state.get('attached_images', []), "conversation_history": state.get('conversation_history', []) + [{"role": "user", "content": state["prompt"]}], "suggestion_type": "user_driven", "model": state.get("model", ""), "aspect_ratio": state.get('aspect_ratio', '')})



# @profile
# def bestModel(state: GraphicDesigner) -> GraphicDesigner:
#     """Recommend the best image-generation model and collect user confirmation.

#     Selects an optimal model based on the requested aspect ratio:

#     * If the aspect ratio is supported by Imagen 4, recommends **Imagen 4**.
#     * Otherwise, defaults to **Nano-Banana Pro**.

#     If the user's selected model and the best_model is the same then it directly goes to prompt enhancer node.

#     If not sends an interrupt to present the recommendation to the user and awaits
#     their response.  The user may accept or override the suggestion.  Sets
#     ``pending_action`` and ``pending_node`` globals so that a subsequent
#     ``decisionHandler`` call can process accept/reject.

#     :param state: The current graph state with ``aspect_ratio`` and
#         ``conversation_history``.
#     :type state: GraphicDesigner
#     :returns: A ``Command`` routing to ``navigator`` with the user's response
#         appended to the conversation history.
#     :rtype: Command
#     :raises Exception: If the interrupt or response parsing fails.
#     """

#     global pending_action, pending_node
#     best_model = ""

#     if state['aspect_ratio'] in imagen4SupportedAspectRatios:
#         best_model = "Imagen 4"
#     else:
#         best_model = "Nano-Banana Pro"

#     if best_model.lower().replace(" ", "-") == state['model']:
#         return Command(goto="promptEnhancer")

#     user_choice = interrupt({
#         "type": "best_model",
#         "question": pystache.render(bestModelNotification, {'best_model': best_model}),
#         "cost": state.get('cost', 0.0)
#     })

#     user_response = parseJson(user_choice)
#     style = user_response.get('style', None)

#     try:
#         pending_action = {
#             "generation": {
#                 "pending_decisions": ["model", "prompt"],
#             }
#         }
#         pending_node = "promptEnhancer"

        
#         return Command(goto = "navigator", update={"prompt": user_response['response'], "style": style, 'cost' : 0.0, 'conversation_history': [*state['conversation_history'], {"role": "system", "content": pystache.render(bestModelNotification, {'best_model': best_model})}, {"role": "user", "content": user_response['response']}]})
        
#     except Exception as e:
#         raise e


# ──────────────────────────────────────────────────────────────────────────
# Generation-pipeline integration (avenue2 reasoning pipeline)
# ──────────────────────────────────────────────────────────────────────────

def resolve_catalog_selection(state: GraphicDesigner):
    """Resolve the canonical catalog (design_goal, platform, aspect_ratio),
    plus whether design_goal is a CUSTOM (off-catalog) goal.

    The frontend sends ``design_goal`` and ``platform`` at project launch.
    ``design_goal`` is validated (case-insensitively) against the service
    catalog, which is ``layout_guidance.json`` -- the same file the generation
    pipeline reasons from (see ``load_catalog``); when it's present but doesn't
    match any supported goal, it is treated as an intentional custom goal and
    preserved VERBATIM (``is_custom_goal=True``) rather than overwritten --
    scopeGate gives custom goals zero backend involvement (no scope-check, no
    CoT). Only a genuinely missing design_goal falls back to the legacy
    placeholder, with a loud warning so the pipeline still runs.
    """
    catalog_goals, catalog_platforms = load_catalog()

    raw_goal = (state.get("design_goal") or "").strip()
    matched_goal = next((g for g in catalog_goals if g.lower() == raw_goal.lower()), "") if raw_goal else ""
    is_custom_goal = False
    if matched_goal:
        design_goal = matched_goal
    elif raw_goal:
        design_goal = raw_goal
        is_custom_goal = True
    else:
        design_goal = "Festive Sale"

    raw_platform = (state.get("platform_canonical") or state.get("platform") or "").strip()
    platform_entry = next((p for p in catalog_platforms if p.get("name", "").lower() == raw_platform.lower()), None) if raw_platform else None
    if platform_entry:
        platform = platform_entry["name"]
    elif raw_platform:
        # Custom goal + custom platform label: keep it verbatim too -- there's
        # no catalog aspect_ratio to fall back on, so state's own aspect_ratio
        # (derived from launch pixel size) below carries the real dimensions.
        platform = raw_platform
    else:
        platform = "Instagram Square Post"

    # aspect_ratio: main.py derives it from the launch 'size' (px) when given;
    # otherwise use the catalog platform's canonical ratio.
    aspect_ratio = state.get("aspect_ratio") or (platform_entry or {}).get("aspect_ratio") or "1:1"
    return design_goal, platform, aspect_ratio, is_custom_goal


def _datauri_to_tempfile(datauri: str) -> Path:
    """Decode a ``data:image/...;base64,...`` URI into a temp file, return its Path."""
    header, _, b64 = datauri.partition(",")
    ext = "png"
    if "image/" in header:
        ext = header.split("image/")[1].split(";")[0] or "png"
    fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix="gp_photo_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(base64.b64decode(b64))
    return Path(path)


def _url_to_tempfile(url: str) -> Path:
    """Download an image URL into a temp file, return its Path."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    ext = imghdr.what(None, resp.content) or "png"
    fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix="gp_photo_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(resp.content)
    return Path(path)


def _to_localfile(src: str, dest_dir: Path, name: str) -> Path:
    """Materialize a data URI or URL into ``dest_dir/{name}.{ext}``.

    Same decode logic as :func:`_to_tempfile`, but written into a caller-chosen
    directory instead of the system temp dir, so the caller can delete it
    (see :func:`v32Generation`'s ``work_dir``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    data = _image_bytes(src)
    ext = "png"
    if src.startswith("data:"):
        header = src.partition(",")[0]
        if "image/" in header:
            ext = header.split("image/")[1].split(";")[0] or "png"
    else:
        ext = imghdr.what(None, data) or "png"
    out = dest_dir / f"{name}.{ext}"
    out.write_bytes(data)
    return out


def _image_bytes(src: str) -> bytes:
    """The bytes of a data URI or URL, in memory."""
    if src.startswith("data:"):
        return base64.b64decode(src.partition(",")[2])
    resp = requests.get(src, timeout=30)
    resp.raise_for_status()
    return resp.content


def _logo_zone_position_description(zone_pct: dict | None) -> str:
    """``{"x_pct": [x1,x2], "y_pct": [y1,y2]}`` -> a natural-language position
    ("top-center", "bottom-right", ...), by quantizing the rect's centroid onto
    a 3x3 grid. Falls back to "top-center" (the same default
    ``_resolve_logo_zone_pct`` itself falls back to) when the zone is missing
    or malformed.
    """
    try:
        x1, x2 = zone_pct["x_pct"]
        y1, y2 = zone_pct["y_pct"]
        cx, cy = (float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2
    except (TypeError, KeyError, ValueError, IndexError):
        return "top-center"
    col = "left" if cx < 33 else "right" if cx > 66 else "center"
    row = "top" if cy < 33 else "bottom" if cy > 66 else "middle"
    return f"{row}-{col}" if row != "middle" or col != "center" else "center"


def _v32_reference_images(state: GraphicDesigner, dest_dir: Path) -> list[dict]:
    """Build v33's ``[{"path", "role"}]`` reference list from whatever this
    project already has attached.

    v33 (``reference_roles.normalise()``) accepts only real local files and
    silently DROPS anything that is not one -- an S3 URL passed through
    unconverted yields a plausible, reference-free graphic with no error. So
    every source here is downloaded/decoded to disk before being handed over,
    and every result is asserted to exist.

    Files are written into ``dest_dir``, the caller's scratch directory for
    one v33 call, which the caller deletes when the call returns.

    Priority order, hard-capped at v33's limit of 5 (a 6th raises ``ValueError``,
    which is not a ``PipelineError`` and must not escape the caller):

    1. ``logo_asset``                                    -> ``logo``
    2. ``attached_images[0]``, when a real photo was supplied -> ``product``
       (not ``hero``: INTEGRATION.md's product rules are the ones that stopped
       a supplied photo rendering as a framed card on a separately generated
       scene, which is what the hero rules were doing).
    3. brand templates (:func:`_get_filtered_templates`)  -> ``style``
    4. brand color palette (:func:`generate_palette_reference_image`) -> ``style``

    Deliberately excludes the host's selected wireframe/layout variant: that is
    v5_4_layout_matcher's own zone diagram, built for a CoT stage that no
    longer runs. v33 makes its own composition decision internally from its
    own ``composition_menu``, and its ``layout`` role is documented as "must
    not leak in a subject or palette" -- handing it a zone-labeled debug
    diagram is the opposite of what that role is for, and risks the exact
    zone-label/coordinate-text artifacts ``_ANTI_ARTIFACT_GUARD`` exists to
    keep off the canvas.
    """
    refs: list[dict] = []

    def _add(src: str, role: str, name: str) -> bool:
        if len(refs) >= 5 or not (isinstance(src, str) and src):
            return False
        try:
            path = _to_localfile(src, dest_dir, name)
        except Exception as e:
            logging.getLogger("graphic_designer.v32Generation").warning(
                "reference image %r (%s) could not be materialized: %s -- dropping it "
                "rather than risk a silent v33 drop", name, role, e)
            return False
        if not path.exists():
            logging.getLogger("graphic_designer.v32Generation").warning(
                "materialized reference %s does not exist on disk -- dropping it", path)
            return False
        refs.append({"path": str(path), "role": role})
        return True

    # strict_paste deliberately does NOT get the logo as a v33 reference: the
    # real logo is pasted in afterward by logoCompositor, onto whatever zone
    # v33 leaves clean (see v32Generation's prompt instruction for that case).
    # Handing v33 the logo file here as well would have it try to render its
    # OWN copy into the composition -- either a duplicate, or a render that
    # fights the post-paste for the same zone. Only single_call (logo folded
    # into this same generation call, no separate paste step) uses it.
    if (state.get("logo_mode") == "provided" and state.get("logo_asset")
            and state.get("logo_insertion_mode") == "single_call"):
        _add(state["logo_asset"], "logo", "logo")

    attached = state.get("attached_images") or []
    if attached and state.get("photo_mode") in ("as_is", "reference"):
        _add(attached[0], "product", "product")

    for i, tmpl in enumerate(_get_filtered_templates(state)):
        if len(refs) >= 5:
            break
        _add(tmpl, "style", f"template_{i}")

    brand_dna = state.get("brand_dna")
    if brand_dna and "color_palettes" in brand_dna and len(refs) < 5:
        palette_b64 = generate_palette_reference_image(brand_dna["color_palettes"])
        if palette_b64:
            _add(palette_b64, "style", "palette")

    return refs


@profile
def scopeGate(state: GraphicDesigner) -> Command:
    """Phase 1.0 scope gate — does the prompt's implied deliverable/platform match
    the catalog selection?

    A design_goal outside the supported catalog (a CUSTOM goal) bypasses this
    entire gate: no scope-check call, no mismatch dialog, straight to
    generation with the raw prompt untouched (see is_custom_goal branch
    below) — the whole CoT apparatus exists only for our own catalog goals.

    For catalog goals, a mismatch fires a blocking dialog with three branches:
    proceed with the detected (supported) goal, proceed as a freeform bypass
    (detected goal unsupported), or keep the original selection. Never breaks
    the flow — any failure degrades to the current behaviour (proceed).

    The gate is split across three nodes so that neither dialog can re-run the
    scope check: THIS node owns the ``check_request_scope`` call and
    checkpoints its output in ``scope_result``; :func:`scopeMismatchGate` owns
    the proceed/decline interrupt; :func:`scopeAbortGate` owns the follow-up
    interrupt after a decline. Both dialog nodes read everything they need out
    of state, so the code they re-run on resume is pure state reads.
    """
    print("[scopeGate] checking whether the prompt matches the catalog selection")
    cotProgress("scopeGate")
    # Running unbilled cost since the last interrupt. Every interrupt below
    # emits it and resets it to 0, so each interrupt bills only its own delta.
    cost = state.get("cost", 0.0)
    design_goal, platform, aspect_ratio, is_custom_goal = resolve_catalog_selection(state)

    if is_custom_goal:
        # The frontend selected a design goal outside our supported catalog.
        # Custom goals get ZERO backend involvement: no scope-check against
        # the prompt, no mismatch dialog, no CoT/manifest/asset gates -- the
        # user's prompt goes straight to image generation exactly as typed.
        custom = {
            **state,
            "design_goal": design_goal,
            "platform_canonical": platform,
            "aspect_ratio": aspect_ratio,
            "is_freeform": True,
            "fused_prompt": state["prompt"],
        }
        return Command(goto=_freeformNext(custom), update=custom)

    try:
        scope = check_request_scope(state["prompt"], design_goal, platform,
                                    role="classify_extract")
        _usage = scope.get("_usage", {})
        _total = _usage.get("prompt_tokens", 0) + _usage.get("completion_tokens", 0)
        if _total:
            try:
                cost += calculateCost("gemini-3-flash-preview", _total)
            except Exception:
                pass
    except Exception as e:
        scope = {"match": True}

    # The resolved catalog selection is committed here (not only on the happy
    # path) so both dialog nodes can read it straight out of state.
    base = {
        **state,
        "design_goal": design_goal,
        "platform_canonical": platform,
        "aspect_ratio": aspect_ratio,
        "cost": cost,
    }

    if not scope.get("match", True):
        # Hand the dialog to scopeMismatchGate. The check result rides along in
        # state, so resuming that interrupt costs nothing to replay.
        return Command(goto="scopeMismatchGate", update={**base, "scope_result": scope})

    return Command(goto="intentClarifier", update={
        **base,
        "pending_platforms": scope.get("pending_platforms", []),
        "is_freeform": False,
        "scope_result": {},
    })


def _platform_aspect_ratio(platform: str, fallback: str) -> str:
    """The catalog's canvas for a platform, or ``fallback`` if it has none.

    Switching platform has to take the aspect ratio with it -- an Instagram
    Story is 9:16 and a Square Post is 1:1, so keeping the old ratio would lay
    the new design out on the wrong canvas.
    """
    try:
        return platform_ratio(platform) or fallback
    except Exception:
        return fallback


def _scope_mismatch_dialog(scope: dict, design_goal: str, platform: str) -> tuple:
    """What the scope mismatch actually IS, and what accepting it will do.

    Returns ``(action, question)`` where action is one of:

    * ``"switch_goal"``     -- the prompt asks for a different deliverable we
      support; accepting adopts that goal (and its platform).
    * ``"switch_platform"`` -- the goal is fine, the prompt just names a
      different supported platform; accepting keeps the goal and moves to that
      platform. This is the common case ("Instagram Story" selected, prompt
      says "Instagram post") and it used to be UNSAYABLE: detected_goal is
      null by contract whenever the goal matches, so the question rendered
      "your prompt looks like 'something different'" and accepting dropped the
      whole run into freeform.
    * ``"freeform"``        -- what they asked for is outside the catalog;
      accepting runs their prompt as typed with no layout or CoT.
    * ``None``              -- nothing nameable conflicts; caller should not ask.
    """
    detected_goal = (scope.get("detected_goal") or "").strip()
    goal_supported = scope.get("detected_goal_supported")
    goal_conflict = bool(scope.get("goal_conflict"))
    platform_conflict = bool(scope.get("platform_conflict"))
    primary = (scope.get("primary_platform") or platform or "").strip()
    unsupported_platforms = [p for p in (scope.get("unsupported_platforms") or []) if p]
    pending = [p for p in (scope.get("pending_platforms") or []) if p]

    if goal_conflict and detected_goal and goal_supported:
        note = ""
        if primary and primary.lower() != (platform or "").lower():
            note = SCOPE_SWITCH_GOAL_NOTE.format(primary=primary)
        return "switch_goal", SCOPE_SWITCH_GOAL_ASK.format(
            design_goal=design_goal, platform=platform, detected_goal=detected_goal, note=note)

    if goal_conflict and detected_goal:
        return "freeform", SCOPE_FREEFORM_GOAL_ASK.format(
            detected_goal=detected_goal, platform=platform, design_goal=design_goal)

    if platform_conflict and primary and primary.lower() != (platform or "").lower():
        extra = SCOPE_SWITCH_PLATFORM_EXTRA.format(pending=", ".join(pending)) if pending else ""
        return "switch_platform", SCOPE_SWITCH_PLATFORM_ASK.format(
            platform=platform, primary=primary, extra=extra, design_goal=design_goal)

    if platform_conflict and unsupported_platforms:
        named = ", ".join(f"'{p}'" for p in unsupported_platforms)
        return "freeform", SCOPE_FREEFORM_PLATFORM_ASK.format(
            named=named, design_goal=design_goal, platform=platform)

    # Flagged as a mismatch but nothing nameable came back -- never ask a
    # question we cannot phrase.
    return None, ""


@profile
def scopeMismatchGate(state: GraphicDesigner) -> Command:
    """Scope stage 2 of 3: the blocking proceed/decline dialog, reached only
    when :func:`scopeGate` found the prompt's implied deliverable doesn't match
    the catalog selection.

    Everything above the ``interrupt()`` is a state read (the scope check
    itself already ran and is checkpointed in ``scope_result``), so the resume
    replay of this node is free.
    """
    print("[scopeMismatchGate] asking whether to proceed with the detected goal")
    cost = state.get("cost", 0.0)
    scope = state.get("scope_result") or {}
    design_goal = state.get("design_goal") or ""
    platform = state.get("platform_canonical") or ""
    aspect_ratio = state.get("aspect_ratio") or "1:1"

    detected_goal = scope.get("detected_goal")
    detected_supported = scope.get("detected_goal_supported")
    primary_platform = scope.get("primary_platform") or platform

    action, question = _scope_mismatch_dialog(scope, design_goal, platform)
    if action is None:
        # Nothing nameable conflicts, so there is no question worth asking --
        # carry on with the user's selection rather than showing a dialog that
        # cannot describe itself.
        print("[scopeMismatchGate] mismatch flagged but nothing nameable — proceeding as a match")
        return Command(goto="intentClarifier", update={
            **state,
            "pending_platforms": scope.get("pending_platforms", []),
            "is_freeform": False,
            "scope_result": {},
            "cost": cost,
        })

    resume = parseJson(interrupt({
        "type": "text_input",
        "question": question,
        # What accepting will actually do: 'switch_goal' | 'switch_platform' |
        # 'freeform'. Sent so the frontend can label the confirm button with the
        # real outcome instead of a generic yes.
        "scope_action": action,
        "detected_goal": detected_goal,
        "detected_goal_supported": detected_supported,
        "primary_platform": primary_platform,
        "pending_platforms": scope.get("pending_platforms", []),
        "cost": cost,
    }))
    cost = 0.0
    choice = str(resume.get("response", "")).strip().lower()
    proceed = (choice in ("yes", "y", "proceed", "true")
               or resume.get("proceed_with_detected") is True)

    if not proceed:
        # User declined the detected goal: ABORT this generation request (never
        # silently continue with a selection the prompt doesn't match) and let
        # scopeAbortGate ask what they want to do next. scope_result stays in
        # state — that node still needs pending_platforms off it.
        return Command(goto="scopeAbortGate", update={**state, "cost": cost})

    is_freeform = False
    if action == "switch_platform":
        # The deliverable was never in question -- only the format. Keep the
        # goal and the whole layout/CoT pipeline; just move to the platform
        # they actually asked for, and take its canvas with it. This path used
        # to fall into the freeform branch below, silently dropping the layout,
        # the CoT and every asset gate for what is only a format change.
        platform = primary_platform
        aspect_ratio = _platform_aspect_ratio(platform, aspect_ratio)
        print(f"[scopeMismatchGate] switching platform to '{platform}' ({aspect_ratio}), goal unchanged")
    elif action == "switch_goal":
        design_goal = detected_goal
        platform = primary_platform
        aspect_ratio = _platform_aspect_ratio(platform, aspect_ratio)
        # The layout-asset warm-up that used to run on both branches is gone
        # with the wireframes: layout facts now come from the guidance file
        # already in memory, so there is no generation miss left to pre-pay.
    else:
        is_freeform = True

    update = {
        **state,
        "design_goal": design_goal,
        "platform_canonical": platform,
        "aspect_ratio": aspect_ratio,
        "pending_platforms": scope.get("pending_platforms", []),
        "is_freeform": is_freeform,
        "scope_result": {},
        "cost": cost,
    }
    if action == "switch_goal":
        # A new design goal is a new design: whatever was collected for the
        # previous one (field answers, product photo, chat) is flushed, the
        # same as when the goal is changed in the editor (generationManager).
        print(f"[scopeMismatchGate] goal switched to {design_goal!r} -- flushing previous inputs")
        update.update(fresh_generation_reset(state))
        update["generation_subject"] = {**(state.get("generation_subject") or {}),
                                        "design_goal": design_goal}
    if is_freeform:
        # No layout / no CoT — the raw prompt goes straight to image generation
        # (via the logo gate first when the prompt asks for one).
        update["fused_prompt"] = state["prompt"]
        return Command(goto=_freeformNext(update), update=update)
    return Command(goto="intentClarifier", update=update)


@profile
def scopeAbortGate(state: GraphicDesigner) -> Command:
    """Scope stage 3 of 3: the generation was cancelled because the user
    declined the detected goal, so ask what they'd like instead.

    Their reply becomes the new prompt and re-enters :func:`scopeGate` as a
    fresh scope check, so a reply that still conflicts with the selection is
    caught again rather than generated blindly. An empty reply keeps the
    original catalog selection and continues the pipeline.
    """
    print("[scopeAbortGate] generation cancelled, asking what to do instead")
    cost = state.get("cost", 0.0)
    scope = state.get("scope_result") or {}
    design_goal = state.get("design_goal") or ""
    platform = state.get("platform_canonical") or ""

    followup = parseJson(interrupt({
        "type": "text_input",
        "generation_aborted": True,
        "question": SCOPE_ABORT_ASK.format(design_goal=design_goal, platform=platform),
        "cost": cost,
    }))
    cost = 0.0
    new_prompt = str(followup.get("response") or followup.get("prompt") or "").strip()
    if new_prompt:
        return Command(goto="scopeGate", update={
            **state, "prompt": new_prompt, "cost": cost, "scope_result": {}})

    # Empty reply: keep the original selection and carry on.
    return Command(goto="intentClarifier", update={
        **state,
        "pending_platforms": scope.get("pending_platforms", []),
        "is_freeform": False,
        "scope_result": {},
        "cost": cost,
    })


@profile
def intentClarifier(state: GraphicDesigner) -> Command:
    """Stage 1 of the decomposed reasoning pipeline (replaces the monolithic
    ``generationPipeline``): resolve the catalog selection, materialize photo/logo
    assets, and run ``clarify_intent()``.

    Each reasoning-pipeline stage now lives in its own node so that a later
    stage's interrupt doesn't force this (LLM-calling) stage to re-run on
    resume — see GENERATION_FLOW_INTERRUPTS.md. ``run_generation_pipeline()``
    has no passthrough params to accept pre-computed stage results, so this
    calls the pipeline's individual stage functions directly instead.
    """
    print("[intentClarifier] clarifying the user's intent and resolving assets")
    cotProgress("intentClarifier")
    cost = state.get("cost", 0.0)
    design_goal = state.get("design_goal") or "Festive Sale"
    platform = state.get("platform_canonical") or "Instagram Square Post"
    aspect_ratio = state.get("aspect_ratio") or "1:1"

    # Layout facts come from v33's bundled layout_guidance.json, not from a
    # fetched wireframe layout profile: v33 picks its own composition from its
    # own menu, so there is no wireframe to fetch, render or match against.
    layout_profile = v33_bridge.layout_facts(design_goal, platform, aspect_ratio)
    # Which assets this combo wants (asset tiers + their ask copy), derived
    # from the same layout_guidance.json record as the layout facts above --
    # not from layout_requirements.json or the S3 layout store, which were
    # derived twins of it and could disagree with the rules v33 renders from.
    frontend_manifest = {}
    try:
        frontend_manifest = v33_bridge.frontend_manifest(design_goal, platform) or {}
    except Exception as e:
        print(f"[intentClarifier] no manifest for ({design_goal!r}, {platform!r}): {e}")
    variants = layout_profile.get("layout_variants", [])
    variant = variants[0] if variants else {}
    canvas_px = layout_profile.get("canvas_px", [1080, 1080])

    attached = state.get("attached_images") or []
    has_photo = bool(attached)
    photo_mode = "as_is" if has_photo else "reference"

    # Resolve the logo into a base-64 asset kept in state so it survives every
    # later Command(goto=...) hop. Priority: a logo already in state (seeded
    # from the project-launch assets payload) wins over the brand-kit logo.
    logo_asset = state.get("logo_asset") or ""
    if logo_asset and not logo_asset.startswith("data:"):
        try:
            logo_asset = (convert_urls_to_b64([logo_asset]) or [""])[0]
        except Exception as e:
            logo_asset = ""
    if not logo_asset:
        logo_asset = _brand_kit_logo(state)

    # single_call is verified best when a real photo is also present (forces
    # nano-banana-pro downstream); strict_paste is the deterministic fallback
    # when the logo is the only asset.
    logo_insertion_mode = "single_call" if has_photo else "strict_paste"

    clarified = clarify_intent(
        raw_prompt=state["prompt"],
        design_goal=design_goal,
        platform=platform,
        layout=variant.get("name", ""),
        required_fields=layout_profile.get("shared_rules", {}).get("required_content_fields"),
        role="classify_extract",
    )
    usage = clarified.get("_usage", {})
    total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    if total:
        try:
            cost += calculateCost("gemini-3-flash-preview", total)
        except Exception:
            pass

    return Command(goto="intentRouter", update={
        **state,
        "cost": cost,
        "design_goal": design_goal,
        "platform_canonical": platform,
        "aspect_ratio": aspect_ratio,
        "layout_profile": layout_profile,
        "frontend_manifest": frontend_manifest,
        "selected_variant_dict": variant,
        "canvas_px": canvas_px,
        "clarify_result": clarified,
        "clarified_intent": clarified.get("clarified_text", ""),
        "photo_mode": photo_mode,
        "logo_asset": logo_asset,
        "logo_insertion_mode": logo_insertion_mode,
    })


@profile
def intentRouter(state: GraphicDesigner) -> Command:
    """Stage 2: ``route_intent()`` plus the two deterministic hard gates
    (structural ambiguity flags, fabrication-risk missing fields). The gates
    are cheap pure Python, so they're folded into this node rather than given
    their own — nothing is lost by redoing them on a resume."""
    print("[intentRouter] routing intent and applying the deterministic gates")
    cotProgress("intentRouter")
    cost = state.get("cost", 0.0)
    clarified = state.get("clarify_result") or {}
    layout_profile = state.get("layout_profile") or {}
    has_photo = bool(state.get("attached_images"))

    brand_profile = None
    if state.get("brand_dna"):
        try:
            brand_profile = json.dumps(state["brand_dna"])
        except Exception:
            brand_profile = state.get("brand_name")

    router_res = route_intent(
        raw_prompt=state["prompt"],
        clarified_intent=clarified.get("clarified_text", ""),
        brand_profile=brand_profile,
        layout_profile=layout_profile,
        has_photo=has_photo,
        clarity_score=clarified.get("clarity_score", 50),
        clarity_level=clarified.get("clarity_level", "MEDIUM"),
        missing_fields=clarified.get("missing_fields", []),
        ambiguity_flags=clarified.get("ambiguity_flags", []),
        role="classify_extract",
    )
    usage = router_res.get("_usage", {})
    total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    if total:
        try:
            cost += calculateCost("gemini-3-flash-preview", total)
        except Exception:
            pass

    # Hard gate 1 (structural): a structural ambiguity flag forces
    # clarification even when the router said proceed.
    _STRUCTURAL_FLAGS = {"CONTRADICTORY", "WRONG_GOAL_TYPE", "OVERLOADED"}
    _structural = [
        f for f in clarified.get("ambiguity_flags", [])
        if f.get("type", "").upper() in _STRUCTURAL_FLAGS
    ]
    if _structural and not (router_res.get("routing_action") == "CLARIFY_FIRST"
                            or router_res.get("requires_clarification")):
        router_res["requires_clarification"] = True
        router_res.setdefault("clarification_questions", [])
        for f in _structural:
            router_res["clarification_questions"].append(
                f"Your request seems to conflict with itself or the selected design goal: "
                f"{f.get('description', '')} — how would you like to proceed?"
            )

    # Hard gate 2 (fabrication-risk backstop): deterministic, independent of
    # the router's own judgment call.
    _fab_at_risk = at_risk_missing_fields(clarified.get("missing_fields", []))
    if _fab_at_risk:
        router_res["requires_clarification"] = True
        router_res.setdefault("clarification_questions", [])
        router_res["clarification_questions"].extend(
            build_fabrication_clarification_questions(_fab_at_risk)
        )

    # A regenerate has already been through manifestFieldsGate,
    # photoRequiredGate and clarificationGate once -- that is how the design
    # being regenerated came to exist. Re-entering them would re-ask answered
    # questions (and photoRequiredGate would nag for a photo the project
    # already has), which is exactly what makes a Regenerate button feel
    # broken. route_intent above still ran, because its detected_photo_usage /
    # detected_layout_intent are read off THIS turn's instruction and are the
    # reason a newly attached image lands as "use directly" or "use as
    # reference".
    return Command(goto="imageAnalyzerLayoutSelector" if state.get("is_regenerate")
                          else "manifestFieldsGate",
                   update={
        **state,
        "cost": cost,
        "router_flags": router_res,
    })


def _manifest_field_label(manifest: dict, field_id: str) -> str:
    """Label for a field_id across all three manifest field lists."""
    for bucket in ("mandatory_fields", "important_fields", "optional_fields"):
        for f in manifest.get(bucket) or []:
            if f.get("field_id") == field_id:
                return f.get("label") or field_id
    return field_id


# recommended_assets keys with dedicated zone-aware gates downstream
# (photoRequiredGate / logoDecisionGate). Every OTHER key a combo's manifest
# declares is a generic reference asset handled by manifestFieldsGate.
_GATE_OWNED_ASSET_KEYS = {"product_imagery", "photo", "hero_image", "product_image",
                          "background", "logo", "brand_logo"}

# Some manifests also list the photo/logo as a *content field* (e.g. a
# mandatory 'product_photo' or 'brand_logo' field_id). Those are assets, not
# text — the zone-aware photo/logo gates own collecting them (and the launch
# payload already seeds asset_prefs), so manifestFieldsGate must never ask
# them as fields or the user gets asked for an image they already provided.
_ASSET_FIELD_WORDS = ("logo", "photo", "image", "screenshot", "mockup",
                      "headshot", "picture")
# ...unless the id marks a textual field ABOUT the asset (description/brief/
# colour/etc.), which really is typed content.
_ASSET_FIELD_TEXT_WORDS = ("description", "brief", "type", "color", "colour",
                           "subject", "theme", "text", "name")


def _is_gate_owned_asset_field(field_id: str) -> bool:
    fid = str(field_id).lower()
    if any(w in fid for w in _ASSET_FIELD_TEXT_WORDS):
        return False
    return any(w in fid for w in _ASSET_FIELD_WORDS)


def _brand_kit_field_answers(state: GraphicDesigner, manifest: dict = None) -> dict:
    """Manifest content fields an attached brand kit already answers.

    The kit carries the brand's identity and palette, so asking "What's your
    brand name?" right after the user attached the brand kit that names it is
    just the gate not looking. Only fields the kit answers UNAMBIGUOUSLY are
    filled -- e.g. brand_tagline is left alone, because a kit has no tagline
    field and brand_context is a description, not a strapline.

    Deliberately NOT persisted into ``field_answers``: these are re-derived on
    every run, so switching the project's brand mid-way can never leave the
    previous brand's name answered in state.

    Scoped to the fields ``manifest`` actually declares -- otherwise a combo
    that only asks for a brand name still gets the palette and tone pushed
    into its brief under raw field ids nobody asked about.
    """
    dna = state.get("brand_dna")
    dna = dna if isinstance(dna, dict) else {}

    def _text(value):
        return value.strip() if isinstance(value, str) and value.strip() else ""

    out = {}
    name = _text(state.get("brand_name")) or _text(dna.get("brand_name"))
    if name:
        # All three ask the same question of a user who has a brand kit attached.
        for field_id in ("brand_name", "company_name", "business_name"):
            out[field_id] = name

    primary = _text(dna.get("color_primary"))
    if primary:
        out["brand_color"] = primary

    secondary = dna.get("color_secondary")
    palette = [primary, _text(dna.get("color_accent"))]
    palette += [_text(c) for c in (secondary if isinstance(secondary, list) else [])]
    palette = [c for c in dict.fromkeys(palette) if c]
    if palette:
        out["color_palette"] = ", ".join(palette)

    tone = _text(dna.get("tone"))
    if tone:
        out["design_tone"] = tone

    if manifest is not None:
        declared = {f.get("field_id") for f in (manifest.get("mandatory_fields") or [])
                    if isinstance(f, dict)}
        out = {k: v for k, v in out.items() if k in declared}
    return out


def _manifestFieldsResult(state: GraphicDesigner, manifest: dict, answers: dict,
                          prefs: dict, cost: float, extra_brief: str,
                          kit_answers: dict = None) -> Command:
    """Shared tail of manifestFieldsGate/manifestFieldsAsk: resolve the
    combo-specific reference assets, fold every known answer into the brief,
    and advance to photoRequiredGate.

    Both the "nothing left to ask" path and the post-interrupt path land here,
    so the brief is assembled identically whether or not the user was asked.
    """
    # Every provided combo-specific asset (launch payload or just asked) is
    # delivered to the model call as a reference image via extra_assets —
    # including recommended/optional-tier ones that were provided at launch
    # even though only 'required'-tier assets are ever asked.
    extra_assets = {k: (prefs.get(k) or {}).get("asset", "")
                    for k in (manifest.get("recommended_assets") or {})
                    if k not in _GATE_OWNED_ASSET_KEYS
                    and (prefs.get(k) or {}).get("status") == "provided"
                    and (prefs.get(k) or {}).get("asset")}

    # Fold every real answer (launch, extracted, or just asked) into the brief.
    # clarified_intent is rebuilt by intentClarifier each generation run, so
    # this append never accumulates across runs.
    # Brand-kit answers ride in the brief alongside the real ones (the design
    # still has to show the brand name), but only `answers` is persisted.
    briefed = {**(kit_answers or {}), **answers}
    lines = [f"- {_manifest_field_label(manifest, fid)}: {val}"
             for fid, val in briefed.items()
             if str(val).strip() and val != "skipped"]
    clarified = state.get("clarified_intent", "")
    if lines:
        clarified += "\n\nUser-provided content fields (must appear in the design as specified):\n" + "\n".join(lines)
    if extra_assets:
        names = ", ".join(k.replace("_", " ") for k in extra_assets)
        clarified += (f"\n\nUser-provided reference assets attached as additional images "
                      f"(incorporate them as intended): {names}")
    if extra_brief:
        clarified += f"\n\nAdditional user notes: {extra_brief}"

    return Command(goto="photoRequiredGate", update={
        **state,
        "cost": cost,
        "field_answers": answers,
        "asset_prefs": prefs,
        "extra_assets": extra_assets,
        "clarified_intent": clarified,
        # The ask payload has been consumed — never let a stale one survive
        # into the next generation run.
        "manifest_pending_fields": [],
        "manifest_pending_assets": [],
    })


@profile
def manifestFieldsGate(state: GraphicDesigner) -> Command:
    """Stage 2.5: blocking, skippable. Gathers the combo's manifest-mandated
    content fields (frontend_manifest.mandatory_fields) when the project-
    creation screen never collected them — the conversational-session path.

    Resolution order per field: launch payload / earlier answers in
    ``field_answers`` -> LLM extraction from the prompt+brief (values the user
    already typed are never re-asked) -> ONE skippable
    ``manifest_fields_request`` interrupt carrying the manifest's own question
    strings, so the frontend can render structured inputs. Declined fields are
    recorded as ``"skipped"`` for the life of the project (never re-asked,
    mirroring asset_prefs). All known answers are appended to
    ``clarified_intent`` so the CoT fusion designs with them.

    Asset generalization: any ``recommended_assets`` key the manifest declares
    beyond photo/logo (those have their own zone-aware gates) is gathered here
    the same way — launch payload wins, else asked in the same interrupt, with
    skips remembered in ``asset_prefs`` for the life of the project. Provided
    extras land in ``extra_assets`` and are attached to the model call as
    reference images.

    The gate is split across two nodes: THIS one resolves what is still
    missing — including the LLM extraction pass — and checkpoints the ask
    payload in ``manifest_pending_fields``/``manifest_pending_assets``;
    :func:`manifestFieldsAsk` owns the interrupt. Resuming the ask therefore
    never re-runs the extraction call. When nothing is left to ask, this node
    finishes the brief itself and the ask node is never entered.
    """
    print("[manifestFieldsGate] gathering the combo's mandated content fields")
    manifest = state.get("frontend_manifest") or {}
    mandatory = [f for f in (manifest.get("mandatory_fields") or [])
                 if f.get("field_id") and not _is_gate_owned_asset_field(f["field_id"])]
    answers = dict(state.get("field_answers") or {})

    # Combo-specific assets beyond photo/logo. Only 'required'-tier assets are
    # ever ASKED during the CoT — recommended/optional tiers are used when the
    # launch payload provides them but never interrupt the session for them.
    prefs = dict(state.get("asset_prefs") or {})
    extra_specs = {k: (v or {}) for k, v in (manifest.get("recommended_assets") or {}).items()
                   if k not in _GATE_OWNED_ASSET_KEYS
                   and (v or {}).get("tier") == "required"}
    missing_assets = [k for k in extra_specs
                      if (prefs.get(k) or {}).get("status") not in ("provided", "skipped")]

    # Launch-provided combo-specific assets of ANY tier still get delivered to
    # the model call, so check them before deciding this gate has no work.
    has_provided_extras = any(
        k not in _GATE_OWNED_ASSET_KEYS
        and (prefs.get(k) or {}).get("status") == "provided"
        and (prefs.get(k) or {}).get("asset")
        for k in (manifest.get("recommended_assets") or {}))

    if not mandatory and not extra_specs and not has_provided_extras:
        return Command(goto="photoRequiredGate", update={
            **state, "manifest_pending_fields": [], "manifest_pending_assets": []})

    cost = state.get("cost", 0.0)
    # An attached brand kit already answers some of these. Resolved before the
    # extraction pass so a kit-answered field is never sent to the LLM to be
    # extracted, and never reaches the ask payload.
    kit_answers = _brand_kit_field_answers(state, manifest)
    missing = [f for f in mandatory
               if not str(answers.get(f["field_id"], "") or "").strip()
               and not str(kit_answers.get(f["field_id"], "") or "").strip()]

    # Extraction pass: values the user already stated in the prompt/brief are
    # captured as answers instead of being asked back at them.
    if missing:
        try:
            fields_json = json.dumps(
                [{"field_id": f["field_id"], "label": f.get("label", "")} for f in missing]
            )
            resp, model_used = invoke_with_fallback([{
                "role": "user",
                "content": MANIFEST_FIELD_EXTRACTOR_PROMPT.format(
                    fields_json=fields_json,
                    raw_prompt=state["prompt"],
                    clarified_intent=state.get("clarified_intent", ""),
                ),
            }], temperature=0, role="classify_extract")
            cost += calculateCost(model_used, resp.usage_metadata['total_tokens'])
            extracted = parseJson(resp.content.strip())
            for f in missing:
                v = extracted.get(f["field_id"])
                if isinstance(v, (str, int, float)) and str(v).strip() and str(v).strip().lower() not in ("null", "none"):
                    answers[f["field_id"]] = str(v).strip()
        except Exception as e:
            pass
        missing = [f for f in missing if f["field_id"] not in answers]

    # Anything the extraction pass could not resolve becomes the ask payload,
    # shaped here (while extra_specs is in scope) and handed to
    # manifestFieldsAsk through state.
    ask = [{"field_id": f["field_id"], "label": f.get("label", ""),
            "question": f.get("question", ""), "example": f.get("example", "")}
           for f in missing]
    ask_assets = [{"asset_key": k, "tier": extra_specs[k].get("tier", ""),
                   "reason": extra_specs[k].get("reason", "")}
                  for k in missing_assets]

    if ask or ask_assets:
        return Command(goto="manifestFieldsAsk", update={
            **state,
            "cost": cost,
            # Extraction results are committed BEFORE the interrupt node runs,
            # so a resume never re-derives them.
            "field_answers": answers,
            "asset_prefs": prefs,
            "manifest_pending_fields": ask,
            "manifest_pending_assets": ask_assets,
        })

    # Nothing left to ask — finish the brief here and skip the ask node.
    return _manifestFieldsResult(state, manifest, answers, prefs, cost, "", kit_answers)


@profile
def manifestFieldsAsk(state: GraphicDesigner) -> Command:
    """Stage 2.5b: blocking, skippable. The single ``manifest_fields_request``
    interrupt that collects whatever :func:`manifestFieldsGate` could not
    resolve from the launch payload or the extraction pass.

    Reached only when there is genuinely something to ask. Everything above the
    ``interrupt()`` is a state read, so the resume replay of this node costs
    nothing — the extraction call that produced the ask payload lives upstream
    and is already checkpointed.
    """
    print("[manifestFieldsAsk] asking for the manifest fields we could not resolve")
    manifest = state.get("frontend_manifest") or {}
    answers = dict(state.get("field_answers") or {})
    prefs = dict(state.get("asset_prefs") or {})
    cost = state.get("cost", 0.0)
    missing = list(state.get("manifest_pending_fields") or [])
    ask_assets = list(state.get("manifest_pending_assets") or [])
    missing_assets = [a["asset_key"] for a in ask_assets if a.get("asset_key")]

    # Defensive: an empty payload means there is nothing to ask (stale state or
    # a direct entry) -- never fire a blank interrupt at the user.
    if not missing and not missing_assets:
        return _manifestFieldsResult(state, manifest, answers, prefs, cost, "",
                                     _brand_kit_field_answers(state, manifest))

    extra_brief = ""
    question_bits = [q["question"] or MANIFEST_FIELD_FALLBACK.format(label=q['label']) for q in missing]
    question_bits += [a["reason"] or MANIFEST_FIELD_FALLBACK.format(label=a['asset_key'].replace('_', ' '))
                      for a in ask_assets]
    r = parseJson(interrupt({
        "type": "text_input",
        "skippable": True,
        # Plain-text fallback for renderers that only show 'question'.
        "question": " ".join(question_bits),
        "fields": missing,
        "assets": ask_assets,
        "cost": cost,
    }))
    cost = 0.0
    # Asset replies: {"assets": {asset_key: <src>|"skipped"}}. Anything not
    # answered is marked skipped for the life of the project (same contract
    # as photo/logo at launch).
    got_assets = r.get("assets") if isinstance(r.get("assets"), dict) else {}
    for k in missing_assets:
        src = (got_assets or {}).get(k)
        if isinstance(src, str) and src and src.lower() != "skipped":
            prefs[k] = {"status": "provided", "asset": src}
        else:
            prefs[k] = {"status": "skipped"}
    got = r.get("field_answers") if isinstance(r.get("field_answers"), dict) else {}
    for f in missing:
        v = (got or {}).get(f["field_id"])
        if isinstance(v, (str, int, float)) and str(v).strip():
            answers[f["field_id"]] = str(v).strip()
    # Free-text reply (old frontend / user typed an answer): extract once
    # against the reply; anything it doesn't resolve still reaches the
    # brief verbatim below, so no user input is lost.
    free = str(r.get("response") or "").strip()
    still = [f for f in missing if f["field_id"] not in answers]
    if free and free.lower() not in ("skip", "skipped", "no") and still:
        try:
            fields_json = json.dumps(
                [{"field_id": f["field_id"], "label": f.get("label", "")} for f in still]
            )
            resp2, model_used2 = invoke_with_fallback([{
                "role": "user",
                "content": MANIFEST_FIELD_EXTRACTOR_PROMPT.format(
                    fields_json=fields_json, raw_prompt=free, clarified_intent="",
                ),
            }], temperature=0)
            cost += calculateCost(model_used2, resp2.usage_metadata['total_tokens'])
            extracted2 = parseJson(resp2.content.strip())
            for f in still:
                v = extracted2.get(f["field_id"])
                if isinstance(v, (str, int, float)) and str(v).strip() and str(v).strip().lower() not in ("null", "none"):
                    answers[f["field_id"]] = str(v).strip()
        except Exception as e:
            pass
        if any(f["field_id"] not in answers for f in missing):
            extra_brief = free
    # Whatever is still unanswered was declined -- remember the skip for
    # the life of the project so it is never asked again.
    for f in missing:
        answers.setdefault(f["field_id"], "skipped")

    return _manifestFieldsResult(state, manifest, answers, prefs, cost, extra_brief,
                                 _brand_kit_field_answers(state, manifest))


@profile
def photoRequiredGate(state: GraphicDesigner) -> Command:
    """Stage 3: blocking, skippable. No-op pass-through unless the layout's
    ``shared_rules.real_photo_required`` is set and no photo is attached.

    Asset memory: the outcome is recorded in ``asset_prefs['photo']``. A skip
    suppresses the ask for the LIFE OF THE PROJECT (never re-ask a skipped
    recommended asset); an asset never yet asked ('unset') is gathered
    conversationally when a layout needs it. A provided photo is silently
    reused in any generation.

    When the combo has a frontend_manifest, its per-goal
    ``recommended_assets.product_imagery.tier`` is the authority on whether a
    real photo must be asked for ('required' asks; 'recommended'/'optional'
    don't — e.g. App Download Promo works with AI imagery even though a layout
    may set real_photo_required). shared_rules is the fallback when no
    manifest exists.
    """
    print("[photoRequiredGate] checking whether a real photo is required")
    cost = state.get("cost", 0.0)
    layout_profile = state.get("layout_profile") or {}
    shared_rules = layout_profile.get("shared_rules", {})
    prefs = dict(state.get("asset_prefs") or {})
    photo_pref = prefs.get("photo") or {}

    # A product image the user actually provided (at launch via asset_prefs.photo,
    # or earlier this project) must be USED — even when the layout only marks
    # product imagery 'recommended'/'optional' rather than strictly required.
    # Per-turn input frequently wipes attached_images (see generateMoreManager),
    # so if asset memory says a photo was provided but the live state lost it,
    # re-attach it here BEFORE the tier check. Without this a 'recommended'-tier
    # layout (e.g. Flash Sale) silently falls through to AI/template imagery and
    # the user's real product is dropped.
    state_out = state
    if (not state.get("attached_images")
            and photo_pref.get("status") == "provided" and photo_pref.get("asset")):
        state_out = {**state, "attached_images": [photo_pref["asset"]], "photo_mode": "as_is"}
    has_photo = bool(state_out.get("attached_images"))

    manifest_photo = ((state.get("frontend_manifest") or {}).get("recommended_assets") or {}).get("product_imagery") or {}
    if manifest_photo.get("tier"):
        photo_required = manifest_photo["tier"] == "required"
        ask_reason = manifest_photo.get("reason") or shared_rules.get(
            "image_requirement_reason", PHOTO_REQUIRED_DEFAULT)
    else:
        photo_required = shared_rules.get("real_photo_required", False)
        ask_reason = shared_rules.get("image_requirement_reason", PHOTO_REQUIRED_DEFAULT)

    if not (photo_required and not has_photo):
        return Command(goto="clarificationGate", update=state_out)

    if photo_pref.get("status") == "skipped":
        return Command(goto="clarificationGate", update=state_out)

    r = parseJson(interrupt({
        "type": "text_input",
        "skippable": True,
        "question": ask_reason,
        "cost": cost,
    }))
    cost = 0.0
    update = {**state}
    provided_src = ""
    for key in ("photo", "photo_url", "current_image_url"):
        if r.get(key):
            provided_src = r[key]
            break
    else:
        imgs = r.get("attached_images") or []
        if imgs:
            provided_src = imgs[0]

    if provided_src:
        update["attached_images"] = [provided_src] + list(state.get("attached_images") or [])
        update["photo_mode"] = "as_is"
        prefs["photo"] = {"status": "provided", "asset": provided_src}
    else:
        # user skipped -- proceed with no photo, AI-generates that zone;
        # remember the skip so we never ask again in this project
        prefs["photo"] = {"status": "skipped"}
    update["asset_prefs"] = prefs
    update["cost"] = cost

    return Command(goto="clarificationGate", update=update)


@profile
def clarificationGate(state: GraphicDesigner) -> Command:
    """Stages 4-6: blocking, not skippable. No-op pass-through unless the
    router (or either hard gate in ``intentRouter``) flagged genuine ambiguity."""
    print("[clarificationGate] handling clarification when the router flagged ambiguity")
    cost = state.get("cost", 0.0)
    router_res = state.get("router_flags") or {}
    clarified = state.get("clarify_result") or {}

    if not (router_res.get("routing_action") == "CLARIFY_FIRST" or router_res.get("requires_clarification")):
        return Command(goto="imageAnalyzerLayoutSelector", update=state)

    # Manifest-driven sessions ask ONLY the manifest's mandatory fields
    # (manifestFieldsGate). The router/clarifier analysed the RAW prompt
    # before those answers existed, so its LLM-generated questions largely
    # re-ask what was just collected — never interrupt for them. Anything
    # genuinely unanswered stays protected by the anti-fabrication directive
    # instead of a second round of questions.
    if (state.get("frontend_manifest") or {}).get("mandatory_fields"):
        qs = router_res.get("clarification_questions", [])
        directive = ""
        if qs:
            directive = (
                "\n\nANTI-FABRICATION HARD RULE: the following details were not provided by the "
                "user: " + " | ".join(qs) + ". Do NOT invent a specific value, claim, number, "
                "date, name, address, or award/certification to fill any of these gaps. Omit the "
                "detail entirely, or use general non-specific language, instead of guessing."
            )
        return Command(goto="imageAnalyzerLayoutSelector",
                       update={**state, "fabrication_directive": directive})

    r = parseJson(interrupt({
        "type": "text_input",
        "skippable": False,
        "question": router_res.get("clarification_questions", []),
        "reason": router_res.get("image_notes", ""),
        "clarity_score": clarified.get("clarity_score", 50),
        "cost": cost,
    }))
    cost = 0.0

    update = {**state}
    fabrication_directive = ""
    updated_prompt = r.get("updated_prompt") or r.get("response")
    if updated_prompt:
        update["prompt"] = updated_prompt
        design_goal = state.get("design_goal") or "Festive Sale"
        platform = state.get("platform_canonical") or "Instagram Square Post"
        layout_profile = state.get("layout_profile") or {}
        variant = state.get("selected_variant_dict") or {}
        # Re-derive clarified_intent from the UPDATED prompt -- without this the
        # CoT would still see the stale "MISSING" brief even though the user
        # just answered, making the whole clarification round pointless.
        reclarified = clarify_intent(
            raw_prompt=updated_prompt, design_goal=design_goal, platform=platform,
            layout=variant.get("name", ""),
            required_fields=layout_profile.get("shared_rules", {}).get("required_content_fields"),
            role="classify_extract",
        )
        usage = reclarified.get("_usage", {})
        total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if total:
            try:
                cost += calculateCost("gemini-3-flash-preview", total)
            except Exception:
                pass
        update["clarify_result"] = reclarified
        update["clarified_intent"] = reclarified.get("clarified_text", "")
    else:
        # Re-check AFTER the answer -- asking is not enough on its own. Covers
        # whatever ended up in clarification_questions regardless of which
        # gate put it there (router's own judgment, structural conflict, or
        # the fabrication-prone missing-fields gate).
        _asked = router_res.get("clarification_questions", [])
        if _asked:
            fabrication_directive = (
                "\n\nANTI-FABRICATION HARD RULE: the user was explicitly asked the following and did "
                "not answer: " + " | ".join(_asked) + ". Do NOT invent a specific value, claim, "
                "number, date, name, address, or award/certification to fill any of these gaps, and "
                "do NOT invent new supporting facts to make a resolution sound more specific or "
                "authoritative than what was actually provided. Omit the detail entirely, or use "
                "general non-specific language, instead of guessing."
            )

    for key in ("photo", "photo_url", "current_image_url"):
        if r.get(key):
            update["attached_images"] = [r[key]] + list(state.get("attached_images") or [])
            update["photo_mode"] = "as_is"
            break

    update["cost"] = cost
    update["fabrication_directive"] = fabrication_directive
    return Command(goto="imageAnalyzerLayoutSelector", update=update)


@profile
def imageAnalyzerLayoutSelector(state: GraphicDesigner) -> Command:
    """Stage 7: resolve how a supplied photo is to be used.

    Wireframe matching is gone with the wireframes. v33 chooses its own
    composition from ``layout_guidance.json``'s menu, so there is no variant to
    shape-match a photo against, no photo zone to compute a merge directive
    for, and no logo zone to pick a logo-friendly variant by -- the whole
    ``select_wireframe`` / ``dominant_photo_zone`` / ``resolve_merge_directive``
    pass has no consumer left.

    What survives is the one decision downstream still reads: whether an
    attached photo is the subject (``as_is``) or just a reference
    (``reference``). :func:`_v32_reference_images` turns that into v33's
    ``product`` reference role, and the freeform bypass turns it into
    :func:`_photo_integration_instruction`.
    """
    print("[imageAnalyzerLayoutSelector] resolving how the supplied photo is used")
    cotProgress("imageAnalyzerLayoutSelector")
    router_res = state.get("router_flags") or {}

    attached = state.get("attached_images") or []
    has_photo = bool(attached)
    photo_mode = state.get("photo_mode") or ("as_is" if has_photo else "reference")

    # The user's own instruction outranks the default: "use this as a
    # reference" turns a supplied photo from the subject into an influence.
    if photo_mode == "as_is" and router_res.get("detected_photo_usage") == "USE_AS_REFERENCE":
        photo_mode = "reference"

    return Command(goto="logoDecisionGate", update={
        **state,
        "photo_mode": photo_mode,
        # Consumed by the retired wireframe matcher only; cleared so no stale
        # value from an older checkpoint reaches anything.
        "merge_directive": "",
        "photo_zone_fullcanvas": False,
        "force_new_variant": False,
    })


# Conventional fallback logo spot (top-center band) for the rare layout that
# offers no explicit logo zone yet the user supplied a logo that must still be
# placed — compositeLogo would otherwise default to the whole canvas.
_DEFAULT_LOGO_ZONE_PCT = {"x_pct": [38, 62], "y_pct": [3, 12]}


def _resolve_logo_zone_pct(variant: dict) -> dict:
    """Where to composite a SUPPLIED logo in ``strict_paste`` mode.

    Always the conventional top-center band now. Per-variant logo zones came
    from the wireframe layout profile, which is gone: ``layout_guidance.json``
    deliberately carries no zone geometry (INTEGRATION.md §6), and v33 decides
    placement itself. ``v32Generation`` tells v33 in words to keep this
    position clear (see ``_logo_zone_position_description``), so the paste and
    the render agree on one fixed spot rather than a negotiated rectangle.

    The parameter is kept so the stale-checkpoint case still works: a variant
    dict left over from before the wireframes were removed still has its zone
    honoured rather than being silently relocated mid-session.
    """
    lz = variant_logo_zone(variant) if variant else None
    if lz and lz.get("x_pct") and lz.get("y_pct"):
        return {"x_pct": lz.get("x_pct"), "y_pct": lz.get("y_pct")}
    return dict(_DEFAULT_LOGO_ZONE_PCT)


_LOGO_TERM_RE = re.compile(
    r"\b(?:logos?|logotypes?|wordmarks?|brand\s?marks?|brand\s+icons?)\b", re.I)
"""What counts as the user naming a logo."""

_LOGO_NEG_BEFORE_RE = re.compile(
    r"\b(?:no|not|without|skip|skipping|omit|omitting|exclude|excluding|avoid|avoiding|"
    r"remove|removing|delete|deleting|drop|dropping|hide|hiding|don'?t|do\s+not|"
    r"never|leave\s+out)\b[^,;.!?]{0,40}$", re.I)
"""Negation sitting just before the logo word in the same clause."""

_LOGO_NEG_AFTER_RE = re.compile(
    r"^[^,;.!?]{0,25}?\b(?:is\s+not|isn'?t|are\s+not|aren'?t|not)\s+"
    r"(?:needed|required|necessary|wanted)\b", re.I)
"""Negation trailing the logo word ("logo is not needed")."""


def _logo_request_text(prompt: str) -> str:
    """The user's own words asking for a logo, normalised — or ``""``.

    Clause-by-clause so a negation in one part of a sentence ("don't touch the
    background, add our logo") cannot cancel a request in another. A clause
    that names a logo counts as a request unless it is negated ("no logo",
    "skip the logo", "the logo is not needed").

    The text itself is returned, not just a flag, so the gate can tell a NEW
    request apart from the same prompt riding along unchanged through a
    regeneration -- see ``explicit_decline`` in :func:`logoDecisionGate`.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return ""
    for clause in re.split(r"[,;.!?\n]", prompt):
        for m in _LOGO_TERM_RE.finditer(clause):
            if _LOGO_NEG_BEFORE_RE.search(clause[:m.start()]):
                continue
            if _LOGO_NEG_AFTER_RE.search(clause[m.end():]):
                continue
            return " ".join(prompt.lower().split())
    return ""




def _brand_kit_logo(state: GraphicDesigner) -> str:
    """The attached brand kit's primary logo as a data URI, or ``""``.

    ``brand_dna['logo']`` holds URLs already checked to resolve (see
    ``fetch_brand_dna``); the first one is the primary.
    """
    urls = (state.get("brand_dna") or {}).get("logo") or []
    if not urls:
        return ""
    try:
        return (convert_urls_to_b64([urls[0]]) or [""])[0] or ""
    except Exception:
        return ""


def _logo_asset_request_payload(cost: float) -> dict:
    """The interrupt for "you asked for a logo, now give me the file".

    Deliberately NOT a yes/no question -- the user already said they want one,
    so the only thing left is the asset. Uses the same structured contract
    :func:`manifestFieldsAsk` uses for every other missing asset ('assets'
    spec out, ``{"assets": {"logo": <src>}}`` back), so the frontend renders
    its normal upload control instead of a question card.
    """
    return {
        "type": "logo_ask",
        "skippable": True,
        # Plain-text fallback for renderers that only show 'question'.
        "question": LOGO_ASK_REASON,
        "assets": [{"asset_key": "logo", "tier": "required", "reason": LOGO_ASK_REASON}],
        "logo_requested": True,
        "cost": cost,
    }


def _logo_asset_from_reply(r: dict) -> str:
    """Pull a logo out of an interrupt reply, as a base-64 data URI.

    Reads the structured asset reply first, then every shape the logo gate has
    historically accepted, so an older frontend keeps working. Returns ``""``
    when the user skipped or sent nothing usable.
    """
    candidates = []
    got_assets = r.get("assets") if isinstance(r.get("assets"), dict) else {}
    for key in ("logo", "brand_logo"):
        src = got_assets.get(key)
        if isinstance(src, str) and src and src.strip().lower() != "skipped":
            candidates.append(src)
    candidates += [r[key] for key in ("logo", "logo_url", "current_image_url") if r.get(key)]
    candidates.extend(r.get("attached_images") or [])
    for src in candidates:
        try:
            b64 = src if src.startswith("data:") else (convert_urls_to_b64([src]) or [None])[0]
            if b64:
                return b64
        except Exception:
            pass
    return ""


def _afterLogoGate(state: GraphicDesigner) -> str:
    """Where logoDecisionGate hands off. Normally straight to the
    ``"generation"`` node (v33: reasoning + render in one call); on the
    freeform bypass there is no layout and no reasoning at all, so it goes
    straight to :func:`runGptImageModel` (see :func:`_freeformNext`)."""
    if state.get("is_freeform"):
        return "runGptImageModel"
    return "generation"


def _freeformNext(state: GraphicDesigner) -> str:
    """Where a freeform request goes instead of the layout/CoT chain.

    Straight to image generation -- unless the prompt itself asks for a logo.
    Freeform skips every asset gate, so "make me a poster with our logo" used
    to reach the image model as words alone and come back with an invented
    brand mark. When a logo is asked for, logoDecisionGate runs first to
    collect the real file and routes back here through :func:`_afterLogoGate`.

    Only an explicit request diverts: a freeform prompt that never mentions a
    logo behaves exactly as before, gate-free.

    Goes straight to :func:`runGptImageModel`, not the ``"generation"`` node --
    that name now means "v33, with a known catalog goal x platform pair",
    which a freeform/custom-goal request by definition does not have.
    """
    return "logoDecisionGate" if _logo_request_text(state.get("prompt", "")) else "runGptImageModel"


@profile
def logoDecisionGate(state: GraphicDesigner) -> Command:
    """Stage 9: no-op unless the selected layout has a logo zone and no logo
    has already been resolved (a brand-kit logo counts as already resolved,
    matching the behavior of skipping the ask when one is on file) -- or the
    user asked for a logo in the prompt itself.

    Asset memory: the outcome is recorded in ``asset_prefs['logo']``. A decline
    suppresses the ask for the LIFE OF THE PROJECT (never re-ask a skipped
    recommended asset); if never asked ('unset') it is gathered conversationally
    when a layout has a logo zone. A previously provided logo is silently
    reused in any generation.

    An EXPLICIT request ("put our logo top-right", see
    :func:`_logo_request_text`) is the one thing that outranks all of that:
    it asks even when the layout has no logo zone, even when the manifest tier
    is below 'required', and even when ``asset_prefs['logo']`` is already
    'skipped' -- because that skip is usually just an un-uploaded logo at
    project creation, where the user declined nothing. Declining THAT ask
    stores the request text in ``asset_prefs['logo']['explicit_decline']``, so
    the same prompt carried into a regeneration does not nag while a freshly
    typed request later does ask again.

    That request-driven ask is shaped as an ASSET REQUEST rather than the
    yes/no question below it -- the user already answered "do you want a
    logo", so the only thing left to collect is the file. It carries the same
    ``assets`` spec :func:`manifestFieldsAsk` uses for every other missing
    asset and reads the same ``{"assets": {"logo": <src>}}`` reply, so the
    frontend renders its normal upload control.
    """
    print("[logoDecisionGate] deciding whether to include the brand logo")
    cost = state.get("cost", 0.0)
    variant = state.get("selected_variant_dict") or {}
    logo_asset = state.get("logo_asset") or ""
    logo_insertion_mode = state.get("logo_insertion_mode") or "strict_paste"

    prefs = dict(state.get("asset_prefs") or {})
    logo_pref = prefs.get("logo") or {}
    # Resolve a logo the user actually PROVIDED (launch logo_asset, or one stored
    # earlier this project) BEFORE the "does this layout have a logo zone?" test.
    # A supplied logo must be used even when the layout's design-time
    # asset_requirements.logo.present is False (parallel to the supplied-product-
    # photo fix) — otherwise provided logos are silently dropped.
    if not logo_asset and logo_pref.get("status") == "provided" and logo_pref.get("asset"):
        src = logo_pref["asset"]
        try:
            logo_asset = src if src.startswith("data:") else (convert_urls_to_b64([src]) or [""])[0]
        except Exception as e:
            pass

    # The user asking for a logo IN WORDS is a first-class signal: it outranks
    # the layout having no logo zone, a manifest tier that would not normally
    # ask, and a project-level skip (which is what an un-uploaded logo at
    # project creation leaves behind -- the user never actually declined
    # anything there). Without this, "put our logo top-right" was silently
    # dropped whenever no logo file happened to be attached.
    logo_request = _logo_request_text(state.get("prompt", ""))

    # Whether this combination wants a logo at all. Was
    # ``logo_recommended(variant)`` -- read off the selected wireframe's logo
    # zone, which no longer exists. The manifest's own asset tier says the same
    # thing without one, and it is what the ask policy below already keys on.
    manifest_assets = (state.get("frontend_manifest") or {}).get("recommended_assets") or {}
    logo_tier = ((manifest_assets.get("logo") or manifest_assets.get("brand_logo") or {}).get("tier"))
    wants_logo = logo_tier in ("required", "recommended")

    # Nothing supplied, this combo doesn't call for one, and nobody asked ->
    # this design genuinely has no logo.
    if not logo_asset and not wants_logo and not logo_request:
        return Command(goto=_afterLogoGate(state), update={**state, "logo_mode": "none"})

    # Only-ask-mandatory policy: interrupt for a logo only when none was supplied
    # and the combo's manifest marks it 'required'. 'recommended'/'optional'
    # tiers never ask. No manifest (legacy/on-the-fly combos) keeps the
    # conversational ask.
    may_ask_logo = logo_tier is None or logo_tier == "required"

    if logo_asset:
        should_ask = False  # already supplied -- use it, never ask
    elif logo_request:
        # Ask unless we already put this exact request to them and they passed.
        # Storing the request TEXT (not just a flag) is what separates "the
        # same prompt rode along into a regeneration" -- don't nag -- from the
        # user typing a fresh logo request later on, which deserves a new ask.
        should_ask = logo_request != (logo_pref.get("explicit_decline") or "")
    else:
        should_ask = logo_pref.get("status") != "skipped" and may_ask_logo

    if should_ask:
        if logo_request:
            payload = _logo_asset_request_payload(cost)
        else:
            payload = {
                "type": "logo_ask",
                "skippable": True,
                "question": LOGO_ASK,
                "cost": cost,
            }
        r = parseJson(interrupt(payload))
        cost = 0.0
        choice = str(r.get("response", "")).strip().lower()
        # An explicit request IS the yes: that reply carries the file, so an
        # upload with no "yes" anywhere in it must still count.
        want_logo = (choice in ("yes", "y", "true") or r.get("want_logo") is True
                     or bool(logo_request))
        if want_logo:
            logo_asset = _logo_asset_from_reply(r) or logo_asset
        if logo_asset:
            prefs["logo"] = {"status": "provided", "asset": logo_asset}
        else:
            prefs["logo"] = {"status": "skipped"}
            if logo_request:
                prefs["logo"]["explicit_decline"] = logo_request

    logo_mode = "omitted"
    logo_directive = ""
    logo_zone_pct = {}
    if logo_asset:
        logo_mode = "provided"
        logo_zone_pct = _resolve_logo_zone_pct(variant)
        if logo_insertion_mode != "strict_paste":
            # single_call: the logo goes to v33 as a `logo` reference image
            # (see _v32_reference_images), whose own role rules already say to
            # reproduce it exactly as supplied. The directive is the text
            # version of that for the freeform bypass, which sends no refs.
            logo_directive = build_logo_directive(variant) if variant else (
                "BRAND LOGO — SUPPLIED, INSERT FAITHFULLY. Reproduce it exactly "
                "as supplied: same shape, colors, and proportions. Do not "
                "redraw, restyle, recolor, or reinterpret it."
            )
    # The zone-stripping that used to run here is gone with the wireframes:
    # there is no variant whose logo zone could reserve a blank box for a logo
    # that never arrives. v33 is told in words instead -- either to keep the
    # paste position clear (strict_paste, see v32Generation) or not to invent a
    # mark at all (logo_mode 'omitted', see logoDecisionGate's freeform branch
    # and _logo_integration_instruction).

    update = {
        **state,
        "selected_variant_dict": variant,
        "logo_mode": logo_mode,
        "logo_asset": logo_asset if logo_mode == "provided" else "",
        "logo_directive": logo_directive,
        "logo_zone_pct": logo_zone_pct,
        "asset_prefs": prefs,
        "cost": cost,
    }

    # Freeform has no layout, no CoT and therefore no routing_instruction --
    # the raw prompt IS the fused prompt. The equivalent guidance has to ride
    # on the prompt itself, or the model letters its own brand mark from the
    # words the user typed.
    if state.get("is_freeform") and logo_request:
        if logo_mode == "provided":
            update["fused_prompt"] = (state.get("fused_prompt") or state["prompt"]) + (
                "\n\nLOGO AREA — LEAVE CLEAN: the real brand logo is placed into the top-center "
                "area afterwards. Do NOT draw any logo, brand mark, lettering, box, or placeholder "
                "there; leave that area as clean, uncluttered background."
            )
        else:
            update["fused_prompt"] = (state.get("fused_prompt") or state["prompt"]) + (
                "\n\nNO LOGO FILE: the request mentions a logo but none was supplied. Do NOT "
                "invent, draw, letter, or approximate any brand mark, wordmark, emblem, or "
                "monogram; use that space as part of the composition."
            )

    return Command(goto=_afterLogoGate(state), update=update)
def _get_filtered_templates(state: GraphicDesigner) -> list[str]:
    """Filter templates based on aspect ratio."""
    brand_images = state.get("brand_dna", {}).get('templates', []) if state.get("brand_dna") else []
    if not brand_images:
        return []
        
    aspect_ratio = state.get('aspect_ratio', '1:1')
    
    # Square templates are identified by having 'square' in the URL/filename
    if aspect_ratio in ['1:1', '3:4', '4:3']: # "1:1 or similar"
        filtered = [img for img in brand_images if 'square' in img]
        return filtered if filtered else brand_images
    else: # "other aspect ratios that are not similar like 1:1 or 3:4"
        filtered = [img for img in brand_images if 'square' not in img]
        return filtered if filtered else brand_images

def _attach_palette_image(state: GraphicDesigner, attached_images: list | None) -> list | None:
    brand_dna = state.get('brand_dna')
    if brand_dna and 'color_palettes' in brand_dna:
        palette_b64 = generate_palette_reference_image(brand_dna['color_palettes'])
        if palette_b64:
            new_images = attached_images.copy() if attached_images is not None else []
            new_images.append(palette_b64)
            return new_images
    return attached_images


def _attach_extra_assets(state: GraphicDesigner, attached_images: list | None) -> list | None:
    """Append the combo-specific manifest assets (state['extra_assets'], e.g. a
    QR code or background texture the manifest asked for) as extra reference
    images for the model call. Appended AFTER attached_images[0] so the primary
    as-is photo keeps its meaning for zone analysis and integration prompts."""
    extras = state.get('extra_assets') or {}
    if not extras:
        return attached_images
    new_images = attached_images.copy() if attached_images is not None else []
    for src in extras.values():
        if not (isinstance(src, str) and src):
            continue
        if not src.startswith("data:"):
            try:
                src = (convert_urls_to_b64([src]) or [""])[0]
            except Exception as e:
                continue
        if src:
            new_images.append(src)
    return new_images or attached_images


_ANTI_ARTIFACT_GUARD = (
    "[CRITICAL: Do not render any structural guide artifacts -- no dashed/dotted "
    "borders, no zone label text (e.g. 'HERO ZONE'), no percentage/coordinate text, "
    "no crosshatch fill patterns, no grey placeholder rectangles. Render a finished, "
    "polished, production-quality graphic.]\n\n"
)


def _final_generation_prompt(state: GraphicDesigner, negative_native: bool = False) -> str:
    """Assemble the text actually sent to the vendor: fused prompt + anti-artifact
    guard (skipped for the scopeGate freeform bypass, since no CoT/wireframe
    reasoning ran) + negative_prompt folded as text unless the caller passes it
    natively (Imagen / Stable Diffusion)."""
    base = state.get('fused_prompt') or state['prompt']
    text = base if state.get('is_freeform') else _ANTI_ARTIFACT_GUARD + base
    if not _user_asked_cta(state):
        # House rule: no call to action unless the user asked for one. Image
        # models add a "SHOP NOW" button by habit, so it has to be said.
        text += ("\n\nNO CALL TO ACTION: do not add any button, pill or tab, and no "
                 "call-to-action text such as 'Shop now', 'Buy now', 'Order now' or 'Learn more'.")
    if not negative_native and state.get('negative_prompt'):
        text += "\n\nSTRICTLY AVOID in the output image: " + state['negative_prompt']
    return text


def _photo_integration_instruction(state: GraphicDesigner) -> str:
    """photo_mode branching for the freeform renderer's flat text prompt."""
    if not state.get('attached_images'):
        return ""
    photo_mode = state.get('photo_mode')
    merge_line = f"\nMERGE DIRECTIVE: {state['merge_directive']}" if state.get('merge_directive') else ""
    if photo_mode == "as_is" and state.get('photo_zone_fullcanvas'):
        return ("\n\n[SUPPLIED HERO PHOTO — FULL-BLEED BACKGROUND, SEAMLESS INTEGRATION: "
                "fill the ENTIRE canvas edge-to-edge with it; keep its MAIN SUBJECT faithful "
                "and recognizable, extend/outpaint its own background to fill the canvas; "
                "no separate new background, no box/frame/letterbox." + merge_line + "]")
    elif photo_mode == "as_is":
        return ("\n\n[SUPPLIED HERO PHOTO — SUBJECT-FAITHFUL, SEAMLESS INTEGRATION: "
                "keep its MAIN SUBJECT faithful and recognizable; integrate it seamlessly, "
                "fill its zone edge-to-edge, feather/grade its edges into adjacent areas, "
                "harmonize lighting and color with the surrounding design." + merge_line + "]")
    elif photo_mode == "reference":
        return "\n\n[REFERENCE PHOTO — hero/product visual for contextual integration]"
    return ""


def _logo_integration_instruction(state: GraphicDesigner) -> str:
    """Instruction for folding a supplied brand logo into the same generation call
    (single_call mode only -- strict_paste inserts the logo as a post-processing
    step instead, see logoCompositor)."""
    if state.get('logo_mode') != 'provided' or state.get('logo_insertion_mode') != 'single_call':
        return ""
    directive = state.get('logo_directive') or (
        "BRAND LOGO — SUPPLIED, INSERT FAITHFULLY into its designated logo zone. "
        "Reproduce it exactly as supplied: same shape, colors, and proportions. "
        "Do not redraw, restyle, recolor, or reinterpret it."
    )
    return f"\n\n[{directive}]"


def _label_references(state: GraphicDesigner, attached_images: list | None, logo_appended: bool = False) -> list | None:
    """Bind per-image role labels to the two references the image model tends to
    confuse -- the product hero photo (index 0) and the brand logo (appended
    last for single_call) -- so nano-banana stops guessing which uploaded image
    is which (the cause of the random "logo becomes wallpaper, product ignored"
    failures).

    Labeled items become ``{"label", "data"}`` dicts; every other reference
    (brand templates, palette, extra manifest assets) is left as a plain
    data-URI string, so their behavior is unchanged. Both forms are understood
    by :func:`services._build_image_gen_parts`.
    """
    if not attached_images:
        return attached_images
    labeled = list(attached_images)
    # Product hero photo sits at index 0 only when the user supplied one.
    # _photo_integration_instruction returns "" when no photo was supplied, so a
    # brand template sitting at index 0 is correctly left unlabeled.
    photo_label = _photo_integration_instruction(state).strip()
    if photo_label and state.get('attached_images'):
        labeled[0] = {"label": photo_label, "data": labeled[0]}
    # Brand logo is the last-appended reference in single_call mode. Guard on
    # logo_appended so a trailing palette/extra asset is never mislabeled.
    if logo_appended:
        logo_label = _logo_integration_instruction(state).strip()
        if logo_label:
            last = labeled[-1]
            data = last["data"] if isinstance(last, dict) else last
            labeled[-1] = {"label": logo_label, "data": data}
    return labeled

@profile
def runGptImageModel(state: GraphicDesigner) -> GraphicDesigner:
    """Generate images using OpenAI's configured image model (``gpt-image-2.5-flare``
    by default -- see :func:`_gptImageApiModel` / the ``image`` role in
    :data:`config.MODEL_ROLES`).

    Invokes :func:`gptImageGen` with the enhanced prompt, model name, and any
    attached reference images (product photo, brand templates, palette,
    single_call logo) -- gathered the same way as the Nano-Banana variants,
    so gpt-image actually conditions on them via OpenAI's ``images.edit``
    instead of ignoring them. Supports single and multiple image generation.
    The resulting base-64 image data is decoded, uploaded to S3, and the
    public URL(s) are stored in ``state['current_image_url']`` (pipe-delimited
    for multiples).

    :param state: The current graph state with ``prompt``, ``model``,
        ``num_images``, and optionally ``attached_images``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``logoCompositor`` with the
        generated image URL(s) and updated cost.
    :rtype: Command
    :raises Exception: If the OpenAI image API call or S3 upload fails.
    """
    print("[runGptImageModel] generating the image with GPT Image")
    cost = state.get('cost', 0.0)
    try:
        brand_images = _get_filtered_templates(state)
        if brand_images:
            b64_images = convert_urls_to_b64(brand_images)
            attached_images = state.get('attached_images', []) + b64_images
        else:
            attached_images = state.get('attached_images', None)

        attached_images = _attach_extra_assets(state, _attach_palette_image(state, attached_images))

        # single_call logo: fold the supplied logo into THIS call as a reference.
        logo_appended = bool(
            state.get('logo_mode') == 'provided'
            and state.get('logo_insertion_mode') == 'single_call'
            and state.get('logo_asset')
        )
        if logo_appended:
            attached_images = (attached_images or []) + [state['logo_asset']]

        if attached_images is not None and len(attached_images) == 0:
            attached_images = None

        generation_prompt = (
            _final_generation_prompt(state)
            + _photo_integration_instruction(state)
            + _logo_integration_instruction(state)
        )
        # Always the configured model: a stale frontend id ("gpt-image-1", ...)
        # must not reach OpenAI or the price table.
        api_model = _gptImageApiModel()
        print(f"[runGptImageModel] using model: {api_model}, prompt: {generation_prompt}, aspect_ratio: {state['aspect_ratio']}, num_images: {state['num_images']}, references: {len(attached_images) if attached_images else 0}")
        response = gptImageGen(generation_prompt, api_model, state['aspect_ratio'], state['num_images'], attached_images)

        if state["num_images"] == 1:
            image_data = base64.b64decode(response[0])
            image_file = BytesIO(image_data)
            file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
            s3_client.upload_fileobj(
                tagged(image_file, state, "generated", api_model), DOCUMENT_BUCKET, file_name, ExtraArgs={
                    "ContentType": "image/webp", 'ACL': 'public-read'}
            )
            file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
            cost += calculateCost(api_model, 1000)  # approximate token count for image data
        elif state["num_images"] > 1:
            file_url = ''
            for b64 in response:
                image_data = base64.b64decode(b64)
                image_file = BytesIO(image_data)
                file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
                s3_client.upload_fileobj(
                    tagged(image_file, state, "generated", api_model), DOCUMENT_BUCKET, file_name, ExtraArgs={
                        "ContentType": "image/webp", 'ACL': 'public-read'}
                )
                file_url += f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}|"
                cost += calculateCost(api_model, 1000)  # approximate token count for image data


        return Command(goto="logoCompositor", update={**state, "current_image_url": file_url, "previous_image_url": state['current_image_url'], "base_prompt": state['prompt'], 'cost': cost,
                       "image_models": _with_image_models(state, file_url.split("|"), api_model)})

    except Exception as e:
        # Was a hardcoded goto that bypassed tried_models: the next node
        # could route straight back here forever. Route through the
        # chain so every attempt is recorded exactly once.
        return _generation_fallback(state, "gpt-image", str(e))

@profile
def _enforce_target_size(state: GraphicDesigner) -> GraphicDesigner:
    """Disabled: no longer crops or resizes the generated image.

    This used to center-crop to ``target_size``'s aspect ratio (cutting into
    real content near the edges) and LANCZOS-resize to the exact pixel size.
    Per product decision, generated images are never cropped or stretched
    after the fact -- the model's own output is used as-is, even if its pixel
    dimensions don't exactly match ``target_size``. Kept as a no-op rather
    than removed so the ``logoCompositor`` call site and docstrings referring
    to it stay valid.
    """
    return state


@profile
def logoCompositor(state: GraphicDesigner) -> Command:
    """Insert a supplied logo into the just-generated image, when needed.

    Logo pass is a no-op unless ``logo_mode == 'provided'`` and
    ``logo_insertion_mode == 'strict_paste'`` (the ``single_call`` mode already
    folded the logo into the main generation call, so there's nothing to do
    here for it). Downloads the generated image, pastes the logo into
    ``logo_zone_pct`` via :func:`compositeLogo`, and re-uploads the result.
    The ``_enforce_target_size`` call afterwards is a no-op (generated images
    are never cropped or stretched to fit a target pixel size).

    :param state: The current graph state with ``current_image_url``,
        ``logo_mode``, ``logo_insertion_mode``, ``logo_asset``, and
        ``logo_zone_pct``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``interruptBeforeFidelitySuggestion``.
    :rtype: Command
    """
    print("[logoCompositor] compositing the logo and enforcing target size")
    update = {**state}
    if state.get('logo_mode') == 'provided' and state.get('logo_insertion_mode') == 'strict_paste':
        try:
            current_image_url = state['current_image_url']
            img_response = makeApiRequest("GET", current_image_url, timeout=30)
            composited_bytes = compositeLogo(img_response.content, state['logo_asset'], state.get('logo_zone_pct') or {})

            file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
            # The pasted logo changes no wording, so the generated file's copy
            # deck carries over -- the critique checks the rendered text against it.
            s3_client.upload_fileobj(
                tagged(BytesIO(composited_bytes), state, "logo-composited",
                       result=carried(read_embedded(img_response.content))), DOCUMENT_BUCKET, file_name, ExtraArgs={
                    "ContentType": "image/webp", 'ACL': 'public-read'}
            )
            update["current_image_url"] = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
            # Pasting the logo renders nothing new: the model that made the
            # design is still the one that made this copy.
            update["image_models"] = _with_image_models(
                state, update["current_image_url"], (state.get("image_models") or {}).get(current_image_url))
        except Exception as e:
            pass

    print(f"[logoCompositor] image URL before enforcing target size: {update.get('current_image_url')}")
    try:
        update = _enforce_target_size(update)
    except Exception as e:
        pass
    print(f"[logoCompositor] image URL after enforcing target size: {update.get('current_image_url')}")

    final_url = update.get("current_image_url")
    if final_url:
        update["last_generated_image_url"] = final_url
        # A composited/resized copy is still this lineage's graphic -- keep
        # "create more" on it varying the stored brief, not re-reading it.
        urls = state.get("v32_lineage_urls")
        if urls is not None and state.get("current_image_url") in urls and final_url not in urls:
            update["v32_lineage_urls"] = list(urls) + [final_url]
        # The pasted logo moves no text, so the typeset layers still describe
        # the copy; they now belong to the composited file.
        layers = state.get("text_layers") or {}
        if layers.get("image_url") == state.get("current_image_url") and final_url != layers.get("image_url"):
            update["text_layers"] = {**layers, "image_url": final_url,
                                     "layers": {**(layers.get("layers") or {}), "image": final_url}}

    return Command(goto="interruptBeforeFidelitySuggestion", update=update)

def _postGenerationUpdate(state: GraphicDesigner, user_response: dict, selected_url: str) -> dict:
    """State carried out of the generation_completed reply into the next flow."""
    return {**state, "model": user_response.get('model', state.get('model', 'nano-banana')),
    "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '1:1')), "style": user_response.get('style', state.get('style', '')), "current_image_url": selected_url, "previous_image_url": state['previous_image_url'], "base_prompt": state['prompt']}


def _autoSuggestionUpdate(state: GraphicDesigner, user_response: dict, selected_url: str) -> dict:
    """State for an automatic critique of a freshly produced graphic.

    Used when generation_completed hands the new graphic to the Critic Agent.
    """
    return {**_postGenerationUpdate(state, user_response, selected_url), **_AUTO_SUGGESTION}


# Starts an automatic critique chain on a freshly produced graphic.
_AUTO_SUGGESTION = {
    "suggestion_type": "auto",
    # Each automatic critique picks its own target: this design may have
    # gained another graphic (a "create more") since the last one.
    "suggestion_target_confirmed": False,
    # Brand-new critique chain on a freshly generated image -- suggestionMoreAsk's
    # own "yes" branch is the only other place this gets written, so a fresh
    # generation always starts with a full 2-ask budget.
    "suggestion_more_round": 0,
}


@profile
def slopManager(state: GraphicDesigner) -> Command:
    """Enter the AI slop remover subgraph.

    Reached from the generation_completed offer (``slop_origin`` already
    ``'post_generation'``) or from the navigator's ``slop_removal`` route.
    The slop subgraph is mounted directly as a node on ``graph`` like the
    layering one, so it resumes where it paused; its static edge leads to
    :func:`slopResultRouter`.
    """
    print("[slopManager] entering the AI slop remover")
    cost = state.get('cost', 0.0)
    image_url = state.get("current_image_url") or state.get("last_generated_image_url") or ""
    if not image_url:
        parsed = parseJson(interrupt({
            "type": "text_input",
            "question": NO_IMAGE_FOR_SLOP_REMOVAL,
            "cost": cost,
        }))
        return Command(goto="navigator", update={
            **state,
            "prompt": parsed.get("response", state.get("prompt", "")),
            "current_image_url": parsed.get("current_image_url", ""),
            "cost": 0.0,
        })
    return Command(goto="slopSubgraph", update={
        "name": "slop_subgraph",
        "slop_origin": state.get("slop_origin") or "navigator",
        "slop_verdict": "",
        "current_image_url": image_url,
        "cost": cost,
    })


@profile
def slopResultRouter(state: GraphicDesigner) -> Command:
    """Route out of the (now-completed) slop subgraph.

    Its output is already merged into the live graph state. ``slop_verdict``:

    * ``create_more`` / ``new_request`` -- the reply goes to the navigator
      as-is, on the cleaned (or re-selected) image.
    * ``continue`` (or anything unexpected) -- the cleaned image goes on to
      the Critic Agent, the same flow a declined offer takes.
    """
    verdict = state.get("slop_verdict") or "continue"
    update = {"name": "main_graph", "slop_origin": "", "slop_verdict": ""}
    print(f"[slopResultRouter] verdict={verdict}")
    if verdict in ("create_more", "new_request"):
        return Command(goto="navigator", update={**state, **update})
    return Command(goto="suggestionManager", update={**state, **_AUTO_SUGGESTION, **update})


@profile
def interruptBeforeFidelitySuggestion(state: GraphicDesigner) -> GraphicDesigner:
    """Present generated images to the user and await their next action.

    Sends an interrupt with the current and previous image URLs, giving the
    user a chance to review the generated images.  The user's response is
    parsed and the graph routes to ``navigator`` with the updated prompt,
    conversation history, model, and aspect ratio.

    :param state: The current graph state with ``current_image_url`` and
        ``previous_image_url``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` with the user's response.
    :rtype: Command
    """
    print("[interruptBeforeFidelitySuggestion] presenting the generated image to the user")
    slop_offer = settings.SLOP_REMOVAL_ENABLED
    frame = {
        "type": "generation_completed",
        "question": SLOP_REMOVAL_OFFER if slop_offer else GENERATION_COMPLETED,
        "cost": state.get('cost', 0.0),
        "prompt": state['enhanced_prompt'],
        "enhanced_prompt": state.get('prompt', ''),
        "current_image_url": state.get('current_image_url', ''),
    }
    if slop_offer:
        # Tells the frontend this frame asks a question: it must not
        # auto-reply "yes" to it as it does to start the critique.
        frame["offer"] = "slop_removal"
    layers = state.get("text_layers") or {}
    if layers.get("layers") and layers.get("image_url") == frame["current_image_url"]:
        # v33 typeset this graphic's text in real fonts: the editor can recreate
        # each element as editable text (schema kognie.text_layers.v2).
        frame["text_layers"] = layers["layers"]
        if layers.get("notices"):
            # A font the user named that Kognie does not have yet, and what was
            # used instead. Said before the question so it is not missed.
            frame["typography_notices"] = layers["notices"]
            frame["question"] = " ".join(layers["notices"]) + "\n\n" + frame["question"]
    user_response = interrupt(frame)

    cost = 0.0
    user_response = parseJson(user_response)
    # Same reasoning as interruptBeforeLayering: honour the image selected on
    # the canvas at reply time, not the one this run happened to produce.
    _selected_url = user_response.get('current_image_url') or state.get('current_image_url', '')
    reply = user_response.get('response', '')

    if slop_offer:
        verdict, cost = judge_slop_offer(reply, state.get('conversation_history', []))
        if verdict is None:
            # Decisions API down: a plain "yes" still accepts; anything else
            # goes through the free-text handling below.
            verdict = "accept" if reply.strip().lower() == "yes" else "other"
        if verdict == "accept":
            return Command(goto="slopManager", update={**_postGenerationUpdate(state, user_response, _selected_url), 'cost': cost,
            "slop_origin": "post_generation",
            "conversation_history": [*state.get('conversation_history', []), {"role": "system", "content": SLOP_REMOVAL_OFFER}, {"role": "user", "content": reply}]})
        if verdict == "decline":
            return Command(goto="suggestionManager", update={**_autoSuggestionUpdate(state, user_response, _selected_url), 'cost': cost})
    elif reply.lower() == "yes":
        return Command(goto="suggestionManager", update={**_autoSuggestionUpdate(state, user_response, _selected_url), 'cost': cost})

    # Per-process navigator: vague dissatisfaction with the generated graphic
    # goes to the global criticismHandler instead of being taken as a literal
    # next prompt (which used to send "it feels cheap" straight to navigator).
    is_criticism, detect_cost = detectCriticism(state, reply, 'generation')
    if is_criticism:
        return Command(goto="criticismHandler", update={**state, "criticism_origin": "generation",
        "criticism_text": reply, "base_prompt": state['prompt'], 'cost': cost + detect_cost,
        "current_image_url": _selected_url,
        "conversation_history": [*state.get('conversation_history', []), {"role": "user", "content": reply}]})

    # Same attachment forwarding as the edit/expand handlers: an image attached
    # to THIS reply is what "add this to the design" refers to on the next turn.
    _reply_refs = _reply_attachments(user_response)
    return Command(goto="navigator", update={**state, "model": user_response.get('model', state.get('model', 'nano-banana')),
    "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '1:1')), "style": user_response.get('style', state.get('style', '')), "current_image_url": _selected_url, "previous_image_url": state['previous_image_url'], "base_prompt": state['prompt'], 'cost': cost, "prompt": user_response.get('response', ''), "attached_images": _reply_refs or state.get('attached_images', []), "edit_reference_images": _reply_refs, "conversation_history": [*state.get('conversation_history', []), {"role": "user", "content": user_response.get("response", "")}]})


@profile
def generation(state: GraphicDesigner) -> GraphicDesigner:
    """Route to the appropriate image-generation executor based on the model.

    Acts as a conditional-edge resolver that reads ``state['model']`` and
    returns a routing key to select the correct ``run*Model`` node:

    * ``'dall-e-2'`` or ``'dall-e-3'`` → ``'dall-e'``
    * ``'flux-pro-1.1'`` → ``'flux'``
    * ``'nano-banana'`` → ``'nanoBanana'``
    * ``'nano-banana-pro'`` → ``'nanoBananaPro'``
    * ``'nano-banana-2'`` → ``'nanoBanana2'``
    * ``'gpt-image*'`` → ``'gpt-image'``
    * ``'stable-diffusion'`` → ``'stableDiff'``

    :param state: The current graph state with ``model`` set.
    :type state: GraphicDesigner
    :returns: A routing string used by the ``StateGraph`` to select the
        generation executor node.
    :rtype: str
    """
    print("[generation] routing to the selected model executor")
    if 'dall-e' in state['model']:
        return "dall-e"
    elif 'sd' in state['model']:
        return "stable-diffusion"
    elif 'flux' in state['model']:
        return "flux"
    elif 'nano-banana-pro' in state['model']:
        return "nano-banana-pro"
    elif 'nano-banana-2' in state['model']:
        return "nano-banana-2"
    elif 'nano-banana' in state['model']:
        return "nano-banana"
    elif 'gpt' in state['model']:
        return "gpt-image"


def _selected_style(state: GraphicDesigner) -> str:
    """The art style the user picked for this project, or ``""``.

    Carried per message the same way ``model`` and ``aspect_ratio`` are (every
    interrupt handler echoes it back), so changing the picker mid-session takes
    effect on the next operation.
    """
    style = state.get("style")
    return style.strip() if isinstance(style, str) and style.strip() else ""


def _style_directive(state: GraphicDesigner, scope: str) -> str:
    """Instruction text applying the selected style, or ``""`` when none is set.

    Two scopes, because the same style means different things to the two
    operations:

    * ``"edit"`` -- the image already exists and the user asked for one
      change. New or modified content adopts the style; the rest of the design
      is left alone. Restyling the whole image off the back of "make the
      background darker" would be a much bigger edit than was asked for.
    * ``"regenerate"`` -- the whole image is being produced again, so the
      style applies to all of it. This one has to say so explicitly: the
      regenerate prompt otherwise instructs the model to reproduce the
      previous version's style EXACTLY, which would silently ignore a style
      the user just changed.
    * ``"regenerate_new_photo"`` -- a regenerate in which the user also
      supplied a new photo. Same as ``"regenerate"`` except it must not say
      "keep the previous version's subject unchanged": the subject is the one
      thing that IS changing, and telling the model otherwise reintroduces the
      contradiction that makes it ignore the attachment.
    """
    style = _selected_style(state)
    if not style:
        return ""
    # A preset id ("bauhaus") on its own is a label, not direction -- the
    # generation path expands the same selection into all nine of its axes
    # (style_presets.apply_style), so name the style and say what it looks
    # like here too, or an edit drifts off the style the design was made in.
    preset = style_presets.find_preset(style)
    if preset is not None:
        style = f"{preset.name} — {preset.to_spec()}"
    # Same precedence the generation block states (style_presets.priority_block):
    # the request's own instruction first, then the brand's assets, then the
    # style. Without it an edit will happily recolour a logo to match a palette.
    tail = (" The instruction in this request outranks the style: where the two "
            "disagree, follow the instruction.")
    if _style_brand_kit(state):
        tail += (" The attached brand outranks the style as well — keep its "
                 "colours, fonts, wording, logo and product photography exactly "
                 "as they are, and never recolour, redraw or texture them to "
                 "match the style.")
    if scope == "edit":
        return (f"\n\nSELECTED STYLE — \"{style}\": anything you add or change in this edit must be "
                f"rendered in that style so it sits naturally in the design. Do NOT restyle parts of "
                f"the image the instruction does not touch.{tail}")
    if scope == "regenerate_new_photo":
        return (f"\n\nSELECTED STYLE — \"{style}\": render the whole design in that style, including "
                f"how the newly supplied photo is presented. Where this conflicts with reproducing the "
                f"previous version's look, the selected style WINS — match it, and keep the previous "
                f"version's wording and logo unchanged. The newly supplied photo stays the subject; "
                f"style how it is presented, but keep the product/subject itself recognisable.{tail}")
    return (f"\n\nSELECTED STYLE — \"{style}\": render the whole design in that style. Where this "
            f"conflicts with reproducing the previous version's look, the selected style WINS — match "
            f"it, and keep the previous version's content (subject, wording, logo) unchanged.{tail}")


# ── Style switching from the edit flow ("change the style to bauhaus") ──────
# The style word on its own is not enough to act on: "change the style of the
# headline to bold" is a typography edit, not a restyle, so a request only
# counts as a switch when it names the whole design's style AND a target. Every
# request that does not clearly match falls through to editIntentClassifier,
# which already has a Global Style Transfer class for it.
_SWITCH_VERB = (r"(?:change|switch|convert|turn|make|set|update|redo|re-?do|"
                r"re-?style|restyle|apply|give|render|want|need|would\s+like)")
_STYLE_WORD = (r"(?:style|styling|aesthetic|art[\s-]?direction|"
               r"look(?:\s+and\s+feel)?|vibe|feel)")
# A design element in the span means the style belongs to that element, not to
# the design: scoped edits must keep their normal classify -> validate path.
_ELEMENT_NOUN = re.compile(
    r"\b(text|words?|wording|headline|title|subtitle|sub-?heading|caption|copy|"
    r"font|typeface|lettering|logo|background|backdrop|border|button|badge|"
    r"price|label|icon|image|photo|picture|product|person|model|frame|shadow)\b",
    re.I)
# "a bit more minimal" is a degree, not a different style: the user wants less
# clutter in the design they have, not the Minimal preset applied to all of it.
_DEGREE = re.compile(r"\b(more|less|bit|slightly|little|somewhat|extra|very)\b", re.I)
_SPAN = r"[^.?!;\n]"
_CONNECTOR = r"(?:\b(?:to|into|as|like|in)\b)?"
# "change the style to bauhaus" / "change the style into something bauhaus"
_SWITCH_STYLE_FIRST = re.compile(
    rf"\b{_SWITCH_VERB}\b(?P<mid>{_SPAN}{{0,40}}?)\b{_STYLE_WORD}\b"
    rf"(?P<mid2>{_SPAN}{{0,30}}?)\b(?:to|into|as|like)\b(?P<target>{_SPAN}{{1,60}})",
    re.I)
# "switch to swiss style", "apply the brutalist aesthetic" (connector optional)
_SWITCH_TARGET_FIRST = re.compile(
    rf"\b{_SWITCH_VERB}\b(?P<mid>{_SPAN}{{0,30}}?){_CONNECTOR}"
    rf"(?P<target>{_SPAN}{{1,40}}?)\b{_STYLE_WORD}\b", re.I)
# "make it bauhaus" -- no style word at all, so this one is trusted only when
# the target resolves to a preset we actually have (see _style_switch_target).
_SWITCH_BARE = re.compile(
    rf"\b{_SWITCH_VERB}\b(?P<mid>(?:{_SPAN}{{0,30}}?)){_CONNECTOR}"
    rf"\s*(?P<target>{_SPAN}{{1,40}})", re.I)
_TARGET_FILLER = re.compile(
    rf"^(?:a|an|the|some|something|more|full|proper|pure|all|everything|it|this|"
    rf"that|whole|entire|design|image|poster|graphic|please)\b"
    rf"|\b(?:please|thanks|thank\s+you|now|instead|throughout|everywhere|"
    rf"print|poster|design|image|graphic|one|{_STYLE_WORD})$", re.I)


_TARGET_TAIL = re.compile(
    r"\b(?:but|and|while|whilst|though|although|except|however|only|keeping|"
    r"keep|without|with|plus|also|then)\b|[,;]", re.I)


def _clean_style_target(target: str) -> str:
    """The style name inside a matched span, or ``""`` when it is not one.

    Rejects a span naming a design element ("the headline text") and anything
    longer than a style name ever is: a long tail means the regex swallowed
    the rest of a sentence, and acting on that would restyle a design the user
    only asked a small change of.
    """
    # A switch is often asked for with a condition attached ("...to bauhaus but
    # keep the red headline"). Only the style name is read here; the condition
    # still reaches the model, verbatim and ranked first, through
    # style_presets.restyle_prompt.
    target = _TARGET_TAIL.split(re.sub(r"[\"'`]", " ", target or ""), 1)[0]
    target = target.strip(" ,-–—:")
    for _ in range(4):
        stripped = _TARGET_FILLER.sub("", target).strip(" ,-–—:")
        if stripped == target:
            break
        target = stripped
    if not target or _ELEMENT_NOUN.search(target) or len(target.split()) > 5:
        return ""
    return target


def _style_switch_target(prompt: str) -> str:
    """The style this edit asks to switch the whole design to, or ``""``.

    Deterministic on purpose: it decides whether an edit skips the classifier,
    so a borderline request has to come out empty rather than guess. The bare
    form ("make it bauhaus") is accepted only when the target is a style this
    service has, because nothing else in that sentence says it is about style
    at all -- "change it to blue" must stay a normal edit.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return ""
    for pattern in (_SWITCH_STYLE_FIRST, _SWITCH_TARGET_FIRST):
        m = pattern.search(prompt)
        if not m:
            continue
        scope = (m.group("mid") or "") + (m.groupdict().get("mid2") or "")
        if _ELEMENT_NOUN.search(scope):
            continue
        target = _clean_style_target(m.group("target"))
        if target:
            return target
    m = _SWITCH_BARE.search(prompt)
    if m:
        scope = m.group("mid") or ""
        target = m.group("target")
        if not _ELEMENT_NOUN.search(scope) and not _DEGREE.search(scope + " " + target):
            target = _clean_style_target(target)
            if target and (style_presets.find_preset(target)
                           or style_presets.find_in_text(target)):
                return target
    return ""


def editStyleSwitch(state: GraphicDesigner) -> Command:
    """Turn "change the style to bauhaus" into a real restyle, or pass through.

    Sits between :func:`editAssetGate` and :func:`editIntentClassifier` and is
    a pure pass-through unless the request asks for the whole design's style
    to change (:func:`_style_switch_target`). Three outcomes:

    * **not a style switch** -> ``editIntentClassifier``, flow unchanged.
    * **a style this service has** (``styles.json``) -> the project's ``style``
      becomes that preset and the prompt is rebuilt as a full restyle
      instruction (``style_presets.restyle_prompt``): the same nine axes, the
      same BRAND LOCK and the same user > brand > style precedence a
      generation gets. Then straight to the executors -- the classifier would
      only label it Class 6 and the validation subgraph has no entities to
      validate in "make it bauhaus".
    * **a style this service does not have** ("vaporwave") -> the user's own
      words go to the executors untouched. There is no preset to expand, and
      inventing axes for a name we do not know would be worse direction than
      the name itself.

    Both switch outcomes set ``edit_style_switch``, which stops
    :func:`_edit_prompt_guidance` appending the project style on top: in the
    known case the prompt already carries it, and in the unknown case the old
    style is exactly what the user asked to leave behind.
    """
    target = (_style_switch_target(state.get("prompt", ""))
              if settings.STYLE_DIRECTION_ENABLED else "")
    if not target:
        return Command(goto="editIntentClassifier",
                       update={**state, "edit_style_switch": False})

    preset = (style_presets.find_preset(target)
              or style_presets.find_in_text(target))
    if preset is None:
        # The project's stored style no longer describes the image, so it is
        # cleared rather than re-asserted on the next edit. The frontend
        # re-sends its picker value with the next message either way.
        print(f"[editStyleSwitch] {target!r} is not in styles.json -- "
              f"sending the request to the executors as written")
        return Command(goto="aspectRatioHandler",
                       update={**state, "style": "", "edit_style_switch": True})

    print(f"[editStyleSwitch] restyling to {preset.id!r} ({preset.name})")
    return Command(goto="aspectRatioHandler", update={
        **state,
        "style": preset.id,
        "prompt": style_presets.restyle_prompt(
            preset.id, brand_kit=_style_brand_kit(state),
            request=state.get("prompt", "")),
        "edit_style_switch": True,
    })


def _edit_prompt_guidance(prompt: str, state: GraphicDesigner) -> str:
    """Everything appended to an edit's prompt before it reaches a model.

    Applied at the executors rather than at prompt-construction time, so it
    covers both the class-templated path and the Class-0 outlier path, and
    every model in the fallback chain gets identical text.

    * Logo: 'omitted' is the important case -- the user asked for a logo, we
      could not get the file, and the words alone are all the model has, so it
      letters an invented brand mark unless told not to. 'provided' gets the
      mirror instruction: place the supplied file, never letter the request's
      words ("kognie logo") as text.
    * Style: the project's selected art style, scoped to what the edit
      touches (see :func:`_style_directive`) -- unless this edit IS the style
      change (``edit_style_switch``), in which case the prompt already carries
      its own style instruction and a second, narrower one would contradict it.
    """
    if (state.get("edit_logo_mode") or "") == "omitted":
        prompt += (
            "\n\nNO LOGO FILE: the request mentions a logo but none was supplied. Do NOT invent, "
            "draw, letter, or approximate any brand mark, wordmark, emblem, or monogram. Make the "
            "rest of the requested change and leave that space as part of the composition."
        )
    elif (state.get("edit_logo_mode") or "") == "provided":
        prompt += (
            "\n\nLOGO FILE SUPPLIED: wherever this request mentions a logo, it means the supplied "
            "logo image. Place that image itself. Do NOT write the words of the request (such as "
            "\"logo\" or the brand name) as text on the design, and do not letter your own wordmark."
        )
    if state.get("edit_style_switch"):
        return prompt
    return prompt + _style_directive(state, "edit")


_IMAGE_ORDINALS = ("second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth")


def _inline_image_part(data_uri: str) -> dict:
    """One Gemini ``inline_data`` part from a ``data:<mime>;base64,<...>`` URI."""
    return {"inline_data": {"mime_type": data_uri.split(":")[1].split(";")[0],
                            "data": data_uri.split(",")[1]}}


def _is_data_uri(value) -> bool:
    return isinstance(value, str) and value.startswith("data:") and "," in value


def _edit_extra_image_parts(state: GraphicDesigner) -> list:
    """Gemini ``parts`` for every image riding along with an edit besides the
    canvas, or ``[]``.

    Two kinds, in this order:

    * images the user attached WITH this request (``edit_reference_images``),
      whose actual content is to be worked into the design -- "add this
      product to the poster" only means anything if the model can see the
      product;
    * a supplied brand logo (``edit_logo_mode == 'provided'``).

    Each is introduced by its own text part naming its position in the call,
    because the first image is always the design being edited and the model
    has no other way to tell canvas from reference.
    """
    extras = [("reference", src) for src in (state.get("edit_reference_images") or [])
              if _is_data_uri(src)]
    logo = state.get("logo_asset") or ""
    if (state.get("edit_logo_mode") or "") == "provided" and _is_data_uri(logo):
        extras.append(("logo", logo))
    if not extras:
        return []

    parts = [{"text": "IMAGE ROLES: the FIRST image is the design being edited -- it is the canvas, "
                      "and the edit is applied to it. Every image after it is a REFERENCE supplied "
                      "by the user, never the canvas and never something to edit."}]
    for i, (kind, uri) in enumerate(extras):
        ordinal = _IMAGE_ORDINALS[i] if i < len(_IMAGE_ORDINALS) else f"number {i + 2}"
        if kind == "logo":
            label = (f"REFERENCE IMAGE ({ordinal} image): the user's REAL brand logo. Insert it into "
                     "the design exactly where the instruction asks, preserving its exact shape, "
                     "colours, lettering and proportions. Do not redraw, restyle, recolour, or letter "
                     "your own version of it, and do not add any other logo.")
        else:
            label = (f"REFERENCE IMAGE ({ordinal} image): an image the user attached with this "
                     "request. When the instruction refers to \"this\", \"this image\", \"the "
                     "attached image\" or the thing it depicts, it means THIS image. Take its ACTUAL "
                     "content -- the real subject, its shape, colours, markings, text and proportions "
                     "-- and place it into the design as the instruction asks, matching the canvas's "
                     "lighting, perspective and scale so it sits naturally. Do NOT invent your own "
                     "version of what it depicts, and do not use it as the canvas or as a style "
                     "reference unless the instruction says so.")
        parts.append({"text": label})
        parts.append(_inline_image_part(uri))
    return parts


def _reply_attachments(user_response: dict) -> list:
    """Images that came in with THIS reply, in the shapes frontends send them.

    Interrupt handlers pass the result into both ``attached_images`` (the
    project's reference set) and ``edit_reference_images`` (just this turn's),
    so an edit can tell "the image you just gave me" apart from "the product
    photo from project launch".
    """
    if not isinstance(user_response, dict):
        return []
    found = [src for src in (user_response.get("attached_images") or [])
             if isinstance(src, str) and src.strip()]
    for key in ("image", "image_url", "reference_image"):
        src = user_response.get(key)
        if isinstance(src, str) and src.strip() and src not in found:
            found.append(src)
    return found


def _warn_dropped_edit_references(state: GraphicDesigner, model: str) -> None:
    """Say out loud when a fallback model cannot carry this edit's references.

    The Gemini and gpt-image editors take more than one image. Reaching
    flux-kontext-pro means the attached reference (and any supplied
    logo) is not in the call at all, and the edit is text-only -- worth a line
    in the log, because the output will look like the attachment was ignored.
    """
    dropped = len(state.get("edit_reference_images") or [])
    if (state.get("edit_logo_mode") or "") == "provided" and state.get("logo_asset"):
        dropped += 1
    if dropped:
        print(f"⚠️ [{model}] cannot take reference images — {dropped} attached image(s) "
              f"dropped from this edit; falling back to a text-only edit.")


def _resolve_edit_references(state: GraphicDesigner) -> list:
    """This turn's attached images, as data URIs the executors can inline.

    ``edit_reference_images`` is written by whichever interrupt handler took
    the user's reply, and holds ONLY what came in with that message. The
    conversion happens here rather than in the executors so the download is
    paid once per edit instead of once per model in the fallback chain.
    """
    incoming = [src for src in (state.get("edit_reference_images") or [])
                if isinstance(src, str) and src.strip()]
    if not incoming:
        return []
    resolved = []
    for src in incoming:
        if _is_data_uri(src):
            resolved.append(src)
            continue
        try:
            b64 = (convert_urls_to_b64([src]) or [""])[0]
            if b64:
                resolved.append(b64)
        except Exception as e:
            logging.getLogger("graphic_designer.editAssetGate").warning(
                "Could not load attached reference %r: %s", src[:80], e)
    print(f"[editAssetGate] {len(resolved)} attached reference image(s) for this edit")
    return resolved


@profile
def editAssetGate(state: GraphicDesigner) -> Command:
    """Resolve the images this EDIT needs before it runs: attached references,
    and a logo the request asked for.

    **References.** Images the user attached with THIS message are normalised
    to data URIs and handed to the executors as extra reference images (see
    :func:`_edit_extra_image_parts`), so "add this product to the poster"
    works on the actual product instead of the model's guess at one. Only this
    turn's attachments are used: ``attached_images`` also holds the project's
    launch photo, which must not be pasted into every later unrelated edit.

    "Add a logo" / "put our logo bottom-right" on an existing image used to go
    straight to an image model with no logo file anywhere in the call, so the
    model invented a brand mark. This gate runs on every entry into editing
    and is a pure pass-through unless the request actually names a logo:

    * a logo already on file (brand kit, launch upload, or an earlier ask this
      project) is used silently -- never re-asked;
    * otherwise the same asset-request interrupt the generation flow uses
      (:func:`_logo_asset_request_payload`) collects the file;
    * a decline is remembered in ``asset_prefs['logo']['explicit_decline']``
      keyed on the request text, so the same instruction does not ask twice
      while a fresh request later does.

    The outcome rides on ``edit_logo_mode`` ('provided' | 'omitted' | ''),
    which is rewritten on every edit so a logo collected for one edit can
    never leak into the next unrelated one.

    Hands off to :func:`editStyleSwitch`, which is itself a pass-through
    unless the request asks to change the whole design's art style.
    """
    print("[editAssetGate] resolving the images this edit needs")
    references = _resolve_edit_references(state)

    # Attachment-backed edits have an explicit canvas and references already.
    # Send the original prompt and images straight to the configured edit
    # executor chain; do not run edit classification, disclaimer checks, or
    # prompt-validation subgraph for this path.
    if state.get("attachment_edit_direct"):
        return Command(goto="aspectRatioHandler", update={
            **state,
            "edit_reference_images": references,
            # This path never reads the request for a logo, so a 'provided'
            # left by an earlier edit must not ride into this one.
            "edit_logo_mode": "",
            "attachment_edit_direct": False,
        })

    logo_request = _logo_request_text(state.get("prompt", ""))
    if not logo_request:
        return Command(goto="editStyleSwitch", update={
            **state, "edit_logo_mode": "", "edit_reference_images": references})

    cost = state.get("cost", 0.0)
    prefs = dict(state.get("asset_prefs") or {})
    logo_pref = prefs.get("logo") or {}

    logo_asset = state.get("logo_asset") or ""
    if not logo_asset and logo_pref.get("status") == "provided" and logo_pref.get("asset"):
        logo_asset = logo_pref["asset"]
    # Must end up as a data URI: the executors inline it as a second reference
    # image, and a launch-seeded logo_asset is often still a plain URL.
    if logo_asset and not logo_asset.startswith("data:"):
        try:
            logo_asset = (convert_urls_to_b64([logo_asset]) or [""])[0]
        except Exception:
            logo_asset = ""
    # The brand kit's logo counts as on file. Only intentClarifier used to
    # read it, so a thread that reached editing without a generation (or got
    # its brand attached afterwards) asked for a logo the kit already had.
    if not logo_asset:
        logo_asset = _brand_kit_logo(state)

    if not logo_asset and logo_request != (logo_pref.get("explicit_decline") or ""):
        r = parseJson(interrupt(_logo_asset_request_payload(cost)))
        cost = 0.0
        logo_asset = _logo_asset_from_reply(r)
        if logo_asset:
            prefs["logo"] = {"status": "provided", "asset": logo_asset}
        else:
            prefs["logo"] = {"status": "skipped", "explicit_decline": logo_request}

    return Command(goto="editStyleSwitch", update={
        **state,
        "asset_prefs": prefs,
        "logo_asset": logo_asset,
        "edit_logo_mode": "provided" if logo_asset else "omitted",
        "edit_reference_images": references,
        "cost": cost,
    })


@profile
def editIntentClassifier(state: GraphicDesigner) -> GraphicDesigner:
    """Classify the user's editing intent into a numbered edit category.

    Uses a multi-class LLM classifier (Gemini 2.5 Flash with GPT-5-mini
    fallback) to determine the specific type of image edit the user is
    requesting.  The supported classes are:

    * **Class 1A** — Generative Insertion (add/replace/clone objects)
    * **Class 1B** — Spatial Transformation (move/resize/rotate)
    * **Class 2** — Intentional Erasure (remove objects)
    * **Class 3** — Attribute Modification (change colour/material/texture)
    * **Class 4** — Subject Semantic Modification (pose/expression/outfit)
    * **Class 5** — Environment Replacement (change background)
    * **Class 6** — Global Style Transfer (apply artistic style)
    * **Class 7** — Typography & Graphics (text/logo editing)
    * **Class 8** — Canvas Geometry (crop/zoom/pan)
    * **Class 9** — Quality Enhancement (sharpen/upscale/denoise)
    * **Class 10** — Ambience & Mood Modification (lighting/atmosphere)
    * **Class 0** — Outlier (does not match any editing class)

    Based on the classification result, routes to either
    ``knownCategoryPromptHandler`` (for known classes) or a direct editing
    executor (for outlier/class 0). Two disclaimer checks run in parallel with
    the classifier: a brand-conflict check (:func:`_edit_brand_conflict`, only
    when a brand is attached) and an executability check
    (:func:`_edit_executability`). A flagged edit detours through
    ``editBrandWarning`` and/or ``editExecutabilityWarning`` first, then
    continues the same way.

    :param state: The current graph state with ``prompt``,
        ``current_image_url``, and ``conversation_history``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to the appropriate downstream node
        with the classified ``intent`` stored in state.
    :rtype: Command
    :raises Exception: If LLM invocation or response parsing fails.
    """
    print("[editIntentClassifier] classifying the user's edit intent")
    try:
        cost = state.get('cost', 0.0)
        intent_prompt = EDIT_INTENT_CLASSIFIER_PROMPT.format(prompt=state['prompt'])
        # The disclaimer checks read only the prompt (and the brand), so they
        # run alongside the classifier instead of adding a round trip.
        with ThreadPoolExecutor(max_workers=2) as executor:
            brand_future = executor.submit(
                contextvars.copy_context().run, _edit_brand_conflict, state)
            exec_future = executor.submit(
                contextvars.copy_context().run, _edit_executability, state)
            response, model_used = invoke_with_fallback([
                {"role": "user", "content": intent_prompt}
            ], temperature=0)
            brand_conflict, brand_cost = brand_future.result()
            not_executable, exec_cost = exec_future.result()
        cost += calculateCost(model_used, response.usage_metadata['total_tokens']) + brand_cost + exec_cost
        # print(response.content)
        intent = parseJson(response.content.strip())

        update = {"edit_intent": intent['class_id'], 'cost': cost,
                  "edit_brand_conflict": brand_conflict,
                  "edit_not_executable": not_executable}
        if brand_conflict:
            return Command(goto="editBrandWarning", update=update)
        if not_executable:
            return Command(goto="editExecutabilityWarning", update=update)
        return Command(goto=_edit_next_node(intent['class_id']), update=update)

    except Exception as e:
        raise e


def _edit_next_node(edit_intent: str) -> str:
    """Where an edit goes once classified: an outlier (class 0) skips prompt
    validation and goes straight to the executors."""
    return "aspectRatioHandler" if edit_intent == '0' else "knownCategoryPromptHandler"


def _edit_brand_conflict(state: GraphicDesigner) -> tuple:
    """Check whether this edit request contradicts the attached brand.

    Returns ``(reason, cost)``: ``reason`` is a short user-facing sentence when
    the edit clashes with the brand, ``""`` otherwise. No brand attached (or a
    kit with no identity -- only asset URLs) skips the LLM call entirely. Any
    failure answers "no conflict": this only decides whether a disclaimer is
    shown, so it must never block the edit itself.
    """
    from .critiquepipeline.grounding import brandNormalizer
    brand = brandNormalizer(state.get("brand_dna"))
    if not brand:
        return "", 0.0
    pal = brand.get("palette") or {}
    identity = {k: v for k, v in {
        "colours": [f"{c['name']} {c['hex']} ({c['role']})" for c in pal.get("named") or []],
        "personality": brand.get("personality"),
        "typography": {k: v for k, v in (brand.get("fonts") or {}).items()
                       if k in ("families", "headline", "body") and v},
        "visual_style": brand.get("visual_style"),
        "composition": brand.get("composition"),
        "constraints": {k: v for k, v in (brand.get("constraints") or {}).items() if v},
    }.items() if v}
    try:
        prompt = EDIT_BRAND_CONFLICT_PROMPT.format(
            brand_name=brand.get("name") or state.get("brand_name") or "your brand",
            brand=json.dumps(identity, ensure_ascii=False, indent=1),
            prompt=state.get("prompt", ""))
        response, model_used = invoke_with_fallback(
            [{"role": "user", "content": prompt}], temperature=0, role="classify_extract")
        cost = calculateCost(model_used, response.usage_metadata['total_tokens'])
        verdict = parseJson(response.content.strip())
        if verdict.get("conflict") is True:
            return (verdict.get("reason") or "").strip() or "It may clash with your brand.", cost
        return "", cost
    except Exception as e:
        print(f"[editIntentClassifier] brand-conflict check failed (edit continues): {e}")
        return "", 0.0


def editBrandWarning(state: GraphicDesigner) -> Command:
    """Tell the user this edit may go against their brand, then run it anyway.

    Entered from :func:`editIntentClassifier` only when
    :func:`_edit_brand_conflict` flagged the request. The
    ``brand_conflict_warning`` frame is a disclaimer, not a question: ``main.py``
    auto-continues it like ``editing_started``, so the reply is ignored and the
    edit carries on to wherever the classifier would have sent it. Reverting is
    what ``editing_successful``'s ``previous_image_url`` is for.

    Sends ``"cost": 0`` and leaves ``state['cost']`` untouched, so the
    classifier's spend is billed with the edit result like before.
    """
    print("[editBrandWarning] edit contradicts the attached brand; warning the user")
    reason = state.get("edit_brand_conflict", "")
    interrupt({
        "type": "brand_conflict_warning",
        "question": f"{EDIT_BRAND_CONFLICT_WARNING}\n\nWhy: {reason}" if reason else EDIT_BRAND_CONFLICT_WARNING,
        "reason": reason,
        "brand_name": state.get("brand_name", ""),
        "current_image_url": state.get("current_image_url", ""),
        "cost": 0,
    })
    if state.get("edit_not_executable"):
        return Command(goto="editExecutabilityWarning")
    return Command(goto=_edit_next_node(state.get("edit_intent", "")))


def _edit_executability(state: GraphicDesigner) -> tuple:
    """Check whether the image-edit model can reliably do what this edit asks.

    Returns ``(reason, cost)``: ``reason`` is a short user-facing sentence when
    the request is something AI editing handles badly (tiny or measured moves
    like "slightly left" / "10% up", exact hex/px targets, named fonts, dense
    new text, big relayouts), ``""`` otherwise. Same rubric family as the
    critic's executability gate, but the request is never dropped -- this only
    decides whether a disclaimer is shown, so any failure answers "reliable".
    """
    try:
        prompt = EDIT_EXECUTABILITY_PROMPT.format(prompt=state.get("prompt", ""))
        response, model_used = invoke_with_fallback(
            [{"role": "user", "content": prompt}], temperature=0, role="classify_extract")
        cost = calculateCost(model_used, response.usage_metadata['total_tokens'])
        verdict = parseJson(response.content.strip())
        if verdict.get("unreliable") is True:
            return (verdict.get("reason") or "").strip() or "AI edits often miss precise changes like this.", cost
        return "", cost
    except Exception as e:
        print(f"[editIntentClassifier] executability check failed (edit continues): {e}")
        return "", 0.0


def editExecutabilityWarning(state: GraphicDesigner) -> Command:
    """Tell the user this edit is likely to come out badly, then run it anyway.

    Entered from :func:`editIntentClassifier` (or after ``editBrandWarning``)
    only when :func:`_edit_executability` flagged the request. The
    ``edit_executability_warning`` frame is a disclaimer, not a question:
    ``main.py`` auto-continues it, so the edit carries on to wherever the
    classifier would have sent it. The user opts out with the normal
    generation cancel, which this frame's ``"generation"`` scope keeps armed.

    Sends ``"cost": 0`` and leaves ``state['cost']`` untouched, so the
    classifier's spend is billed with the edit result like before.
    """
    print("[editExecutabilityWarning] edit is unlikely to execute well; warning the user")
    reason = state.get("edit_not_executable", "")
    interrupt({
        "type": "edit_executability_warning",
        "question": f"{EDIT_EXECUTABILITY_WARNING}\n\nWhy: {reason}" if reason else EDIT_EXECUTABILITY_WARNING,
        "reason": reason,
        "current_image_url": state.get("current_image_url", ""),
        "cost": 0,
    })
    return Command(goto=_edit_next_node(state.get("edit_intent", "")))

@profile
def knownCategoryPromptHandler(state: GraphicDesigner) -> GraphicDesigner:
    """Run the prompt-validation subgraph for a known edit category.

    Invokes the ``validatorSubGraph`` (from
    :mod:`graphic_designer.subgraphs.validation`) to extract entities, validate
    them against the image, ask clarification questions if needed, and produce
    a refined editing prompt.  Handles four possible outcomes from the
    subgraph:

    * **new_edit** — The user introduced a new edit intent during
      clarification; routes back to ``editingManager``.
    * **layering** — The user asked to break the image into layers mid-
      clarification; routes directly to ``layeringManager``.
    * **cancel** — The user cancelled the edit; routes to ``navigator``.
    * **max_attempts_reached** — Clarification loop exhausted; routes to
      ``aspectRatioHandler`` with the original prompt.
    * **completed** — Validated prompt ready; routes to
      ``aspectRatioHandler`` with the refined prompt.

    :param state: The current graph state with ``prompt``, ``intent``,
        ``current_image_url``, and ``conversation_history``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to the appropriate downstream node
        based on the subgraph result.
    :rtype: Command
    :raises Exception: If the subgraph invocation fails.
    """
    # A supplied logo is placed from the file itself. The class templates
    # rebuild the prompt from extracted fields, and the Typography template
    # turns "add the kognie logo" into `Text String: "kognie logo"` -- which
    # the model then letters onto the design instead of placing the file.
    # The user's own words plus the labelled logo image are the right input.
    if (state.get("edit_logo_mode") or "") == "provided":
        print(f"[knownCategoryPromptHandler] logo supplied (class {state.get('edit_intent')}) "
              "-- keeping the user's prompt, skipping the template rewrite")
        return Command(goto="aspectRatioHandler")

    print("[knownCategoryPromptHandler] validating the edit prompt via the subgraph")
    try:

        cost = state.get('cost', 0.0)

        subGraph_response = valGraph.invoke({
            "original_prompt": state['prompt'],
            "prompt": state['prompt'],
            "conversation_history": state.get('conversation_history', []),
            "current_image_url": state['current_image_url'],
            "intent": state['edit_intent'],
            "name": "validation_subgraph",
            "cost": cost
        })
        
        # Check for extracted_fields with status codes
        if 'extracted_fields' in subGraph_response and isinstance(subGraph_response['extracted_fields'], dict):
            status = subGraph_response['extracted_fields'].get('status')
            
            if status == 'new_edit':
                return Command(goto='editingManager', update={"prompt": subGraph_response['extracted_fields']['prompt'], "cost": subGraph_response.get('cost', cost)})

            elif status == 'layering':
                return Command(goto='layeringManager', update={
                    "prompt": subGraph_response['extracted_fields']['prompt'],
                    "cost": subGraph_response.get('cost', cost),
                    "conversation_history": subGraph_response.get('conversation_history', state.get('conversation_history', [])),
                })
            
            elif status == 'cancel':
                user_response = interrupt({
                    "type": "text_input",
                    "question": EDIT_CANCELLED,
                    "cost": subGraph_response.get('cost', cost),
                    "current_image_url": state['current_image_url']
                })
                user_response = parseJson(user_response)
                return Command(goto='navigator', update={
                    "prompt": user_response['response'], 
                    "cost": 0.0, 
                    "model": user_response.get('model', state.get('model', 'nano-banana-pro')),
                    "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', '')),
                    "current_image_url": user_response.get('current_image_url', state['current_image_url']),
                    "conversation_history": state.get('conversation_history', []) + [{
                        "role": "assistant",
                        "content": EDIT_CANCELLED
                    }, {
                        "role": "user",
                        "content": user_response['response']
                    }]
                })
            
            elif status == 'max_attempts_reached':
                return Command(goto='aspectRatioHandler', update={
                    "prompt": subGraph_response['extracted_fields']['prompt'],
                    "cost": subGraph_response.get('cost', cost)
                })

        # Normal completion - proceed with validated prompt
        return Command(goto="aspectRatioHandler", update={**state, "prompt": subGraph_response['prompt'], "cost": subGraph_response.get('cost', cost)})

    except Exception as e:
        raise e

# Execution nodes

@profile
def bananaProExecutor(state: GraphicDesigner) -> GraphicDesigner:
    """Execute an image-editing operation using the Gemini 3 Pro Image API.

    Sends the current image (as a base-64 data URI) along with the editing
    prompt to the ``gemini-3-pro-image-preview`` endpoint.  The returned
    image is uploaded to S3 and the URL is stored as
    ``state['current_image_url']``.

    Falls back to :func:`bananaExecutor` on failure.

    :param state: The current graph state with ``b64_image_url``, ``prompt``,
        ``user_id``, and ``session_id``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``successfulEditHandler`` on success,
        or ``bananaExecutor`` on failure.
    :rtype: Command
    :raises Exception: If the API call, S3 upload, or response parsing fails.
    """

    print("[bananaProExecutor] editing the image with Nano-Banana Pro")
    cost = state.get('cost', 0.0)

    try:
        if not GEMINI_API_KEY:
            raise Exception("GEMINI_API_KEY not set")
        data_url = state['b64_image_url']

        payload = {
            "contents": [{
                "parts": [
                    {"text": _edit_prompt_guidance(state['prompt'], state)},
                    {"inline_data": {"mime_type": data_url.split(":")[1].split(";")[0],
                                     "data": data_url.split(",")[1]}}
                ] + _edit_extra_image_parts(state)
            }],
            "generationConfig": {"responseModalities": ["IMAGE"]}
        }
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-3-pro-image-preview:generateContent"
               f"?key={GEMINI_API_KEY}")

        apiLog("google", "nano-banana-pro", "bananaProExecutor", status="start")
        response = makeApiRequest('POST', url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}:", response.text)
        response_data = response.json()
        base64_data = response_data["candidates"][0]["content"]["parts"][-1]["inlineData"]["data"]
        base64_data = convert_png_to_webp_base64(base64_data)
        image_data = base64.b64decode(base64_data)
        image_file = BytesIO(image_data)
        file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
        s3_client.upload_fileobj(
            tagged(image_file, state, "edited", "nano-banana-pro"), DOCUMENT_BUCKET, file_name, ExtraArgs={
                "ContentType": "image/webp", 'ACL': 'public-read'}
        )
        file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
        _usage = response_data.get("usageMetadata", {}) or {}
        apiLog("google", "nano-banana-pro", "bananaProExecutor", status="ok",
               tokens_in=_usage.get("promptTokenCount"), tokens_out=_usage.get("candidatesTokenCount"))
        cost += calculateCost("nano-banana-pro", 1000)  # approximate token count for image data
        return Command(goto="successfulEditHandler", update={**state, "cost": cost, "previous_image_url": state['current_image_url'], "current_image_url": file_url, "base_prompt": state['prompt'], "image_models": _with_image_models(state, file_url, "nano-banana-pro")})

    except Exception as e:
        apiLog("google", "nano-banana-pro", "bananaProExecutor", status="failed", detail=str(e))
        return Command(goto="bananaExecutor", update={**state, })
    

@profile
def bananaExecutor(state: GraphicDesigner) -> GraphicDesigner:
    """Execute an image-editing operation using the Gemini 2.5 Flash Image API.

    Sends the current image (as a base-64 data URI) along with the editing
    prompt to the ``gemini-2.5-flash-image`` endpoint.  The returned image is
    uploaded to S3 and the URL is stored as ``state['current_image_url']``.

    Falls back to :func:`fluxExecutor` on failure.

    :param state: The current graph state with ``b64_image_url``, ``prompt``,
        ``user_id``, and ``session_id``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``successfulEditHandler`` on success,
        or ``fluxExecutor`` on failure.
    :rtype: Command
    :raises Exception: If the API call, S3 upload, or response parsing fails.
    """

    print("[bananaExecutor] editing the image with Nano-Banana")
    cost = state.get('cost', 0.0)

    try:
        if not GEMINI_API_KEY:
            raise Exception("GEMINI_API_KEY not set")
        data_url = state['b64_image_url']

        payload = {
            "contents": [{
                "parts": [
                    {"text": _edit_prompt_guidance(state['prompt'], state)},
                    {"inline_data": {"mime_type": data_url.split(":")[1].split(";")[0],
                                     "data": data_url.split(",")[1]}}
                ] + _edit_extra_image_parts(state)
            }],
            "generationConfig": {"responseModalities": ["IMAGE"]}
        }
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-2.5-flash-image:generateContent"
               f"?key={GEMINI_API_KEY}")

        apiLog("google", "nano-banana", "bananaExecutor", status="start")
        response = makeApiRequest('POST', url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}:", response.text)
        response_data = response.json()

        base64_data = response_data["candidates"][0]["content"]["parts"][-1]["inlineData"]["data"]
        base64_data = convert_png_to_webp_base64(base64_data)
        image_data = base64.b64decode(base64_data)
        image_file = BytesIO(image_data)
        file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
        s3_client.upload_fileobj(
            tagged(image_file, state, "edited", "nano-banana"), DOCUMENT_BUCKET, file_name, ExtraArgs={
                "ContentType": "image/webp", 'ACL': 'public-read'}
        )
        file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"

        _usage = response_data.get("usageMetadata", {}) or {}
        apiLog("google", "nano-banana", "bananaExecutor", status="ok",
               tokens_in=_usage.get("promptTokenCount"), tokens_out=_usage.get("candidatesTokenCount"))
        cost += calculateCost("nano-banana", 1000)  # approximate token count for image data
        return Command(goto="successfulEditHandler", update={**state, "cost": cost, "previous_image_url": state['current_image_url'], "current_image_url": file_url, "base_prompt": state['prompt'], "image_models": _with_image_models(state, file_url, "nano-banana")})
    except Exception as e:
        apiLog("google", "nano-banana", "bananaExecutor", status="failed", detail=str(e))
        return Command(goto="fluxExecutor", update={**state, })
    

@profile
def fluxExecutor(state: GraphicDesigner) -> GraphicDesigner:
    """Execute an image-editing operation using the Flux Kontext Pro API.

    Sends the current image and editing prompt to the Flux Kontext Pro
    endpoint.  Polls for the result until the status is ``Ready``, then
    fetches the generated image, uploads it to S3, and stores the URL.

    Falls back to :func:`extremeFallback` on failure (including moderation
    rejections).

    :param state: The current graph state with ``b64_image_url``, ``prompt``,
        ``user_id``, and ``session_id``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``successfulEditHandler`` on success,
        or ``extremeFallback`` on failure.
    :rtype: Command
    :raises Exception: If the API call fails, times out, or is moderated.
    """

    print("[fluxExecutor] editing the image with Flux Kontext Pro")
    cost = state.get('cost', 0.0)

    try:
        url = "https://api.us.bfl.ai/v1/flux-kontext-pro"
        # Flux Kontext takes exactly one input image, so neither an attached
        # reference nor a supplied logo can ride along here -- only the
        # "don't invent one" guidance can.
        _warn_dropped_edit_references(state, "flux-kontext-pro")
        prompt = _edit_prompt_guidance(state['prompt'], state)
        data = {
            "prompt": prompt,
            "input_image" : state['b64_image_url'].split(",")[1],
            "output_format": "webp",
        }
        headers={
        'accept': 'application/json',
        'x-key': FLUX_API_KEY,
        'Content-Type': 'application/json',
        }

        apiLog("bfl", "flux-kontext-pro", "fluxExecutor", status="start")
        response = makeApiRequest('POST', url,json=data, headers=headers, timeout=60)
        id = response.json()['id']
        url = "https://api.us.bfl.ai/v1/get_result"
        params = {
            "id": id
        }
        headers={
        'accept': 'application/json',
        'x-key': FLUX_API_KEY,
        }
        while True:
            response = makeApiRequest('GET', url, params=params, headers=headers, timeout=60)
            res = response.json()
            if res['status'] == "Ready":
                # Fetch the image over HTTP
                response = makeApiRequest('GET', res['result']['sample'], timeout=60)
                # Convert the raw bytes to a base64 string
                base64_image = base64.b64encode(response.content).decode('utf-8')

                image_data = base64.b64decode(base64_image)
                image_file = BytesIO(image_data)
                file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
                s3_client.upload_fileobj(
                    tagged(image_file, state, "edited", "flux-kontext-pro"), DOCUMENT_BUCKET, file_name, ExtraArgs={
                        "ContentType": "image/webp", 'ACL': 'public-read'}
                )
                file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"

                apiLog("bfl", "flux-kontext-pro", "fluxExecutor", status="ok")
                cost += calculateCost("flux-kontext-pro", 1000)  # approximate token count for image data
                return Command(goto="successfulEditHandler", update={**state, "cost": cost, "previous_image_url": state['current_image_url'], "current_image_url": file_url, "base_prompt": state['prompt'], "image_models": _with_image_models(state, file_url, "flux-kontext-pro")})

            if res['status'] == 'Pending':
                time.sleep(5)

            if res['status'] == 'Task not found' or res['status'] == 'Request Moderated' or res['status'] == 'Error' or res['status'] == 'Content moderated':
                raise Exception("Error in generating image", res)
    except Exception as e:
        apiLog("bfl", "flux-kontext-pro", "fluxExecutor", status="failed", detail=str(e))
        return Command(goto="extremeFallback", update={**state, })
    

# A canvas at least this transparent is flattened onto a matte before a
# gpt-image edit. Below it, stray transparent pixels are not worth the round trip.
_EDIT_TRANSPARENT_MIN_SHARE = 0.02
# RGB distance from the matte colour: at/below LO a result pixel is pure matte
# (fully transparent), at/above HI it is solid content, linear in between.
_EDIT_MATTE_KEY_LO = 18.0
_EDIT_MATTE_KEY_HI = 60.0


def _flatten_transparent_canvas(canvas_uri: str):
    """Flatten a transparent canvas onto a solid matte that contrasts with its
    visible content.

    gpt-image cannot see content over transparency: near-white text on a
    transparent canvas reads as nothing, and the edit endpoint also treats the
    transparent pixels as the region it may repaint, so a mostly-transparent
    design came back as an empty transparent image. A dark matte behind light
    content (and a light one behind dark content) makes the content visible.

    :returns: ``(flat_data_uri, matte_rgb, transparent_share)``, or
        ``(None, None, share)`` when the canvas is not meaningfully transparent.
    """
    img = Image.open(BytesIO(base64.b64decode(canvas_uri.split(",", 1)[1])))
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if not has_alpha:
        return None, None, 0.0
    rgba = np.asarray(img.convert("RGBA")).astype(np.float32)
    alpha = rgba[..., 3:4] / 255.0
    share = float((alpha[..., 0] < 0.98).mean())
    if share < _EDIT_TRANSPARENT_MIN_SHARE:
        return None, None, share

    visible = alpha[..., 0] > 0.5
    lum = (0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2])
    light_content = (float(lum[visible].mean()) if visible.any() else 255.0) >= 128.0
    matte = np.array([24, 24, 24] if light_content else [232, 232, 232], dtype=np.float32)

    flat = rgba[..., :3] * alpha + matte * (1.0 - alpha)
    buf = BytesIO()
    Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8), "RGB").save(buf, format="PNG")
    return ("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            matte, share)


def _key_out_matte(image_bytes: bytes, matte: np.ndarray) -> bytes:
    """Turn the matte of a flattened-then-edited image back into transparency.

    The matte colour is re-measured from the result's border first, since the
    model rarely returns it exactly. Opacity grows with each pixel's distance
    from that colour, and the colour is un-mixed from the matte so soft edges
    do not keep a dark (or light) fringe.

    :returns: PNG bytes with an alpha channel.
    """
    out = Image.open(BytesIO(image_bytes))
    rgba = np.asarray(out.convert("RGBA")).astype(np.float32)
    # A result that already carries transparency is judged on the matte, as sent.
    a_in = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * a_in + matte * (1.0 - a_in)

    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    near = border[np.linalg.norm(border - matte, axis=1) < _EDIT_MATTE_KEY_HI]
    if len(near):
        matte = np.median(near, axis=0)

    dist = np.linalg.norm(rgb - matte, axis=2)
    alpha = np.clip((dist - _EDIT_MATTE_KEY_LO) / (_EDIT_MATTE_KEY_HI - _EDIT_MATTE_KEY_LO), 0.0, 1.0)
    safe = np.maximum(alpha, 1e-3)[..., None]
    fg = np.clip(matte + (rgb - matte) / safe, 0, 255)
    fg = np.where((alpha < 0.15)[..., None], rgb, fg)

    buf = BytesIO()
    Image.fromarray(np.dstack([fg, alpha * 255.0]).astype(np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


@profile
def gptImageExecutor(state: GraphicDesigner) -> GraphicDesigner:
    """First editor in the edit chain: OpenAI's configured image model
    (``gpt-image-2.5-flare`` by default -- see :func:`_gptImageApiModel`).

    Sends the current design through ``images.edit`` with every image riding
    along with this edit -- the canvas first, then the user's attached
    references and a supplied logo, introduced by the same IMAGE ROLES text the
    Gemini editors get.

    The image goes as-is, with no ``size`` in the request and no resize of the
    result: the model returns the input's own aspect ratio.

    A transparent canvas is the one exception: it is flattened onto a
    contrasting matte first (see :func:`_flatten_transparent_canvas`), and the
    matte is keyed back out of the result, which is uploaded as a transparent PNG.

    Chain: gptImageExecutor -> bananaProExecutor -> bananaExecutor ->
    fluxExecutor -> extremeFallback.

    :param state: The current graph state with ``b64_image_url``, ``prompt``,
        ``user_id``, and ``session_id``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``successfulEditHandler`` on success,
        or ``bananaProExecutor`` on failure.
    :rtype: Command
    """

    print("[gptImageExecutor] editing the image with GPT Image")
    cost = state.get('cost', 0.0)
    # Same resolver runGptImageModel uses, so an IMAGE_MODEL change (or a
    # registry fallback) updates both generation and editing together.
    api_model = _gptImageApiModel()
    try:
        canvas_uri = state.get('b64_image_url') or ""
        if not _is_data_uri(canvas_uri):
            raise ValueError("no b64_image_url to edit (aspectRatioHandler could not load the image)")

        flat_uri, matte, transparent_share = _flatten_transparent_canvas(canvas_uri)
        if flat_uri:
            print(f"[gptImageExecutor] canvas {transparent_share:.0%} transparent -- "
                  f"flattened onto matte {tuple(int(c) for c in matte)}")
            canvas_uri = flat_uri

        extra_parts = _edit_extra_image_parts(state)
        images = [_data_uri_to_openai_file(canvas_uri, 0)] + [
            _data_uri_to_openai_file(part_uri, i + 1)
            for i, part_uri in enumerate(
                f"data:{p['inline_data']['mime_type']};base64,{p['inline_data']['data']}"
                for p in extra_parts if "inline_data" in p)
        ]
        prompt = _edit_prompt_guidance(state['prompt'], state)
        role_text = "\n\n".join(p["text"] for p in extra_parts if "text" in p)
        if role_text:
            prompt = f"{role_text}\n\n{prompt}"
        if flat_uri:
            hex_matte = "#{:02x}{:02x}{:02x}".format(*(int(c) for c in matte))
            prompt += (f"\n\nBACKGROUND: the flat {hex_matte} background of the design is a placeholder "
                       "for transparency. Keep it exactly that flat colour everywhere it is now -- no "
                       "gradient, texture, shadow, glow or new element on it. Apply the edit only to "
                       "the content.")

        apiLog("openai", api_model, "gptImageExecutor", status="start",
               detail=f"images={len(images)}")
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=IMAGE_API_TIMEOUT_SECONDS)
        result = client.images.edit(model=api_model, image=images, prompt=prompt,
                                    n=1, output_format='webp', quality='medium')

        result_bytes = base64.b64decode(result.data[0].b64_json)
        ext, content_type = "webp", "image/webp"
        if flat_uri:
            result_bytes = _key_out_matte(result_bytes, matte)
            ext, content_type = "png", "image/png"
            out_alpha = np.asarray(Image.open(BytesIO(result_bytes)))[..., 3]
            print(f"[gptImageExecutor] matte keyed out -- result {float((out_alpha < 250).mean()):.0%} "
                  f"transparent, {float((out_alpha > 0).mean()):.0%} visible")
        out = BytesIO(result_bytes)
        file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.{ext}"
        s3_client.upload_fileobj(
            tagged(out, state, "edited", api_model), DOCUMENT_BUCKET, file_name, ExtraArgs={
                "ContentType": content_type, 'ACL': 'public-read'}
        )
        file_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"

        usage = getattr(result, "usage", None)
        apiLog("openai", api_model, "gptImageExecutor", status="ok",
               tokens_in=getattr(usage, "input_tokens", None),
               tokens_out=getattr(usage, "output_tokens", None))
        cost = cost + calculateCost(api_model, 1000)  # approximate token count for image data
        return Command(goto="successfulEditHandler", update={**state, "cost": cost, "previous_image_url": state['current_image_url'], "current_image_url": file_url, "base_prompt": state['prompt'], "image_models": _with_image_models(state, file_url, api_model)})
    except Exception as e:
        apiLog("openai", api_model, "gptImageExecutor", status="failed", detail=str(e))
        return Command(goto="bananaProExecutor", update={**state, })

@profile
def successfulExpandHandler(state: GraphicDesigner) -> GraphicDesigner:
    """Present a successfully edited image and await the user's next action.

    Sends an interrupt showing both the **current** (edited) image and the
    **previous** image, along with the accumulated cost.  The user's response
    is parsed and the graph routes back to ``navigator`` with the updated
    conversation history, model, and aspect-ratio preferences.

    :param state: The current graph state with ``current_image_url`` and
        ``previous_image_url``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` with the user's
        follow-up input.
    :rtype: Command
    """
    print("[successfulExpandHandler] presenting the expanded image to the user")
    user_input = interrupt({
        "type": "expand_successful",
        "question": EXPAND_SUCCESSFUL,
        "prompt": state.get('prompt', ''),
        "current_image_url": state.get('current_image_url', ''),
        "previous_image_url": state.get('previous_image_url', ''),
        "cost": state.get('cost', 0.0)
    })
    user_response = parseJson(user_input)

    # Per-process navigator (expand flow).
    is_criticism, detect_cost = detectCriticism(state, user_response['response'], 'expand')
    if is_criticism:
        return Command(goto="criticismHandler", update={**state, "criticism_origin": "expand",
        "criticism_text": user_response['response'], "cost": state.get('cost', 0.0) + detect_cost,
        "current_image_url": user_response.get('current_image_url', state.get('current_image_url', '')),
        "conversation_history": state.get('conversation_history', []) + [{"role": "user", "content": user_response['response']}]})

    return Command(goto="navigator", update={"prompt": user_response['response'], "cost": state.get('cost', 0.0), "conversation_history": state.get('conversation_history', []) + [{"role": "assistant", "content": EDIT_SUCCESSFUL}] +[{"role": "user", "content": user_response['response']}], "current_image_url": user_response.get('current_image_url',""), "attached_images": _reply_attachments(user_response) or state.get('attached_images', []), "edit_reference_images": _reply_attachments(user_response), "base_prompt": None, "model": user_response.get('model', state.get('model', '')), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))})


@profile
def successfulEditHandler(state: GraphicDesigner) -> GraphicDesigner:
    """Present a successfully edited image and await the user's next action.

    Sends an interrupt showing both the **current** (edited) image and the
    **previous** image, along with the accumulated cost.  The user's response
    is parsed and the graph routes back to ``navigator`` with the updated
    conversation history, model, and aspect-ratio preferences.

    :param state: The current graph state with ``current_image_url`` and
        ``previous_image_url``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` with the user's
        follow-up input.
    :rtype: Command
    """
    print("[successfulEditHandler] presenting the edited image to the user")
    user_input = interrupt({
        "type": "editing_successful",
        "question": EDIT_SUCCESSFUL,
        "prompt": state.get('prompt', ''),
        "current_image_url": state.get('current_image_url', ''),
        "previous_image_url": state.get('previous_image_url', ''),
        "cost": state.get('cost', 0.0)
    })
    user_response = parseJson(user_input)

    # Two checks on the reply, run together so neither adds a round trip:
    # did it say the edit went to the wrong image, and is it vague criticism?
    # Per-process navigator (editing flow). Note a blunt but concrete report
    # like "the cup still looks like tea" is NOT criticism by design — it
    # already names the fix and stays on the normal editing route.
    with ThreadPoolExecutor(max_workers=2) as executor:
        wrong_future = executor.submit(
            contextvars.copy_context().run, judge_wrong_image, state.get('prompt', ''),
            user_response['response'], state.get('conversation_history', []))
        is_criticism, detect_cost = detectCriticism(state, user_response['response'], 'editing')
        wrong_image, wrong_cost = wrong_future.result()
    detect_cost += wrong_cost
    if wrong_image:
        return Command(goto="editWrongImageHandler", update={
            "request_type": "editing",
            "cost": state.get('cost', 0.0) + detect_cost,
            "conversation_history": state.get('conversation_history', [])
                + [{"role": "assistant", "content": EDIT_SUCCESSFUL},
                   {"role": "user", "content": user_response['response']}],
        })
    if is_criticism:
        return Command(goto="criticismHandler", update={**state, "criticism_origin": "editing",
        "criticism_text": user_response['response'], "cost": state.get('cost', 0.0) + detect_cost,
        "current_image_url": user_response.get('current_image_url', state.get('current_image_url', '')),
        "conversation_history": state.get('conversation_history', []) + [{"role": "user", "content": user_response['response']}]})

    return Command(goto="navigator", update={"prompt": user_response['response'], "cost": state.get('cost', 0.0), "conversation_history": state.get('conversation_history', []) + [{"role": "assistant", "content": EDIT_SUCCESSFUL}] +[{"role": "user", "content": user_response['response']}], "current_image_url": user_response.get('current_image_url',""), "attached_images": _reply_attachments(user_response) or state.get('attached_images', []), "edit_reference_images": _reply_attachments(user_response), "base_prompt": None, "model": user_response.get('model', state.get('model', '')), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))})

def editWrongImageHandler(state: GraphicDesigner) -> Command:
    """Apologise for editing the wrong image and ask for the right one.

    Entered from :func:`successfulEditHandler` when the reply said the edit
    went to a different image than the one meant (:func:`judge_wrong_image`)
    -- typically an edit sent with nothing selected, which ran on the
    project's latest graphic. Nothing is re-run here: the user selects the
    image on the canvas and sends the request again, and that reply goes to
    the navigator like any new message. The edited version stays a separate
    graphic, so the original image is untouched.

    No image work happened, so the frame carries ``"cost": 0`` and the check's
    spend rides ``state['cost']`` forward to the next real interrupt.
    """
    print("[editWrongImageHandler] edit went to the wrong image; asking for the right one")
    user_response = parseJson(interrupt({
        "type": "text_input",
        "kind": "reselect_image",
        "question": EDIT_WRONG_IMAGE_RESELECT,
        "current_image_url": state.get("current_image_url", ""),
        "cost": 0,
    }))
    reply = user_response.get("response", "")
    return Command(goto="navigator", update={
        "prompt": reply,
        "cost": state.get("cost", 0.0),
        "conversation_history": state.get("conversation_history", [])
            + [{"role": "assistant", "content": EDIT_WRONG_IMAGE_RESELECT},
               {"role": "user", "content": reply}],
        "current_image_url": user_response.get("current_image_url", ""),
        "attached_images": _reply_attachments(user_response) or state.get("attached_images", []),
        "edit_reference_images": _reply_attachments(user_response),
        "base_prompt": None,
        "model": user_response.get("model", state.get("model", "")),
        "aspect_ratio": user_response.get("aspect_ratio", state.get("aspect_ratio", "")),
        "style": user_response.get("style", state.get("style", "")),
    })


@profile
def criticismHandler(state: GraphicDesigner) -> GraphicDesigner:
    """Turn vague dissatisfaction into concrete, actionable options.

    The global landing point for every per-process criticism detector (see
    :func:`graphic_designer.utils.detectCriticism`).  The user has said they
    don't like the graphic but not what to change; this node infers the likely
    axis of complaint (palette, typography, layout, copy, …), offers 2-3
    specific directions, and waits for the user to pick one.

    Routing after the user replies follows the origin of the criticism:

    * ``'suggestion'`` — the suggestions themselves missed the mark, so the
      only useful fix is to regenerate them; routes to ``suggestionManager``.
    * everything else (``'generation'``, ``'editing'``, ``'expand'``,
      ``'layering'``) — the graphic exists and can be corrected in place, so
      routes to ``navigator``, which classifies the chosen remedy (normally
      into the editing flow).

    The offered options go out as plain prose in ``question``, matching every
    other ``text_input`` interrupt — the user replies free-form and their reply
    becomes the next ``prompt``.  The offer itself is appended to
    ``conversation_history`` so the downstream flow can resolve a short pick
    ("the gold one") against what was actually offered.

    :param state: The current graph state with ``criticism_text`` and
        ``criticism_origin`` set by the detecting flow.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``suggestionManager`` or ``navigator``.
    :rtype: Command
    """
    origin = state.get('criticism_origin', 'generation')
    criticism_text = state.get('criticism_text', state.get('prompt', ''))
    print(f"[criticismHandler] handling criticism raised in the {origin} flow")

    cost = state.get('cost', 0.0)

    try:
        prompt = CRITICISM_HANDLER_PROMPT.format(
            origin=origin,
            criticism_text=criticism_text,
            base_prompt=state.get('base_prompt') or state.get('prompt', ''),
            chat_history=json.dumps(state.get('conversation_history', [])[-6:])
        )
        llm_response, model_used = invoke_with_fallback(
            [{"role": "user", "content": prompt}], temperature=0.3
        )
        cost += calculateCost(model_used, llm_response.usage_metadata['total_tokens'])
        message = llm_response.content.strip()

    except Exception as e:
        # Never dead-end on a failed inference — fall back to asking plainly.
        print(f"[criticismHandler] option inference failed ({e}); asking the user directly")
        message = ""

    if not message:
        message = CRITICISM_FALLBACK

    user_response = interrupt({
        "type": "text_input",
        "question": message,
        "cost": cost,
        "current_image_url": state.get('current_image_url', ''),
    })

    user_response = parseJson(user_response)
    reply = user_response.get('response', '') or ''

    history = state.get('conversation_history', []) + [
        {"role": "assistant", "content": message},
        {"role": "user", "content": reply},
    ]

    update = {
        **state,
        "prompt": reply,
        "conversation_history": history,
        "cost": 0.0,
        "current_image_url": user_response.get('current_image_url', state.get('current_image_url', '')),
        "model": user_response.get('model', state.get('model', '')),
        "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', '')),
        # Clear the criticism context so a later turn can't re-enter on stale state.
        "criticism_origin": "",
        "criticism_text": "",
    }

    if origin == 'suggestion':
        # The suggestions themselves were the problem — regenerate them rather
        # than trying to edit the graphic against a rejected suggestion set.
        return Command(goto="suggestionManager", update={
            **update, "s_stack": [], "suggestion_type": "user_driven",
            "suggestion_request": reply, "regenerate_suggestions": True,
            # A rebuilt stack is a fresh chain -- full 2-ask budget again.
            "suggestion_more_round": 0,
        })

    return Command(goto="navigator", update={**update, "name": "main_graph"})


@profile
def extremeFallback(state: GraphicDesigner) -> GraphicDesigner:
    """Terminal fallback when all editing executors have failed.

    Presents an apology message to the user asking them to retry from
    scratch.  The user's response is parsed and the graph routes back to
    ``navigator`` to begin a fresh interaction loop.

    :param state: The current graph state.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` with the user's new
        prompt.
    :rtype: Command
    """
    print("[extremeFallback] recovering after all executors failed")
    user_response = interrupt({
        "type": "execution_failed",
        "question": EXTREME_FALLBACK,
        "cost": state.get('cost', 0.0)
    })
    user_response = parseJson(user_response)
    return Command(goto="navigator", update={"prompt": user_response['response'], "cost": state.get('cost', 0.0), "model": user_response.get("model", state.get("model", "")), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', '')), "current_image_url": user_response['current_image_url'], "conversation_history": state.get('conversation_history', []) + [{"role": "assistant", "content": EXTREME_FALLBACK}] +[{"role": "user", "content": user_response['response']}]})


@profile
def suggestionHandler(state: GraphicDesigner) -> Command:
    """Present suggestions to the user, collect responses, and route execution.

    Sends an interrupt displaying the ``s_stack`` suggestions along with
    the current image.  When the user responds:

    * If **any suggestion was accepted**, routes to ``suggestionExecutors``,
      which runs that one suggestion immediately and comes back here with
      the edited image and whatever is still undecided. Suggestions are
      never batched and merged into one prompt any more — an accept is
      executed on its own, the moment it arrives.
    * If **all suggestions are empty** (user cleared them), routes to
      ``navigator`` to continue the conversation.
    * If **nothing is accepted and nothing is left pending** (everything was
      rejected), there is nothing to spend on — routes straight to
      ``interruptBeforeLayering``.
    * If **some suggestions are still pending**, invokes the
      ``suggestionSubGraph`` to classify the user's response and update
      the suggestion stack accordingly. An edit to a suggestion only
      rewrites that suggestion and comes back here for a decision.

    :param state: The current graph state with ``s_stack`` and
        ``current_image_url``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``suggestionExecutors``,
        ``navigator``, or back to the suggestion loop.
    :rtype: Command
    """
    print("[suggestionHandler] presenting suggestions and routing the response")
    cost = state.get('cost', 0.0)
    user_response = interrupt({
        "type": "text_input",
        "question": pystache.render(suggestionHandlerMessage, {"username": state.get("username", "User")}),
        "suggestions": state['s_stack'],
        "suggestion_type": state['suggestion_type'],
        "cost": cost,
        "current_image_url": state.get('current_image_url', ''),
    })

    user_response = parseJson(user_response)
    cost = 0.0
    suggestions = user_response["suggestions"]
    accepted = [i for i in suggestions if i['status'] == 'accepted']
    pending = [i for i in suggestions if i['status'] == 'pending']

    if accepted:
        # Execute on accept, one unit at a time. The undecided units ride
        # along so suggestionExecutors can hand them straight back here
        # together with the image it just produced.
        return Command(goto='suggestionExecutors', update={**state, "s_stack": accepted + pending, "cost": cost})

    elif not pending and suggestions != []:
        # Everything was rejected: nothing to execute, so skip the executor
        # entirely rather than paying it to do nothing.
        return Command(goto='interruptBeforeLayering', update={**state, "s_stack": [], "cost": cost})

    elif any(i['status'] not in ('accepted', 'pending') for i in suggestions) \
            and not (user_response.get('response') or '').strip():
        # A plain reject click with units still undecided: drop the rejected
        # ones and re-present the rest. No classifier call -- there is no text
        # to classify, and sending an empty prompt to it used to be read as an
        # ambiguous reply.
        return Command(goto='suggestionHandler', update={**state, "s_stack": pending, "cost": cost,
            "current_image_url": user_response.get('current_image_url', state.get('current_image_url', ''))})

    elif suggestions == []:
        # Per-process navigator (suggestion flow): the user cleared the whole
        # stack. If they did so with a plain "none of these are any good",
        # regenerate the suggestions rather than sending that to navigator as a
        # literal prompt.
        is_criticism, detect_cost = detectCriticism(state, user_response['response'], 'suggestion')
        if is_criticism:
            return Command(goto="criticismHandler", update={**state, "s_stack": [],
            "criticism_origin": "suggestion", "criticism_text": user_response['response'],
            "cost": cost + detect_cost,
            "current_image_url": user_response.get('current_image_url', state.get('current_image_url', '')),
            "conversation_history": state.get('conversation_history', []) + [{"role": "user", "content": user_response['response']}]})

        return Command(goto='navigator', update={**state, "s_stack": [], "prompt": user_response['response'], "cost": cost, "current_image_url": user_response.get('current_image_url', ''), "attached_images": user_response['attached_images'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "user",
            "content": user_response['response']
        }], "suggestion_type": "user_driven", "model": user_response.get("model", state.get("model", "")), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))})

    else:
        subGraph_response = sugGraph.invoke({
            "name": 'suggestion_subgraph',
            "prompt": user_response['response'],
            "s_stack": user_response['suggestions'],
            "conversation_history": [],
            "cost": cost,
            "current_image_url": state.get('current_image_url', '')
        })

        cost += subGraph_response.get('cost', 0.0)
        if subGraph_response['s_stack'] == []:
            return Command(goto='navigator', update={**state, "s_stack": [], "prompt": subGraph_response['prompt'], "cost": cost, "current_image_url": subGraph_response.get('current_image_url', ''), "name": "main_graph", "conversation_history": state.get('conversation_history', []) + [{
                "role": "user",
                "content": user_response['response']
            }], "suggestion_type": "user_driven", "model": user_response.get("model", state.get("model", "")), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))})
        else:
            # The subgraph exits the moment a unit is accepted, so an accept
            # made in free text executes right away too. Anything else (an
            # edited suggestion, for instance) comes back to this node so the
            # user decides on it.
            sub_stack = subGraph_response['s_stack']
            goto = 'suggestionExecutors' if any(
                i['status'] == 'accepted' for i in sub_stack) else 'suggestionHandler'
            return Command(goto=goto, update={**state, "s_stack": sub_stack, "cost": cost, "current_image_url": subGraph_response.get('current_image_url', ''), "name": "main_graph", "conversation_history": state.get('conversation_history', []) + [{
                "role": "user",
                "content": user_response['response']
            }]})


@profile
def suggestionExecutors(state: GraphicDesigner) -> Command:
    """Execute ONE accepted suggestion using the appropriate model API.

    Suggestions are executed as they are accepted, never merged: this node
    takes the first ``accepted`` unit off ``s_stack``, runs it against the
    current image, and then either loops (another unit is already accepted),
    hands the fresh image plus the still-undecided units back to
    :func:`suggestionHandler`, or exits to ``interruptBeforeLayering`` when
    the stack is empty. Each unit therefore edits the result of the one
    before it, which is also why no conflict check is needed up front — a
    later unit simply overrides an earlier one it disagrees with.

    Also:

    * Downloads the current image from its URL **once** (not per-operation).
    * Passes the in-memory base-64 directly from EDIT → UPSCALE,
      eliminating a redundant S3 round-trip.
    * Uploads only the **final** result to S3.

    :param state: The current graph state with ``s_stack``,
        ``current_image_url``, ``user_id``, and ``session_id``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``suggestionExecutors`` (next accepted
        unit), ``suggestionHandler`` (units left to decide), or
        ``interruptBeforeLayering``.
    :rtype: Command
    """
    print("[suggestionExecutors] executing the accepted suggestion")
    cost = state.get('cost', 0.0)
    s_stack = state['s_stack']
    original_image_url = state['current_image_url']
    updated_image_url = original_image_url

    # ── Step 1: Pick the single unit to run now ──
    # Rejected units are dropped here; everything still pending survives so it
    # can go back to the user with the edited image.
    unit = next((s for s in s_stack if s.get('status') == 'accepted'), None)
    remaining = [s for s in s_stack if s is not unit and s.get('status') != 'rejected']

    if unit is None:
        print("[suggestionExecutors] nothing accepted — nothing to execute")
        return Command(goto="interruptBeforeLayering", update={**state, "s_stack": [], "cost": cost})

    # ── Step 2: Categorise into typed buckets ──
    edit_item = None
    gen_item = None
    upscale_items = []

    # gpt-image-2.5-flare runs EVERY suggestion edit, whatever the aspect ratio.
    # nano-banana-2 is only the fallback, for when the gpt call itself fails.
    _sug_aspect = state.get('aspect_ratio') or '1:1'
    if unit['type'] == 'EDIT':
        edit_item = {"type": "EDIT", "nano_prompt": unit['nano_prompt'],
                     "model": "gpt-image-2.5-flare", "topic": unit.get('topic')}
    elif unit['type'] == 'GENERATION':
        gen_item = unit
    elif unit['type'] == 'UPSCALE':
        upscale_items.append(unit)

    # ── Step 2b: DEBUG — the exact prompt the image model is about to receive ──
    # Single place where the prompt being sent is known. When an edit comes back wrong (a garbled
    # word, a half-applied fix), the first question is always whether the prompt asked for the
    # right thing or the model failed to follow it, and that is unanswerable without this.
    print("\n" + "=" * 78)
    print("[suggestionExecutors] EDIT PROMPT")
    print("=" * 78)
    print(f"remaining    : {len(remaining)} unit(s) left on the stack")
    if edit_item:
        print(f"model        : {edit_item['model']}   aspect: {_sug_aspect}")
        print(f"accepted     : [{unit.get('topic')}]")
        print("-" * 78)
        print(edit_item['nano_prompt'])
    else:
        print("(no EDIT unit)")
    if gen_item:
        print("-" * 78)
        print(f"GENERATION   : {gen_item.get('nano_prompt')}")
    if upscale_items:
        print(f"UPSCALE      : {len(upscale_items)} item(s)")
    print("=" * 78 + "\n")

    # ── Step 3: Download current image ONCE (shared by EDIT & UPSCALE) ──
    current_b64 = None
    if edit_item or upscale_items:
        resp = makeApiRequest('GET', original_image_url, timeout=30)
        current_b64 = base64.b64encode(resp.content).decode('utf-8')

    # ── Step 4: Launch EDIT + GENERATION in parallel (they are independent) ──
    edit_result = None
    gen_result = None

    # A single Context object cannot be entered by two threads at once --
    # `_ctx.run` raced here ("cannot enter context: ... is already entered")
    # once EDIT and GENERATION started running concurrently often enough for
    # both futures to actually overlap. Each submission gets its OWN copy
    # (still captures the same node name / spend session at submission time)
    # so the two workers never fight over one Context.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if edit_item:
            if edit_item['model'] == 'gpt-image-2.5-flare':
                futures['edit'] = executor.submit(
                    contextvars.copy_context().run, sugGptImageEditor, current_b64,
                    edit_item['nano_prompt'], edit_item['model'], _sug_aspect)
            else:
                futures['edit'] = executor.submit(
                    contextvars.copy_context().run, sugBananaEditor, current_b64,
                    edit_item['nano_prompt'])
        if gen_item:
            # Suggestion stacks carry no model of their own: techAgent, which
            # used to attach one, is disabled, so `gen_item['model']` raised a
            # KeyError the moment a GENERATION suggestion was accepted. Default
            # to nano-banana-2 and write it back, so the cost line and the
            # webp conversion below read the model that actually ran.
            gen_item['model'] = gen_item.get('model') or 'nano-banana-2'
            futures['gen'] = executor.submit(
                contextvars.copy_context().run, nanoBanana2ImageGen,
                gen_item['nano_prompt'],
                gen_item['model'],
                state['aspect_ratio'],
                state.get('attached_images', None),
            )

        # Collect results (blocks until each future completes). The gen helpers
        # now raise on failure, so capture the exception into the result -- the
        # isinstance(..., Exception) fallbacks below expect a value, not a throw.
        if 'edit' in futures:
            try:
                edit_result = futures['edit'].result()
            except Exception as e:
                edit_result = e
        if 'gen' in futures:
            try:
                gen_result = futures['gen'].result()
            except Exception as e:
                gen_result = e

    # ── Step 4b: Fallback for EDIT — whichever editor did NOT run above ──
    # `edit_item['model']` is rewritten to whatever actually produced the image, so the cost
    # line and the webp conversion downstream read the model that really ran.
    if edit_item and (edit_result is None or isinstance(edit_result, Exception)):
        _log = logging.getLogger('graphic_designer.suggestion')
        try:
            if edit_item['model'] == 'gpt-image-2.5-flare':
                _log.warning("sugGptImageEditor failed (%s), falling back to sugBananaEditor "
                             "(nano-banana-2)", edit_result)
                edit_result = sugBananaEditor(current_b64, edit_item['nano_prompt'])
                edit_item['model'] = 'nano-banana-2'
            else:
                _log.warning("sugBananaEditor failed (%s), falling back to sugGptImageEditor "
                             "(gpt-image-2.5-flare)", edit_result)
                edit_result = sugGptImageEditor(current_b64, edit_item['nano_prompt'],
                                                'gpt-image-2.5-flare', _sug_aspect)
                edit_item['model'] = 'gpt-image-2.5-flare'
        except Exception as e:
            _log.error("edit fallback also failed: %s", e)
            edit_result = None

    # ── Step 4c: Fallback for GENERATION — if bananaProImageGen failed, try gptImageGen ──
    if gen_item and (gen_result is None or isinstance(gen_result, Exception)):
        logging.getLogger('graphic_designer.suggestion').warning(
            "bananaProImageGen failed (%s), falling back to gptImageGen (gpt-image-2.5-flare)", gen_result
        )
        try:
            # gptImageGen returns list[str] (n images); the rest of this
            # function expects a single base-64 string like every other
            # editor/generator here, so unwrap the sole result (n=1).
            gen_result = gptImageGen(
                gen_item['nano_prompt'],
                'gpt-image-2.5-flare',
                state['aspect_ratio'],
                1,
            )[0]
            gen_item['model'] = 'gpt-image-2.5-flare'
        except Exception as e:
            logging.getLogger('graphic_designer.suggestion').error(
                "gptImageGen fallback also failed: %s", e
            )
            gen_result = None

    # ── Step 5: EDIT → keep base-64 in memory (NO intermediate S3 upload) ──
    if edit_result and not isinstance(edit_result, Exception):
        # check if the image is generated by nano banana model, if so, convert to webp format
        if edit_item['model'] == 'nano-banana-2':
            edit_result = convert_png_to_webp_base64(edit_result)
        cost += calculateCost(edit_item['model'], 1000)

        # No post-edit verify/retry pass any more: it flagged partial/regressed
        # results that were, on inspection, fine -- reverting or re-editing a
        # correct image is worse than leaving it as the model returned it.
        current_b64 = edit_result   # pass straight to UPSCALE

    # ── Step 6: UPSCALE chain (uses in-memory base-64 — no re-download) ──
    for uitem in upscale_items:
        current_b64 = upscaleImage(current_b64)
        cost += calculateCost("upscale-fast", 1000)

    # ── Step 7: Single S3 upload for the final EDIT / UPSCALE image ──
    image_models = state.get("image_models") or {}
    if edit_result or upscale_items:
        # The editor drew the content; an upscale on top only resamples it.
        edit_model = edit_item['model'] if edit_result else "upscale-fast"
        image_data = base64.b64decode(current_b64)
        image_file = BytesIO(image_data)
        file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
        s3_client.upload_fileobj(
            tagged(image_file, state, "suggestion", edit_model), DOCUMENT_BUCKET, file_name,
            ExtraArgs={"ContentType": "image/webp", 'ACL': 'public-read'},
        )
        updated_image_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
        image_models = _with_image_models(state, updated_image_url, edit_model)

    # ── Step 8: Handle GENERATION result ──
    if gen_result:
        # check if the image is generated by nano banana model, if so, convert to webp format
        if gen_item['model'] == 'nano-banana-2':
             gen_result = convert_png_to_webp_base64(gen_result)
        image_data = base64.b64decode(gen_result)
        image_file = BytesIO(image_data)
        file_name = f"{state['user_id']}/{state['session_id']}/image_{str(datetime.now().timestamp())}.webp"
        s3_client.upload_fileobj(
            tagged(image_file, state, "suggestion", gen_item["model"]), DOCUMENT_BUCKET, file_name,
            ExtraArgs={"ContentType": "image/webp", 'ACL': 'public-read'},
        )
        gen_url = f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{file_name}"
        cost += calculateCost(gen_item['model'], 1000)
        # Exits through suggestionMoreAsk, the same terminal step the
        # EDIT/UPSCALE path uses now -- the loop just executed a real
        # suggestion, so the user is offered another pass before falling
        # through to interruptBeforeLayering's normal "what next" step.
        return Command(goto="suggestionMoreAsk", update={
            **state,
            "prompt": gen_item['nano_prompt'],
            "current_image_url": gen_url,
            "previous_image_url": updated_image_url,   # EDIT output (or original if no EDIT)
            "base_prompt": state['prompt'],
            "s_stack": [],
            "cost": cost,
            "image_models": _with_image_models({"image_models": image_models}, gen_url, gen_item['model']),
        })

    # ── Default return (EDIT / UPSCALE only) ──
    update = {**state, "current_image_url": updated_image_url, "s_stack": remaining, "cost": cost,
              "image_models": image_models}
    if edit_result:
        update["previous_image_url"] = original_image_url

    # Another unit is already accepted — run it against the image this pass
    # just produced, without going back to the user in between.
    if any(s.get('status') == 'accepted' for s in remaining):
        return Command(goto="suggestionExecutors", update=update)

    # Units left to decide on — show the user the new image with them.
    # `prompt` / `base_prompt` stay untouched here: they are only settled when
    # the stack is finally empty, so a mid-loop pass can't overwrite the
    # original design prompt with one suggestion's nano_prompt.
    if remaining:
        return Command(goto="suggestionHandler", update=update)

    update["prompt"] = unit.get('nano_prompt') or state['prompt']
    update["s_stack"] = []
    if edit_result:
        update["base_prompt"] = state['prompt']

    # The loop just executed a real suggestion (EDIT and/or UPSCALE), so the
    # user is offered another critique pass on the result before falling
    # through to interruptBeforeLayering's normal "what next" step.
    return Command(goto="suggestionMoreAsk", update=update)


MAX_SUGGESTION_MORE_ASKS = 1
"""Cap on how many times suggestionMoreAsk offers another round for one
suggestion chain. Asked once; the round that follows a "yes" runs with no
2nd ask after it -- see suggestionMoreAsk."""


@profile
def suggestionMoreAsk(state: GraphicDesigner) -> Command:
    """Offer another suggestion pass on the image the suggestion loop just produced.

    Entered only when the loop actually EXECUTED something and the stack is
    now empty (every accepted unit ran, or a GENERATION unit replaced the
    canvas) -- an all-rejected pass with nothing executed skips this and goes
    straight to :func:`interruptBeforeLayering`, since there is no new image
    to critique yet.

    Asked at most :data:`MAX_SUGGESTION_MORE_ASKS` (2) times per chain --
    ``suggestion_more_round`` counts how many times this has already been
    asked, and the 2nd time it's said yes to runs with no 3rd ask after it;
    the cap check below then skips straight to ``interruptBeforeLayering``
    with no interrupt at all. Reset to 0 at every FRESH entry into the
    suggestion flow (interruptBeforeFidelitySuggestion, a typed "give me
    suggestions", criticismHandler's rebuild) -- only this node's own "yes"
    branch increments it, so a continued chain keeps counting.

    * "yes" (while under the cap) -- routes back to :func:`suggestionManager`
      for another automatic critique of THIS image. ``suggestion_type`` is
      set to ``'auto'`` (same forced-general path
      :func:`interruptBeforeFidelitySuggestion` uses) and
      ``suggestion_target_confirmed`` is set True, since the user just
      finished editing this exact image -- suggestionManager's "which
      graphic" ask only fires when that flag is unset.
    * anything else, or the cap already reached -- falls through to
      :func:`interruptBeforeLayering`, the normal "what do you want to do
      next?" step, unchanged.

    :param state: The current graph state with ``current_image_url``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``suggestionManager`` or
        ``interruptBeforeLayering``.
    :rtype: Command
    """
    cost = state.get('cost', 0.0)
    asked_so_far = state.get('suggestion_more_round', 0)

    if asked_so_far >= MAX_SUGGESTION_MORE_ASKS:
        print(f"[suggestionMoreAsk] already asked {asked_so_far} time(s) -- not asking again")
        return Command(goto="interruptBeforeLayering", update={**state, "cost": cost})

    print("[suggestionMoreAsk] asking whether to run another suggestion pass "
          f"({asked_so_far + 1}/{MAX_SUGGESTION_MORE_ASKS})")
    user_response = interrupt({
        "type": "text_input",
        "question": SUGGESTION_MORE_ASK,
        "cost": cost,
        "current_image_url": state.get('current_image_url', ''),
    })

    user_response = parseJson(user_response)
    cost = 0.0
    # Same reasoning as interruptBeforeLayering: honour whatever is selected
    # on the canvas at reply time, not just the image this run produced.
    _selected_url = user_response.get('current_image_url') or state.get('current_image_url', '')
    conversation_history = state.get('conversation_history', []) + [
        {"role": "system", "content": SUGGESTION_MORE_ASK},
        {"role": "user", "content": user_response.get('response', '')},
    ]

    if user_response.get("response", "").strip().lower() == "yes":
        return Command(goto="suggestionManager", update={
            **state,
            "current_image_url": _selected_url,
            "cost": cost,
            "suggestion_type": "auto",
            "suggestion_target_confirmed": True,
            "suggestion_more_round": asked_so_far + 1,
            "conversation_history": conversation_history,
        })

    return Command(goto="interruptBeforeLayering", update={
        **state,
        "current_image_url": _selected_url,
        "cost": cost,
        "conversation_history": conversation_history,
    })


# LAYERING NODES


@profile
def interruptBeforeLayering(state: GraphicDesigner) -> Command:
    """Present post-suggestion results and collect the user's next action.

    Sends an interrupt displaying the current image (after suggestion
    execution) along with a prompt asking "What do you want to do next?".
    It collects the user's response and routes back to ``navigator`` with
    updated state including the user's new prompt and conversation history.

    This node is typically used after a suggestion subgraph has completed
    its execution and the system is ready for the next user command.

    :param state: The current graph state with ``current_image_url``,
        and ``cost``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` with the user's
        follow-up input and updated state.
    :rtype: Command
    """
    print("[interruptBeforeLayering] offering to split the image into layers")
    cost = state.get('cost', 0.0)
    # Logic to proceed to the next step
    user_response = interrupt({
        "type": "suggestions_executed",
        "question": LAYERING_OFFER,
        "prompt": state.get('prompt', ''),
        "cost": cost,
        "current_image_url": state.get('current_image_url', '')
    })

    user_response = parseJson(user_response)

    # The reply carries whatever is selected on the canvas NOW, which is not
    # necessarily the image this run just produced -- the user is free to pick a
    # different one before typing "give me suggestions" again. Reading it here
    # (like successfulEditHandler / suggestionHandler already do) is what keeps
    # the next flow from re-analysing the stale executed image. Fall back to the
    # state value so a frontend that omits the key doesn't blank the selection.
    _reply_refs = _reply_attachments(user_response)
    return Command(goto='navigator', update={
        **state,
        # The suggestion loop is over by the time this node runs: every unit
        # was executed or dropped. Clearing it here means a stale stack can
        # never be carried into the next turn and re-presented as if those
        # suggestions were still open.
        "s_stack": [],
        "prompt": user_response['response'],
        "current_image_url": user_response.get('current_image_url') or state.get('current_image_url', ''),
        "attached_images": _reply_refs or state.get('attached_images', []),
        "edit_reference_images": _reply_refs,
        "model": user_response.get('model', state.get('model', '')),
        "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')),
        "style": user_response.get('style', state.get('style', '')),
        "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": LAYERING_OFFER
        }] + [{
            "role": "user",
            "content": user_response['response']
        }],
        })





