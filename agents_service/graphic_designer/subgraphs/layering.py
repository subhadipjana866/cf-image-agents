from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from ..states import LayeringSubGraph
from langgraph.types import Command, interrupt
from ..utils import (parseJson, calculateCost, profile, invoke_with_fallback,
                     markImageWork, requestedLayerTargets)
from ..common_nodes import chitChatHandler, questionHandler, unclearHandler
from ..prompts import LAYERING_HANDLER_PROMPT
from ..config import LAYERING_OBJECT_ASK, LAYERING_SKIPPED
import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import requests

import settings

from layers.detection import (
    # Classes
    DetectionResult,
    # Functions
    start as detection_start
)
from layers.segmentation import (
    # Functions
    start as segmentation_start
)


# ``DetectionResult`` holds raw numpy image/depth arrays, which cannot be
# checkpoint-serialized (LangGraph's per-task pending-write persistence
# requires msgpack-serializable values). Since this subgraph is mounted
# directly as a node on the main graph -- so a resume continues exactly where
# it paused instead of replaying detection from scratch -- putting the raw
# object in graph state would silently break that resume tracking on the very
# next round trip and force a full replay anyway. Keep it in a process-local
# cache keyed by (user_id, session_id) instead; only that (already
# serializable) identity travels through the checkpointed state.
_detection_cache: dict = {}

# Pixel crops of each detected text region, taken from the pristine
# pre-erasure image before the text-removal pre-pass runs. Text-removal
# happens unconditionally during detection, so restoring a text later means
# having kept its original pixels around.
#
# "restore" holds the texts the user did NOT point at, when they pointed at one
# at all -- see ``_apply_text_selection``. Pointing at no text keeps every text
# layered and leaves this empty, which is the common case and makes
# ``pasteBackTexts`` a no-op.
#
# cache_key -> {"crops": {text_id: (crop_ndarray, (x1, y1, x2, y2))}, "restore": [text_dict, ...]}
_text_crop_cache: dict = {}


def _detection_cache_key(user_id, session_id) -> tuple:
    """Same normalization as :func:`sessionArtifactsDir`, so the two never disagree."""
    return (str(user_id or 'unknown_user'), str(session_id or 'unknown_session'))


def _text_bbox_rect(bbox_quad):
    """Axis-aligned (x1, y1, x2, y2) rect enclosing a text's quad bbox corners."""
    if not isinstance(bbox_quad, (list, tuple)) or len(bbox_quad) < 3:
        return None
    xs = [p[0] for p in bbox_quad]
    ys = [p[1] for p in bbox_quad]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


@profile
def layeringHandler(state: LayeringSubGraph) -> LayeringSubGraph:
    """
    Handles user response and bboxes
    """
    # Manual mode was already chosen (and boxes already drawn) at the
    # layering_started interrupt in layeringManager -- there's no ambiguity
    # left to classify, so skip straight to processing those boxes.
    if state.get("layering_mode") == "manual":
        if state.get("manual_bboxes"):
            print("[layeringHandler] manual mode -- routing to manualLayering")
            return Command(goto="manualLayering", update={**state})
        print("[layeringHandler] manual mode chosen but no boxes were drawn")
        return Command(goto="layDoNotLayer", update={**state})

    # layeringManager already read the targets out of this same prompt with
    # this same classifier and skipped the ask because of what it found.
    # Re-running it here would pay for the identical call twice and could
    # answer differently the second time.
    if state.get("layering_targets_resolved"):
        print("[layeringHandler] targets already resolved at the gate -- "
              "routing straight to detection")
        return Command(goto="bboxGenerator", update={
            **state, "layering_targets_resolved": False})

    print("[layeringHandler] classifying the user's layering intent")
    try:
        cost = state.get("cost",0)

        prompt = LAYERING_HANDLER_PROMPT.format(
            conversation_history=json.dumps(state.get("conversation_history", [])),
            user_response=state.get("prompt", "")
        )

        response, model_used = invoke_with_fallback([
            {"role": "user", "content": prompt}
        ], temperature=0, thinking_budget=0)

        cost += calculateCost(model_used, response.usage_metadata['total_tokens'])
        response_data = parseJson(response.content)
        response = response_data.get("class", "")
        if response == "proceed_to_layering":
            requested_objects, requested_texts, detect_all = requestedLayerTargets(response_data)
            # Naming a text is as complete an answer as naming an object --
            # "layer the SALE headline" says exactly what the user wants, so
            # asking them again which objects they mean turns one request into
            # two. Only a request that names neither falls through to the ask.
            if (requested_objects or requested_texts) and not detect_all:
                return Command(goto="bboxGenerator", update={
                    **state, "cost": cost,
                    "requested_objects": requested_objects,
                    "requested_texts": requested_texts})
            if detect_all:
                return Command(goto="bboxGenerator", update={
                    **state, "cost": cost,
                    "requested_objects": [], "requested_texts": []})
            return Command(goto="layObjectAsk", update={**state, "cost": cost})
        elif response == "do_not_layer":
            return Command(goto="layDoNotLayer", update={**state, "cost": cost})
        elif response == "new_edit":
            return Command(goto="layNewEdit", update={**state, "cost": cost})
        elif response == "new_generation":
            return Command(goto="layNewGeneration", update={**state, "cost": cost})
        elif response == "suggestions":
            return Command(goto="laySuggestions", update={**state, "cost": cost})
        elif response == "question":
            return Command(goto="layQuestion", update={**state, "cost": cost})
        elif response == "chitchat":
            return Command(goto="layChitchat", update={**state, "cost": cost})
        elif response == "unclear":
            return Command(goto="layUnclear", update={**state, "cost": cost})
    except Exception as e:
        logging.error(f"Error in layeringHandler: {e}")
        return Command(goto=END, update={**state, "bboxes": []})

    

@profile
def bboxGenerator(state: LayeringSubGraph) -> Command:
    """Generate bounding boxes for the current image to enable layering.

    Runs the full detection pipeline (Gemini concept generation → Depth
    reordering → Florence-2 phrase grounding) on the image pointed to by
    ``state['current_image_url']``.

    The pipeline is configured using the settings already imported into
    ``config.py`` (DEVICE, FLORENCE_MODEL, STABILITY_API_KEY, etc.).

    Detection artefacts (concept JSONs, depth maps, text-removal outputs)
    are written to a per-session directory under ``layers/output/``.

    What detection finds goes **straight to segmentation** -- there is no
    ``bboxes_generated`` confirmation on either path any more. The user already
    said what they wanted, either by naming the objects or by drawing the
    boxes, and re-asking them to approve the machine's reading of that turned
    one request into two.

    :param state: The current graph state — must contain ``current_image_url``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``segmentGenerator`` with the detection
        results cached, or to ``layDoNotLayer`` when nothing was found.
    :rtype: Command
    """
    print("[bboxGenerator] running detection to generate bounding boxes")
    cost = state.get('cost', 0.0)

    image_url = state.get('current_image_url', '')
    if not image_url:
        logging.getLogger('graphic_designer.bboxGenerator').error(
            "No current_image_url in state — cannot run detection."
        )
        return Command(goto="nextStep", update={**state, "cost": cost})

    # Per-session output directory for detection artefacts. Same builder the
    # websocket teardown uses, so the two cannot disagree about the path.
    output_dir = str(sessionArtifactsDir(state.get('user_id'), state.get('session_id')))

    
    detection_result: DetectionResult = detection_start(
        image_url, output_dir,
        requested_objects=state.get('requested_objects') or None,
        requested_texts=state.get('requested_texts') or None,
    )
    # Detection runs on the GPU server rather than through apiLog, so it has to
    # raise the "this turn touched pixels" flag itself for the interrupt that
    # eventually bills it to be marked show_cost.
    if (detection_result.cost or 0) > 0:
        markImageWork()
    cache_key = _detection_cache_key(state.get('user_id'), state.get('session_id'))
    concepts = detection_result.ordered_concepts or []
    crops = _capture_text_crops(detection_result)
    # Whether they asked for a text *by name* -- not merely whether the image
    # has any. "Layer the chair" with no chair found is a failed run, even on
    # a poster covered in text; handing back the text nobody asked for would
    # be worse than saying nothing was layered.
    text_requested = bool(detection_result.selected_text_ids)
    restore = _apply_text_selection(detection_result)
    if not concepts and not text_requested:
        # Nothing to layer. There is no confirmation step left to show that
        # at, so end the run cleanly rather than handing segmentation an
        # empty list.
        logging.getLogger('graphic_designer.bboxGenerator').warning(
            "Detection found nothing to layer — skipping segmentation."
        )
        _cleanup_local_artifacts(detection_result)
        return Command(goto="layDoNotLayer", update={
            **state, "cost": cost + detection_result.cost, "current_image_url": image_url})

    _detection_cache[cache_key] = detection_result
    _text_crop_cache[cache_key] = {"crops": crops, "restore": restore,
                                   **_matte_cache_fields(detection_result)}
    print(f"[bboxGenerator] {len(concepts)} object(s), "
          f"{len(detection_result.extracted_texts or [])} text(s) → segmentation")
    # `bboxes` is what layering_executed reports back, so it carries the boxes
    # that were actually segmented. The cost rides along rather than resetting:
    # no interrupt happens here, so nothing has billed it yet.
    return Command(goto="segmentGenerator", update={
        **state,
        "bboxes": concepts,
        "cost": cost + detection_result.cost,
        "current_image_url": image_url,
    })


@profile
def layObjectAsk(state: LayeringSubGraph) -> Command:
    """Ask which objects to layer when the user didn't name any (e.g. bare
    "turn into layer"). Loops back into layeringHandler with the reply so the
    same classifier extracts the named objects -- reusing the exact same
    proceed_to_layering / do_not_layer / etc. handling for whatever they say.
    """
    print("[layObjectAsk] asking which objects to turn into layers")
    # Same wording as the layering_started question in layeringManager, from
    # the one constant -- this is the re-ask when the first answer named
    # nothing, and two differently-worded versions of the same question read
    # like the system did not hear the first answer.
    question = LAYERING_OBJECT_ASK
    user_response = parseJson(interrupt({
        "type": "text_input",
        "question": question,
        "accepts": ["object_names", "bboxes"],
        "auto_text": True,
        "cost": state.get('cost', 0.0),
        "current_image_url": state.get('current_image_url', '')
    }))
    reply = str(user_response.get('response') or '')
    return Command(goto="layeringHandler", update={
        **state,
        "prompt": reply,
        "conversation_history": state.get('conversation_history', []) + [
            {"role": "system", "content": question},
            {"role": "user", "content": reply},
        ],
        "cost": 0.0,
    })


@profile
def manualLayering(state: LayeringSubGraph) -> Command:
    """Run the whole layering pass on the boxes the user drew in the frontend.

    Each drawn box is recursed through Gemini + Florence over its own crop (see
    ``detection._concepts_from_frontend_bboxes``), and the objects found inside
    them go **straight to segmentation** -- no ``bboxes_generated``
    confirmation. The user already said what they want by drawing the boxes;
    the objects the recursion finds inside one are an implementation detail of
    getting a clean SAM2 mask, not a second decision to put in front of them,
    and segmentation composes them back into one layer per drawn box anyway.

    The auto path still confirms: there the boxes come from Gemini's reading of
    the whole image, which the user has not seen and may want to correct.

    Text is decided per drawn box: detection OCRs each box on its own crop and
    a box that is mostly text becomes a text layer, while the rest go to SAM2
    (see ``detection._run_per_box_ocr``). Text OUTSIDE the drawn boxes is never
    read or erased, so ``pasteBackTexts`` finds no restore list and no-ops.
    """
    print("[manualLayering] processing frontend-drawn bounding boxes")
    cost = state.get('cost', 0.0)

    image_url = state.get('current_image_url', '')
    if not image_url:
        logging.getLogger('graphic_designer.manualLayering').error(
            "No current_image_url in state — cannot process manual boxes."
        )
        return Command(goto="nextStep", update={**state, "cost": cost})

    output_dir = str(sessionArtifactsDir(state.get('user_id'), state.get('session_id')))

    detection_result: DetectionResult = detection_start(
        image_url, output_dir, frontend_bboxes=state.get('manual_bboxes') or []
    )
    cache_key = _detection_cache_key(state.get('user_id'), state.get('session_id'))
    concepts = detection_result.ordered_concepts or []
    crops = _capture_text_crops(detection_result)
    # Boxing only a text is a legitimate ask -- "layer just this headline" --
    # and leaves no objects to segment, so the guard has to look at both. What
    # counts is that a box pointed AT a text, not that the image contains one.
    text_requested = bool(detection_result.selected_text_ids)
    restore = _apply_text_selection(detection_result)
    if not concepts and not text_requested:
        logging.getLogger('graphic_designer.manualLayering').warning(
            "Nothing usable from %d drawn box(es) — skipping segmentation.",
            len(state.get('manual_bboxes') or []),
        )
        _cleanup_local_artifacts(detection_result)
        return Command(goto="layDoNotLayer", update={
            **state, "cost": cost + detection_result.cost, "current_image_url": image_url})

    _detection_cache[cache_key] = detection_result
    _text_crop_cache[cache_key] = {"crops": crops, "restore": restore,
                                   **_matte_cache_fields(detection_result)}
    print(f"[manualLayering] {len(concepts)} object(s), "
          f"{len(detection_result.extracted_texts or [])} text(s) → segmentation")
    # `bboxes` is what layering_executed reports back, so it carries the boxes
    # that were actually segmented. The cost rides along rather than resetting:
    # no interrupt happens here, so nothing has billed it yet (segmentGenerator
    # adds to it).
    return Command(goto="segmentGenerator", update={
        **state,
        "bboxes": concepts,
        "cost": cost + detection_result.cost,
        "current_image_url": image_url,
    })


def _capture_text_crops(detection_result: DetectionResult) -> dict:
    """Tag each detected text with a stable id and crop its region out of the
    pristine pre-erasure image, so it can be pasted back later if the user
    un-selects it. Mutates ``detection_result.extracted_texts`` in place to
    add the id -- that list is what flows to the frontend and, unchanged,
    through to segmentation, so the id survives the whole round trip.
    """
    crops = {}
    extracted_texts = detection_result.extracted_texts or []
    original_rgb = detection_result.original_rgb
    for i, text in enumerate(extracted_texts):
        text['id'] = i
        if original_rgb is None:
            continue
        rect = _text_bbox_rect(text.get('bbox'))
        if rect is None:
            continue
        # A text's own 'bbox' is a QUAD (four corner points, straight from
        # OCR), while an object's is [x1, y1, x2, y2]. Both shapes travel in
        # the same layering_executed payload, so hand the frontend the
        # axis-aligned rect too rather than making it discover the difference.
        text['rect'] = list(rect)
        x1, y1, x2, y2 = rect
        crop = original_rgb[y1:y2, x1:x2]
        if crop.size:
            crops[i] = (crop.copy(), rect)
    return crops

def _apply_text_selection(detection_result: DetectionResult) -> list:
    """Split the detected texts into the ones that stay layered and the ones
    that go back onto the background.

    Detection reports which texts the user pointed at -- by drawing a box round
    one, or by naming it. Pointing at none (the common case) means
    ``selected_text_ids`` is ``None`` and every text stays layered, exactly as
    before. Pointing at one narrows it to that one: the rest are pasted back at
    the end of the run.

    Mutates ``detection_result.extracted_texts`` down to the kept list, since
    that is what segmentation passes through into the final state.

    :returns: The text dicts to restore, ready for ``_text_crop_cache``.
    """
    texts = detection_result.extracted_texts or []
    selected = detection_result.selected_text_ids
    restore_ids = detection_result.restore_text_ids

    if restore_ids and texts:
        # Demand-driven detection already decided this, per text, from WHY it
        # was erased -- so honour the split verbatim. The fallback below cannot
        # express it: a run where the user named no text at all but an object
        # had text across its edge leaves ``selected`` empty, and "empty" there
        # means "they pointed at nothing, keep everything", which would layer
        # text nobody asked for instead of pasting it back.
        kept = [t for i, t in enumerate(texts) if i not in restore_ids]
        restore = [t for i, t in enumerate(texts) if i in restore_ids]
        detection_result.extracted_texts = kept
        print(f"[textSelection] keeping {len(kept)} text(s) as layers, "
              f"pasting {len(restore)} back onto the background")
        return restore

    if not selected or not texts:
        return []

    # ``_capture_text_crops`` numbers texts by their position in this same
    # list, so a selected index and its crop's id are the same number.
    kept = [t for i, t in enumerate(texts) if i in selected]
    restore = [t for i, t in enumerate(texts) if i not in selected]
    if not kept:
        # Nothing actually matched -- keep every text rather than silently
        # erasing them all.
        return []

    detection_result.extracted_texts = kept
    print(f"[textSelection] keeping {len(kept)} text(s) as layers, "
          f"pasting {len(restore)} back onto the background")
    return restore


LAYER_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / 'layers' / 'output'
"""Path: Root of the on-disk detection/segmentation artifacts, laid out as
``layers/output/<user_id>/<session_id>/``."""


def sessionArtifactsDir(user_id, session_id) -> Path:
    """The artifact directory for one session. Single source of truth for the
    layout, so the node that writes it and the teardown that removes it cannot
    drift apart."""
    return LAYER_OUTPUT_ROOT / str(user_id or 'unknown_user') / str(session_id or 'unknown_session')


def _remove_artifact_dir(output_dir: Path) -> bool:
    """Delete one session folder and prune the user folder if it is left empty.

    Never raises: cleanup runs on teardown paths where an exception would mask
    the error that caused the teardown.
    """
    removed = False
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
            removed = True
        except Exception as e:
            logging.getLogger('graphic_designer.cleanup').warning(
                "Failed to delete session folder %s: %s", output_dir, e
            )

    parent_dir = output_dir.parent
    try:
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()
    except Exception as e:
        logging.getLogger('graphic_designer.cleanup').warning(
            "Failed to prune user folder %s: %s", parent_dir, e
        )
    return removed


def cleanupSessionArtifacts(user_id, session_id) -> bool:
    """Remove a session's layering artifacts by identity rather than by
    ``detection_result``.

    The in-flow cleanup below only fires once a layering run reaches its own
    success or failure path. A socket that drops mid-run — or a layering
    interrupt the user never answers — leaves the folder behind for good, so
    the websocket teardown calls this to close that gap.

    :returns: True if a directory was actually removed.
    :rtype: bool
    """
    _detection_cache.pop(_detection_cache_key(user_id, session_id), None)
    _text_crop_cache.pop(_detection_cache_key(user_id, session_id), None)
    return _remove_artifact_dir(sessionArtifactsDir(user_id, session_id))


def _cleanup_local_artifacts(detection_result) -> None:
    """Remove local detection/segmentation artifacts after results are in state.

    Deletes the per-session output directory (e.g. layers/output/<user_id>/<session_id>/)
    and prunes the parent user directory if it is left empty.
    """
    if detection_result is None:
        return
    output_dir = getattr(detection_result, 'output_dir', None)
    if output_dir is None:
        return
    _remove_artifact_dir(Path(output_dir) if not isinstance(output_dir, Path) else output_dir)


@profile
def segmentGenerator(state: LayeringSubGraph) -> LayeringSubGraph:
    """
    Generating layers using the bboxes
    """
    print("[segmentGenerator] generating layers from the bounding boxes")
    cache_key = _detection_cache_key(state.get('user_id'), state.get('session_id'))
    detection_result = _detection_cache.get(cache_key)
    try:
        cost = state.get("cost",0.0)
        current_image_url = state.get("current_image_url","")
        segmentation_result, extracted_texts = segmentation_start(detection_result)
        # Same as detection above: real pixel work, invisible to apiLog.
        markImageWork()
        state["segmentation_result"] = segmentation_result
        state["extracted_texts"] = extracted_texts
        # Added to what came in, not assigned over it. Neither path
        # interrupts between detection and segmentation any more, so the
        # detection spend is still pending in state and would otherwise be
        # wiped here and never billed. It rides to the next interrupt
        # (layering_executed) instead.
        state["cost"] = cost + 0.2 * (len(segmentation_result)-1)
        state["current_image_url"] = current_image_url
        _detection_cache.pop(cache_key, None)

        return state

    except Exception as e:
        logging.getLogger('graphic_designer.segmentGenerator').exception(
            "Segmentation pipeline failed: %s", e
        )
        # Clean up local detection artifacts on failure since segmentation.py's
        # cleanup (which runs on success) never executed.
        _cleanup_local_artifacts(detection_result)
        _detection_cache.pop(cache_key, None)
        state['segmentation_result'] = None
        state['extracted_texts'] = []
        state["bboxes"] = []
        return state


@profile
def pasteBackTexts(state: LayeringSubGraph) -> LayeringSubGraph:
    """Restore original pixels for any detected text the user un-selected.

    Runs after object segmentation/extraction (unchanged) so it edits the
    final background layer rather than the mid-pipeline working image --
    object erasure never sees or touches these regions either way.
    """
    print("[pasteBackTexts] restoring un-selected text regions")
    cache_key = _detection_cache_key(state.get('user_id'), state.get('session_id'))
    cache_entry = _text_crop_cache.pop(cache_key, None)
    restore_entries = (cache_entry or {}).get("restore") or []
    segmentation_result = state.get("segmentation_result") or []
    mattes = (cache_entry or {}).get("mattes")
    if mattes is not None:
        # LLM_TEXT_LAYERING run: the unpicked texts travel as difference mattes.
        if mattes and segmentation_result:
            try:
                _paste_back_mattes(segmentation_result, mattes, cache_entry.get("original"))
            except Exception as e:
                logging.getLogger('graphic_designer.pasteBackTexts').exception(
                    "Failed to paste texts back onto final image: %s", e
                )
        return state
    if not restore_entries or not segmentation_result:
        return state
    try:
        _restore_texts_on_background(segmentation_result, restore_entries, (cache_entry or {}).get("crops") or {})
    except Exception as e:
        logging.getLogger('graphic_designer.pasteBackTexts').exception(
            "Failed to paste un-selected text back onto final image: %s", e
        )
    return state


GLYPH_ALPHA_FLOOR = 0.35
"""Projection value below which a pixel counts as pure backdrop and is not
pasted back at all."""

GLYPH_ALPHA_CEIL = 0.80
"""Projection value at/above which a pixel counts as solid glyph. The gap
between the two is the anti-aliasing ramp."""


def _hex_to_rgb(value):
    """``"#rrggbb"`` (or ``"#rgb"``) -> float32 RGB triple; None if unparseable."""
    if not isinstance(value, str):
        return None
    v = value.strip().lstrip('#')
    if len(v) == 3:
        v = ''.join(c * 2 for c in v)
    if len(v) != 6:
        return None
    try:
        return np.array([int(v[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
    except ValueError:
        return None


def _otsu_text_colors(crop_rgb: np.ndarray):
    """(foreground, background) RGB estimated from the crop itself.

    The four CORNERS decide which Otsu class is the backdrop, not the full
    border. A bold, tightly-cropped headline can have glyph strokes reaching
    every edge of its own bbox -- a full-border vote reads that as "border is
    mostly glyph" and calls the glyphs the background. The corners of a tight
    text bbox are essentially never a glyph stroke (letters are letter-shaped,
    not corner-filling rectangles), so they stay a reliable backdrop sample
    even when the border as a whole is not.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = otsu > 0
    h, w = bright.shape
    patch = max(1, min(h, w) // 8)
    corner = np.zeros_like(bright)
    corner[:patch, :patch] = corner[:patch, -patch:] = True
    corner[-patch:, :patch] = corner[-patch:, -patch:] = True
    bg_mask = bright if bright[corner].mean() >= 0.5 else ~bright
    fg_mask = ~bg_mask
    if not fg_mask.any() or not bg_mask.any():
        return None, None
    pixels = crop_rgb.reshape(-1, 3).astype(np.float32)
    return (pixels[fg_mask.reshape(-1)].mean(axis=0),
            pixels[bg_mask.reshape(-1)].mean(axis=0))


def _project_glyph_alpha(crop_rgb: np.ndarray, fg, bg):
    """Project every pixel onto the background→foreground colour axis.

    Solid glyph pixels land near 1, backdrop near 0, and anti-aliased glyph
    edges in between -- which is what keeps the type looking smooth once it
    is composited onto a background it was never rendered against.
    """
    if fg is None or bg is None:
        return None
    axis = fg - bg
    denom = float(axis @ axis)
    if denom < 1e-6:
        return None
    return np.clip(((crop_rgb.astype(np.float32) - bg) @ axis) / denom, 0.0, 1.0)


def _border_mean(alpha: np.ndarray) -> float:
    """Mean alpha around the rim of the crop. The rim is backdrop by
    construction, so a high value means the mask has the polarity backwards
    (or is selecting the whole rectangle) and should not be trusted."""
    if alpha.shape[0] < 3 or alpha.shape[1] < 3:
        return float(alpha.mean())
    return float(np.concatenate([
        alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]
    ]).mean())


def _text_glyph_alpha(crop_rgb: np.ndarray, entry: dict) -> np.ndarray:
    """Soft 0..1 mask of the GLYPHS inside a text crop.

    Uses the colours the style analyzer already measured at extraction time,
    and falls back to an Otsu split of the crop when those are missing, too
    close together to separate anything by, or produce a mask that bleeds
    onto the crop's rim. The final ramp drops the low-alpha haze -- the
    backdrop the glyph edges were originally blended against -- so the old
    scene is not smeared back onto the new background.
    """
    fg = _hex_to_rgb(entry.get('font_color')) if isinstance(entry, dict) else None
    bg = _hex_to_rgb(entry.get('background_color')) if isinstance(entry, dict) else None
    alpha = None
    if fg is not None and bg is not None and float(np.linalg.norm(fg - bg)) >= 24.0:
        alpha = _project_glyph_alpha(crop_rgb, fg, bg)

    if alpha is None or _border_mean(alpha) > 0.5:
        fallback = _project_glyph_alpha(crop_rgb, *_otsu_text_colors(crop_rgb))
        if fallback is not None and (alpha is None or _border_mean(fallback) <= _border_mean(alpha)):
            alpha = fallback

    if alpha is None:
        # Nothing separable in this crop (single flat colour): fall back to
        # the old whole-rectangle paste rather than dropping the text.
        return np.ones(crop_rgb.shape[:2], dtype=np.float32)

    alpha = np.clip((alpha - GLYPH_ALPHA_FLOOR) / (GLYPH_ALPHA_CEIL - GLYPH_ALPHA_FLOOR), 0.0, 1.0)
    return alpha.astype(np.float32)


def _restore_texts_on_background(segmentation_result: list, restore_entries: list, crops: dict) -> None:
    """Download the final background layer, paste each restored text's
    original GLYPHS back onto it, and re-upload to the same S3 key so no
    downstream URL needs to change.

    Only the glyph pixels are composited where that is possible, never the
    whole crop rectangle by preference: the rectangle also holds whatever
    surrounded the text in the ORIGINAL image, and by this point that surround
    is stale -- objects have been peeled out from under it and the hole
    inpainted -- so stamping it whole printed a patch of the old scene (a ghost
    of the peeled object) around every restored text.

    NOTHING HERE IS ALLOWED TO FAIL QUIETLY. These texts were erased on the
    user's behalf, not at their request: they were lying over an object being
    peeled and had to go so SAM2 could cut a clean edge. If the restore drops
    one, copy the user wrote is gone from their design. Every path that used to
    ``return``/``continue`` silently now says so, and the one judgement call in
    here -- the glyph mask -- degrades to a full-rectangle paste rather than
    pasting nothing, because a small stale surround is recoverable and missing
    text is not.
    """
    log = logging.getLogger('graphic_designer.pasteBackTexts')
    wanted = [e for e in restore_entries if isinstance(e, dict)]
    if not wanted:
        return

    bg_url = next((u for u in segmentation_result if isinstance(u, str) and 'background' in u.lower()), None)
    if not bg_url:
        log.error("no background layer in %r — %d erased text(s) CANNOT be restored: %s",
                  segmentation_result, len(wanted), [e.get('text') for e in wanted])
        return

    resp = requests.get(bg_url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    bg_image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if bg_image is None:
        log.error("background %s did not decode — %d erased text(s) CANNOT be restored: %s",
                  bg_url, len(wanted), [e.get('text') for e in wanted])
        return

    h, w = bg_image.shape[:2]
    restored, failed = [], []
    for entry in wanted:
        label = (entry.get('text') or '')[:40]
        cached = crops.get(entry.get('id'))
        if not cached:
            failed.append((label, f"no cached crop for id {entry.get('id')!r}"))
            continue
        crop_rgb, (x1, y1, x2, y2) = cached
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            failed.append((label, f"rect {(x1, y1, x2, y2)} outside the {w}x{h} background"))
            continue
        target_h, target_w = y2c - y1c, x2c - x1c
        if crop_rgb.shape[:2] != (target_h, target_w):
            crop_rgb = cv2.resize(crop_rgb, (target_w, target_h))

        alpha = _text_glyph_alpha(crop_rgb, entry)
        # A mask that selects (almost) nothing means the glyph/backdrop split
        # failed on this crop -- flat art, an anti-aliased hairline, a colour
        # pair the projection cannot separate. Paste the rectangle instead of
        # dropping the text: this is the branch that used to `continue`.
        how = "glyphs"
        if alpha is None or not alpha.any() or float(alpha.mean()) < 0.002:
            alpha = np.ones(crop_rgb.shape[:2], dtype=np.float32)
            how = "full rect (glyph mask degenerate)"

        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        region = bg_image[y1c:y2c, x1c:x2c, :3].astype(np.float32)
        a = alpha[..., None]
        bg_image[y1c:y2c, x1c:x2c, :3] = np.clip(
            crop_bgr * a + region * (1.0 - a), 0, 255).astype(np.uint8)
        if bg_image.shape[2] == 4:
            # The background layer is opaque anyway; never make it LESS opaque
            # than it already was where a glyph lands.
            bg_image[y1c:y2c, x1c:x2c, 3] = np.maximum(
                bg_image[y1c:y2c, x1c:x2c, 3], (alpha * 255).astype(np.uint8))
        restored.append((label, how))

    for label, why in failed:
        log.error("could not restore erased text %r: %s", label, why)
    print(f"[pasteBackTexts] restored {len(restored)}/{len(wanted)} erased text(s)"
          + (f"; FAILED: {[l for l, _ in failed]}" if failed else ""))
    for label, how in restored:
        print(f"   ✏️  {label!r} via {how}")

    if not restored:
        log.error("not one of %d erased text(s) could be restored — background left as-is",
                  len(wanted))
        return

    ok, encoded = cv2.imencode('.png', bg_image)
    if not ok:
        log.error("re-encoding the background failed — %d restored text(s) LOST", len(restored))
        return

    from graphic_designer.config import s3_client, DOCUMENT_BUCKET
    key = bg_url.split(".amazonaws.com/", 1)[-1]
    try:
        s3_client.put_object(
            Bucket=DOCUMENT_BUCKET, Key=key, Body=encoded.tobytes(),
            ContentType='image/png', ACL='public-read',
        )
    except Exception as e:
        # Caught and named here rather than surfacing as pasteBackTexts' generic
        # "failed to paste" -- this specific failure means the restored pixels
        # exist but never reached the bucket, so the user sees text missing.
        log.error("re-uploading the restored background to %s failed (%s) — "
                  "%d text(s) LOST from the delivered image", key, e, len(restored))
        raise


def _matte_cache_fields(detection_result) -> dict:
    """Paste-back data an LLM_TEXT_LAYERING run leaves for ``pasteBackTexts``.
    Empty for the other paths, which keep using crops/restore."""
    mattes = getattr(detection_result, 'text_restore_mattes', None)
    if mattes is None:
        return {}
    fields = {"mattes": mattes}
    if mattes and getattr(settings, 'LLM_TEXT_PASTE_BACK', 'matte') == 'llm':
        fields["original"] = detection_result.original_rgb
    return fields


def _paste_back_mattes(segmentation_result: list, mattes: list, original_rgb) -> None:
    """Put the texts the user did not pick back onto the final background layer
    and re-upload it under the same S3 key.

    ``matte`` mode composites each text's own glyph pixels from the original,
    so where the background did not change the result is the original exactly.
    A text lying over a peeled object lands on the background, under that
    object's layer. ``llm`` mode asks nano-banana-2 to redraw them, clamped to
    the text regions, and falls back to ``matte`` if the call fails.
    """
    from layers import text_layering as tl
    log = logging.getLogger('graphic_designer.pasteBackTexts')

    bg_url = next((u for u in segmentation_result if isinstance(u, str) and 'background' in u.lower()), None)
    if not bg_url:
        log.error("no background layer in %r — %d text(s) CANNOT be pasted back: %s",
                  segmentation_result, len(mattes), [m.get('text') for m in mattes])
        return

    resp = requests.get(bg_url, timeout=30)
    resp.raise_for_status()
    bg_bgr = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bg_bgr is None:
        log.error("background %s did not decode — %d text(s) CANNOT be pasted back", bg_url, len(mattes))
        return
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    h, w = bg_rgb.shape[:2]

    pasted = None
    if original_rgb is not None and original_rgb.shape[:2] == (h, w):
        from graphic_designer.config import GEMINI_API_KEY
        raw = tl.llm_paste_back(original_rgb, bg_rgb, [m.get('text') for m in mattes], GEMINI_API_KEY)
        if raw is not None:
            markImageWork()
            region = np.zeros((h, w), dtype=np.uint8)
            for m in mattes:
                x1, y1, x2, y2 = m["rect"]
                region[y1:y2, x1:x2] = np.maximum(region[y1:y2, x1:x2],
                                                  (m["rgba"][..., 3] > 0).astype(np.uint8) * 255)
            pasted = tl.clamp_to_mask(bg_rgb, raw, region)
            print(f"[pasteBackTexts] {len(mattes)} text(s) redrawn by nano-banana-2")
        else:
            log.warning("LLM paste-back failed — falling back to matte composite")

    if pasted is None:
        pasted = bg_rgb.copy()
        failed = [m.get('text') for m in mattes if not tl.composite_matte(pasted, m)]
        for label in failed:
            log.error("could not paste back text %r: rect outside the %dx%d background", label, w, h)
        print(f"[pasteBackTexts] composited {len(mattes) - len(failed)}/{len(mattes)} text(s)")

    ok, encoded = cv2.imencode('.png', cv2.cvtColor(pasted, cv2.COLOR_RGB2BGR))
    if not ok:
        log.error("re-encoding the background failed — %d pasted-back text(s) LOST", len(mattes))
        return

    from graphic_designer.config import s3_client, DOCUMENT_BUCKET
    key = bg_url.split(".amazonaws.com/", 1)[-1]
    s3_client.put_object(
        Bucket=DOCUMENT_BUCKET, Key=key, Body=encoded.tobytes(),
        ContentType='image/png', ACL='public-read',
    )


@profile
def layeringSkipped(state: LayeringSubGraph) -> LayeringSubGraph:
    """
    If layering is skipped, we set bboxes and segmentation results to empty and move to next step
    """

    print("[layeringSkipped] skipping the layering workflow")
    user_response = interrupt({
        "type": "text_input",
        "question": LAYERING_SKIPPED,
        "prompt": state.get('prompt', ''),
        "cost": state.get('cost', 0.0),
        "current_image_url": state.get('current_image_url', '')
    })

    user_response = parseJson(user_response)

    return Command(goto=END, update={**state, "prompt": user_response.get('response', ''), "cost": state.get('cost', 0.0), "current_image_url": state.get('current_image_url', ''), "attached_images": state.get('attached_images', []), "conversation_history": state.get('conversation_history', []) + [{"role": "user", "content": user_response.get('response', '')}], "suggestion_type": "user_driven", "model": state.get("model", ""), "aspect_ratio": state.get('aspect_ratio', ''), "bboxes": [], "segmentation_result": None})


@profile
def layExit(state: LayeringSubGraph) -> Command:
    """Leave the layering workflow without re-asking what to do next.

    Used for new_edit / new_generation / suggestions: the message that got
    classified into one of those already states the user's next action, so
    (unlike layeringSkipped) this doesn't interrupt again — it just clears
    bboxes/segmentation_result and ends, which sends layeringManager straight
    to the navigator to classify state['prompt'] itself.
    """
    print("[layExit] leaving the layering workflow without a re-ask")
    return Command(goto=END, update={**state, "bboxes": [], "segmentation_result": None})


layeringGraph = StateGraph(LayeringSubGraph)

layeringGraph.add_node("layeringHandler", layeringHandler)
layeringGraph.add_node("bboxGenerator", bboxGenerator)
layeringGraph.add_node("layObjectAsk", layObjectAsk)
layeringGraph.add_node("manualLayering", manualLayering)
layeringGraph.add_node("segmentGenerator", segmentGenerator)
layeringGraph.add_node("pasteBackTexts", pasteBackTexts)
layeringGraph.add_node("layQuestion", questionHandler)
layeringGraph.add_node("layChitchat", chitChatHandler)
layeringGraph.add_node("layUnclear", unclearHandler)
layeringGraph.add_node("layDoNotLayer", layeringSkipped)
layeringGraph.add_node("layNewEdit", layExit)
layeringGraph.add_node("layNewGeneration", layExit)
layeringGraph.add_node("laySuggestions", layExit)

layeringGraph.add_edge(START, "layeringHandler")
layeringGraph.add_edge("segmentGenerator", "pasteBackTexts")
layeringGraph.add_edge("pasteBackTexts", END)

lgb = layeringGraph.compile()