"""Critique graph node: robust, parallel, instrumented ``suggestionCreator``.

Drop-in replacement for the four near-duplicate branches in the old
``suggestionCreator`` (nodes.py:1816-1931). Reuses your EXISTING ``techAgent`` and mirrors
the existing flatten loop so the produced ``s_stack`` is identical in shape to today's.
"""

import json
import contextvars
from concurrent.futures import ThreadPoolExecutor

from langgraph.types import Command

from ..utils import parseJson, profile
from ..image_metadata import read_url
from .telemetry import Telemetry
from .grounding import resolveGrounding, mergeConstraints, genericLimits
from .agents import (
    caseAgent, integrityEval, compositionEval, contentEval, aestheticEval,
    mergeFindings, arbiterAgent, executabilityGate, copyFreeze,
)

# CEILING, not a target: the user is shown the top 0-3 AI-executable suggestions, never padded.
# The count is decided by the quality floor (MIN_DISPLAY_CONFIDENCE / aesthetic cap) and the
# executability gate in agents.py — a clean design correctly returns 0. If a batch keeps pinning
# to 3, the floor is too permissive — raise MIN_DISPLAY_CONFIDENCE rather than lowering this.
MAX_SUGGESTIONS = 3


def _run_tech_agent(arbiter_json, tele):
    """Call the existing techAgent (model + executability metadata).

    Kept as a lazy import so this package still imports/tests standalone; if the real
    techAgent is unavailable it passes findings through unchanged.
    """
    try:
        from ..subgraphs.suggestion import techAgent          # your existing agent
        with tele.span("techAgent", "existing") as box:
            resp, cost = techAgent(arbiter_json)
            box["cost"] = cost or 0.0
        return resp
    except Exception:
        return arbiter_json


def structureSuggestions(tech_json):
    """Flatten the arbiter/techAgent output into the exact s_stack contract the frontend
    consumes. Internal fields (source_lens, checklist_category, confidence, evidence, ...) are
    dropped; only the frontend fields are emitted, in order. ``edits`` = total suggestion count.
    """
    data = parseJson(tech_json) if isinstance(tech_json, str) else tech_json
    images = data.get("images", []) or []
    edits = sum(len(im.get("recommendations", []) or []) for im in images)
    stack, idx = [], 0
    for image in images:
        image_id = image.get("image_id", "image_1")
        for rec in image.get("recommendations", []) or []:
            stack.append({
                "image_id": image_id,
                "s_id": idx,
                "status": "pending",
                "focus": False,
                "user_response": None,
                "edits": edits,
                "type": rec.get("type", "EDIT"),
                "tier": rec.get("tier"),
                "topic": rec.get("topic"),
                "user_friendly_text": rec.get("user_friendly_text"),
                "nano_prompt": rec.get("nano_prompt"),
            })
            idx += 1
    return stack


def _isEditInstruction(base_prompt):
    """True when base_prompt is a compiled EDIT instruction rather than a generation spec.
    compilerAgent always opens with this anchor (SUGGESTION_COMPILATION_PROMPT, STEP 2)."""
    return (base_prompt or "").lstrip().lower().startswith("using the current image as reference")


def _withGraphicBrief(state):
    """With the frame's graphic loaded (``active_graphic``), its stored brief is the spec for
    THIS artwork. The session-wide base_prompt may belong to whichever graphic ran last (or be
    gone after a reconnect), so it is replaced: a generated graphic's brief, none for an upload.
    Frames without a graphic keep base_prompt as it is."""
    active = state.get("active_graphic")
    if not active:
        return state
    return {**state, "base_prompt": active.get("brief") if active.get("source") == "generated" else None}


def _copyDeck(images, src):
    """The copywriter's text for each generated image, read from the XMP the generation step
    embedded in the file: {image_id: {slot: string}}. An image without one (edited since — an
    edit can change the wording — or produced before the copywriter) is simply absent."""
    if src != "url":
        return {}
    deck = {}
    for i, url in enumerate(images, 1):
        if not str(url).startswith(("http://", "https://")):
            continue
        locked = read_url(url).get("locked_text")
        if isinstance(locked, dict):
            slots = {str(k): str(v) for k, v in locked.items() if str(v or "").strip()}
            if slots:
                deck[f"image_{i}"] = slots
    return deck


@profile
def suggestionCreator(state):
    tele = Telemetry()
    base_cost = state.get("cost", 0.0)

    state = _withGraphicBrief(state)

    # After an accepted edit, base_prompt holds the edit instruction ("Remove the two 'SHOP AT
    # NORTHFIELD.CO' lines ..."). It describes the image BEFORE the edit, so lenses that read it
    # as a spec re-reported defects the edit had already fixed — live, both re-critiques did.
    # Only a real generation spec is a brief to check fidelity against.
    if _isEditInstruction(state.get("base_prompt")):
        state = {**state, "base_prompt": None}

    images = [i for i in state.get("current_image_url", "").split("|") if i] \
        or state.get("attached_images", []) or []
    src = "url" if state.get("current_image_url") else "b64"

    # -- Grounding (main thread; may interrupt to ask an upload for goal/platform) --------
    g = resolveGrounding(state)
    layout, brand, eff, avail = g["layout"], g["brand"], g["eff"], g["avail"]
    # COPY FREEZE: a generated design's wording came from our copywriter, so its critique only
    # repairs defects in the copy, checked against the deck. Uploads keep the full copy review.
    copy_deck = _copyDeck(images, src) if avail.get("provenance") == "generated" else {}
    avail = {**avail, "copy_deck": copy_deck}
    tele.note("copy_freeze", {"active": avail.get("provenance") == "generated",
                              "deck_images": sorted(copy_deck)})
    neg_tags = (layout or {}).get("shared_rules", {}).get("negative_prompt_tags", [])
    # Whether brand grounding actually reached the lenses. A brand kit in state that normalizes
    # to nothing (unknown shape, assets only) shows up here as present=False.
    tele.note("brand", {
        "attached": bool(state.get("brand_dna")),
        "present": brand is not None,
        "colours": len(((brand or {}).get("palette") or {}).get("named") or []),
        "fonts": bool(((brand or {}).get("fonts") or {}).get("families")),
        "voice": bool((brand or {}).get("personality")),
    })

    # -- Lens gating: Integrity + Composition + Aesthetic always; Content conditional -----
    # NOTE on "skip lenses that can't contribute": Content already self-skips (returns []) when
    # there's nothing to say. We don't spend an extra caseAgent call to pre-gate it in grounded
    # mode too — these design goals (flash sale, product launch, etc.) virtually always carry
    # text, so the extra paid vision call would rarely change the outcome. caseAgent is reserved
    # for generic mode, where "is there even text here" is a genuinely open question.
    run_integrity = True
    run_composition = True
    run_aesthetic = True
    if avail["layout"] or avail["brand"] or avail["platform"]:
        run_content = True                       # clearly worth a content pass -> skip caseAgent
    else:
        run_content = caseAgent(images, src, tele)["run_content"]

    # -- Phase B: all four lenses, fully parallel ----------------------------------------
    # This used to be two sequential phases (B1 then B2) because composition/aesthetic had to
    # wait on matchLayoutVariant to pick which wireframe's zones to ground against. The critique
    # no longer references wireframes at all, so that dependency — and the paid vision call that
    # created it — is gone, and every lens can start at once.
    lens_results = []
    # ThreadPoolExecutor does not carry ContextVars into its workers, so the
    # node name (and spend-session, for the breaker) are captured here and
    # re-applied inside each lens. A single Context object can't be entered by
    # two threads at once though -- `_ctx.run` shared across concurrent
    # submissions here raced ("cannot enter context: ... is already entered"),
    # which silently dropped a lens's findings (caught below as []) whenever
    # two futures actually overlapped. Each submission gets its own fresh copy
    # (still captures the same node name / spend session at submission time).
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = []
        if run_integrity:
            futs.append(ex.submit(contextvars.copy_context().run, integrityEval, images, src,
                                  state.get("base_prompt"), neg_tags, avail, state, tele, brand))
        if run_content:
            futs.append(ex.submit(contextvars.copy_context().run, contentEval, images, src, layout,
                                  brand, eff, avail, state, tele))
        if run_composition:
            futs.append(ex.submit(contextvars.copy_context().run, compositionEval, images, src,
                                  layout, brand, eff, avail, state, tele))
        if run_aesthetic:
            futs.append(ex.submit(contextvars.copy_context().run, aestheticEval, images, src,
                                  layout, brand, eff, avail, state, tele))
        for f in futs:
            try:
                lens_results.append(f.result())
            except Exception:
                lens_results.append([])          # one lens failing must not kill the critique

    merged = copyFreeze(mergeFindings(lens_results), avail.get("provenance"), copy_deck, tele)
    total_recs = sum(len(im.get("recommendations", [])) for im in merged)

    # -- Phase C: arbiter, SKIPPED when a single lens already returned <= N ---------------
    lenses_run = sum([run_integrity, run_composition, run_content, run_aesthetic])
    if lenses_run <= 1 and total_recs <= MAX_SUGGESTIONS:
        # No arbiter to curate, but the executability gate still applies — the user must only
        # ever see suggestions the edit model can actually carry out.
        kept, not_exec = executabilityGate(
            [r for im in merged for r in im.get("recommendations", [])], tele)
        ok = {id(r) for r in kept}
        for im in merged:
            im["recommendations"] = [{k: v for k, v in r.items() if k != "fid"}
                                     for r in im.get("recommendations", []) if id(r) in ok]
        tele.note("not_executable", not_exec)
        arbiter_json = json.dumps({"status": "success", "case": 1, "images": merged})
    else:
        arbiter_json = arbiterAgent(merged, brand, eff, avail, MAX_SUGGESTIONS, tele)

    # -- techAgent (existing) -> final frontend structure ---------------------------------
    tech_json = _run_tech_agent(arbiter_json, tele)
    s_stack = structureSuggestions(tech_json)

    # Distribution inputs for tuning the quality floor: how many findings the lenses raised vs how
    # many actually reached the user, plus the grounding context that shaped them. If final_count
    # keeps pinning to MAX_SUGGESTIONS across a batch, the floor is still too permissive.
    tele.note("suggestion_counts", {
        "raw_findings": total_recs,
        "final_count": len(s_stack),
        "max_suggestions": MAX_SUGGESTIONS,
        "provenance": avail.get("provenance"),
        "mode": avail.get("mode"),
        "lenses_run": lenses_run,
    })

    # Job cost + time record for the DB write after generation.
    job = tele.summary()                         # {cost, wall_s, tokens, phases:[...], notes:{}}
    return Command(goto="suggestionHandler", update={
        "base_prompt": None,
        # Persist the origin decision so later critiques of this same artwork (after an edit,
        # which rewrites base_prompt) don't re-derive it from signals that no longer describe
        # where the image came from. Cleared by v32Generation/v32CreateMore on a genuinely new image.
        "critique_provenance": avail.get("provenance"),
        "s_stack": s_stack,
        "cost": base_cost + job["cost"],
        "critique_telemetry": job,               # persist this row: cost, wall_s, tokens, phases
    })
