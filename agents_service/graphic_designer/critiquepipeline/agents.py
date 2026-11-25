"""Critique lens + arbiter wrappers.

Each ``*Eval`` builds a grounded prompt, runs one vision LLM call, and returns the parsed
``{"images": [...]}`` findings (tagged with its ``source_lens``). ``arbiterAgent`` merges
the lenses, curates, and returns a creator-compatible JSON string so your EXISTING
``techAgent`` + flatten loop consume it unchanged.

Every call is wrapped in ``tele.span(...)`` for cost + time tracking.
"""

import json
import re
import base64

from ..config import GEMINI_API_KEY
from ..utils import parseJson, makeApiRequest, calculateCost, apiLog, messageText
from .prompts import (
    render, DEGRADATION_BLOCK, SHARED_OUTPUT_NOTE, NANO_PROMPT_RULES, EDIT_RISK_RULE,
    CASE_AGENT_PROMPT, INTEGRITY_EVALUATOR_PROMPT, COMPOSITION_EVALUATOR_PROMPT,
    CONTENT_COMPLIANCE_EVALUATOR_PROMPT, AESTHETIC_EVALUATOR_PROMPT,
    ARBITER_PROMPT, SUGGESTION_COMPILATION_PROMPT, EDIT_MODE_BLOCK, REGEN_MODE_BLOCK,
    VERIFY_EDIT_PROMPT, EXECUTABILITY_HINT, EXECUTABILITY_PROMPT, intentBlock, brandBlock,
    copyFreezeBlock,
)

# Per-phase model + thinking budget. Thinking is capped to kill the latency runaway while
# protecting quality: the 3 vision lenses + arbiter keep a SMALL budget (real judgment);
# the mechanical caseAgent + compiler get 0. Pure-text steps use the faster flash-lite tier.
LENS_MODEL = "gemini-3-flash-preview"  # vision-capable; the quality-critical lens calls
LENS_THINKING = 512                    # keep judgment quality, cap the runaway
CASE_MODEL = "gemini-3-flash-preview"  # vision (text/logo detection)
CASE_THINKING = 0                      # mechanical
ARBITER_MODEL = "gemini-3-flash-preview"  # curation is real judgment — reliability > speed
ARBITER_THINKING = 512                 # dedupe/rank/keep-decisions
COMPILER_MODEL = "gemini-3.1-flash-lite"  # mechanical fusion: fast lite tier is fine
COMPILER_THINKING = 0
EXEC_MODEL = "gemini-3-flash-preview"  # executability gate: text-only, but a real judgement call
EXEC_THINKING = 256


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _img_type(raw):
    """Detect image type from magic bytes (imghdr was removed in Py3.13). Defaults to png."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "png"


def _to_data_uri(img, src):
    """Normalize a single image (URL, b64 string, or raw bytes) to a ``data:image/...`` URI."""
    if isinstance(img, (bytes, bytearray)):
        b64 = base64.b64encode(img).decode("utf-8")
        return f"data:image/{_img_type(img)};base64,{b64}"
    if src == "url":
        raw = makeApiRequest("GET", img, timeout=30).content
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:image/{_img_type(raw)};base64,{b64}"
    # src == 'b64'
    if isinstance(img, str) and img.startswith("data:"):
        return img
    return f"data:image/png;base64,{img}"


def _model(name, thinking_budget=None):
    # Lazy import so the module loads without langchain (e.g. offline tests / alt transports).
    from langchain_google_genai import ChatGoogleGenerativeAI
    kw = {"model": name, "temperature": 0, "google_api_key": GEMINI_API_KEY}
    if thinking_budget is not None:
        kw["thinking_budget"] = thinking_budget     # cap Gemini 2.5 thinking latency
    return ChatGoogleGenerativeAI(**kw)


def _vision_invoke(phase, model_name, prompt_text, images, src, tele, thinking_budget=None):
    """One multimodal call: text + N image parts -> parsed JSON dict."""
    from langchain_core.messages import HumanMessage
    content = [{"type": "text", "text": prompt_text}]
    for img in images:
        content.append({"type": "image_url", "image_url": _to_data_uri(img, src)})
    apiLog("google", model_name, f"critic:{phase}", status="start",
           detail=f"{len(images)} image(s)")
    with tele.span(phase, model_name) as box:
        resp = _model(model_name, thinking_budget).invoke([HumanMessage(content=content)])
        usage = resp.usage_metadata or {}
        box["tokens"] = usage.get("total_tokens", 0)
        box["cost"] = calculateCost(model_name, box["tokens"])
    # Vision calls carry image tokens and run three-wide in parallel, so they are
    # the critique's real spend — logged per call, not just in the telemetry roll-up.
    # tokens_in/tokens_out (not just the total) let apiLog price actual_cost off
    # the real per-direction provider rate instead of a flat input-rate floor.
    apiLog("google", model_name, f"critic:{phase}", cost=box.get("cost"),
           tokens_in=usage.get("input_tokens"), tokens_out=usage.get("output_tokens"))
    return parseJson(messageText(resp).strip())


def _text_invoke(phase, model_name, prompt_text, tele, thinking_budget=None):
    apiLog("google", model_name, f"critic:{phase}", status="start")
    with tele.span(phase, model_name) as box:
        resp = _model(model_name, thinking_budget).invoke([{"role": "user", "content": prompt_text}])
        usage = resp.usage_metadata or {}
        box["tokens"] = usage.get("total_tokens", 0)
        box["cost"] = calculateCost(model_name, box["tokens"])
    apiLog("google", model_name, f"critic:{phase}", cost=box.get("cost"),
           tokens_in=usage.get("input_tokens"), tokens_out=usage.get("output_tokens"))
    return messageText(resp).strip()


def _degradation(avail):
    # provenance defaults to "generated" so an avail dict built without it (older callers,
    # tests) keeps the pre-existing behaviour rather than silently disabling real checks.
    return render(
        DEGRADATION_BLOCK,
        mode=avail["mode"],
        brand_state="present" if avail["brand"] else "absent",
        base_state="present" if avail["base_prompt"] else "absent",
        provenance=avail.get("provenance", "generated"),
        copy_freeze=copyFreezeBlock(avail.get("provenance", "generated"), avail.get("copy_deck")),
    )


def _output(lens, image_count):
    return render(SHARED_OUTPUT_NOTE, lens=lens, image_count=image_count,
                  nano_rules=NANO_PROMPT_RULES, edit_risk_rule=EDIT_RISK_RULE,
                  executability_hint=EXECUTABILITY_HINT)


def _tag(result, lens):
    """Ensure every recommendation carries source_lens + edit_risk, and deterministically force
    type="GENERATION" for structural-risk items regardless of whether the LLM set it correctly
    (routing must not depend on prompt compliance alone)."""
    imgs = (result or {}).get("images", []) or []
    for im in imgs:
        for rec in im.get("recommendations", []) or []:
            rec.setdefault("source_lens", lens)
            rec.setdefault("edit_risk", "local")
            if rec.get("edit_risk") == "structural" and rec.get("type") != "UPSCALE":
                rec["type"] = "GENERATION"
    return imgs


# --------------------------------------------------------------------------- #
# orchestrator pre-check (only called when no layout / brand / platform signal exists)
# --------------------------------------------------------------------------- #
def caseAgent(images, src, tele):
    try:
        out = _vision_invoke("caseAgent", CASE_MODEL, CASE_AGENT_PROMPT, images, src, tele,
                             CASE_THINKING)
        return {"run_content": bool(out.get("has_text_or_logo")),
                "comparison_intent": bool(out.get("comparison_intent"))}
    except Exception:
        return {"run_content": True, "comparison_intent": False}   # fail open


# --------------------------------------------------------------------------- #
# NO WIREFRAME GROUNDING (deliberate)
# --------------------------------------------------------------------------- #
# The critique NEVER references wireframes — i.e. `layout_variants[].zones[]`, the same zone
# data v5_3_wireframe_renderer draws. A wireframe is one of several ranked ways the design COULD
# have been laid out; it is not a contract the finished artwork owes anything to. Grounding the
# critique on it produced suggestions phrased against an invisible template ("move the headline
# into the designated overlay zone", "add a button in the designated CTA zone") — meaningless to
# a user looking at their image, and it pushed the arbiter toward whole-canvas relayouts, the
# single highest-risk edit category (see _capHighRiskConfidence).
#
# Removed with it: matchLayoutVariant (a paid vision call whose ONLY consumer was zone selection)
# and _compact_zones.
#
# STILL USED from layout_profile — these are `shared_rules`, not wireframes, and describe the
# design goal rather than one candidate arrangement: typography_scales, negative_rules,
# required_content_fields, mandatory_elements, reading_order_description, plus the derived
# limits (max_colors, contrast_floor, logo_max_pct, safe_zone).
#
# compact grounding — keep every RULE/label/constraint, drop verbose prose so the
# lens prompts stay small (speed) without losing any signal (quality).
def _compact_fields(fields):
    return [{"field_id": f.get("field_id"), "label": f.get("label"),
             "required_level": f.get("required_level"), "can_infer": f.get("can_infer")}
            for f in (fields or [])]


def _arbiterRules(eff):
    """effective_rules for the arbiter minus the brand identity copies — the arbiter gets the
    brand through brandBlock (named colours, voice, style), so repeating the raw palette/font/mood
    dicts here only adds tokens and a second, unnamed view of the same brand."""
    return {k: v for k, v in (eff or {}).items()
            if k not in ("palette_identity", "font_identity", "mood_identity")}


# --------------------------------------------------------------------------- #
# lenses
# --------------------------------------------------------------------------- #
def integrityEval(images, src, base_prompt, artifact_tags, avail, state, tele, brand=None):
    prompt = render(
        INTEGRITY_EVALUATOR_PROMPT,
        degradation=_degradation(avail),
        base_prompt=base_prompt or "",
        user_intent_block=intentBlock(avail.get("intent")),
        brand_block=brandBlock(brand),
        purpose=state.get("purpose", ""),
        platform=avail.get("platform", ""),
        artifact_tags=json.dumps(artifact_tags or []),
        output=_output("integrity", len(images)),
    )
    return _tag(_vision_invoke("integrity", LENS_MODEL, prompt, images, src, tele, LENS_THINKING),
                "integrity")


def compositionEval(images, src, layout, brand, eff, avail, state, tele):
    sr = (layout or {}).get("shared_rules", {})
    prompt = render(
        COMPOSITION_EVALUATOR_PROMPT,
        degradation=_degradation(avail),
        user_query=state.get("prompt", "") or "",
        user_intent_block=intentBlock(avail.get("intent")),
        clarification=state.get("clarification_question", "") or "",
        purpose=state.get("purpose", ""),
        platform=avail.get("platform", ""),
        reading_order=sr.get("reading_order_description", ""),
        typography_scales=json.dumps(sr.get("typography_scales", {})),
        negative_rules=json.dumps(sr.get("negative_rules", [])),
        max_colors=eff.get("effective_max_colors", ""),
        contrast_floor=eff.get("contrast_floor", ""),
        logo_max_pct=eff.get("logo_max_pct", ""),
        safe_zone=eff.get("safe_zone", ""),
        brand_block=brandBlock(brand),
        output=_output("composition", len(images)),
    )
    return _tag(_vision_invoke("composition", LENS_MODEL, prompt, images, src, tele, LENS_THINKING),
                "composition")


def contentEval(images, src, layout, brand, eff, avail, state, tele):
    sr = (layout or {}).get("shared_rules", {})
    b = brand or {}
    prompt = render(
        CONTENT_COMPLIANCE_EVALUATOR_PROMPT,
        degradation=_degradation(avail),
        design_goal=(layout or {}).get("design_goal", avail.get("goal", "")),
        user_intent_block=intentBlock(avail.get("intent")),
        purpose=state.get("purpose", ""),
        platform=avail.get("platform", ""),
        canvas_px=json.dumps((layout or {}).get("canvas_px", [])),
        aspect_ratio=(layout or {}).get("aspect_ratio", state.get("aspect_ratio", "")),
        required_content_fields=json.dumps(_compact_fields(sr.get("required_content_fields", []))),
        mandatory_elements=json.dumps(sr.get("mandatory_elements", [])),
        negative_rules=json.dumps(sr.get("negative_rules", [])),
        artifact_tags=json.dumps(sr.get("negative_prompt_tags", [])),
        brand_logo_url=b.get("logo_url", "") or "",
        logo_max_pct=eff.get("logo_max_pct", ""),
        brand_block=brandBlock(brand),
        output=_output("content", len(images)),
    )
    return _tag(_vision_invoke("content", LENS_MODEL, prompt, images, src, tele, LENS_THINKING),
                "content")


def aestheticEval(images, src, layout, brand, eff, avail, state, tele):
    """Senior-designer polish lens — elevates visual appeal WITHIN brand/text rules."""
    prompt = render(
        AESTHETIC_EVALUATOR_PROMPT,
        degradation=_degradation(avail),
        user_intent_block=intentBlock(avail.get("intent")),
        purpose=state.get("purpose", ""),
        platform=avail.get("platform", ""),
        brand_block=brandBlock(brand),
        output=_output("aesthetic", len(images)),
    )
    return _tag(_vision_invoke("aesthetic", LENS_MODEL, prompt, images, src, tele, LENS_THINKING),
                "aesthetic")


# --------------------------------------------------------------------------- #
# merge + arbiter
# --------------------------------------------------------------------------- #
def mergeFindings(lens_image_lists):
    """Combine per-lens image lists into one findings list keyed by image_id.

    Assigns a stable ``fid`` to every finding so the arbiter can select by id (compact
    output) and the full recommendation is reassembled in code.
    """
    by_id = {}
    order = []
    counter = 0
    for imgs in lens_image_lists:
        for im in imgs:
            iid = im.get("image_id", "image_1")
            if iid not in by_id:
                by_id[iid] = {"image_id": iid, "overall_assessment": "", "recommendations": []}
                order.append(iid)
            if im.get("overall_assessment"):
                prev = by_id[iid]["overall_assessment"]
                by_id[iid]["overall_assessment"] = (prev + " " + im["overall_assessment"]).strip()
            for rec in im.get("recommendations", []) or []:
                rec["fid"] = f"f{counter}"
                counter += 1
                by_id[iid]["recommendations"].append(rec)
    return [by_id[i] for i in order]


_STOPWORDS = {"text", "image", "design", "area", "this", "that", "with", "from", "into",
              "graphic", "canvas", "element", "elements", "some", "when", "than", "which",
              "using", "current", "reference", "preserve", "leave", "untouched", "existing",
              "instead", "before", "after", "without", "ensure", "keep", "make", "only",
              "also", "still", "much", "other", "around", "toward", "across"}


def _keywords(rec):
    text = f"{rec.get('topic', '')} {rec.get('evidence', '')}".lower()
    return {w for w in re.findall(r"[a-z]{4,}", text) if w not in _STOPWORDS}


# --------------------------------------------------------------------------- #
# TEXT FIXES ARE PROTECTED
# --------------------------------------------------------------------------- #
# Repairing what the words SAY (and whether they can be read at all) is the single highest-value
# thing this critique does — a clipped headline, a typo, or a leftover "SN_404_NOT_FOUND" is
# objectively broken, not a matter of taste. But four separate mechanisms below were quietly
# eating them: _predupe merged two different text findings that happened to share a keyword,
# the structural cap dropped a text reflow, the quality floor dropped a low-confidence one, and
# the top-N similarity check treated "headline clipped at the edge" and "tagline overlapping the
# edge" as the same suggestion. That is why a poster with two obvious text faults kept coming
# back with only one of them addressed.
# Rule now: a text fix is never dropped by dedupe/cap/floor/similarity, and it sorts to the top.
# Two text fixes are only ever merged when they target the SAME string.
_TEXT_NOUNS = {"text", "headline", "headlines", "tagline", "copy", "caption", "wording", "word",
               "words", "letter", "letters", "letterform", "letterforms", "glyph", "glyphs",
               "label", "subhead", "subheadline", "title", "typo", "spelling", "sentence",
               "phrase", "line", "paragraph", "footer", "price", "cta"}
_TEXT_PROBLEMS = {"clipped", "cropped", "truncated", "truncation", "cut", "cutoff", "overflow",
                  "overflowing", "overlap", "overlapping", "illegible", "unreadable", "legible",
                  "legibility", "readable", "readability", "garbled", "gibberish", "misspelled",
                  "misspelling", "spelling", "typo", "placeholder", "duplicated", "doubled",
                  "obscured", "hidden", "missing", "broken", "incomplete",
                  # "duplicate" (not just "duplicated") — live test: a 4-lens-corroborated duplicated
                  # CTA wasn't recognised as a text repair and lost its slot to a polish note.
                  "duplicate", "duplicates", "repeated", "twice"}
# Markers strong enough on their own — these only ever describe rendered copy.
_TEXT_STRONG = {"placeholder", "typo", "misspelled", "misspelling", "spelling", "garbled",
                "gibberish", "truncated", "clipped", "illegible", "unreadable"}


def _isTextFix(rec):
    """True when the recommendation repairs rendered copy — its wording, spelling, completeness,
    or whether it can be read. Deliberately NOT matched by pure taste calls about type size or
    hierarchy, which are style preferences rather than something factually broken."""
    blob = " ".join(str(rec.get(k, "") or "") for k in
                    ("topic", "evidence", "checklist_category", "user_friendly_text", "nano_prompt")).lower()
    words = set(re.findall(r"[a-z]+", blob))
    if words & _TEXT_STRONG:
        return True
    return bool((words & _TEXT_NOUNS) and (words & _TEXT_PROBLEMS))


_QUOTED_RE = re.compile(r"'([^']{2,40})'|\"([^\"]{2,40})\"")


def _textTargets(rec):
    """The literal strings a text fix names (quoted copy, e.g. 'LEVEL', 'SN_404_NOT_FOUND').

    Two text fixes naming DIFFERENT strings are different repairs however similar their wording
    looks, which is what stops the similarity check from collapsing them into one.
    """
    blob = " ".join(str(rec.get(k, "") or "") for k in ("topic", "evidence", "user_friendly_text", "nano_prompt"))
    out = set()
    for a, b in _QUOTED_RE.findall(blob):
        s = (a or b).strip().lower()
        if s:
            out.add(s)
    return out


def _sameTextTarget(a, b):
    """True only when two text fixes clearly address the same literal string."""
    ta, tb = _textTargets(a), _textTargets(b)
    if not ta or not tb:
        return False          # can't prove they're the same -> treat as distinct, keep both
    return bool(ta & tb)


def _keywords_full(rec):
    """Broader than _keywords: also looks at nano_prompt, since two DIFFERENT-looking findings
    can still resolve to a near-identical edit (used for top-N similarity, not pre-arbiter dedup
    of the same underlying defect)."""
    text = f"{rec.get('topic', '')} {rec.get('evidence', '')} {rec.get('nano_prompt', '')}".lower()
    return {w for w in re.findall(r"[a-z]{4,}", text) if w not in _STOPWORDS}


def _tooSimilar(a, b, threshold=0.4, min_shared=2):
    """True if two suggestions are similar enough that showing both in the top N is redundant —
    e.g. they'd target the same visible text/element even if two different lenses phrased it
    differently. Requires both a meaningful overlap RATIO and a minimum shared-keyword COUNT so
    two unrelated suggestions don't collide on one generic shared word."""
    # Two text repairs are only "the same" when they name the same literal string. Without this,
    # "the clipped 'OF SOUND' headline" and "the overlapping 'STAY TUNED' tagline" collide on
    # generic words (edge, legibility, overlap) and the second is silently dropped.
    if _isTextFix(a) and _isTextFix(b):
        return _sameTextTarget(a, b)
    ka, kb = _keywords_full(a), _keywords_full(b)
    shared = ka & kb
    if len(shared) < min_shared:
        return False
    return len(shared) / min(len(ka), len(kb)) >= threshold


_ELEMENT_COUNT_RE = re.compile(r"requires moving (\d+) elements?", re.IGNORECASE)


def _capHighRiskConfidence(recs, min_elements=3, cap=0.5):
    """Deterministic enforcement of the lens's own 'REQUIRES MOVING N ELEMENTS' self-report
    (EDIT_RISK_RULE). A multi-element structural fix is the single highest-risk change category
    (confirmed live: it duplicates/garbles content or produces seam artifacts even with a fully
    complete, well-named instruction) — this must not depend on the arbiter LLM noticing a phrase
    in free text. Caps confidence in code so the arbiter's own CONFIDENCE FILTER (drop < 0.45)
    and ranking naturally deprioritize it, regardless of what the lens set confidence to.

    COUPLING (deliberate): `cap` sits just BELOW MIN_DISPLAY_CONFIDENCE, so a self-reported 3+
    element relayout is not merely ranked last — the quality floor removes it. That is the
    intended outcome: this category is documented above as breaking the image more often than it
    fixes it, so proposing it is worse than staying quiet. Raising MIN_DISPLAY_CONFIDENCE keeps
    them dropped; lowering it below `cap` starts surfacing them again. Change the two together,
    on purpose."""
    for r in recs:
        if r.get("edit_risk") != "structural":
            continue
        m = _ELEMENT_COUNT_RE.search(r.get("evidence", "") or "")
        if m and int(m.group(1)) >= min_elements:
            r["confidence"] = min(r.get("confidence", 0) or 0, cap)
    return recs


def _textProblems(rec):
    blob = " ".join(str(rec.get(k, "") or "") for k in ("topic", "evidence", "nano_prompt")).lower()
    return set(re.findall(r"[a-z]+", blob)) & _TEXT_PROBLEMS


def _sameTextRepair(a, b):
    """Two text fixes for the SAME literal string AND the same kind of problem — the same repair
    reported by different lenses, whatever checklist_category each filed it under. (Same string
    with a different problem, e.g. a typo vs. low contrast on one line, stays separate.)"""
    return _sameTextTarget(a, b) and bool(_textProblems(a) & _textProblems(b))


def _predupe(recs):
    """Deterministic pre-arbiter dedup: collapse findings that share a checklist_category AND
    have overlapping topic/evidence keywords, keeping the highest-confidence one. Guarantees
    obvious duplicate/multi-lens-corroborated grouping in CODE rather than hoping the LLM
    arbiter catches it — less arbiter load, less run-to-run variance.
    Text repairs also merge ACROSS categories when they are the same repair (_sameTextRepair):
    live test, four lenses flagged the same duplicated 'SHOP AT NORTHFIELD.CO' under four
    different categories, so it never got its multi-lens corroboration boost."""
    groups = []
    for r in recs:
        cat = r.get("checklist_category")
        kw = _keywords(r)
        is_text = _isTextFix(r)
        placed = False
        for g in groups:
            if is_text and g["is_text"] and any(_sameTextRepair(r, it) for it in g["items"]):
                g["items"].append(r)
                g["keywords"] |= kw
                placed = True
                break
            if g["category"] == cat and kw and (g["keywords"] & kw):
                # Two text repairs only merge when they name the SAME string. "the clipped
                # headline" and "the overlapping tagline" share category and keywords but are
                # two separate broken things; merging them silently discarded one.
                if is_text or g["is_text"]:
                    if not any(_sameTextTarget(r, it) for it in g["items"]):
                        continue
                g["items"].append(r)
                g["keywords"] |= kw
                placed = True
                break
        if not placed:
            groups.append({"category": cat, "keywords": kw, "items": [r], "is_text": is_text})
    out = []
    for g in groups:
        items = g["items"]
        if len(items) == 1:
            out.append(items[0])
            continue
        items.sort(key=lambda r: r.get("confidence", 0) or 0, reverse=True)
        kept = dict(items[0])
        lenses = sorted({it.get("source_lens") for it in items if it.get("source_lens")})
        if len(lenses) > 1:
            kept["evidence"] = f"{kept.get('evidence', '')} (corroborated by: {', '.join(lenses)})"
            kept["confidence"] = max(kept.get("confidence", 0) or 0, 0.9)  # multi-lens = high-confidence
        out.append(kept)
    return out


def _applyStructuralCap(recs, max_structural=1):
    """Deterministic bias to fewer, safer edits: keep at most `max_structural` structural-risk
    recs, and only when no local rec already targets the same checklist_category. Belt-and-
    suspenders on top of the arbiter's own STRUCTURAL CAP instruction, enforced in code so it
    can't be skipped by LLM non-compliance."""
    # Text repairs are exempt: a clipped or misspelled word is factually broken, and dropping the
    # fix because it happens to need reflowing leaves the defect on screen.
    local = [r for r in recs if r.get("edit_risk") != "structural" or _isTextFix(r)]
    structural = [r for r in recs if r.get("edit_risk") == "structural" and not _isTextFix(r)]
    local_categories = {r.get("checklist_category") for r in local if not _isTextFix(r)}
    kept_structural = []
    for r in structural:
        if len(kept_structural) >= max_structural:
            break
        if r.get("checklist_category") in local_categories:
            continue                      # a local fix already covers this category — skip the riskier one
        kept_structural.append(r)
    keep_ids = {id(r) for r in local} | {id(r) for r in kept_structural}
    return [r for r in recs if id(r) in keep_ids]


# Final-display quality floor. The arbiter prompt's own CONFIDENCE FILTER drops < 0.45, which is
# low enough that marginal findings still reach the user: with four lenses each told to "emit every
# MAJOR finding (do not cap)", the POOL is essentially always full, so the top-N slice was always
# full too — which is exactly why a 7-image test batch came back with 4/4/4/4/4/4/4 suggestions.
# Enforced in code so it holds regardless of LLM compliance, same as the structural cap below.
MIN_DISPLAY_CONFIDENCE = 0.55
# Aesthetic is the subjective-craft lens — its own prompt tells it to stay <= 0.7 "unless obvious",
# and there is ALWAYS some polish note available on any design, so it is the single largest source
# of filler. Hold it to a strictly higher bar than the objective lenses.
MIN_AESTHETIC_CONFIDENCE = 0.7
MAX_AESTHETIC_SUGGESTIONS = 1
# A lens that omitted "confidence" entirely gets the benefit of the doubt: it clears the general
# floor but not the stricter aesthetic one. Without this, a schema-non-compliant lens response
# would be silently wiped out by the floor.
_UNSCORED_CONFIDENCE = 0.6


def _confidence(rec):
    c = rec.get("confidence")
    return _UNSCORED_CONFIDENCE if c is None else c


def _note(tele, key, value):
    """Best-effort telemetry note. Diagnostics must never be able to break the critique path,
    so a telemetry object without .note (older instance, a test double) is silently tolerated."""
    try:
        tele.note(key, value)
    except Exception:
        pass


def _applyQualityFloor(recs):
    """Drop findings not confident enough to be worth spending a user's attention on.

    This is the anti-padding guarantee. It is deliberately applied BEFORE top-N selection so a
    weak item can never occupy a slot just because the pool had room — the user sees fewer, better
    suggestions, and a genuinely clean design correctly yields zero.
    """
    out = []
    for r in recs:
        if _isTextFix(r):
            out.append(r)     # never floor-drop a text repair — broken copy is not a taste call
            continue
        floor = (MIN_AESTHETIC_CONFIDENCE if r.get("source_lens") == "aesthetic"
                 else MIN_DISPLAY_CONFIDENCE)
        if _confidence(r) >= floor:
            out.append(r)
    return out


def _applyAestheticCap(recs, max_aesthetic=MAX_AESTHETIC_SUGGESTIONS):
    """Keep at most `max_aesthetic` subjective-polish items. The arbiter is already instructed to
    do this ("keep at most ONE") but it was never enforced, so a run where the aesthetic lens was
    chatty could fill half the user's list with taste notes ahead of real defects."""
    kept, out = 0, []
    for r in recs:
        if r.get("source_lens") == "aesthetic":
            if kept >= max_aesthetic:
                continue
            kept += 1
        out.append(r)
    return out


def _curate(pool_recs, N):
    """Deterministic post-arbiter curation, in the order the guarantees must compose:
    structural cap -> quality floor -> aesthetic cap -> similarity-aware top-N.
    Returns ``(final_recs, dropped)`` where `dropped` lists everything the three caps/floor
    removed (with its confidence), so a deliberately short suggestion list is explainable in
    telemetry rather than looking like a pipeline failure."""
    before = list(pool_recs)
    recs = _applyStructuralCap(pool_recs, max_structural=1)
    recs = _applyQualityFloor(recs)
    recs = _applyAestheticCap(recs)
    # Text repairs float to the top, keeping the arbiter's relative order within each group
    # (sort is stable). Broken copy is the most concrete, highest-value thing we can report, and
    # it must never lose a slot to a spacing or polish note.
    recs.sort(key=lambda r: not _isTextFix(r))
    kept_ids = {id(r) for r in recs}
    dropped = [{"fid": r.get("fid"), "lens": r.get("source_lens"), "topic": r.get("topic"),
                "confidence": r.get("confidence")}
               for r in before if id(r) not in kept_ids]
    final = _selectDiverseTopN(recs, N)
    final_ids = {id(r) for r in final}
    # also log what the similarity check / top-N cut removed, so a missing finding is traceable
    dropped += [{"fid": r.get("fid"), "lens": r.get("source_lens"), "topic": r.get("topic"),
                 "confidence": r.get("confidence"), "reason": "top-N / similarity"}
                for r in recs if id(r) not in final_ids]
    return final, dropped


def _selectDiverseTopN(ranked_recs, N):
    """Deterministic top-N selection with similarity backfill. Walks `ranked_recs` (already in
    score/rank order, best first) and keeps a rec only if it is not too similar to one already
    kept. This IS "remove the lower-scored duplicate and supplement with the next-closest-score
    suggestion" — because we walk in rank order, a collision always drops the lower-ranked item
    and the loop naturally continues to the next candidate by score. If fewer than N survive
    (or there are no candidates left), fewer are shown — never padded."""
    selected = []
    for cand in ranked_recs:
        if len(selected) >= N:
            break
        if any(_tooSimilar(cand, kept) for kept in selected):
            continue
        selected.append(cand)
    return selected


# --------------------------------------------------------------------------- #
# AI-EXECUTABILITY GATE
# --------------------------------------------------------------------------- #
# Every suggestion the user accepts is carried out by the image-edit model and nothing else, so a
# suggestion it can't carry out is worse than none: the user accepts it, the edit misfires, and
# the critique loses their trust. The gate runs on the arbiter's ranked POOL *before* top-N
# selection — a dropped item is backfilled by the next-best executable one instead of leaving a
# hole. Unlike the caps/floor in _curate, text repairs get no exemption here: a text fix the model
# can't perform (e.g. "replace with the real product code") is still one the user can't use.
#
# Two layers: (1) deterministic rules for failure classes that are unambiguous from the text alone,
# (2) an LLM judge (EXECUTABILITY_PROMPT) for everything else. The judge fails OPEN — if the call
# errors, rule-passed items are kept — because a transport failure must not empty every critique.
_LOGO_ADD_RE = re.compile(
    r"\b(add|draw|redraw|re-draw|restore|recreate|re-create|insert|place|render|re-render|"
    r"regenerate|generate)\b[^.;]{0,40}\b(logo|logos|logomark|wordmark|brand mark|brandmark)\b",
    re.IGNORECASE)
_TRANSPARENT_RE = re.compile(r"\btransparent (background|backdrop|bg)\b", re.IGNORECASE)
_PIXEL_EXACT_RE = re.compile(r"\bpixel[- ](perfect|identical|exact)\b", re.IGNORECASE)


def _ruleNotExecutable(rec):
    """Reason string when a deterministic rule proves the edit model can't do this, else None.
    Checks only nano_prompt (what the model is actually told) — `evidence` legitimately quotes
    internal rules and thresholds, so matching it would drop valid suggestions."""
    if rec.get("type") == "UPSCALE":
        return None                                   # routed to the upscaler, not the edit model
    nano = rec.get("nano_prompt", "") or ""
    if _LOGO_ADD_RE.search(nano):
        return "rule: adds or redraws a logo"
    m = _ELEMENT_COUNT_RE.search(rec.get("evidence", "") or "")
    if m and int(m.group(1)) >= 3:
        return f"rule: relayout of {m.group(1)} elements"
    if _TRANSPARENT_RE.search(nano):
        return "rule: transparent background (edit output is never transparent)"
    if _PIXEL_EXACT_RE.search(nano):
        return "rule: demands pixel-exact result"
    return None


def executabilityGate(recs, tele):
    """Split `recs` into (executable, dropped), preserving rank order. `recs` may span images;
    items are matched to the judge's verdicts by `fid`, so each must carry one (arbiterAgent and
    mergeFindings guarantee that). `dropped` entries are telemetry records, not recommendations."""
    dropped, candidates = [], []
    for r in recs:
        reason = _ruleNotExecutable(r)
        if reason:
            dropped.append({"fid": r.get("fid"), "lens": r.get("source_lens"),
                            "topic": r.get("topic"), "reason": reason})
        else:
            candidates.append(r)

    judged = [r for r in candidates if r.get("type") != "UPSCALE" and r.get("fid")]
    verdicts = {}
    if judged:
        compact = [{"fid": r["fid"], "topic": r.get("topic"), "edit_risk": r.get("edit_risk"),
                    "nano_prompt": r.get("nano_prompt"), "evidence": r.get("evidence")}
                   for r in judged]
        try:
            out = parseJson(_text_invoke("executability", EXEC_MODEL,
                                         render(EXECUTABILITY_PROMPT, candidates=json.dumps(compact)),
                                         tele, EXEC_THINKING))
            verdicts = {v.get("fid"): v for v in (out.get("verdicts") or []) if v.get("fid")}
        except Exception as e:
            _note(tele, "executability_error", str(e)[:150])    # fail open: keep rule-passed items

    kept = []
    for r in candidates:
        v = verdicts.get(r.get("fid"))
        if v is not None and v.get("executable") is False:
            dropped.append({"fid": r.get("fid"), "lens": r.get("source_lens"),
                            "topic": r.get("topic"), "reason": f"judge: {v.get('reason', '')}"})
        else:
            kept.append(r)                     # executable, UPSCALE, or no verdict (fail open)
    return kept, dropped


# --------------------------------------------------------------------------- #
# USER TEXT MUST BE AN INSTRUCTION
# --------------------------------------------------------------------------- #
# The user accepts or rejects on user_friendly_text alone, so it must say what the edit WILL DO.
# Live test: the arbiter's "reword" turned "Remove the placeholder 'Tickets: TBC'" into "ensure
# this is updated with actual booking information before posting" — the user was promised real
# details and the edit then deleted the line. The prompt now forbids that; this enforces it.
_ACTION_VERBS = {
    "fix", "remove", "replace", "correct", "move", "shrink", "enlarge", "recolor", "recolour",
    "brighten", "darken", "lighten", "rewrite", "reword", "shorten", "add", "straighten",
    "separate", "delete", "erase", "refit", "resize", "bring", "lift", "lower", "raise", "shift",
    "nudge", "increase", "reduce", "make", "change", "swap", "tighten", "simplify", "clean",
    "sharpen", "align", "center", "centre", "restore", "redraw", "repair", "update", "boost",
    "soften", "strengthen", "split", "condense", "spell", "set", "turn", "give", "put", "place",
    "widen", "narrow", "crop", "trim", "clear", "tidy", "balance", "space", "unclutter", "declutter",
    "render", "rerender", "re-render", "recreate", "keep", "reposition", "relocate",
}
_TELL_USER_RE = re.compile(r"\b(ensure|make sure|be sure|remember to|before posting|you should|"
                           r"you'll need|you will need|consider)\b", re.IGNORECASE)


def _isInstruction(text):
    """True when `text` opens with an action verb and doesn't hand the work back to the user."""
    words = re.findall(r"[A-Za-z]+", text or "")
    return bool(words) and words[0].lower() in _ACTION_VERBS and not _TELL_USER_RE.search(text)


def _userText(reword, original, nano):
    """Pick the user-facing text: the lens's original, else the arbiter's reword, else the
    verb-led clause of the nano_prompt (which NANO_PROMPT_RULES requires to open with a verb).
    The lens wrote its text alongside the nano_prompt while looking at the image, so it is the
    most faithful description of what the edit will do; the arbiter never sees the nano_prompt,
    and live its rewords drifted ("remove the duplicate URLs from the upper-left AND bottom-left"
    for an edit that deliberately keeps the bottom-left one). Reword is only a repair.
    Returns (text, source) so telemetry can show how often each path fires."""
    if _isInstruction(original):
        return original, "original"
    if reword and _isInstruction(reword):
        return reword, "reword"
    first = re.split(r"(?<=[.!?])\s+", (nano or "").strip(), maxsplit=1)[0]
    # "In the lower-left corner, change X to Y" -> the verb-led clause is after a comma
    parts = first.split(", ")
    for i in range(len(parts)):
        clause = ", ".join(parts[i:])
        if _isInstruction(clause):
            return clause[0].upper() + clause[1:], "nano_prompt"
    return reword or original, "unfixed"


# --------------------------------------------------------------------------- #
# LENS SCOPE — enforced in code, because prompt rules alone did not hold live
# --------------------------------------------------------------------------- #
# (1) Only Content (and Integrity's brief-fidelity tier) may propose ADDING content. Composition
#     and Aesthetic are told "NEVER report an element as missing" / "add NO new text or elements",
#     yet composition asked for a 'Shop Now' button on a poster whose corner already read
#     'aurora-audio.com' — and it outranked a real contrast defect.
# (2) A deadline/offer line is never removed as "redundant". Content suggested dropping
#     'THIS WEEKEND ONLY' from a flash-sale ad twice, despite a prompt example naming that exact
#     case. Removing a genuinely DUPLICATED string (same text twice) is unaffected.
_ADD_LED_RE = re.compile(r"^\s*(?:[^,.]{0,40},\s*)?(add|insert|include|introduce|create|place)\b",
                         re.IGNORECASE)
_REMOVE_LED_RE = re.compile(r"^\s*(?:[^,.]{0,40},\s*)?(remove|delete|erase|drop|cut)\b",
                            re.IGNORECASE)
_REDUNDANT_RE = re.compile(r"redundan|repeats? (the|what)|already (communicat|establish|convey|says)",
                           re.IGNORECASE)
_KEY_COPY_RE = re.compile(r"\b(weekend|today|tonight|only|ends?|until|hours?|days?|limited|last|"
                          r"now|sale|off|deadline|free|save)\b|\d+\s*%", re.IGNORECASE)

# (3) No micro-repositioning. Lenses kept emitting "move the title slightly left" / "move the button
#     10% up": the user can't see the difference, the edit model can't hit a measured offset, and
#     it takes a slot from a real defect. A move verb that opens a clause, paired with a small or
#     measured amount, is a nudge. "Darken the lower-left corner slightly" is not (no move verb at a
#     clause start). Kept only when the element is actually clipped / overlapping / covered.
_MOVE_VERB = (r"(?:move|shift|nudge|lift|lower|raise|push|pull|slide|drag|reposition|relocate|"
              r"offset|bump|align|center|centre|adjust)")
_NUDGE_AMOUNT = (r"(?:slightly|a (?:little|bit|touch|tad|hair)|a few (?:pixels|px|points)|marginally|"
                 r"subtly|fractionally|just a bit|\d+(?:\.\d+)?\s*(?:%|px\b|pixels?\b|percent\b|pts?\b))")
_NUDGE_RE = re.compile(
    rf"(?:^|[.;:]\s*|,\s*|\b(?:and|then)\s+)(?:(?:{_MOVE_VERB})\b[^.;]{{0,80}}?{_NUDGE_AMOUNT}"
    rf"|{_NUDGE_AMOUNT}\s+{_MOVE_VERB}\b|nudge\b)", re.IGNORECASE)
_POSITION_DEFECT_RE = re.compile(
    r"\b(clipped|cropped|truncated|cut[- ]?off|overflow\w*|off[- ]canvas|off the (edge|canvas|frame)|"
    r"runs? off|bleeds? off|overlap\w*|obscur\w*|covered|covers|collid\w*|touch(es|ing) the edge|"
    r"outside the (frame|canvas))\b", re.IGNORECASE)


def _isNudge(rec):
    """True when the finding is a small/measured reposition with no visible defect behind it."""
    if rec.get("type") == "UPSCALE":
        return False
    said = " ".join(str(rec.get(k, "") or "") for k in ("nano_prompt", "user_friendly_text"))
    if not _NUDGE_RE.search(said):
        return False
    why = " ".join(str(rec.get(k, "") or "") for k in ("topic", "evidence", "nano_prompt"))
    return not _POSITION_DEFECT_RE.search(why)


def _outOfScope(rec):
    """Reason string when a finding breaks its lens's scope, else None."""
    nano = rec.get("nano_prompt", "") or ""
    if rec.get("source_lens") in ("composition", "aesthetic") and _ADD_LED_RE.search(nano):
        return f"scope: {rec.get('source_lens')} lens may not add content"
    if _isNudge(rec):
        return "scope: micro-reposition (moves an element slightly / by a measured amount)"
    blob = f"{rec.get('topic', '')} {rec.get('evidence', '')}"
    if _REMOVE_LED_RE.search(nano) and _REDUNDANT_RE.search(blob):
        if any(_KEY_COPY_RE.search(t) for t in _textTargets({"nano_prompt": nano})):
            return "scope: removes deadline/offer copy as 'redundant'"
    return None


def _applyScope(recs, tele):
    kept, dropped = [], []
    for r in recs:
        why = _outOfScope(r)
        if why:
            dropped.append({"fid": r.get("fid"), "lens": r.get("source_lens"),
                            "topic": r.get("topic"), "reason": why})
        else:
            kept.append(r)
    if dropped:
        _note(tele, "out_of_scope", dropped)
    return kept


# --------------------------------------------------------------------------- #
# COPY FREEZE — generated artwork keeps the copywriter's wording
# --------------------------------------------------------------------------- #
# The generation pipeline's copywriter already wrote the best text for a generated design, so
# its critique may only repair DEFECTS in the rendered copy: misspellings and grammar, misprints
# against the copy deck, glitched letters, cut-off text, duplicated lines, placeholder strings,
# and deck lines the render dropped. The lenses are told this (COPY_FREEZE_BLOCK); this filter
# enforces it, for the same reason the scope rules above exist — prompt rules alone do not hold.
# Visual treatment of text (contrast, size, position, overlap) never changes a word and passes.
# A word naming copy, or the opening quote of a quoted string.
_TEXT_NOUN = (r"(?:\b(?:text|copy|words?|wording|lines?|phrases?|sentences?|headlines?|"
              r"subheadlines?|sub-headlines?|taglines?|slogans?|captions?|labels?|titles?|"
              r"subtitles?|paragraphs?|cta|call[- ]to[- ]action)\b|['\"])")
# Things a fix adds, removes or puts behind text WITHOUT touching its words: "add a dark band
# behind the headline", "remove the glow around the title".
_VISUAL_NOUN_RE = re.compile(
    r"\b(band|bar|shadow|drop shadow|glow|outline|stroke|border|backdrop|background|panel|overlay|"
    r"box|gradient|scrim|padding|spacing|space|margin|gap|breathing room|plate|strip|highlight|"
    r"underline|effect|texture|tint|colou?r|contrast|frame|shape|icon|graphic|logo|image|photo)\b",
    re.IGNORECASE)
_TEXT_NOUN_RE = re.compile(_TEXT_NOUN, re.IGNORECASE)
_REWORD_VERB_RE = re.compile(r"\b(re-?word\w*|re-?writ\w*|re-?phras\w*|shorten\w*|condens\w*|"
                             r"punch up|simplif\w*|tone down|re-?tone)\b", re.IGNORECASE)
# "place"/"put" are left out: "place the CTA below the price" moves copy, it adds none.
_ADD_VERB_RE = re.compile(r"\b(add|insert|include|introduce|write|restore)\b", re.IGNORECASE)
_DROP_VERB_RE = re.compile(r"\b(remove|delete|erase|drop|cut)\b", re.IGNORECASE)
_CHANGE_TO_RE = re.compile(
    r"\b(change|replace|swap|update|correct|fix|edit|turn)\b[^.;]{0,80}?"
    r"\b(to|with|into|to read|so it reads)\s+['\"]", re.IGNORECASE)
# What makes a wording change a REPAIR the user allowed rather than a rewrite. Wording defects
# can need new words (a grammar fix, a placeholder replaced); layout defects (cut-off,
# duplicated) never do — the same words are refitted or the extra copy removed.
_WORDING_DEFECT_RE = re.compile(
    r"\b(typos?|misspel\w*|spelling|spelt|grammar\w*|grammatical\w*|misprint\w*|garbled|"
    r"gibberish|glitch\w*|warped|nonsense|scrambled|jumbled|corrupt\w*|placeholder|lorem|"
    r"dangling|stray|leftover|debug|error string)\b", re.IGNORECASE)
_LAYOUT_DEFECT_RE = re.compile(
    r"\b(truncat\w*|clipped|cut[- ]off|runs? off|cropped|duplicat\w*|repeated|repeats|twice|"
    r"doubled)\b", re.IGNORECASE)


def _textOp(nano, verb_re):
    """True when ``verb_re`` (add/remove) acts on COPY in ``nano``: a text noun or quoted string
    follows the verb within its clause, with no visual noun (band, glow, background ...) between
    them — "add a dark band behind the headline" acts on a band, not on the headline's words."""
    for m in verb_re.finditer(nano):
        tail = re.split(r"[.;]", nano[m.end():m.end() + 60], maxsplit=1)[0]
        t = _TEXT_NOUN_RE.search(tail)
        if t and not _VISUAL_NOUN_RE.search(tail[:t.start()]):
            return True
    return False


def _norm(s):
    return " ".join(re.findall(r"[a-z0-9%$£€]+", str(s or "").lower()))


def _inDeck(rec, deck_lines):
    """True when a string this fix quotes is (part of) a copy-deck line — restoring the deck."""
    if not deck_lines:
        return False
    for t in _textTargets(rec):
        n = _norm(t)
        if len(n) >= 3 and any(n in d or (len(d) >= 3 and d in n) for d in deck_lines):
            return True
    return False


def _frozenCopyReason(rec, deck_lines):
    """Reason string when ``rec`` changes the copywriter's wording without repairing a defect,
    else None. ``deck_lines`` = normalised copy-deck strings for this image (may be empty)."""
    if rec.get("type") == "UPSCALE":
        return None
    nano = rec.get("nano_prompt", "") or ""
    blob = " ".join(str(rec.get(k, "") or "") for k in
                    ("topic", "evidence", "checklist_category", "user_friendly_text", "nano_prompt"))
    wording_defect = bool(_WORDING_DEFECT_RE.search(blob))
    restores_deck = _inDeck(rec, deck_lines)
    # Shortening, condensing or re-toning is a rewrite even when the line is also cut off:
    # a cut-off line is refitted with the same words.
    if rec.get("text_change") == "rewrite" or _textOp(nano, _REWORD_VERB_RE):
        if wording_defect or restores_deck:
            return None
        return "copy freeze: rewrites copy that has no defect"
    adds = _textOp(nano, _ADD_VERB_RE)
    changes = bool(_CHANGE_TO_RE.search(nano) or _textOp(nano, _DROP_VERB_RE))
    if not (adds or changes):
        return None                    # visual treatment of text — no word changes
    if wording_defect or restores_deck or _LAYOUT_DEFECT_RE.search(blob):
        return None
    if adds:
        return "copy freeze: adds copy the copy deck does not have"
    return "copy freeze: changes copy that has no defect"


def copyFreeze(merged_images, provenance, deck, tele):
    """Drop, from generated artwork only, every finding that changes the copywriter's wording
    without repairing a defect. ``deck`` maps image_id -> {slot: string}. Uploads pass through."""
    if provenance != "generated":
        return merged_images
    dropped = []
    for im in merged_images:
        lines = [_norm(v) for v in ((deck or {}).get(im.get("image_id", "image_1")) or {}).values()]
        kept = []
        for r in im.get("recommendations", []) or []:
            why = _frozenCopyReason(r, [d for d in lines if d])
            if why:
                dropped.append({"fid": r.get("fid"), "lens": r.get("source_lens"),
                                "topic": r.get("topic"), "reason": why})
            else:
                kept.append(r)
        im["recommendations"] = kept
    if dropped:
        _note(tele, "copy_frozen", dropped)
    return merged_images


def arbiterAgent(merged_images, brand, eff, avail, N, tele):
    """Curate merged findings. The LLM returns only {fid, rank, reword?}; the full
    recommendations are reassembled here from the merged input (small output = fast/cheap).
    A deterministic pre-dedup runs before the LLM call; after it, a deterministic structural-risk
    cap AND a similarity-aware top-N backfill run, so all three guarantees hold regardless of LLM
    compliance. The LLM is asked to rank a POOL larger than N so there is real material to
    backfill from when two of the best-ranked items turn out to be near-duplicates.
    Returns a creator-compatible JSON STRING for techAgent.
    """
    POOL = N + 4
    lookup = {}
    compact = []
    for im in merged_images:
        deduped = _predupe(_capHighRiskConfidence(
            _applyScope(im.get("recommendations", []) or [], tele)))
        cf = []
        for r in deduped:
            lookup[r["fid"]] = r
            cf.append({"fid": r["fid"], "lens": r.get("source_lens"),
                       "category": r.get("checklist_category"), "topic": r.get("topic"),
                       "evidence": r.get("evidence"), "brand_check": r.get("brand_check"),
                       "confidence": r.get("confidence"),
                       "edit_risk": r.get("edit_risk", "local")})
        compact.append({"image_id": im.get("image_id", "image_1"),
                        "overall_assessment": im.get("overall_assessment", ""), "findings": cf})

    prompt = render(
        ARBITER_PROMPT,
        findings=json.dumps({"images": compact}),
        brand_block=brandBlock(brand),
        effective_rules=json.dumps(_arbiterRules(eff)),
        user_intent_block=intentBlock(avail.get("intent")),
        purpose=avail.get("goal", ""),
        platform=avail.get("platform", ""),
        max_output=POOL,
    )
    # Pre-arbiter observability: the compact per-lens findings, recorded BEFORE any curation.
    # Without this, a missing suggestion is undiagnosable — you cannot tell whether a lens never
    # produced the finding (prompt gap) or produced it and curation dropped it (ranking bug).
    _note(tele, "pre_arbiter_findings", compact)

    try:
        sel = parseJson(_text_invoke("arbiter", ARBITER_MODEL, prompt, tele, ARBITER_THINKING))
        pools = []
        for im in sel.get("images", []):
            pool_recs = []
            for k in sorted(im.get("keep", []), key=lambda k: k.get("rank", 999))[:POOL]:
                base = lookup.get(k.get("fid"))
                if not base:
                    continue
                rec = dict(base)                             # keep fid through curation for tracing
                rec["_reword"] = k.get("reword")             # resolved by _fixUserText below
                pool_recs.append(rec)
            pools.append((im.get("image_id", "image_1"), im.get("overall_assessment", ""), pool_recs))
    except Exception:
        # fallback: skip the LLM ranking, rank the pre-deduped findings by confidence instead
        pools = []
        for im in merged_images:
            deduped = _predupe(_capHighRiskConfidence(
            _applyScope(im.get("recommendations", []) or [], tele)))
            deduped.sort(key=lambda r: r.get("confidence", 0) or 0, reverse=True)
            pools.append((im.get("image_id", "image_1"), im.get("overall_assessment", ""), deduped))
    _fixUserText([r for _, _, recs in pools for r in recs], tele)
    return json.dumps({"status": "success", "case": 1,
                       "images": _gateAndCurate(pools, N, tele)})


def _fixUserText(recs, tele):
    """Resolve each rec's user_friendly_text through _userText (in place); log the sources."""
    sources = []
    for r in recs:
        text, src = _userText(r.pop("_reword", None), r.get("user_friendly_text"),
                              r.get("nano_prompt"))
        r["user_friendly_text"] = text
        sources.append({"fid": r.get("fid"), "source": src})
    _note(tele, "user_text_source", sources)


def _gateAndCurate(pools, N, tele):
    """Executability gate over every image's pool in ONE judge call, then per-image curation to
    the final top N. `pools` = [(image_id, overall_assessment, ranked_recs_with_fid)]. Returns
    frontend-bound image dicts with `fid` stripped."""
    executable, not_exec = executabilityGate([r for _, _, recs in pools for r in recs], tele)
    ok = {id(r) for r in executable}
    out, dropped_all = [], []
    for image_id, assessment, recs in pools:
        final_recs, dropped = _curate([r for r in recs if id(r) in ok], N)
        dropped_all.extend(dropped)
        out.append({"image_id": image_id, "overall_assessment": assessment,
                    "recommendations": [{kk: vv for kk, vv in r.items() if kk != "fid"}
                                        for r in final_recs]})
    _note(tele, "not_executable", not_exec)
    _note(tele, "curation_dropped", dropped_all)
    return out


# Edit quality for the image-edit call. Re-rendering LETTERS (a spelling fix, retyped wording) is
# where GPT-Image-2.5 Flare is weakest at "medium": live, correcting 'wirless' -> 'wireless' in a
# small monospace label landed 1/6 times at medium ('wiroless', 'wirıless', 'wirlūcss',
# 'wirbless', no change) and 2/2 at "high" with the identical prompt, the rest of the image
# untouched. High costs ~4x the image output tokens and ~+6s, so it is used only where needed —
# recolors, moves and removals were reliable at medium in every run.
_RETYPE_RE = re.compile(
    r"\b(spell\w*|misspell\w*|typo|garbled|glyphs?|re-?render|retype|rewrite|reword)\b"
    r"|\b(change|replace|correct)\b[^.;]{0,80}\b(to|with)\s+['\"]", re.IGNORECASE)


def editQuality(units):
    """Quality for ONE edit call covering `units`: "high" if any unit re-renders letters."""
    for u in units or []:
        blob = " ".join(str(u.get(k, "") or "") for k in ("topic", "nano_prompt"))
        if _RETYPE_RE.search(blob):
            return "high"
    return "medium"


def compilerAgent(purpose, units, tele):
    """Agent 4: merge accepted EDIT/GENERATION units into one execution prompt.

    ``units`` = list of accepted suggestion dicts (type/tier/topic/user_friendly_text/
    nano_prompt/edit_risk). Picks the EDIT-lane (strict, surgical) or REGEN-lane (permissive,
    still image-referenced) mode block depending on whether any accepted unit is structural-risk
    — still ONE compiled prompt / ONE edit call, no sequential batching.
    Returns the parsed ``{execution_prompt, unit_count, compilation_notes}``.
    """
    editable = [u for u in units if u.get("type") in ("EDIT", "GENERATION")]
    has_structural = any(u.get("edit_risk") == "structural" or u.get("type") == "GENERATION"
                         for u in editable)
    mode_block = REGEN_MODE_BLOCK if has_structural else EDIT_MODE_BLOCK
    payload = {"purpose": purpose, "suggestion_units": editable}
    prompt = render(SUGGESTION_COMPILATION_PROMPT, suggestions=json.dumps(payload), mode_block=mode_block)
    raw = _text_invoke("compiler", COMPILER_MODEL, prompt, tele, COMPILER_THINKING)
    return parseJson(raw)


def verifyEdit(before_image, after_image, accepted_units, tele, pending_units=None):
    """Post-edit verification pass: compares before/after against what was accepted, confirms
    each change landed, and independently scans for NEW regressions the edit introduced (e.g. a
    duplicated text block from a risky structural edit). ``before_image``/``after_image`` may be
    raw bytes, a URL, or a b64 string — all accepted directly (see _to_data_uri).
    ``pending_units`` = accepted fixes scheduled for LATER edit steps (sequential execution), so
    problems they will address are not misreported as regressions of this step.
    Returns {"applied": [...], "regressions": [...], "overall": "clean|partial|regressed"}.
    """
    def _brief(units):
        return [{"topic": u.get("topic"), "user_friendly_text": u.get("user_friendly_text")}
                for u in units or []]
    prompt = render(VERIFY_EDIT_PROMPT, accepted_changes=json.dumps(_brief(accepted_units)),
                    pending_changes=json.dumps(_brief(pending_units)))
    try:
        return _vision_invoke("verify", LENS_MODEL, prompt, [before_image, after_image], "b64",
                              tele, LENS_THINKING)
    except Exception as e:
        return {"applied": [], "regressions": [], "overall": "unknown", "error": str(e)[:150]}
