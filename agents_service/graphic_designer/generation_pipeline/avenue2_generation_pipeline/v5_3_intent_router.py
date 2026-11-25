"""
v5_3_intent_router.py  (v5.4)
------------------------------
Intent routing engine: detects brand violations and layout contradictions.

v5.4 changes:
  - Accepts clarified_intent so routing decisions are made on the parsed brief,
    not the raw user prompt
  - Sends only a trimmed layout summary (variant names + descriptions) instead
    of the full layout_profile JSON -- reduces token cost significantly
  - brand_violation.details is now a specific, actionable explanation that flows
    into the downstream fusion instruction
"""

import json
import os
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MODEL  = os.environ.get("MODEL_PRO", "models/gemini-3.1-pro-preview")

ROUTER_PROMPT = """\
You are an expert Art Director and Routing Engine for an AI graphic design pipeline.
Analyze the user's clarified intent, field validation results, and Brand Profile, then output a structured JSON routing decision.

--- BRAND PROFILE ---
{brand_profile}

--- AVAILABLE LAYOUT VARIANTS ---
{layout_summary}

--- USER REQUEST (RAW) ---
{raw_prompt}

--- CLARIFIED INTENT ---
{clarified_intent}

--- INTENT CLARITY ASSESSMENT ---
Clarity Score: {clarity_score}/100  [{clarity_level}]
Missing Fields: {missing_fields_summary}
Ambiguity Flags: {ambiguity_flags_summary}

--- HAS USER PHOTO ---
{has_photo}

--- ASSET RESOLUTION (Phase 1.2 output) ---
{asset_summary}

Analyze and decide:

1. BRAND VIOLATION: Does the clarified intent explicitly request colors, moods, or styles that the Brand Profile forbids?
   - Only flag if the brand profile is present AND the conflict is explicit and specific.
   - If no brand profile provided, always false.
   - SEVERITY: "HIGH" = explicit style/color override requested | "MEDIUM" = tonal mismatch | "LOW" = minor aesthetic deviation
   - In "details", name the exact brand rule being violated and what the user requested that breaks it.

2. LAYOUT CONTRADICTION: Does the user request a spatial arrangement that none of the available layout variants support?
   - Only flag if the contradiction is genuine and cannot be resolved by fuzzy-matching to an existing variant.
   - In "details", name what the user wants vs what the variants provide.

3. DROP WIREFRAMES: Set true only if layout_contradiction is severe OR if has_photo=true.

4. ROUTING ACTION — choose exactly one:
   "PROCEED"               — Intent clear, no critical issues. Execute pipeline.
   "PROCEED_WITH_OVERRIDE" — Brand or style violation detected but user context suggests it is intentional.
                             Pipeline can proceed with a routing_instruction that limits override scope.
   "CLARIFY_FIRST"         — Clarity score < 50 OR one or more CRITICAL fields are MISSING
                             OR asset resolution shows PENDING_USER_INPUT or clarification_needed.
                             Pipeline MUST pause and surface the clarification_questions before proceeding.
   "REJECT"                — Input is fundamentally incompatible with the design goal. Use when:
                             (a) The clarifier flagged UNRELATED_INPUT AND clarity_score < 30, OR
                             (b) The input is abusive, adversarial, or attempts to misuse the pipeline, OR
                             (c) The input is completely empty or pure random characters.
                             Do NOT use REJECT just because fields are missing — use CLARIFY_FIRST for that.

5. CONFIDENCE (0-100): Your confidence that the routing_action you chose is CORRECT.
   This score MUST be calibrated against the clarity_score provided:
   - If clarity_score >= 80 AND routing_action = "PROCEED" with no violations: 85-95
   - If clarity_score 60-79 AND routing is clear: 70-84
   - If clarity_score < 60 (MEDIUM/LOW): cap confidence at 65 regardless
   - If clarity_score < 40 (LOW): cap confidence at 50
   - If you are routing "CLARIFY_FIRST" due to missing CRITICAL fields: 70-85
     (you are confident in the routing, but the design intent itself is unclear)
   - If routing_action = "REJECT" and there is genuine fundamental incompatibility: 85+
   - If you are routing based on INFERRED information or ambiguity flags: subtract 10-20

6. CLARIFICATION QUESTIONS: If routing_action = "CLARIFY_FIRST", list the exact questions to ask the user
   (taken from the missing_fields question_if_missing values). Maximum 3 questions.

7. DETECTED_PHOTO_USAGE: If has_photo=true, scan the RAW USER REQUEST for explicit signals about how
   the user wants their photo treated. Set to:
   - "USE_AS_REFERENCE" if the user explicitly says any of: "use as inspiration", "reference only",
     "do not embed", "don't embed", "use for style", "use for color mood", "generate fresh image",
     "don't use directly", "as a reference", "inspired by"
   - "USE_DIRECTLY"   if the user explicitly says any of: "use my photo", "embed my photo",
     "feature this photo", "keep my image", "use it as-is", "show my photo"
   - null if no explicit instruction about photo usage (user simply attached a photo without direction)
   Only set a non-null value when the user's intent is EXPLICIT, not inferred.

8. DETECTED_LAYOUT_INTENT: Scan the RAW USER REQUEST for explicit signals about how the hero/product
   image should be composed within the graphic. Set to:
   - "full_bleed"   if the user asks for a full-bleed / edge-to-edge / full-canvas / full-background /
     "cover the whole", "background image", "immersive", "poster-style" treatment.
   - "split_panel"  if the user asks for a split / side-by-side / "photo on the left|right" /
     "half image half text" / editorial-split treatment.
   - "inset"        if the user asks for the photo as a small inset / thumbnail / framed card /
     "in a circle" / "in a corner" treatment.
   - null if the user gives no explicit instruction about how the image should be laid out.
   Only set a non-null value when the user's intent is EXPLICIT, not inferred.

Output ONLY a strict JSON object:
{{
  "brand_violation": {{
    "detected": false,
    "severity": "",
    "details": ""
  }},
  "layout_contradiction": {{
    "detected": false,
    "details": ""
  }},
  "drop_wireframes": false,
  "routing_action": "PROCEED",
  "confidence": 90,
  "requires_clarification": false,
  "clarification_questions": [],
  "image_notes": "<one sentence on the image route if relevant, or empty string>",
  "detected_photo_usage": null,
  "detected_layout_intent": null
}}
"""


def _summarise_layout(layout_profile: dict) -> str:
    """Build a compact layout summary for routing -- no zone coords, no fill instructions."""
    lines = [
        f"Design Goal : {layout_profile.get('design_goal', '')}",
        f"Platform    : {layout_profile.get('platform', '')} ({layout_profile.get('aspect_ratio', '')})",
        "Variants:",
    ]
    for v in layout_profile.get("layout_variants", []):
        lines.append(
            f"  [{v.get('name', '')}]  {v.get('description', '')}  "
            f"Best for: {v.get('use_case', '')}"
        )
    neg = layout_profile.get("shared_rules", {}).get("negative_rules", [])
    if neg:
        lines.append(f"Layout constraints: {'; '.join(neg)}")
    return "\n".join(lines)


def route_intent(
    raw_prompt: str,
    clarified_intent: str,
    brand_profile: str | None,
    layout_profile: dict,
    has_photo: bool,
    clarity_score: int = 50,
    clarity_level: str = "MEDIUM",
    missing_fields: list | None = None,
    ambiguity_flags: list | None = None,
    asset_resolution: dict | None = None,
    model: str | None = None,
    role: str | None = None,
    pin: str | None = None,
) -> dict:
    from llm_fallback import generate_with_fallback
    use_model = model or DEFAULT_MODEL

    brand_str      = brand_profile if brand_profile else "None provided."
    layout_summary = _summarise_layout(layout_profile)

    # Build summary strings for the new context slots
    if missing_fields:
        mf_lines = [
            f"  [{f['level']}] {f['field_id']}: {f['question']}"
            for f in missing_fields
        ]
        missing_fields_summary = "\n" + "\n".join(mf_lines)
    else:
        missing_fields_summary = "none"

    if ambiguity_flags:
        af_lines = [f"  {f['type']}: {f['description']}" for f in ambiguity_flags]
        ambiguity_flags_summary = "\n" + "\n".join(af_lines)
    else:
        ambiguity_flags_summary = "none"

    # Build asset resolution summary for router context
    if asset_resolution:
        src   = asset_resolution.get("source", "UNKNOWN")
        usage = asset_resolution.get("usage") or "PENDING"
        path  = asset_resolution.get("asset_path") or "none"
        real_required = asset_resolution.get("real_photo_required", False)
        interrupt     = asset_resolution.get("interrupt_required", False)
        clarify_img   = asset_resolution.get("clarification_needed", False)
        asset_summary = (
            f"  source: {src}  |  usage: {usage}  |  asset_path: {path}\n"
            f"  real_photo_required: {real_required}\n"
            f"  interrupt_required: {interrupt}  |  clarification_needed: {clarify_img}"
        )
    else:
        asset_summary = "not resolved (asset resolver not called)"

    prompt = ROUTER_PROMPT.format(
        brand_profile           = brand_str,
        layout_summary          = layout_summary,
        raw_prompt              = raw_prompt,
        clarified_intent        = clarified_intent,
        has_photo               = has_photo,
        clarity_score           = clarity_score,
        clarity_level           = clarity_level,
        missing_fields_summary  = missing_fields_summary,
        ambiguity_flags_summary = ambiguity_flags_summary,
        asset_summary           = asset_summary,
    )

    import re
    MAX_RETRIES = 3
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = generate_with_fallback(
                model=use_model,
                prompt_text=prompt,
                temperature=0.1,
                response_mime_type="application/json",
                role=role,
                pin=pin,
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
                if m:
                    raw_text = m.group(1)
            res_json = json.loads(raw_text)
            if response.usage_metadata:
                res_json["_usage"] = {
                    "prompt_tokens":     response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                }
            return res_json
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_retryable = (
                "503" in err_str or "UNAVAILABLE" in err_str
                or "timeout" in err_str.lower()
                or "JSONDecodeError" in type(e).__name__
                or isinstance(e, ValueError)  # json.loads raises ValueError on bad JSON
            )
            if attempt < MAX_RETRIES - 1 and is_retryable:
                import time
                wait = (attempt + 1) * 8
                print(f"  [ROUTER RETRY {attempt+1}/{MAX_RETRIES}] {err_str[:60]}... waiting {wait}s")
                time.sleep(wait)
            else:
                break

    print(f"  [ROUTER ERROR] {last_err} -- using safe defaults")
    return {
        "brand_violation":     {"detected": False, "details": ""},
        "layout_contradiction":{"detected": False, "details": ""},
        "drop_wireframes":     False,
    }
