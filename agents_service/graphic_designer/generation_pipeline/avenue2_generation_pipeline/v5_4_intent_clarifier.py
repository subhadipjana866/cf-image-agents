"""
v5_4_intent_clarifier.py  (v5.4 rev-2)
----------------------------------------
Phase 1.1: Intent Clarification Engine.

Rev-2 changes:
  - Accepts required_content_fields from layout profile (Phase 0 output)
  - Outputs CLARITY_SCORE (0-100) + CLARITY_LEVEL (HIGH/MEDIUM/LOW)
  - Outputs FIELD_STATUS per required field (PRESENT / INFERRED / MISSING)
  - Outputs MISSING_FIELDS with exact clarification questions
  - Outputs AMBIGUITY_FLAGS for contradictory or unclear statements
  - Return dict carries all parsed sections for Phase 1.5 and experiment runner
"""

import os
import re
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("MODEL_PRO", "models/gemini-3.1-pro-preview")

SYSTEM_ROLE = """You are an Intent Clarification and Validation Engine for an AI marketing graphic pipeline.

You have TWO jobs:

JOB 1 — CREATIVE BRIEF EXTRACTION
Rewrite the user's raw prompt into a structured creative brief with:
  SUBJECT, PRODUCTS, OFFER, THEME, STYLE, and PROPOSED TEXT (Headline, Discount, CTA, Terms)

JOB 2 — MANDATORY FIELD VALIDATION
Given the design goal's required content fields, determine whether each field is:
  PRESENT  — the user explicitly stated it
  INFERRED — not stated, but can be reasonably concluded from context (low risk)
  MISSING  — not present and cannot be reliably inferred

Then produce:
  CLARITY_SCORE: integer 0–100
    90-100 = All critical fields present, intent unambiguous
    70-89  = All critical fields present, minor inference or style ambiguity
    40-69  = Some critical fields inferred or one missing; intent mostly readable
    20-39  = Multiple critical fields missing or intent is ambiguous
    0-19   = Fundamental information absent; cannot produce meaningful output

  CLARITY_LEVEL: HIGH (score >= 70) | MEDIUM (40-69) | LOW (< 40)

---

## FIELD STATUS RULES

PRESENT  — Use this when the user's text directly states the field value.
            Example: "20% off" → offer_mechanic = PRESENT: "20% off"

INFERRED — Use this ONLY when ALL of the following are true:
            a) The value is not stated explicitly
            b) It can be derived from context with high confidence (>85%)
            c) It is not a CRITICAL required field (or the design goal makes it obvious)
            Mark inferred values with [INFERRED] and explain the reasoning.
            Never infer a call to action: a CTA appears only when the user asks for one.

MISSING  — Use this when the field is not stated AND cannot be reliably inferred.
            Always provide the exact clarification question from the field definition.

## AMBIGUITY FLAGS

Flag any of these patterns:
  - UNRELATED_INPUT: The prompt has no identifiable relationship to the design goal or marketing context.
    Use when even the most generous interpretation yields nothing actionable — random words, technical jargon,
    placeholder text, accidental input (e.g. "ai hallucination", "test", "asdf", "hello world", "lorem ipsum").
    This is different from VAGUE_OFFER (which is on-topic but missing details) — UNRELATED_INPUT means the
    text is not a marketing prompt at all.
  - CONTRADICTORY: User requests two mutually exclusive things
  - WRONG_GOAL_TYPE: Prompt describes a different design category (e.g., a greeting card instead of a sale)
  - VAGUE_OFFER: An offer is hinted at but not specified (e.g., "amazing deals" without a discount)
  - MULTI_OCCASION: Multiple different holidays/occasions mentioned simultaneously
  - BRAND_MISMATCH: Prompt seems to be for a different brand than the brand profile
  - OVERLOADED: Prompt has too many competing messages for a single graphic

---

## OUTPUT FORMAT

Output the creative brief first, then a separator, then the validation section.

SUBJECT: <brand + category>
PRODUCTS: <specific items, or "not specified">
OFFER: <exact deal text, or "not specified">
THEME: <occasion/event, or "not specified">
STYLE: <visual aesthetic keywords>
PROPOSED TEXT:
  - Headline: "<exact string>"
  - Discount: "<exact string or MISSING>"
  - CTA: "<exact string ONLY if the user explicitly asked for a call to action or a button, otherwise none>"
  - Terms: "<exact string or none>"

--- FIELD VALIDATION ---
CLARITY_SCORE: <0-100>
CLARITY_LEVEL: <HIGH | MEDIUM | LOW>

FIELD_STATUS:
<For each required field, one line:>
  [field_id] STATUS: "<value or explanation>" — <brief reasoning>

MISSING_FIELDS:
<If none: "none">
<If any: list as:>
  [CRITICAL|IMPORTANT|RECOMMENDED] field_id → "<exact question from field definition>"

AMBIGUITY_FLAGS:
<If none: "none">
<If any: list as:>
  TYPE: <flag type> — <specific description of what is ambiguous or contradictory>
"""


def _parse_clarifier_output(text: str, required_fields: list[dict]) -> dict:
    """Parse the structured clarifier output into a dict with all sections."""
    result = {
        "clarified_text": "",
        "clarity_score": 50,
        "clarity_level": "MEDIUM",
        "field_status": {},
        "missing_fields": [],
        "ambiguity_flags": [],
    }

    # Split on the separator
    parts = text.split("--- FIELD VALIDATION ---", 1)
    result["clarified_text"] = parts[0].strip()

    if len(parts) < 2:
        return result

    validation = parts[1].strip()

    # Parse CLARITY_SCORE
    m = re.search(r"CLARITY_SCORE:\s*(\d+)", validation)
    if m:
        result["clarity_score"] = int(m.group(1))

    # Parse CLARITY_LEVEL
    m = re.search(r"CLARITY_LEVEL:\s*(HIGH|MEDIUM|LOW)", validation)
    if m:
        result["clarity_level"] = m.group(1)

    # Parse FIELD_STATUS block
    fs_match = re.search(r"FIELD_STATUS:(.*?)(?=MISSING_FIELDS:|$)", validation, re.DOTALL)
    if fs_match:
        fs_block = fs_match.group(1).strip()
        for line in fs_block.splitlines():
            line = line.strip()
            m = re.match(r"\[(\w+)\]\s+(PRESENT|INFERRED|MISSING):\s*(.*)", line)
            if m:
                result["field_status"][m.group(1)] = {
                    "status": m.group(2),
                    "value": m.group(3).strip(),
                }

    # Parse MISSING_FIELDS
    mf_match = re.search(r"MISSING_FIELDS:(.*?)(?=AMBIGUITY_FLAGS:|$)", validation, re.DOTALL)
    if mf_match:
        mf_block = mf_match.group(1).strip()
        if mf_block.lower() != "none":
            for line in mf_block.splitlines():
                line = line.strip()
                m = re.match(r"\[(CRITICAL|IMPORTANT|RECOMMENDED)\]\s+(\w+)\s*→\s*\"(.+)\"", line)
                if m:
                    result["missing_fields"].append({
                        "level": m.group(1),
                        "field_id": m.group(2),
                        "question": m.group(3),
                    })

    # Parse AMBIGUITY_FLAGS
    af_match = re.search(r"AMBIGUITY_FLAGS:(.*?)$", validation, re.DOTALL)
    if af_match:
        af_block = af_match.group(1).strip()
        if af_block.lower() != "none":
            for line in af_block.splitlines():
                line = line.strip()
                m = re.match(r"([\w_]+):\s*(\w+)\s*[—-]\s*(.+)", line)
                if m:
                    result["ambiguity_flags"].append({
                        "type": m.group(2),
                        "description": m.group(3).strip(),
                    })
                else:
                    m2 = re.match(r"(\w+):\s*(.+)", line)
                    if m2:
                        result["ambiguity_flags"].append({
                            "type": m2.group(1),
                            "description": m2.group(2).strip(),
                        })

    return result


def clarify_intent(
    raw_prompt: str,
    design_goal: str,
    platform: str,
    layout: str,
    required_fields: list[dict] | None = None,
    model: str | None = None,
    role: str | None = None,
    pin: str | None = None,
) -> dict:
    """
    Phase 1.1: Extract structured creative brief + validate mandatory content fields.

    Args:
        raw_prompt: The user's raw input.
        design_goal: The campaign design goal (from config).
        platform: Target platform (from config).
        layout: Selected layout variant name.
        required_fields: List of required_content_fields from layout profile.
        model: Override model (defaults to DEFAULT_MODEL).

    Returns dict with:
        clarified_text  — structured brief string (flows to router + CoT)
        clarity_score   — 0-100
        clarity_level   — HIGH / MEDIUM / LOW
        field_status    — per-field PRESENT/INFERRED/MISSING
        missing_fields  — list of {level, field_id, question}
        ambiguity_flags — list of {type, description}
        _usage          — token counts
    """
    from llm_fallback import generate_with_fallback
    use_model = model or DEFAULT_MODEL

    # Build the required-fields section for the prompt
    if required_fields:
        fields_desc = "\n".join(
            f"  [{f['field_id']}] ({f['required_level'].upper()}) — {f['description']}\n"
            f"    Examples: {', '.join(f.get('examples', []))}\n"
            f"    Can infer: {'yes' if f.get('can_infer') else 'no'}"
            + (f" — {f['inference_note']}" if f.get('can_infer') and f.get('inference_note') else "")
            + f"\n    If missing, ask: \"{f['question_if_missing']}\""
            for f in required_fields
        )
        required_section = f"\n--- REQUIRED CONTENT FIELDS FOR THIS DESIGN GOAL ---\n{fields_desc}\n"
    else:
        required_section = (
            "\n--- REQUIRED CONTENT FIELDS ---\n"
            "No specific required fields provided. Apply general marketing graphic standards:\n"
            "  [offer_mechanic] (critical) — what discount/deal is offered?\n"
            "  [subject_brand]  (critical) — what brand/business is this for?\n"
            "  [cta_action]     (optional) — ONLY if the user asked for a call to action; never MISSING\n"
            "  [occasion_theme] (important) — what event/season/occasion?\n"
            "  [validity_period] (recommended) — when is the offer valid?\n"
        )

    user_message = (
        f"DESIGN GOAL: {design_goal}\n"
        f"PLATFORM: {platform}\n"
        f"LAYOUT: {layout}\n"
        f"{required_section}\n"
        f"USER PROMPT:\n{raw_prompt}"
    )

    try:
        response = generate_with_fallback(
            model=use_model,
            prompt_text=user_message,
            temperature=0.0,
            max_output_tokens=1024,
            system_instruction=SYSTEM_ROLE,
            role=role,
            pin=pin,
        )
        text = response.text.strip()
        parsed = _parse_clarifier_output(text, required_fields or [])

        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
            }
        parsed["_usage"] = usage
        parsed["_raw"] = text
        return parsed

    except Exception as e:
        print(f"  [ERROR] Intent Clarifier failed: {e}")
        return {
            "clarified_text": f"PROMPT: {raw_prompt}\nDESIGN GOAL: {design_goal}",
            "clarity_score": 0,
            "clarity_level": "LOW",
            "field_status": {},
            "missing_fields": [],
            "ambiguity_flags": [{"type": "API_ERROR", "description": str(e)}],
            "_usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "_raw": "",
        }
