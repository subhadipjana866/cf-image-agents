"""
v5_4_fabrication_check.py
----------------------------------------------------------------------
Deterministic backstop against a specific, repeatedly-confirmed failure: the
CoT invents a plausible-sounding SPECIFIC FACT (a price, a discount, a date,
a promo code) to fill a gap the user never specified, instead of leaving it
out or asking. Confirmed across multiple runs this session — "STARTING AT
$12", "$14" for two menu items, "ROOFTOP EXCLUSIVE" as filler — each time on
a prompt with no offer/price given.

WHY NOT JUST TRUST THE ROUTER: the router (v5_3_intent_router.route_intent)
already receives missing_fields and is instructed to return CLARIFY_FIRST
when a CRITICAL field is missing. In practice this is inconsistent — the
same "no price given" gap sometimes triggers CLARIFY_FIRST and sometimes
doesn't, because the router's routing_action is an LLM judgment call
weighed against overall clarity_score, not a hard rule. A well-written
prompt that is missing ONLY the price can score high enough on clarity that
the router lets it through, and the CoT then quietly invents a number.

WHY NO NEW LLM CALL: clarify_intent() ALREADY extracts missing_fields with
levels (CRITICAL/IMPORTANT/RECOMMENDED) and question text — the raw signal
already exists. This module is pure deterministic filtering: is a CRITICAL/
IMPORTANT missing field the KIND that a CoT is likely to paper over with an
invented specific fact, versus a field where vague creative filler is
harmless (e.g. a missing "occasion_theme" just softens the vibe; a missing
"offer_mechanic" tempts the model to invent a price)? Fast, free, and -- by
being deterministic -- provides a hard gate the router's own judgment can't
override, mirroring the existing structural-ambiguity-flag hard gate in
v5_4_pipeline_api.py.
"""

from __future__ import annotations

import re

# Field id / question text patterns strongly associated with the model
# inventing a specific, plausible-sounding fact rather than vague filler.
_FABRICATION_PRONE_RE = re.compile(
    r"offer|discount|price|percent|%|deal|promo|coupon|code\b|"
    r"\bdate\b|valid|expir|deadline|"
    r"address|phone|location\b",
    re.IGNORECASE,
)


def at_risk_missing_fields(missing_fields: list[dict]) -> list[dict]:
    """
    Filter clarify_intent()'s missing_fields down to the ones a CoT is likely
    to fabricate a specific value for. Only CRITICAL/IMPORTANT fields are
    considered -- RECOMMENDED fields (e.g. a footer disclaimer) are low-stakes
    enough that generic filler is an acceptable outcome, not worth a pause.
    """
    at_risk = []
    for f in missing_fields or []:
        if f.get("level") not in ("CRITICAL", "IMPORTANT"):
            continue
        text = f"{f.get('field_id', '')} {f.get('question', '')}"
        if _FABRICATION_PRONE_RE.search(text):
            at_risk.append(f)
    return at_risk


def build_fabrication_clarification_questions(at_risk: list[dict]) -> list[str]:
    """One clear question per at-risk field, explicit that we won't guess."""
    questions = []
    for f in at_risk:
        base = f.get("question") or f"What is the {f.get('field_id', 'missing detail')}?"
        questions.append(
            base + " (If you'd rather not specify, say so and we'll leave it out or use "
            "general language instead of guessing a number.)"
        )
    return questions
