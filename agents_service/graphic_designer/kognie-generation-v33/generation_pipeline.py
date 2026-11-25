"""
armc_v31_pipeline.py
----------------------------------------------------------------------
Arm C v3.1 — subject-grounded generation with typed references.

    GROUND   (facts + locked text; extraction, so thinking stays low)
        |
    SUBJECT  (what this graphic is ACTUALLY about; thinking HIGH)
        |
        +--> TYPOGRAPHER  ] all three run CONCURRENTLY, each with its own
        +--> ART DIRECTOR ] model fallback, so one slow or failed provider
        +--> COMPOSITOR   ] degrades that discipline instead of the request
        |
    DIRECTOR (arbitrates conflicts, compiles ONE prompt; thinking HIGH)
        |
    RENDER   (gpt-image-2.5-flare -> Nano Banana 2)

WHAT CHANGED FROM v2, AND WHY

  1. A SUBJECT NODE NOW EXISTS.
     v2 handed the art director `raw_prompt[:400]` -- the brief, truncated --
     and asked it to invent a "concept". Nothing in the pipeline ever asked
     the plain question "what is this graphic actually depicting?". So a night
     market brief could yield a tasteful gradient with type on it: on-brief by
     the letter, but not a night market. The subject node answers that question
     first, in concrete nouns, and every downstream specialist inherits it.

     It reasons about the subject through three lenses at once, because the
     same subject wants a different picture in each:
       - the SUBJECT itself  (a night market has specific bulbs, steam, signage)
       - the DESIGN GOAL     (a flash sale and a testimonial about one shop are
                              not the same photograph)
       - the PLATFORM        (a story is a vertical close crop; signage is read
                              from four metres away)

  2. GEMINI 3.8 FLASH WITH THINKING AT HIGH.
     v2 ran gemini-3.1-flash-lite for specialists and 3.1-pro for the director.
     Everything reasoning-shaped now runs on 3.8-flash at thinking_level=high.
     GROUND stays low on purpose: it copies strings out of a brief, and giving
     an extraction task a reasoning budget invites it to embellish facts.

  3. GPT-IMAGE-2 IS ACTUALLY REACHED.
     It was always the configured primary, but with no OPENAI_API_KEY every
     render silently fell through to Nano Banana 2, so it had never once run.

  4. UP TO 5 REFERENCE IMAGES.
     images.edit accepts a list. v2 passed a single handle, which capped us at
     one reference and made product+hero briefs impossible.

OBSERVABILITY
    Every provider call is timed and priced as it completes and emitted through
    an on_event callback, so cost and latency are visible live. Thinking tokens
    are billed at the output rate and are counted -- at thinking_level=high they
    are frequently the majority of a call's cost, and a total that ignores them
    understates the real spend several-fold.

CREDENTIALS
    Read from the environment, never from disk:
        GEMINI_API_KEY   reasoning
        OPENAI_API_KEY   gpt-image-2.5-flare
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import time
from pathlib import Path

import brief_store as base   # slug, GROUND_PROMPT, exception types
from brief_store import BriefNotFound, MissingCredential
import typography as ty        # real fonts, legibility, contrast, editable-text metadata
import reference_roles as refroles
import antislop

# --- models ---------------------------------------------------------------
REASONING_MODEL = os.environ.get("ARMC3_MODEL", "gemini-3.8-flash")
REASONING_FALLBACK = os.environ.get("ARMC3_MODEL_FB", "gemini-3.7-flash")
# gpt-image-2.5-flare: the model the typography work was validated on (fonts from
# reference images 24/28 lines; layout-image text within 0-1 px on 2:3), and ~2x
# faster than gpt-image-2.5-sunburst at equal font fidelity. ARMC3_IMAGE_MODEL
# overrides it (e.g. gpt-image-2).
IMAGE_MODEL = os.environ.get("ARMC3_IMAGE_MODEL", "gpt-image-2.5-flare")
IMAGE_FALLBACK_MODEL = os.environ.get("ARMC3_IMAGE_FALLBACK", "gemini-3.1-flash-image")

MAX_REFERENCE_IMAGES = 5

# The single source of layout guidance, identical to the file the shipped
# package reads. Until now this module called base.load_brief(), which reads the
# pre-bundle armc_store briefs -- whose composition_menu was capped at 3 by the
# old builder while the bundle carries 4. Experiments and production were
# therefore reading different data, and a lineage test that "exhausted the menu"
# exhausted a menu one option shorter than the one users would get.
GUIDANCE_PATH = Path(os.environ.get(
    "KOGNIE_LAYOUT_GUIDANCE", Path(__file__).resolve().parent / "layout_guidance.json"))
with open(GUIDANCE_PATH, encoding="utf-8") as _fh:
    GUIDANCE = json.load(_fh)


def combos() -> list:
    """Every (design_goal, platform) pair this package can serve."""
    return sorted((r["design_goal"], r["platform"]) for r in GUIDANCE.values())


def goals() -> list:
    return sorted({r["design_goal"] for r in GUIDANCE.values()})


def platforms() -> list:
    return sorted({r["platform"] for r in GUIDANCE.values()})


def load_guidance(design_goal: str, platform: str) -> dict:
    """Layout guidance for one combination — same contract as the package.

    `authored` resolves the same way it does in the shipped brief_store: a
    hand-written house brief wins where one exists, otherwise the derived
    guidance fills the slot.
    """
    combo = f"{base.slug(design_goal)}__{base.slug(platform)}"
    rec = GUIDANCE.get(combo)
    if rec is None:
        raise BriefNotFound(
            f"no layout guidance for {design_goal} x {platform} ({combo}). "
            f"{len(GUIDANCE)} combinations are available; see combos().")
    out = dict(rec)
    out["authored"] = rec.get("authored") or rec.get("guidance") or {}
    # A call to action is never mandatory: graphics carry one only when the
    # user asks for it (see user_asked_cta). Demoted here, the one place every
    # consumer reads the guidance from, so GROUND never lists "Button Text" as
    # missing and the session never stops to ask for it.
    must = rec.get("must_communicate") or []
    demoted = [f for f in must if f.get("field_id") in CTA_FIELD_IDS]
    if demoted:
        out["must_communicate"] = [f for f in must if f.get("field_id") not in CTA_FIELD_IDS]
        may = list(rec.get("may_include") or [])
        have = {f.get("field_id") for f in may}
        out["may_include"] = may + [f for f in demoted if f.get("field_id") not in have]
    return out

# A reasoning call that never returns is worse than one that fails: the failure
# has a fallback path, the hang has nothing. A single call at thinking=high runs
# 20-70s in practice, so 180s is far outside the normal band while still bounding
# a wedged connection. Enforced twice -- at the SDK's HTTP layer and again as a
# thread ceiling in _call -- because the first only covers the transport.
CALL_TIMEOUT_S = int(os.environ.get("ARMC3_CALL_TIMEOUT_S", "180"))
RENDER_TIMEOUT_S = int(os.environ.get("ARMC3_RENDER_TIMEOUT_S", "300"))

# Per-node reasoning budget, all overridable so latency can be swept rather
# than argued about. The defaults encode a judgement about which stages are
# doing genuine reasoning and which are following rules:
#
#   GROUND      low     copies facts out of a brief; a reasoning budget on an
#                       extraction task invites it to fill gaps, not report them
#   SUBJECT     high    decides what the picture actually contains, and predicts
#                       how a renderer will get this subject wrong. This is the
#                       stage the previous version lacked, so it earns its budget
#   TYPOGRAPHER high    (candidate for reduction -- largely procedural)
#   ART         high    the one specialist genuinely synthesising from the subject
#   COMPOSITOR  high    (candidate for reduction -- largely procedural)
#   DIRECTOR    medium  arbitration plus a must-contain checklist; compressing
#                       this node has silently dropped requirements before
GROUND_THINKING = os.environ.get("ARMC3_GROUND_THINKING", "low")
SUBJECT_THINKING = os.environ.get("ARMC3_SUBJECT_THINKING", "high")
TYPO_THINKING = os.environ.get("ARMC3_TYPO_THINKING", "high")
ART_THINKING = os.environ.get("ARMC3_ART_THINKING", "high")
COMP_THINKING = os.environ.get("ARMC3_COMP_THINKING", "high")
DIRECTOR_THINKING = os.environ.get("ARMC3_DIRECTOR_THINKING", "medium")
# SLOP GUARD (antislop.py): pre-generation rules against the defaults that make a graphic
# read as AI slop, given to the specialists, the DIRECTOR and the negative prompt. Plain
# code, no extra model call. 0 turns it off.
ANTISLOP = os.environ.get("ARMC3_ANTISLOP", "1") not in ("0", "false", "False")
# With the guard on, the Director writes the short prompt (DIRECTOR_FINAL_SHORT).
SHORT_PROMPT = os.environ.get("ARMC3_SHORT_PROMPT", "1") not in ("0", "false", "False")
# With the short prompt on, one fast call tailors the negatives and positives to the prompt.
# It runs on a hard time limit, and any failure leaves the full lists in place.
TAILOR = os.environ.get("ARMC3_TAILOR", "1") not in ("0", "false", "False")
TAILOR_THINKING = os.environ.get("ARMC3_TAILOR_THINKING", "low")
TAILOR_TIMEOUT_S = float(os.environ.get("ARMC3_TAILOR_TIMEOUT_S", "20"))
# The copywriter writes every word on the graphic. On unless switched off; if it fails or
# takes longer than this after SUBJECT is done, the run keeps GROUND's text.
COPYWRITER = os.environ.get("ARMC3_COPYWRITER", "1") not in ("0", "false", "False")
COPY_TIMEOUT_S = float(os.environ.get("ARMC3_COPY_TIMEOUT_S", "60"))
# GROUND only copies strings and normally takes under 10 seconds, so a stalled call moves to
# the fallback model early instead of holding the whole run for the full timeout.
GROUND_TIMEOUT_S = float(os.environ.get("ARMC3_GROUND_TIMEOUT_S", "60"))

# Ground can run on a cheaper, faster model than the rest: it extracts, it does
# not reason. Defaults to the main model so behaviour is unchanged until asked.
GROUND_MODEL = os.environ.get("ARMC3_GROUND_MODEL", "")

# $/1K tokens (input, output, is_estimate). Thinking tokens bill as output.
RATES = {
    "gemini-3.8-flash": (0.0005, 0.003, True),
    "gemini-3.7-flash": (0.0005, 0.003, True),
    "gemini-3.1-flash-image": (0.0005, 0.06, False),
}
# Render quality. "high" was hardcoded and is the single biggest line item;
# medium is the default now. Override with ARMC3_IMAGE_QUALITY.
IMAGE_QUALITY = os.environ.get("ARMC3_IMAGE_QUALITY", "medium")

# $/image by quality tier. ESTIMATES -- but the API returns real token counts in
# usage.output_tokens, which are now recorded next to the estimate, so one line
# from the billing dashboard is enough to replace these with true rates.
IMAGE_PRICE_BY_QUALITY = {"low": 0.02, "medium": 0.07, "high": 0.25}

# gpt-image-2 sizing, from the published parameter spec plus probing the live
# API. Both earlier versions of this were guesses and both were wrong:
#
#   v1  a three-entry lookup table (1024x1024 / 1024x1536 / 1536x1024). This is
#       why every 9:16 brief silently rendered at 2:3.
#   v2  computed, but bounded 1024..2048 on a guess. 1024 is NOT the floor and
#       2048 is NOT the ceiling, so 16:9 was forced to 2048x1152 when 1280x720
#       is an exact fit -- 2.6x the pixels for a thumbnail displayed at 210px.
#
# Measured limits (probe results in the comments):
#   both dimensions divisible by 16          (API rejects otherwise)
#   short edge >= 640                        (1024x576 rejected, 1280x720 OK,
#                                             512x1024 rejected, 640x1024 OK)
#   long edge  <= 3840                       (3840x2160 OK)
#   aspect between 1:3 and 3:1               (1024x3072 OK)
OPENAI_SIZE_MULTIPLE = 16
OPENAI_MIN_EDGE = 640
OPENAI_MAX_EDGE = 3840
OPENAI_MIN_ASPECT, OPENAI_MAX_ASPECT = 1 / 3, 3.0

# Smallest short edge we are willing to ship per platform family. Bigger is not
# better here: render time and payload both grow, and every one of these is
# viewed far below its native size. Signage is the exception -- it is read from
# several metres and genuinely benefits from resolution.
SHORT_EDGE_TARGET = {
    "YouTube Thumbnail": 720,          # displays ~210px wide
    "X / Twitter In-Stream Post": 900,
    "Facebook Feed Landscape": 900,
    "Presentation Slide": 1080,
    "Instagram Story": 1024,
    "Instagram Square Post": 1024,
    "Instagram Portrait Post": 1024,
    "LinkedIn Feed Post": 1024,
    "Pinterest Pin": 1024,
    "Digital Signage Portrait": 1440,  # read from a distance
}
DEFAULT_SHORT_EDGE = 1024
# Above this relative error the frame is visibly the wrong shape, and we hand
# the job to Nano Banana 2, which takes the ratio string directly.
ASPECT_TOLERANCE = 0.02


def openai_size(aspect_ratio: str, platform: str | None = None
                ) -> tuple[str | None, float]:
    """Smallest EXACT fit for a ratio at or above the platform's short edge.

    Returns (size, relative_aspect_error). Smallest rather than largest on
    purpose: aspect accuracy is the thing that matters, and extra pixels cost
    render time and transfer without being seen.
    """
    try:
        aw, ah = (float(x) for x in str(aspect_ratio).split(":"))
        target = aw / ah
    except Exception:
        return f"{DEFAULT_SHORT_EDGE}x{DEFAULT_SHORT_EDGE}", 0.0
    if not (OPENAI_MIN_ASPECT <= target <= OPENAI_MAX_ASPECT):
        return None, 1.0                       # outside what the model accepts

    want_short = SHORT_EDGE_TARGET.get(platform or "", DEFAULT_SHORT_EDGE)
    step = OPENAI_SIZE_MULTIPLE
    best, best_key = None, None
    for h in range(OPENAI_MIN_EDGE, OPENAI_MAX_EDGE + 1, step):
        w = int(round(h * target / step)) * step
        if not (OPENAI_MIN_EDGE <= w <= OPENAI_MAX_EDGE):
            continue
        if min(w, h) < want_short:
            continue
        err = abs((w / h) - target) / target
        # exact first, then the smallest canvas that satisfies the short edge
        key = (round(err, 6), w * h)
        if best_key is None or key < best_key:
            best, best_key = (w, h), key
    if best is None:                           # target short edge unreachable
        for h in range(OPENAI_MIN_EDGE, OPENAI_MAX_EDGE + 1, step):
            w = int(round(h * target / step)) * step
            if not (OPENAI_MIN_EDGE <= w <= OPENAI_MAX_EDGE):
                continue
            err = abs((w / h) - target) / target
            key = (round(err, 6), w * h)
            if best_key is None or key < best_key:
                best, best_key = (w, h), key
    if best is None:
        return None, 1.0
    return f"{best[0]}x{best[1]}", best_key[0]


def cost_of(model: str, usage: dict) -> tuple[float, bool]:
    k = model.replace("models/", "")
    if k not in RATES:
        return 0.0, True
    i, o, est = RATES[k]
    # Thinking tokens are output tokens. Omitting them was understating cost.
    out = usage.get("completion_tokens", 0) + usage.get("thinking_tokens", 0)
    return round(usage.get("prompt_tokens", 0) / 1000 * i + out / 1000 * o, 6), est


SAFE_AREA = {
    "Instagram Story": "Keep all text and the focal subject inside the middle 80% vertically — "
                       "the top ~14% and bottom ~20% are covered by app chrome and the reply bar.",
    "Digital Signage Portrait": "Viewed from several metres away. Keep content inside a generous "
                                "8% margin and size type for distance reading, not for a phone.",
    "YouTube Thumbnail": "The bottom-right corner is covered by the duration stamp — keep it clear. "
                         "Must stay legible at roughly 210x118px.",
    "Presentation Slide": "Projected. Keep a 7% margin; avoid placing anything critical in the "
                          "bottom 10% where heads and furniture obscure it.",
}
DEFAULT_SAFE = ("Keep all text and critical content inside a 6-8% margin on every edge. "
                "Nothing may touch or cross the frame.")


def safe_area(platform: str) -> str:
    return SAFE_AREA.get(platform, DEFAULT_SAFE)


# ---------------------------------------------------------------------------
# Locked text hygiene
# ---------------------------------------------------------------------------

def dedupe_locked(locked: dict) -> tuple[dict, list[str]]:
    """Drop locked strings that are already contained in a longer locked string.

    The schemas themselves create this: company_milestone mandates BOTH
    milestone_year ("25 years") AND milestone_headline ("25 years, over 400
    vessels, still the same yard"). Both were extracted correctly, both were
    handed to the renderer as must-appear-verbatim, and the graphic came back
    with "25 YEARS" twice -- once as a badge, once opening the headline.

    Nothing was wrong with the extraction. What was missing was anyone noticing
    that one string was a substring of another.
    """
    items = [(k, str(v).strip()) for k, v in (locked or {}).items()
             if v and str(v).strip()]
    dropped = []
    keep = {}
    for k, v in items:
        # Whole words only: the brand "Stride" is not already said by the code "STRIDE50".
        inside = re.compile(rf"(?<!\w){re.escape(v)}(?!\w)", re.I)
        swallowed_by = next(
            (k2 for k2, v2 in items
             if k2 != k and len(v2) > len(v) and inside.search(v2)), None)
        if swallowed_by:
            dropped.append(f"{k} ({v!r}) is already contained in {swallowed_by}")
        else:
            keep[k] = v
    return keep, dropped


_STOPWORDS = frozenset(
    "a an the and or of on in at to for with from by our your my its this that these those "
    "is are be get all now just only".split())


def _content_tokens(text: str) -> set:
    # "50%" and "$50" stay whole: a number is the part that makes two lines say
    # the same thing ("50% off everything" vs headline "Summer Sale 50% Off").
    words = re.findall(r"[$£€]?\d[\d.,]*%?|[a-z0-9']+", str(text).lower())
    return {w for w in words if w not in _STOPWORDS}


def dedupe_facts(facts: list, locked: dict) -> tuple[list, list[str]]:
    """Drop other_stated_facts that only restate the locked text.

    GROUND puts a fact in other_stated_facts when no field can hold it, but it
    routinely also copies a phrase that a field already holds: locked
    sale_headline "Summer Flash Sale 50% Off" plus other fact "50% off
    everything". The typographer is told to give every other fact "a real
    place in the hierarchy", so it set that one as the sub-headline -- and the
    graphic said 50% off twice, once in the headline and once right under it.

    A fact is dropped when most of its words (60%+) already appear in the
    locked text, or in a fact kept before it, AND it brings no number of its
    own. A new number ("extra 10% for members") is new information, so it stays.
    """
    seen = set()
    for v in (locked or {}).values():
        seen |= _content_tokens(v)
    kept, dropped = [], []
    for fact in facts or []:
        tokens = _content_tokens(fact)
        if not tokens:
            continue
        new_numbers = {t for t in tokens if any(c.isdigit() for c in t)} - seen
        covered = len(tokens & seen) / len(tokens)
        if not new_numbers and covered >= 0.6:
            dropped.append(f"other fact {fact!r} repeats the locked text "
                           f"({covered:.0%} of its words)")
            continue
        kept.append(fact)
        seen |= tokens
    return kept, dropped


# ---------------------------------------------------------------------------
# Call to action -- only when the user asked for one
# ---------------------------------------------------------------------------
# Graphics used to carry a "SHOP NOW" / "BUY NOW" line or button by default:
# the guidance made Button Text mandatory for some goals, the intent clarifier
# proposed a CTA for every brief, and the typographer set it "as plain
# confident type" when it decided against a button. House rule now: no call to
# action of any kind unless the user's own words ask for one.
CTA_FIELD_IDS = frozenset({"cta_text"})

_CTA_WORDS = r"(?:\bctas?\b|call[\s-]*to[\s-]*actions?|\bbutton\b)"
_CTA_REQUEST = re.compile(_CTA_WORDS, re.I)
_CTA_NEGATED = re.compile(r"\b(?:no|without|don'?t|do not|not|never|skip)\s+(?:\w+\s+){0,3}?"
                          + _CTA_WORDS, re.I)
_CTA_VERBS = re.compile(
    r"^(?:shop|buy|order|book|sign\s*up|register|learn|get|download|apply|join|claim|grab|"
    r"discover|explore|try|start|subscribe|reserve|visit|call|contact|tap|click|swipe|"
    r"enrol|enroll|rsvp|install|see|find|check|save)\b", re.I)


def user_asked_cta(text: str) -> bool:
    """Whether the user's own words ask for a call to action or a button.

    Pass the user's text, not a rewritten brief: the intent clarifier used to
    propose a CTA for every brief, so its output always "mentions" one.
    "no CTA" / "without a button" is not a request.
    """
    text = str(text or "")
    asks = len(_CTA_REQUEST.findall(text))
    return asks > len(_CTA_NEGATED.findall(text))


def _is_cta_key(key: str) -> bool:
    k = str(key).lower()
    return (("cta" in k or "button" in k or "call_to_action" in k)
            and not any(w in k for w in ("url", "link")))


def _is_cta_phrase(text: str) -> bool:
    # "Shop now", "Book your table" -- a short imperative with no number, URL
    # or handle in it. "Apply at kognie.com/jobs" and "Register by 12 May" carry
    # real information and are not treated as a CTA.
    t = str(text).strip().strip("!.").strip()
    return (len(t.split()) <= 4 and bool(_CTA_VERBS.match(t))
            and not re.search(r"\d|@|\.\w|/", t))


def strip_cta(locked: dict, facts: list) -> tuple[dict, list, list[str]]:
    """Remove call-to-action text the user did not ask for."""
    dropped = []
    keep = {}
    for k, v in (locked or {}).items():
        if _is_cta_key(k) or _is_cta_phrase(v):
            dropped.append(f"{k} ({v!r}) is a call to action the user did not ask for")
        else:
            keep[k] = v
    kept_facts = []
    for f in facts or []:
        if _is_cta_phrase(f):
            dropped.append(f"other fact {f!r} is a call to action the user did not ask for")
        else:
            kept_facts.append(f)
    return keep, kept_facts, dropped


CTA_TYPO_OFF = """CALL TO ACTION — NONE. The user did not ask for a call to action, so this graphic has
NO CTA of any kind: no drawn button, pill or tab, and no call-to-action line set as type either
-- no "Shop now", "Buy now", "Order now", "Learn more", "Sign up", "Book now", "Download now" or
anything like them, even where one would be conventional for this goal. Do not invent one to
fill space. This is not the same as actionable information: a promo code with its instruction
("use code X at checkout"), a date, a URL or an address the user gave are content and stay.
Set cta.button=false and cta.text=null, and leave CTA wording out of the hierarchy."""

CTA_TYPO_ON = """CALL TO ACTION — THE USER ASKED FOR ONE. Use the CTA wording the user gave (it is among
the text above); only if they gave no wording, write one short, specific label for this graphic.
If the user asked for a BUTTON, set button=true -- a stated request is a requirement. If they
asked for a call to action without saying button, choose: a drawn button where the action is
specific (register, redeem, book), otherwise the label set as plain confident type."""

CTA_DIRECTOR_OFF = """5. NO CALL TO ACTION. The user did not ask for one. Render no button, pill or tab and no
   call-to-action wording ("Shop now", "Buy now", "Order now", "Learn more", "Sign up"...), and
   STATE EXPLICITLY in the prompt that the graphic contains no button and no call-to-action
   text -- image models add a "SHOP NOW" button by habit unless told not to. A promo-code
   instruction, a date, a URL or an address the user gave is content, not a CTA, and stays."""

CTA_DIRECTOR_ON = """5. HONOUR THE CTA DECISION. The user asked for a call to action. The typographer decided
   whether it is a drawn button: if yes, render it as a button with its exact label; if no, set
   the label as plain type. One call to action only."""

CTA_NEGATIVES = ("call-to-action buttons, button or pill shapes containing text, 'Shop Now' / "
                 "'Buy Now' / 'Order Now' / 'Learn More' / 'Sign Up' text, invented calls to action")


# ---------------------------------------------------------------------------
# SUBJECT — the node v2 did not have
# ---------------------------------------------------------------------------
SUBJECT = """You are a visual researcher. Before anyone designs anything, you establish what this graphic is ACTUALLY DEPICTING, in concrete nouns.

DESIGN GOAL: {goal}          PLATFORM: {platform} ({aspect_ratio})
WHAT THIS FORMAT IS FOR: {format_character}

THE BRIEF, VERBATIM:
\"\"\"{raw_brief}\"\"\"

FACTS ALREADY ESTABLISHED: {facts}
USER'S OWN DIRECTION: {directives}
USER EXCLUSIONS (honour absolutely): {exclusions}
REFERENCE IMAGERY SUPPLIED: {photo_state}
{reference_rules}

WHY YOU EXIST

The previous version of this pipeline skipped straight to "concept" and produced graphics that
were on-brief by the letter and empty in the picture — a brief about a night market could come
back as a tasteful gradient with type on it. Nobody had ever asked the plain question: what is in
this image? You ask it, and you answer in specifics that someone who knows this subject would
recognise as true.

HOW TO THINK

1. NAME THE SUBJECT LITERALLY. Not "food and community" — "a night market stall grilling satay
   skewers over charcoal, strung with bare bulbs". If the brief names a product, a place, a trade
   or a cuisine, that thing must be physically present and identifiable in the frame.

2. GET THE DETAILS RIGHT. List concrete, specific things a knowledgeable person expects to see:
   materials, objects, tools, surfaces, signage, typical light sources, the human activity around
   it. These are what make an image read as observed rather than imagined. Be culturally and
   materially accurate; if the subject belongs to a specific place or tradition, respect its real
   visual vocabulary rather than a generic international version of it.

3. LET THE GOAL CHOOSE THE MOMENT. The same subject wants a different photograph depending on
   why the graphic exists. A flash sale wants urgency and product clarity; a testimonial wants a
   human moment and warmth; a grand opening wants threshold, arrival, first light on a new place;
   a hiring post wants the workplace as somewhere you would want to be. Say which MOMENT of this
   subject the goal calls for.

4. LET THE PLATFORM CHOOSE THE CROP. A vertical story frames one subject close and tall. Signage
   read from four metres needs a bold, simple, high-contrast subject with no fine detail. A
   thumbnail needs one unmistakable focal object that survives being 210px wide. Say how the
   platform reshapes the shot.

5. NAME THE CLICHÉS AND REJECT THEM. For this subject, what is the stock-photo version that a
   generative model will reach for by default? Say it explicitly so the art director can steer
   away from it — the smiling model with a headset, the anonymous flat-lay on white marble, the
   generic glass office tower, the plate lit from directly above.

{photo_rule}

{people_rule}{slop_subject}

Return ONLY strict JSON:
{{
{people_field}  "subject": "<the literal thing depicted, in concrete nouns, one sentence>",
  "domain": "<the trade, cuisine, industry or category it belongs to>",
  "authentic_details": ["<5-8 specific things a knowledgeable person expects to see>"],
  "hero_moment": "<which moment of this subject the DESIGN GOAL calls for, and why>",
  "platform_crop": "<how this PLATFORM reshapes the shot: distance, orientation, detail level>",
  "hero_scene": "<2-3 sentences: the concrete scene to depict, specific enough to photograph>",
  "cliches_to_avoid": ["<the generic default versions of this subject>"],
  "grounding_risk": "<what a renderer is most likely to get factually wrong about this subject>"
}}"""

PHOTO_RULE_WITH = """REFERENCE IMAGERY IS ATTACHED AND IS THE HERO.
Your job is NOT to invent a hero scene — the supplied pixels are the hero. Instead, establish the
subject matter so the surrounding design is coherent with it: what world this image belongs to,
what materials and light sit around it, what the environment should be. Set "hero_scene" to
describe the SUPPORTING environment and treatment only. Never describe what the attached imagery
depicts; naming its contents invites the renderer to rebuild it from words instead of preserving
it."""

PHOTO_RULE_WITHOUT = """NO REFERENCE IMAGERY SUPPLIED — the hero image will be generated, so the
scene you specify IS the graphic. It must be relevant to the brief, to the design goal and to the
platform simultaneously. A hero that is merely decorative, or generically on-theme, is the failure
this node exists to prevent."""

# This guard used to live only on the DIRECTOR, which was safe while the
# director was the first place a scene got invented. This node now commits to a
# scene upstream of it, so the rule has to arrive here too -- without it, a
# testimonial brief produced "a South Asian woman in her late 30s" for a named
# real person and carried it all the way into the render.
PEOPLE_RULE_NAMED = """NAMED REAL PEOPLE, NO REFERENCE IMAGERY ({names}).
You may NOT invent a likeness. Do not specify age, ethnicity, build, hair, face or clothing, and
do not stage them as a depicted individual. Inventing a face for a named real person fabricates
their likeness — a defect no matter how good the picture looks. Ground the subject in their WORK
and their ENVIRONMENT instead: the room, the tools, the surfaces, the evidence of what they do.
Anonymous hands or an out-of-focus figure at work are acceptable; a recognisable invented portrait
is not. Set "hero_scene" accordingly."""

PEOPLE_RULE_NONE = "No named real individuals appear in this brief."


# ---------------------------------------------------------------------------
# SPECIALIST 1 — TYPOGRAPHER
# ---------------------------------------------------------------------------
TYPOGRAPHER = """You are a typographer. You decide the type system for one graphic. You do NOT describe scenery, lighting or layout — other specialists own those.

DESIGN GOAL: {goal}          PLATFORM: {platform} ({aspect_ratio})
PLATFORM CRAFT: {platform_craft}
SUBJECT MATTER: {subject}

TEXT THAT MUST APPEAR (verbatim):
{locked}

ALSO STATED BY THE CLIENT (no template field holds these — give them a real
place in the hierarchy, not a leftover corner):
{other_facts}

USER'S OWN DIRECTION (if any): {directives}

THE USER'S BRIEF, VERBATIM — read it for anything they asked for explicitly:
\"\"\"{raw_brief}\"\"\"

YOUR CRAFT KNOWLEDGE

TYPE SYSTEM -- choose REAL FONTS from the Kognie library. Every font below is supplied to the renderer
as a reference image of its alphabet, so it is drawn faithfully and the text stays editable later.
Previous output described everything as "bold sans-serif" and every graphic looked identical: choose a
pairing with real tension that suits THIS goal's register and the subject's world.
{font_library}
RULES
  - 2 fonts by default, at most 3. The TITLE always has its own font -- no other element shares it.
    Subheading, body, CTA and caption may share one font, differentiated by size, weight or case.
  - Decorative faces (readability "display") only for titles or short lines; long copy needs a "text" face.
  - Text is optional: if nothing must appear, return an empty hierarchy and no fonts.
{font_requests}
State what each face does and why the pairing suits this specific message.

HIERARCHY — exactly one element is primary. Give it real dominance (roughly 3-5x the body size),
and make secondary elements clearly subordinate in size, weight or colour. Equal-weight text
everywhere is the single most common reason a graphic reads as amateur.

LEGIBILITY — for every text element, state how it stays readable against what sits behind it.
Options: place it over a quiet area of the image; a soft directional gradient scrim; a solid
colour field; a knockout panel. Aim for a strong value difference between text and its immediate
backdrop — if the type and background are close in tone, the graphic fails at thumbnail size.
Never rely on a drop shadow alone.

ACTIONABLE INFORMATION — a promo code, deadline or entry mechanic is never a
loose token. Group it with the words that make it usable and with the offer it
unlocks, as a single visual unit. "USE CODE FIRSTPOUR" reads as an instruction;
"FIRSTPOUR" alone reads as noise. If the brief carries a code AND an offer, they
belong together or immediately adjacent.

WITHIN that unit the hierarchy is not flat. The TOKEN is what the reader has to
carry away and type in later, so it is the dominant element — larger, heavier,
or set in the accent. The instruction words around it ("USE CODE", "AT
CHECKOUT", "ENTER AT BASKET") are supporting: set them smaller, lighter, or in
a muted weight so the eye lands on the code first and reads the instruction
second. "use code HARVEST30 at checkout" set at one uniform size wastes the
code; the words are the scaffolding, the code is the payload.

QUOTATIONS — if any locked string is something a named person or named source
SAID, the artwork DRAWS an opening quotation mark before it and a closing one
after it. These are visible glyphs in the design, not the delimiters you use
when writing the string down; a brief that merely says "in quotation marks"
gets read as punctuation in the instruction and disappears from the render.
Say which mark style you want and how it is set. The attribution goes clearly
apart: smaller, lighter, usually preceded by an em dash. The marks may be a
display feature — an oversized opening mark is a legitimate device — but they
must be present as drawn characters, never implied by layout alone.

ONE FACT, ONE ELEMENT — headline, sub-headline, eyebrow, badge and body line
must each tell the reader something the others do not. A sub-headline that
restates the headline's offer, number, product, name or date in other words
("SUMMER FLASH SALE 50% OFF" over "50% off everything") is the most common
defect in these graphics: the reader reads the same thing twice and the second
line wastes the most valuable space after the headline. Before you finish,
compare every pair of entries in your hierarchy. If two of the strings you were
given state the same fact, set it ONCE, in the element where it works hardest,
and leave the other out of the hierarchy entirely; that is not a breach of the
verbatim rule, because its content is still on the graphic. Never invent a new
line to fill the space it leaves.

{slop_typo}

{cta_rule}

Return ONLY strict JSON:
{{
  "fonts": {{"title": "<library key>", "subheading": "<key>", "body": "<key>", "cta": "<key>", "caption": "<key>"}},
  "pairing": "<the faces you chose (by family name) and what each does>",
  "why_this_pairing": "<one sentence tying it to this goal's register and this subject>",
  "hierarchy": [{{"text": "<verbatim string>", "role": "primary|secondary|tertiary", "treatment": "<size relationship, weight, case, colour>"}}],
  "legibility": "<how each text element stays readable against what is behind it>",
  "cta": {{"button": true|false, "reason": "<why>", "text": "<the label, or null>"}}
}}"""


# The ART DIRECTOR's light and material guidance. The SLOP GUARD swaps these for its
# natural-photography versions (antislop.LIGHT_CRAFT / MATERIAL_CRAFT) while it is on.
ART_LIGHT = """LIGHT — commit to ONE decisive design and describe direction, quality and colour temperature.
Hard low sun with long shadows; a single soft window from one side; hard key with a crisp rim
separating subject from ground; bright airy high-key. Prefer light the subject would actually
have — the bulbs, screens, flames or windows named in the authentic details are better motivation
than a studio setup. Include real speculars and controlled falloff. Flat, even, all-round
illumination is the clearest single tell of a generic AI render."""
ART_MATERIAL = """MATERIAL — describe how the specific surfaces in THIS scene behave: how glass refracts and catches
an edge, how seasoned metal throws speculars, how paper stock takes light, how fabric absorbs it,
how condensation beads and runs, how oil sheens, how worn wood reads. Material behaviour is what
makes a render read as a photograph rather than a picture of nothing."""
ART_LIGHT_SCHEMA = "<direction, quality, temperature, speculars, falloff, and what motivates it in-scene>"


# ---------------------------------------------------------------------------
# SPECIALIST 2 — ART DIRECTOR  (now subject-led)
# ---------------------------------------------------------------------------
ART_DIRECTOR = """You are an art director. You decide the imagery, light and colour for one graphic. You do NOT decide typography or element placement — other specialists own those.

DESIGN GOAL: {goal}          PLATFORM: {platform} ({aspect_ratio})
WHAT THIS FORMAT IS FOR: {format_character}

THE SUBJECT — ESTABLISHED, NOT NEGOTIABLE
A visual researcher has already determined what this graphic depicts. You are art-directing THIS
scene. Do not substitute a different concept, and do not abstract it away into mood or pattern:

{subject_block}

USER'S OWN DIRECTION (if any): {directives}
USER EXCLUSIONS (honour absolutely): {exclusions}
{photo_note}
{reference_rules}

ART-DIRECTION PROVOCATIONS for this format (options, not a checklist):
{provocations}

YOUR CRAFT KNOWLEDGE

FIDELITY FIRST — the subject and its authentic details above must be visibly, recognisably
present. Every one of your choices below serves that scene rather than replacing it. If you find
yourself describing a mood with no object in it, you have drifted; go back to the subject.

COLOUR — build a deliberate palette, not an assortment. State the relationship you are using:
analogous for calm and cohesion; complementary for tension and energy; a near-monochrome with one
saturated accent for focus; split-complementary for richness without noise. Give the palette a
VALUE STRUCTURE — one dominant tone across most of the canvas, a mid, and a small high-contrast
accent (roughly 60/30/10). Say which colour carries the accent and where its weight sits. Let the
subject's own real colours lead; invented palettes that fight the subject look artificial.

{light_craft}

{material_craft}

INTEGRATION — the imagery must be built so that type can live ON it. Deliberately design a quiet
region — an area of even tone, shadow or negative space — where text will sit. Do not compose a
busy image and leave the typographer to fight it.

{slop_art}

MARKS AND PRINTED BRANDING — do NOT invent any logo, wordmark, emblem, badge, seal, crest or app
icon, AND do not print any brand name onto objects in the scene. This second half is the one that
keeps failing: across recent renders the client's own name appeared stamped on an apron, on gloves
and on an overshoe, and real third-party equipment brands turned up on a treatment couch and a
goniometer. Both are defects. The client's real logo is composited in a later step, so fabricated
branding collides with it; and reproducing a real company's trademark in a commercial graphic is
not ours to do.

{branding_note}

Return ONLY strict JSON:
{{
{people_field}  "concept": "<one sentence: the image idea, naming the subject explicitly>",
  "palette": {{"relationship": "<analogous|complementary|near-monochrome|split-complementary>", "dominant": "<colour>", "secondary": "<colour>", "accent": "<colour>", "where_accent_sits": "<...>"}},
  "light": "{light_schema}",
  "materials": "<how the key surfaces in this specific scene behave>",
  "quiet_region": "<where in the frame the imagery is deliberately calm so type can sit there>",
  "scene": "<2-3 sentences describing the image concretely, with the subject's real details in it>"
}}"""


# ---------------------------------------------------------------------------
# SPECIALIST 3 — COMPOSITOR
# ---------------------------------------------------------------------------
COMPOSITOR = """You are a compositor. You decide where every element sits in the frame. You do NOT choose typefaces or scenery — other specialists own those.

DESIGN GOAL: {goal}          PLATFORM: {platform} ({aspect_ratio}, {w}x{h}px)
CANVAS: exactly {w}x{h}px at {aspect_ratio}. This is fixed for this platform -- compose for
this frame, never for a different shape or a cropped version of it.
SAFE AREA FOR THIS PLATFORM: {safe}

THE SUBJECT BEING FRAMED: {subject}
HOW THIS PLATFORM WANTS IT CROPPED: {platform_crop}

REFERENCE IMAGERY: {reference_rules}

ELEMENTS TO PLACE:
{elements}

COMPOSITIONS FOR THIS FORMAT -- pick exactly ONE and name the one you picked. These are the
compositions this format is built on; do not invent a different arrangement:
{menu}

YOUR CRAFT KNOWLEDGE

ASYMMETRY — this is the failure being corrected. Previous output was persistently centred, not
because anything asked for centre but because placement was UNDER-SPECIFIED and the renderer
defaults to symmetry. So be explicit and directional for EVERY element. Anchor the composition
off-centre: thirds, a strong vertical or horizontal division, a diagonal, a deliberate corner
weight. Centred symmetry is a legitimate choice only when you argue for it — never as a default.

FRAME INTEGRITY — nothing crosses or touches the edge unless it is a deliberate full-bleed
element. All text sits inside the safe area above. The subject is never stretched, squashed or
distorted to fill the frame; it keeps its true proportions and the composition adapts around it.

BALANCE — a large quiet area needs a small dense one to hold against it. State what balances what.

FOCAL PATH — say what the eye hits first, second, third, and how the arrangement moves it there.
The subject itself should own one of the first two positions; if the type wins every time, the
graphic is a poster about words rather than about the thing.

BREATHING ROOM — generous, uneven margins read as designed; tight, uniform ones read as a
template. Give the primary element room to dominate.

{slop_comp}

Return ONLY strict JSON:
{{
  "structure": "<the underlying division: e.g. 'left third type column against right two-thirds image', 'diagonal split lower-left to upper-right'>",
  "anchor": "<where the composition's weight sits and why it is off-centre>",
  "placements": [{{"element": "<name>", "where": "<specific position in words — 'upper-left, starting a third in from the left edge'>", "size": "<relative dominance>"}}],
  "balance": "<what counterweights what>",
  "focal_path": "<first, second, third>",
  "frame_notes": "<safe-area and proportion notes for the renderer>"
}}"""


# ---------------------------------------------------------------------------
# DIRECTOR
# ---------------------------------------------------------------------------
DIRECTOR = """You are the design director. Three specialists have each solved their own part of one graphic, on top of an established subject. Your job is to ARBITRATE where they conflict and compile ONE image-generation prompt.

DESIGN GOAL: {goal}     PLATFORM: {platform} ({aspect_ratio}, {w}x{h}px)

CANVAS: exactly {w}x{h}px at {aspect_ratio}, and the compositor's chosen composition is the
frame you are compiling for. Neither is negotiable.

THE SUBJECT THIS GRAPHIC DEPICTS — the finished image must show this, recognisably:
{subject_block}

TEXT THAT MUST APPEAR, VERBATIM — reproduce every string exactly:
{locked}

OTHER STATED FACTS — the client said these and no template field could hold them.
They are content, not decoration. Place them unless doing so would genuinely harm
the graphic; if a code's usage is listed here, those words go beside the code:
{other_facts}

NOT PROVIDED — design around these, never invent a value:
{missing}

USER'S OWN DIRECTION — requirements, not suggestions:
{directives}

USER EXCLUSIONS: {exclusions}
{people_note}
{photo_note}
{reference_rules}

{typography_note}

--- TYPOGRAPHER ---
{typo}

--- ART DIRECTOR ---
{art}

--- COMPOSITOR ---
{comp}

YOUR JOB

1. ARBITRATE. The specialists worked independently and may conflict — a display line the column
   cannot hold, an accent colour that fights the palette, a placement that sits over a busy region.
   Resolve every conflict explicitly in favour of the finished graphic. The composition may change
   to serve the type, or the type may change to serve the composition; decide, do not average.

2. PROTECT THE SUBJECT. Where a specialist has drifted into mood, pattern or abstraction and lost
   the actual subject matter, pull it back. The subject and its authentic details are the one thing
   that may not be arbitrated away.

3. KEEP AN ACTION WHOLE. Where the brief contains something the reader is meant
   to ACT ON -- a promo or discount code, a deadline, an entry mechanic, a booking
   step -- the instruction, the token and what it yields form ONE unit and must
   appear together, adjacent, and legible as a group.

   YOU MUST ADD THE INSTRUCTION WORDS. Rule 4 says reproduce every locked string
   exactly; it does NOT forbid you from setting words around one. The locked
   string is a substring of the line you render: locked "HARVEST30" becomes
   "USE CODE HARVEST30 AT CHECKOUT" and the locked string is still intact and
   still appears once. A previous version read the two rules as being in
   conflict, kept the token bare, and enclosed it in a ruled box instead --
   producing a graphic whose offer no reader could actually claim.

   A BOX IS NOT AN INSTRUCTION. A rule, border, dashed outline or tag shape
   around a code says "this is a code" to a designer and nothing at all to a
   customer. The WORDS carry the action; the shape is optional decoration.

   THE TOKEN OUTRANKS ITS INSTRUCTION. Inside that unit the code is the
   dominant element -- larger, heavier or in the accent -- and the words around
   it are set smaller and lighter. The reader's eye should land on HARVEST30
   and pick up "use code" and "at checkout" second. Say so explicitly in the
   prompt; a flat line at one weight throws away the only part they have to
   remember.
   Where OTHER STATED FACTS gives you how the code is used, those exact words
   go into the line. If a deadline or condition governs the action, it travels
   with it. This is not optional polish -- an action a reader cannot complete
   is a failed graphic.

4. SAY EACH THING ONCE. Every locked string appears exactly once in the finished
   graphic. A previous version rendered "25 YEARS" as a badge AND as the opening
   of the headline, because two mandated fields overlapped. Repetition is allowed
   only when it is a deliberate device that adds meaning -- and then you must say
   so explicitly. Otherwise: once, in the place it works hardest.

   THE SAME GOES FOR EACH FACT, not only each string. The headline and the
   sub-headline (and any eyebrow, badge or body line) carry DIFFERENT
   information: a sub-headline never restates the headline's offer, number,
   product, name or date in other words. If two strings you were given say the
   same thing -- "Summer Flash Sale 50% Off" and "50% off everything" -- render
   that fact ONCE, in the stronger element, and drop the other line rather than
   repeat it. Do not invent replacement copy for the line you dropped.

{cta_rule}

{slop_director}

{compile_step}

7. GUARD. No invented facts. No invented logos or marks, and no REAL THIRD-PARTY brand on any
   object, ever — reproducing another company's trademark is not ours to do.

   {branding_note}

   ATTRIBUTED QUOTATIONS CARRY QUOTATION MARKS. Where a locked string is something a named person
   or a named source SAID, the finished artwork shows an opening and a closing quotation mark
   around it, and the attribution reads as an attribution ("— Ines Roca, founder").

   THE MARKS ARE DRAWN GLYPHS, NOT PUNCTUATION IN YOUR SENTENCE. You quote every locked string
   with " when you state it, so saying "set it in quotation marks" is ambiguous and has already
   been read as meaning your own delimiters. Say it in words the renderer cannot mistake: state
   that a visible opening quotation mark is drawn before the first word and a visible closing
   quotation mark after the last, as part of the typography the viewer sees. Adding them does not
   breach verbatim reproduction -- the locked string is intact between them. Without them the
   graphic presents a person's words as the brand's own slogan, which misrepresents both.

   State the branding rule explicitly in the prompt. Nothing crosses
   the frame edge. Text sits inside the safe area with real contrast against whatever is behind it.
   The subject keeps its true proportions and is never stretched to fill the canvas.

Output exactly:

<direction>One sentence: the design in a line, and any conflict you resolved.</direction>

<final_prompt>
{final_spec}
</final_prompt>

<negative_prompt>
Comma-separated. Start from this defect list, add the subject-specific clichés to avoid, and add
ONLY user-stated exclusions. Never ban a subject the user did not object to.
{defect_tags}
ALWAYS INCLUDE: invented logos, fabricated wordmarks, real third-party trademarks
{branding_negatives}
{cta_negatives}
CLICHÉS FOR THIS SUBJECT: {cliches}
</negative_prompt>"""

DIRECTOR_COMPILE = """6. COMPILE one prompt that carries: the subject and its concrete details, the exact text, the type
   pairing and hierarchy, the palette and its value structure, the light, the materials, and the
   SPECIFIC placement of every element. Placement must be explicit and directional — vagueness
   here is what produced centred output."""

DIRECTOR_FINAL = """One flowing paragraph, 340-520 words, ready for an image model. Natural language only — never
coordinates, never "Zone 1", never pixel values, never real font names.

IT MUST CONTAIN ALL OF THE FOLLOWING. A previous version compressed to ~320 words and silently
dropped the last three; length is cheap, omission is not:
  - the SUBJECT, named concretely, with at least three of its authentic physical details
  - every locked string, exactly as given, and EACH ONE EXACTLY ONCE
  - every OTHER STATED FACT listed above, unless including it would genuinely
    harm the graphic -- these are things the client asked for that no template
    field could hold, and dropping them silently is how a brief gets half-served
  - any code, deadline or entry mechanic rendered WITH the words that make it
    actionable and adjacent to the offer it applies to, the token set larger and
    heavier than the instruction words around it
  - any attributed quotation with a VISIBLE opening and closing quotation mark
    DRAWN in the artwork around it, and its attribution visually subordinate
  - no other text of any kind: no repeated phrases, no invented taglines,
    no duplicated words across elements. If a phrase would appear twice,
    it appears once, in the single place it does the most work
  - no two text elements stating the same fact: the sub-headline adds
    information the headline does not carry, never the same offer, number,
    product or date reworded
  - the type pairing, described by letterform character, and which element dominates
  - the palette with its value structure, the lighting design, and key material behaviour
  - EXPLICIT DIRECTIONAL PLACEMENT for every element ("upper-left, a third in from the edge",
    "anchored bottom-right"). Vague placement is what makes the renderer default to centring,
    so give at least four distinct positional statements
  - for EACH text element, how it stays readable against whatever sits behind it
  - a statement that all text sits inside the safe area and no element crosses the frame edge
  - a statement that the subject keeps its true proportions and is never stretched to fill
  - every one of the user's stated visual requirements, none omitted
{slop_final}
End by stating it must be a finished, production-quality graphic."""

# The short prompt (SLOP GUARD v3). Every prompt in the experiments ran 354-434 words because
# DIRECTOR_FINAL demands at least 340. This keeps control of the content, scene, camera and
# light and leaves the rest to the generator; the negative list still follows it.
DIRECTOR_COMPILE_SHORT = """6. COMPILE one short prompt. The generator draws better from a few clear sentences than from a long
   specification: control the content, the scene, the camera and the light, and leave the rest to it."""

SHORT_PROMPT_LINES = """Scene: one or two plain sentences: what is in the frame and where it sits. Name the subject and at
most three details a viewer can see at that distance. Follow "in_frame" (the art director's, otherwise
the subject's). If a body is in frame, say plainly which part, how it is cropped and its natural
position; otherwise describe no action that needs a body. Add one short clause for the branding rule,
and follow any reference rules above.
Camera: distance and angle in a few everyday words, e.g. "eye level, about a metre away". Leave this
line out if the graphic is not a photograph.
Light: the source and direction in a few everyday words.
Text: every locked string and every OTHER STATED FACT, exactly as given, each EXACTLY ONCE, in reading
order; for each give only its role, its letterform in a few words, its colour and its position. A
code, deadline or entry mechanic sits with the words that make it actionable, the token larger and
heavier than those words. An attributed quotation has a VISIBLE opening and closing quotation mark
drawn in the artwork, and a smaller attribution. No other text of any kind.
Layout: one sentence: where the text sits against the subject, on a calm part of the image and
inside the safe area.
Style: what kind of image this is in a few words, plus every one of the user's stated visual
requirements, none omitted.

Leave out: Kelvin numbers, lens and engineering vocabulary, material chemistry, anatomy terms, mood
and quality adjectives ("explosive", "cinematic", "finished"), and sentences about what is NOT in the
image. The negative list handles those."""

DIRECTOR_FINAL_SHORT = """A short plain brief for an image model: 120-200 words, not counting the text strings themselves.
Natural language only — never coordinates, never "Zone 1", never pixel values, never real font names.
Use exactly these labelled lines:

""" + SHORT_PROMPT_LINES

# With TYPESET on, the text arrives as a finished layout image, so the short prompt's Text and
# Layout lines would ask the Director to restate what ty.director_note tells it not to.
_SHORT_TEXT_LINE = SHORT_PROMPT_LINES[SHORT_PROMPT_LINES.index("Text: "):SHORT_PROMPT_LINES.index("Style: ")]
assert _SHORT_TEXT_LINE.startswith("Text: ") and "Layout: " in _SHORT_TEXT_LINE
DIRECTOR_FINAL_SHORT_TYPESET = DIRECTOR_FINAL_SHORT.replace(_SHORT_TEXT_LINE, """Text: only the sentence the TYPOGRAPHY IS ALREADY SET note above gives you, word for word. Quote no
text string, name no typeface and describe no text placement.
Layout: one sentence: where the subject sits so the typeset text blocks stay on calm ground close to
their backdrop colours, inside the safe area.
""")


# ---------------------------------------------------------------------------
# Call plumbing
# ---------------------------------------------------------------------------

def _usage(it) -> dict:
    u = getattr(it, "usage", None)
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0}
    return {"prompt_tokens": getattr(u, "total_input_tokens", 0) or 0,
            "completion_tokens": getattr(u, "total_output_tokens", 0) or 0,
            "thinking_tokens": getattr(u, "total_thought_tokens", 0) or 0}


def _thought_summary(it) -> str:
    """The Interactions API returns the model's own reasoning summary. Keeping
    it makes a bad graphic diagnosable: you can read why a node chose what it
    chose instead of inferring it from the output."""
    out = []
    for s in getattr(it, "steps", []) or []:
        if getattr(s, "type", "") == "thought" and getattr(s, "summary", None):
            for b in s.summary:
                if getattr(b, "type", "") == "text" and getattr(b, "text", ""):
                    out.append(b.text)
    return "\n".join(out)


def _call(client, prompt, label, on_event, thinking="high",
          temp=0.6, as_json=True, model=None, timeout=None):
    """One reasoning call on the Interactions API, with a model-level fallback.

    JSON is requested through response_format rather than the older
    response_mime_type -- the API rejects the mime type on its own with
    'responseFormat must be set when responseMimeType is set'. No schema is
    attached on purpose: each prompt declares its own shape, and pinning a
    schema here would silently drop keys a prompt adds later.
    """
    last = None
    chain = (model, REASONING_MODEL) if model else (REASONING_MODEL, REASONING_FALLBACK)
    for attempt, m in enumerate(chain):
        t0 = time.time()
        try:
            # The SDK timeout below covers the transport; this covers everything
            # else, including an SDK that decides not to honour it. A TimeoutError
            # lands in the same except as any other failure, so the fallback model
            # is tried exactly as it would be for a 500.
            kw = {"model": m, "input": prompt,
                  "generation_config": {"thinking_level": thinking,
                                        "thinking_summaries": "auto",
                                        "temperature": temp}}
            if as_json:
                kw["response_format"] = {"type": "text",
                                         "mime_type": "application/json"}
            # Not a `with` block: leaving one waits for the worker, so a wedged call would
            # still hold the run however early this timeout fired.
            ex = cf.ThreadPoolExecutor(max_workers=1)
            try:
                it = ex.submit(lambda: client.interactions.create(**kw)
                               ).result(timeout=timeout or CALL_TIMEOUT_S)
            finally:
                ex.shutdown(wait=False)
            txt = (it.output_text or "").strip()
            data = txt
            if as_json:
                if txt.startswith("```"):
                    mm = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL)
                    txt = mm.group(1) if mm else txt
                data = json.loads(txt)
                if not isinstance(data, dict):  # every node reads this with .get()
                    raise ValueError(f"expected a JSON object, got {type(data).__name__}")
            u = _usage(it)
            c, est = cost_of(m, u)
            meta = {"node": label, "model": m, "thinking": thinking,
                    "seconds": round(time.time() - t0, 2),
                    "tokens": u["prompt_tokens"] + u["completion_tokens"],
                    "thinking_tokens": u["thinking_tokens"],
                    "cost": c, "estimated": est, "fell_back": attempt > 0,
                    "thought_summary": _thought_summary(it)[:1800]}
            on_event({"event": "node_done", **{k: v for k, v in meta.items()
                                               if k != "thought_summary"}})
            return data, meta
        except Exception as e:
            last = e
            on_event({"event": "node_retry", "node": label, "model": m,
                      "error": f"{type(e).__name__}: {str(e)[:90]}"})
    raise RuntimeError(f"{label} failed on both models: {last}")


def tailor(client, final_prompt, negative_prompt, guard, protect, drawn, emit, metas, writing=False):
    """The TAILOR node: fit the negatives and positives to this prompt with one fast call.
    It may cost seconds, never the graphic: on a timeout or failure the full lists stand.
    Returns the prompt, the negatives, whether it applied, and what it was offered."""
    if not (TAILOR and SHORT_PROMPT and guard["active"]):
        return final_prompt, negative_prompt, False, None
    inputs = antislop.tailor_inputs(negative_prompt, guard["active"], protect, drawn, writing)
    emit({"event": "node_start", "node": "tailor"})
    pool = cf.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_call, client, antislop.tailor_prompt(final_prompt, inputs), "tailor", emit,
                      TAILOR_THINKING, 0.1)
    try:
        data, m = fut.result(timeout=TAILOR_TIMEOUT_S)
        metas.append(m)
    except Exception as e:
        emit({"event": "node_failed", "node": "tailor", "error": f"{type(e).__name__}: {str(e)[:120]}"})
        return final_prompt, negative_prompt, False, inputs
    finally:
        pool.shutdown(wait=False)
    return (*antislop.apply_tailor(final_prompt, negative_prompt, inputs, data), inputs)


def finish_prompt(client, raw_prompt, raw_negative, guard, exclusions, who, deciders, locked,
                  has_logo, cta_requested, emit, metas):
    """Everything between the Director's text and the renderer, for run() and create_more().
    Returns the prompt, the negatives and a record of each conditional decision."""
    parts = antislop.body_parts(who, raw_prompt, *deciders)
    final_prompt, softened, frame_bans = antislop.finish(raw_prompt, guard, who, parts, locked)
    negative, protect, gated = antislop.prepare_negatives(
        raw_negative, guard, frame_bans, exclusions, who, parts, raw_prompt, has_logo, cta_requested)
    drawn = antislop.is_drawn(raw_prompt)
    before = negative
    final_prompt, negative, tailored, offered = tailor(
        client, final_prompt, negative, guard, protect, drawn, emit, metas,
        antislop.shows_writing(raw_prompt, guard["active"]))
    chosen = set(negative.split(", "))
    return final_prompt, negative, {
        "softened": softened, "in_frame": who, "body_parts": parts, "drawn": drawn,
        "tailored": tailored, "frame_bans": frame_bans,
        "text_restored": antislop.missing_text(raw_prompt, locked) if guard["active"] else [],
        "negatives": {
            "gated_out": gated,
            "before_tailor": len(before.split(", ")) if before else 0,
            "final": len(negative.split(", ")) if negative else 0,
            "kept": (offered or {}).get("keep"),
            "tailor_chose": [t for t in (offered or {}).get("free", []) if t in chosen],
            "tailor_left_out": [t for t in (offered or {}).get("free", []) if t not in chosen],
        },
        "positives": {
            "offered": (offered or {}).get("positives"),
            "chosen": [p for p in (offered or {}).get("positives", [])
                       if tailored and p in final_prompt.rsplit("Also:", 1)[-1]],
        },
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render_openai(prompt: str, aspect_ratio: str, out_png: Path,
                   refs: list[Path], quality: str = None,
                   platform: str | None = None) -> dict:
    """gpt-image-2.5-flare. Uses images.edit when references are supplied so the user's
    own subject is preserved rather than reinterpreted."""
    quality = quality or IMAGE_QUALITY
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return {"ok": False, "detail": "OPENAI_API_KEY not set", "price": 0.0}
    size, err = openai_size(aspect_ratio, platform)
    if size is None:
        return {"ok": False, "detail": f"no valid size for {aspect_ratio}", "price": 0.0}
    try:
        from openai import OpenAI
        # RENDER_TIMEOUT_S was defined and never used: the SDK default is ten minutes per try.
        client = OpenAI(api_key=key, timeout=RENDER_TIMEOUT_S, max_retries=1)
        if refs:
            # input_fidelity was carried over from gpt-image-1, which accepts
            # it. gpt-image-2 rejects it outright with a 400, so EVERY
            # reference-image render was falling through to the secondary
            # provider -- the one scenario with a photo never once reached the
            # primary endpoint. Rather than hardcode which model takes which
            # parameter, drop whatever the API names as unsupported and retry:
            # self-healing when the next model changes its surface again.
            kw = {"model": IMAGE_MODEL, "image": None, "prompt": prompt,
                  "size": size, "quality": quality, "input_fidelity": "high",
                  "n": 1}
            r = None
            for _ in range(len(kw)):
                handles = [open(x, "rb") for x in refs]
                kw["image"] = handles
                try:
                    r = client.images.edit(**kw)
                    break
                except Exception as exc:
                    m = re.search(r"does not support the '(\w+)' parameter", str(exc))
                    if not m or m.group(1) not in kw:
                        raise
                    kw.pop(m.group(1))
                finally:
                    for fh in handles:
                        fh.close()
            if r is None:
                return {"ok": False, "detail": "images.edit exhausted retries",
                        "price": 0.0}
        else:
            r = client.images.generate(model=IMAGE_MODEL, prompt=prompt,
                                       size=size, quality=quality, n=1)
        b64 = r.data[0].b64_json
        if not b64:
            return {"ok": False, "detail": "no image payload returned", "price": 0.0}
        out_png.write_bytes(base64.b64decode(b64))
        u = getattr(r, "usage", None)
        return {"ok": True, "detail": IMAGE_MODEL,
                "price": IMAGE_PRICE_BY_QUALITY.get(quality, 0.07),
                "size": size, "aspect_error": round(err, 5), "quality": quality,
                # Real token counts, so the price estimate above can be replaced
                # with a measured rate from a single billing line.
                "output_tokens": getattr(u, "output_tokens", None) if u else None}
    except Exception as e:
        # Logged so a rejected render (e.g. a 400 on a reference image) says why before
        # the fallback renderer takes over; the return value is unchanged.
        ref_info = [(Path(x).name, Path(x).stat().st_size if Path(x).exists() else 0)
                    for x in refs]
        print(f"[v33 render] {IMAGE_MODEL} failed, falling back: {type(e).__name__}: "
              f"{str(e)[:500]} | references (name, bytes): {ref_info}")
        return {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:160]}", "price": 0.0}


def _render_gemini(client, types, prompt: str, aspect_ratio: str, out_png: Path,
                   refs: list[Path]) -> dict:
    """Nano Banana 2. Takes the aspect ratio string directly, so it is the safe
    home for any ratio gpt-image-2.5-flare cannot hit within ASPECT_TOLERANCE."""
    try:
        parts = [types.Part(text=prompt)]
        for pth in refs:
            import io as _io
            from PIL import Image as PILImage
            img = PILImage.open(pth).convert("RGB")
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            parts.insert(0, types.Part.from_bytes(data=buf.getvalue(),
                                                  mime_type="image/png"))
        cfg = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size="2K"))
        r = client.models.generate_content(
            model=IMAGE_FALLBACK_MODEL,
            contents=[types.Content(parts=parts, role="user")], config=cfg)
        for prt in (r.parts or []):
            if prt.inline_data is not None:
                prt.as_image().save(str(out_png))
                return {"ok": True, "detail": IMAGE_FALLBACK_MODEL, "price": 0.06,
                        "size": aspect_ratio, "aspect_error": 0.0, "quality": "2K"}
        return {"ok": False, "detail": "no image part in response", "price": 0.0}
    except Exception as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:160]}", "price": 0.0}


def render(client, types, prompt: str, negative: str, aspect_ratio: str,
           out_png: Path, refs: list[Path], platform: str | None = None) -> dict:
    """Route to whichever endpoint can actually hold this aspect ratio.

    gpt-image-2.5-flare leads whenever it has a size within ASPECT_TOLERANCE of the
    target -- which, now that sizing is computed rather than looked up, is every
    ratio we currently use. If a future platform asks for a shape it cannot hit,
    Nano Banana 2 leads instead, because a graphic delivered in the wrong shape
    is a defect no amount of render quality fixes.

    The negative list is appended as prose because neither endpoint takes a
    separate negative-prompt parameter -- both are instruction-following image
    models rather than diffusion samplers with CFG-negative conditioning.
    """
    full = prompt + ("\n\nDo not include any of the following: " + negative if negative else "")
    size, err = openai_size(aspect_ratio, platform)
    openai_fits = size is not None and err <= ASPECT_TOLERANCE
    routing = (f"{IMAGE_MODEL} leads: {size} is {err * 100:.2f}% off {aspect_ratio}"
               if openai_fits else
               f"Nano Banana 2 leads: {IMAGE_MODEL}'s best for {aspect_ratio} is "
               f"{size} ({err * 100:.2f}% off, over the "
               f"{ASPECT_TOLERANCE * 100:.0f}% tolerance)")

    def run_openai():
        return _render_openai(full, aspect_ratio, out_png, refs,
                              platform=platform)

    def run_gemini():
        return _render_gemini(client, types, full, aspect_ratio, out_png, refs)

    order = [(IMAGE_MODEL, run_openai), (IMAGE_FALLBACK_MODEL, run_gemini)]
    if not openai_fits:
        order.reverse()

    attempts = []
    res = {"ok": False, "detail": None, "price": 0.0}
    for name, fn in order:
        res = fn()
        attempts.append({"model": name, "ok": res["ok"], "detail": res["detail"]})
        if res["ok"]:
            break
    return {"ok": res["ok"],
            "rendered_by": res["detail"] if res["ok"] else None,
            "attempts": attempts,
            "price": res.get("price", 0.0) if res["ok"] else 0.0,
            "routing": routing, "size": res.get("size"),
            "quality": res.get("quality"),
            "aspect_error": res.get("aspect_error"),
            "output_tokens": res.get("output_tokens")}


# ---------------------------------------------------------------------------

def run(raw_prompt: str, goal: str, platform: str,
        reference_images=None,          # [path] | [(path, role)] | [{path, role}]
        generate_image: bool = True, out_dir: Path = Path("armc_v3_runs"),
        on_event=None, cta_requested: bool | None = None, copywriter=None) -> dict:
    """``cta_requested``: whether the user asked for a call to action. Pass it
    from the user's own words (see :func:`user_asked_cta`) when ``raw_prompt``
    is a rewritten brief; ``None`` detects it from ``raw_prompt``.

    ``copywriter``: optional callable ``(client, ground=, locked=, other_facts=,
    cta_requested=, emit=) -> {"locked": {slot: text}, "metas": [...], ...}``.
    It starts as soon as GROUND's text hygiene is done and runs beside SUBJECT
    (neither needs the other); its deck then replaces the locked text and the
    other stated facts for the specialists and the DIRECTOR. If it fails, the
    run degrades to GROUND's own locked text. ``None`` uses the default one, built
    from ``raw_prompt``, unless ``ARMC3_COPYWRITER=0``; ``False`` skips it."""
    from google import genai
    from google.genai import types

    typed = refroles.normalise(reference_images)
    if len(typed) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"at most {MAX_REFERENCE_IMAGES} reference images, got {len(typed)}")
    refs = [t["path"] for t in typed]
    # EVERY supplied reference goes to the renderer, logo included.
    #
    # An earlier version withheld the logo, on the theory that it would be
    # composited afterwards and that handing it to an image model invites a
    # redraw. That was wrong in the worst way: the compiled prompt still said
    # "the supplied logo, reproduced exactly as provided", so both renderers
    # were told about a file they never received and had no option but to
    # invent one. The brief supplied a brown circular mark with BLOOM & BEAN in
    # heavy sans; both models drew a coffee-branch illustration in a serif.
    #
    # A reference the prompt talks about must be a reference the renderer has.
    gen_refs = [t["path"] for t in typed]

    events = []
    def emit(e):
        e["t"] = round(time.time() - T0, 2)
        events.append(e)
        if on_event:
            on_event(e)

    T0 = time.time()
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise MissingCredential(
            "GEMINI_API_KEY is not set - the pipeline cannot reason without it")
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=CALL_TIMEOUT_S * 1000))
    brief = load_guidance(goal, platform)
    authored = brief.get("authored") or {}
    metas = []

    # ---- GROUND ------------------------------------------------------------
    # thinking=low on purpose: this copies strings out of a brief. Giving an
    # extraction task a reasoning budget invites it to embellish facts.
    emit({"event": "node_start", "node": "ground"})
    g_prompt = base.GROUND_PROMPT.format(
        design_goal=goal, platform=platform, aspect_ratio=brief["aspect_ratio"],
        must="\n".join(f"  - {f['label']} ({f['field_id']})" for f in brief["must_communicate"]),
        optional="\n".join(f"  - {f['label']}" for f in brief["may_include"]),
        raw_prompt=raw_prompt.strip(),
        has_photo="yes" if any(t["role"] in ("hero","product") for t in typed) else "no")
    ground, m = _call(client, g_prompt, "ground", emit,
                      thinking=GROUND_THINKING, temp=0.1,
                      model=GROUND_MODEL or None, timeout=GROUND_TIMEOUT_S)
    metas.append(m)
    if ground.get("blocking_question"):
        return {"status": "needs_input", "question": ground["blocking_question"],
                "events": events, "metas": metas}

    font_requests = [r for r in (ground.get("font_requests") or []) if str(r.get("requested", "")).strip()]
    font_refs = [t for t in typed if t["role"] == "font"]
    locked_raw = ground.get("locked_text") or {}
    locked, dropped_locked = dedupe_locked(locked_raw)
    if dropped_locked:
        emit({"event": "locked_deduped", "dropped": dropped_locked})
    locked_s = "\n".join(f'  {k}: "{v}"' for k, v in locked.items()) or "  (none)"
    directives = ground.get("visual_directives") or []
    dir_s = "; ".join(directives) if directives else "none — the art direction is yours"

    # Facts the client stated that no schema field could hold, plus how an offer
    # is actually redeemed. Both are extracted by GROUND and were previously
    # dropped on the floor between GROUND and the DIRECTOR.
    other_facts = [str(x).strip() for x in (ground.get("other_stated_facts") or [])
                   if str(x).strip()]
    other_facts, dropped_facts = dedupe_facts(other_facts, locked)
    if dropped_facts:
        emit({"event": "facts_deduped", "dropped": dropped_facts})
        dropped_locked = dropped_locked + dropped_facts
    if cta_requested is None:
        cta_requested = user_asked_cta(raw_prompt)
    if not cta_requested:
        locked, other_facts, dropped_cta = strip_cta(locked, other_facts)
        if dropped_cta:
            emit({"event": "cta_removed", "dropped": dropped_cta})
            dropped_locked = dropped_locked + dropped_cta
        locked_s = "\n".join(f'  {k}: "{v}"' for k, v in locked.items()) or "  (none)"
    red = ground.get("redemption") or {}
    if red.get("token") and red.get("how"):
        other_facts.append(f'the code {red["token"]} is used {red["how"]}')
    other_s = ("\n".join(f"  - {x}" for x in other_facts) if other_facts
               else "  (none — every stated fact has a field above)")
    ex = ground.get("user_exclusions") or {}
    ex_terms = (ex.get("explicit_absences") or []) + (ex.get("style_contradictions") or [])
    ex_s = ", ".join(ex_terms) if ex_terms else "none"
    w, h = (brief.get("canvas_px") or [1080, 1080])[:2]

    # Blank surfaces exist to stop the pipeline INVENTING a wordmark, not to stop
    # the client's real mark appearing on their own product. Once a logo file is
    # supplied that reasoning inverts: dressing the scene with it is the correct
    # answer and refusing to is the defect.
    has_logo = any(t["role"] == "logo" for t in typed)
    branding_note = (
        "BRANDED SURFACES ARE ALLOWED HERE, because the client supplied their logo. Packaging, "
        "bags, cups, garments, signage and labels in the scene MAY carry that supplied mark, "
        "reproduced from the attached file exactly as given — same shapes, same proportions, same "
        "colours, same lettering — at whatever scale and angle the surface calls for, and reading "
        "as printed ON the material rather than pasted over it.\n"
        ""
        "THE SUPPLIED MARK IS THE ONLY TEXT ALLOWED ON THOSE SURFACES. No product names, no "
        "origins or varietals, no weights, no specifications, no tasting notes, no descriptors, "
        "no sub-brands, no slogans, no invented label copy of any kind — a rendered label reading "
        "'ETHIOPIA YIRGACHEFFE, 1,900-2,200 MASL' is a factual claim about a product that does "
        "not exist.\n"
        "SO: the supplied mark DOES appear on the primary package or surface in shot — that is "
        "the point of having it — and NOTHING ELSE does. Do not resolve the tension by leaving "
        "every surface bare: a blank bag next to a client who gave you their logo is a wasted "
        "asset. Mark on, copy off. Any secondary panel that would carry descriptive text is "
        "blank or turned away. No real third-party trademark appears anywhere."
        if has_logo else
        "NO BRANDED SURFACES. Garments, packaging, product labels, equipment housings, signage and "
        "screens must be BLANK or turned away from camera. Specify plain unbranded surfaces "
        "explicitly — an unmarked apron, a plain bottle, a label rotated out of view. No brand name "
        "is printed on any object: not the client's own name, not a real third-party brand. The "
        "only text in the image is the locked strings set as typography.")

    photo_note = (
        f"{len(typed)} reference image(s) attached: {refroles.summary(typed)}. "
        "Each has a declared ROLE and the rules for it are stated below; follow "
        "them exactly. Never describe the contents of an attached image."
        if typed else "No reference imagery supplied — all imagery is generated.")

    # Named-person guard, computed BEFORE the subject node because the subject
    # node is the first place a scene gets invented. A supplied hero or product
    # image means we are not inventing a likeness, so the guard lifts; a logo,
    # layout or style reference tells us nothing about the person and does not.
    people = [p.get("name") for p in (ground.get("named_people") or []) if p.get("name")]
    protect_people = bool(people) and not any(
        t["role"] in ("hero", "product") for t in typed)
    people_note = ""
    if protect_people:
        people_note = (f"NAMED PERSON WITHOUT A PHOTO ({', '.join(people)}): do NOT render a "
                       "photorealistic face or an invented likeness. Represent them through "
                       "their work and environment, or typographically — name set with real "
                       "presence, initials, or a monogram.")

    # ---- SLOP GUARD (code only) --------------------------------------------
    # Read off what the user and their chosen style asked for, so a rule steps aside when
    # the brief wants that very thing ("neon rim light", "confetti"). Built before SUBJECT,
    # which is the first place a scene gets invented.
    guard_text = f"{raw_prompt}\n{dir_s}"
    guard = antislop.build(guard_text, enabled=ANTISLOP, short_prompt=SHORT_PROMPT)
    short_prompt = SHORT_PROMPT and bool(guard["active"])
    emit({"event": "antislop", "active": guard["active"], "exempted": guard["exempted"]})

    # ---- COPYWRITER, beside SUBJECT ----------------------------------------
    copy_pool = copy_fut = None
    if copywriter is None and COPYWRITER:
        import copywriter as _copy_module  # imported here: it imports this module
        copywriter = _copy_module.make_hook(goal=goal, platform=platform, raw_prompt=raw_prompt,
                                            clarifications=[])
    if copywriter:
        emit({"event": "node_start", "node": "copywriter"})
        copy_pool = cf.ThreadPoolExecutor(max_workers=1)
        copy_fut = copy_pool.submit(copywriter, client, ground=ground, locked=dict(locked),
                                    other_facts=list(other_facts), cta_requested=bool(cta_requested),
                                    emit=emit)

    # ---- SUBJECT -----------------------------------------------------------
    emit({"event": "node_start", "node": "subject"})
    facts = {k: v for k, v in (ground.get("facts") or {}).items()} if isinstance(
        ground.get("facts"), dict) else ground.get("facts")
    s_prompt = SUBJECT.format(
        goal=goal, platform=platform, aspect_ratio=brief["aspect_ratio"],
        format_character=authored.get("format_character", f"A {goal} graphic."),
        raw_brief=raw_prompt.strip(),
        facts=json.dumps(facts, ensure_ascii=False)[:600] if facts else "(none extracted)",
        directives=dir_s, exclusions=ex_s,
        photo_state=refroles.summary(typed) if typed else "none — hero will be generated",
        reference_rules=refroles.block(typed, refroles.SUBJECT_RULE),
        photo_rule=PHOTO_RULE_WITH if any(t["role"] in ("hero", "product") for t in typed)
                   else PHOTO_RULE_WITHOUT,
        people_rule=(PEOPLE_RULE_NAMED.format(names=", ".join(people))
                     if protect_people else PEOPLE_RULE_NONE),
        slop_subject=guard["subject"], people_field=guard["people_field"])
    subject, m = _call(client, s_prompt, "subject", emit,
                       thinking=SUBJECT_THINKING, temp=0.7)
    metas.append(m)

    copy_result = None
    if copy_fut:
        emit({"event": "copywriter_wait_start"})
        try:
            copy_result = copy_fut.result(timeout=COPY_TIMEOUT_S)
            metas.extend(copy_result.get("metas") or [])
            if copy_result.get("locked"):
                # The deck carries every stated fact it keeps, so nothing is left over.
                locked = dict(copy_result["locked"])
                locked_s = "\n".join(f'  {k}: "{v}"' for k, v in locked.items())
                other_s = "  (none — the copy deck above carries every fact that goes on the graphic)"
        except Exception as e:
            emit({"event": "node_failed", "node": "copywriter", "error": f"{type(e).__name__}: {str(e)[:120]}"})
        finally:
            copy_pool.shutdown(wait=False)
        emit({"event": "copywriter_wait_done"})

    subject_block = json.dumps(subject, indent=2, ensure_ascii=False)
    subject_line = subject.get("subject", raw_prompt.strip()[:200])
    cliches = ", ".join(subject.get("cliches_to_avoid") or [])
    # Clichés are whole scene descriptions. They already steered SUBJECT and the ART DIRECTOR;
    # as negatives they would only spell the unwanted picture out to the renderer.
    if short_prompt:
        cliches = ""

    # ---- THREE SPECIALISTS, IN PARALLEL ------------------------------------
    typo_p = TYPOGRAPHER.format(
        goal=goal, platform=platform, aspect_ratio=brief["aspect_ratio"],
        platform_craft=authored.get("platform_craft") or safe_area(platform),
        subject=subject_line, locked=locked_s, directives=dir_s, other_facts=other_s,
        raw_brief=raw_prompt.strip(), slop_typo=guard["typo"],
        cta_rule=CTA_TYPO_ON if cta_requested else CTA_TYPO_OFF,
        font_library=ty.library_block(),
        font_requests=("  - The client named these typefaces: " + json.dumps(font_requests)
                       + ". Supported ones are REQUIRED for the role they named; for any not in the "
                       "library pick the closest supported style." if font_requests else
                       "  - The client named no typeface."))
    art_p = ART_DIRECTOR.format(
        goal=goal, platform=platform, aspect_ratio=brief["aspect_ratio"],
        format_character=authored.get("format_character", f"A {goal} graphic."),
        subject_block=subject_block, directives=dir_s, exclusions=ex_s,
        photo_note=photo_note,
        reference_rules=refroles.block(typed, refroles.ART_RULE),
        branding_note=branding_note, slop_art=guard["art"], people_field=guard["people_field"],
        light_craft=guard["light_craft"] or ART_LIGHT,
        material_craft=guard["material_craft"] or ART_MATERIAL,
        light_schema=guard["light_schema"] or ART_LIGHT_SCHEMA,
        provocations="\n".join(f"  - {p}" for p in (authored.get("art_direction_prompts") or [])[:6])
                     or "  (none supplied — invent your own)")
    comp_p = COMPOSITOR.format(
        goal=goal, platform=platform, aspect_ratio=brief["aspect_ratio"], w=w, h=h,
        safe=safe_area(platform), subject=subject_line,
        platform_crop=subject.get("platform_crop", "(not specified)"),
        reference_rules=refroles.block(typed, refroles.COMP_RULE),
        elements="\n".join(f"  - {k}: \"{v}\"" for k, v in locked.items()) or "  (text minimal)",
        menu="\n".join(f"  - {c['idea']} ({c['note']})" for c in brief["composition_menu"]),
        slop_comp=guard["comp"])

    emit({"event": "parallel_start", "nodes": ["typographer", "art_director", "compositor"]})
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_call, client, typo_p, "typographer", emit, TYPO_THINKING, 0.6): "typo",
            pool.submit(_call, client, art_p, "art_director", emit, ART_THINKING, 0.8): "art",
            pool.submit(_call, client, comp_p, "compositor", emit, COMP_THINKING, 0.7): "comp",
        }
        got = {}
        for f in cf.as_completed(futs):
            name = futs[f]
            try:
                data, m = f.result()
                got[name] = data
                metas.append(m)
            except Exception as e:
                # A dead specialist degrades that discipline, not the request.
                got[name] = {"_failed": f"{type(e).__name__}: {str(e)[:120]}"}
                emit({"event": "node_failed", "node": name, "error": got[name]["_failed"]})
    emit({"event": "parallel_done"})

    # ---- TYPESET -----------------------------------------------------------
    # Sets the exact copy in real fonts BEFORE the director writes the prompt, so
    # the director composes the artwork around finished typography and the
    # editable-text metadata exists before a single pixel is generated.
    typeset_ts, layout_png, user_font = None, None, None
    canvas_size, _ = openai_size(brief["aspect_ratio"], platform)
    if ty.ENABLED and canvas_size and (locked or other_facts):
        emit({"event": "node_start", "node": "typeset"})
        cw, ch = (int(v) for v in canvas_size.split("x"))
        if font_refs:
            try:
                import typography_ocr as tocr
                user_font = (tocr.match_user_font_image(font_refs[0]["path"]) if tocr.available()
                             else {"nearest": None, "confidence": None, "approximate": True,
                                   "reason": "typography_ocr unavailable (pip install easyocr)"})
            except Exception as e:
                user_font = {"nearest": None, "confidence": None, "approximate": True, "reason": str(e)[:120]}
        try:
            plan, m = _call(client, ty.typeset_prompt(
                goal=goal, platform=platform, W=cw, H=ch, locked=locked_s, other_facts=other_s,
                typo=got.get("typo"), comp=got.get("comp"), art=got.get("art"),
                font_requests=font_requests, user_font=user_font),
                "typeset", emit, thinking="medium", temp=0.4)
            metas.append(m)
            typeset_ts = ty.enforce(plan, platform=platform, W=cw, H=ch,
                                    font_requests=font_requests, user_font=user_font)
            typeset_ts["user_font"] = user_font
            if not typeset_ts["elements"]:
                typeset_ts = None
        except Exception as e:
            # A failed typesetter degrades to the pre-typography behaviour, never to a failed request.
            emit({"event": "node_failed", "node": "typeset", "error": f"{type(e).__name__}: {str(e)[:120]}"})
            typeset_ts = None
        if typeset_ts:
            emit({"event": "typeset_done", "elements": len(typeset_ts["elements"]),
                  "fonts": [ty.FONTS[k]["family"] for k in typeset_ts["fonts_used"]],
                  "notices": typeset_ts["user_notices"]})

    # ---- DIRECTOR ----------------------------------------------------------
    d_prompt = DIRECTOR.format(
        goal=goal, platform=platform, aspect_ratio=brief["aspect_ratio"], w=w, h=h,
        subject_block=subject_block, locked=locked_s,
        missing=", ".join(ground.get("missing_mandatory") or []) or "nothing — brief was complete",
        directives=dir_s, exclusions=ex_s, people_note=people_note, photo_note=photo_note,
        other_facts=other_s, branding_note=branding_note,
        branding_negatives=(
            "ALSO BAN: invented product names, invented origins or varietals, invented "
            "specifications, tasting notes or descriptors on any label or package"
            if has_logo else
            "ALSO BAN: brand names printed on clothing, branded packaging, product labels with "
            "text, manufacturer names on equipment"),
        reference_rules=refroles.block(typed, refroles.DIRECTOR_RULE),
        cta_rule=CTA_DIRECTOR_ON if cta_requested else CTA_DIRECTOR_OFF,
        cta_negatives="" if cta_requested else "ALSO BAN: " + CTA_NEGATIVES,
        typography_note=ty.director_note(typeset_ts),
        typo=json.dumps(got.get("typo", {}), indent=2, ensure_ascii=False),
        art=json.dumps(got.get("art", {}), indent=2, ensure_ascii=False),
        comp=json.dumps(got.get("comp", {}), indent=2, ensure_ascii=False),
        defect_tags=", ".join(brief["defect_tags"]),
        cliches=cliches or "(none identified)",
        slop_director=guard["director"],
        compile_step=DIRECTOR_COMPILE_SHORT if short_prompt else DIRECTOR_COMPILE,
        final_spec=(DIRECTOR_FINAL_SHORT_TYPESET if typeset_ts else DIRECTOR_FINAL_SHORT)
                   if short_prompt else DIRECTOR_FINAL.format(
            slop_final=("  - the SLOP GUARD above, each point as a positive instruction\n"
                        if guard["director"] else "")))

    emit({"event": "node_start", "node": "director"})
    def tag(n, d=""):
        mm = re.search(rf"<{n}>(.*?)</{n}>", text, re.DOTALL)
        return mm.group(1).strip() if mm else d

    # With no prompt the renderer would draw from the negative list alone. One more try, then
    # a plain failure the caller already handles.
    for attempt in range(2):
        text, m = _call(client, d_prompt, "director", emit,
                        thinking=DIRECTOR_THINKING, temp=0.85, as_json=False)
        metas.append(m)
        if tag("final_prompt"):
            break
        emit({"event": "node_retry", "node": "director", "error": "no <final_prompt> in the reply"})
    else:
        raise RuntimeError("director returned no final prompt on two attempts")

    # Hype words ("gleaming", "flawless", "hyper-realistic") pull the render toward the
    # over-polished look; the guard tones them down in code.
    # A supplied hero or product image decides who is in the frame, not the specialists.
    who = (None if any(t["role"] in ("hero", "product") for t in typed)
           else antislop.in_frame(got.get("art"), subject))
    # With a typeset layout the text reaches the renderer as image 1 and the Director was told not
    # to restate it, so the guard must not restore the "lost" strings into the prompt: they would
    # be drawn a second time.
    final_prompt, negative_prompt, conditional = finish_prompt(
        client, tag("final_prompt"), tag("negative_prompt", ", ".join(brief["defect_tags"])), guard,
        ex_terms, who, (got.get("art"), subject), None if typeset_ts else locked, has_logo,
        bool(cta_requested), emit, metas)

    result = {
        "status": "ok", "goal": goal, "platform": platform,
        "aspect_ratio": brief["aspect_ratio"],
        "reference_images": [{"path": str(t["path"]), "role": t["role"]} for t in typed],
        "locked_text_dropped": dropped_locked,
        "cta_requested": bool(cta_requested),
        "copy": ({k: v for k, v in copy_result.items() if k != "metas"} if copy_result else None),
        "ground": ground, "subject": subject,
        "typographer": got.get("typo"), "art_director": got.get("art"),
        "compositor": got.get("comp"),
        "direction": tag("direction"),
        "final_prompt": final_prompt,
        # The guard's terms are added in code, so they reach the renderer even when the
        # Director's own list leaves them out.
        "negative_prompt": negative_prompt,
        "antislop": {"brief_text": guard_text, "active": guard["active"],
                     "exempted": guard["exempted"], "negatives": guard["negatives"],
                     "short_prompt": short_prompt, **conditional},
        "typography": ({"user_notices": typeset_ts["user_notices"], "policy_log": typeset_ts["policy_log"],
                        "fonts_used": [ty.FONTS[k]["family"] for k in typeset_ts["fonts_used"]],
                        "user_font_image": user_font} if typeset_ts else None),
        "metas": metas,
        "llm_calls": len(metas),
        "cost_llm": round(sum(x["cost"] for x in metas), 6),
        "thinking_tokens": sum(x.get("thinking_tokens", 0) for x in metas),
    }
    if copy_result and copy_result.get("locked"):
        # create_more() reads the text from here; without it, it falls back to GROUND's.
        result["locked_text"] = locked

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{base.slug(goal)}__{base.slug(platform)}_{int(time.time())}"

    if generate_image:
        emit({"event": "node_start", "node": "render"})
        png = out_dir / f"{stem}.png"
        t0 = time.time()
        render_prompt, render_refs = result["final_prompt"], gen_refs
        if typeset_ts:
            try:
                render_prompt, render_refs, layout_png = typeset_render_inputs(
                    typeset_ts, result["final_prompt"], typed, out_dir / f"{stem}.layout.png")
            except Exception as e:
                # No layout image after all: the prompt has no strings in it, so put them back.
                emit({"event": "node_failed", "node": "typeset",
                      "error": f"layout image: {type(e).__name__}: {str(e)[:120]}"})
                typeset_ts = None
                render_prompt = result["final_prompt"] = restore_text(result["final_prompt"], locked)
        r = render(client, types, render_prompt, result["negative_prompt"],
                   brief["aspect_ratio"], png, render_refs, platform=platform)
        if r["ok"]:
            result["image"] = str(png)
        result["render"] = r
        result["cost_image"] = round(r["price"], 6)
        emit({"event": "node_done", "node": "render", "model": r.get("rendered_by"),
              "seconds": round(time.time() - t0, 2), "cost": round(r["price"], 6),
              "size": r.get("size"), "quality": r.get("quality"),
              "routing": r.get("routing"),
              "fell_back": bool(r["ok"]) and not r["attempts"][0]["ok"]})
        if typeset_ts and r["ok"]:
            result["text_layers"] = run_text_qa(png, typeset_ts, bool(font_refs), emit)
            (out_dir / f"{stem}.text_layers.json").write_text(ty.dumps(result["text_layers"]),
                                                              encoding="utf-8")
            result["layout_image"] = str(layout_png)
    elif typeset_ts:
        # no render: the pre-generation metadata is still the deliverable
        result["text_layers"] = ty.text_layers(typeset_ts, "")

    result["cost_total"] = round(result["cost_llm"] + result.get("cost_image", 0), 6)
    result["wall_s"] = round(time.time() - T0, 2)
    result["events"] = events
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return result



# ---------------------------------------------------------------------------
# Typography plumbing shared by run() and create_more()
# ---------------------------------------------------------------------------

def typeset_render_inputs(ts: dict, final_prompt: str, typed: list, layout_png: Path):
    """Layout image first, then font references, then the client's own references.
    A client 'font' reference is the ONLY font reference when present: no Kognie
    sample is attached alongside it."""
    W, H = ts["canvas"]["width"], ts["canvas"]["height"]
    user_font_mode = any(t["role"] == "font" for t in typed)
    ty.render_layout(ts["elements"], W, H, ts["background_hex"], layout_png, glyphs=not user_font_mode)
    if user_font_mode:
        font_files = [t["path"] for t in typed if t["role"] == "font"]
        names = ["the client's own lettering sample"] * len(font_files)
    else:
        font_files = [ty.sample_path(k) for k in ts["fonts_used"]]
        names = [ty.FONTS[k]["family"] for k in ts["fonts_used"]]
    client = [t["path"] for t in typed if t["role"] != "font"]
    header = ty.render_header(ts, names, len(client), user_font_mode)
    return header + final_prompt, [layout_png] + font_files + client, layout_png


def restore_text(final_prompt: str, locked: dict) -> str:
    """Add back every locked string a prompt written for a typeset layout left out,
    for when that layout never reaches the renderer."""
    lost = antislop.missing_text(final_prompt, locked)
    if not lost:
        return final_prompt
    return final_prompt + "\n\nThe text in the graphic is exactly: " + "; ".join(f'"{x}"' for x in lost) + "."


def run_text_qa(png: Path, ts: dict, user_font_mode: bool, emit) -> dict:
    """Confirm or correct every text element against the rendered pixels. A QA failure
    never costs the render: the planned metadata comes back, marked unverified."""
    try:
        return _run_text_qa(png, ts, user_font_mode, emit)
    except Exception as e:
        emit({"event": "node_failed", "node": "text_qa", "error": f"{type(e).__name__}: {str(e)[:120]}"})
        layers = ty.text_layers(ts, png.name)
        for el in layers.get("elements") or []:
            el.setdefault("qa", {})["status"] = "unverified"
        return layers


def _run_text_qa(png: Path, ts: dict, user_font_mode: bool, emit) -> dict:
    t0 = time.time()
    try:
        import typography_ocr as tocr
        ocr = tocr if tocr.available() else None
    except Exception:
        ocr = None
    qa = ty.text_qa(png, ts, user_font_mode, ocr=ocr)
    layers = ty.text_layers(qa, png.name)
    statuses = [e["qa"]["status"] for e in layers["elements"]]
    emit({"event": "text_qa_done", "seconds": round(time.time() - t0, 2),
          "statuses": statuses,
          "contrast_fail": [e["id"] for e in layers["elements"]
                            if (e["contrast"]["after_render"] or {}).get("pass") is False],
          "legibility_fail": [e["id"] for e in layers["elements"]
                              if (e["qa"].get("legibility_after") or {}).get("pass") is False]})
    return layers

def _live(e):
    if e["event"] == "node_done":
        fb = "  (fallback)" if e.get("fell_back") else ""
        c = f"  ${e['cost']:.5f}" if e.get("cost") is not None else ""
        th = f"  think={e['thinking_tokens']}" if e.get("thinking_tokens") else ""
        print(f"    [{e['t']:>6.2f}s] {e['node']:<13} {e.get('seconds','?')}s{c}{th}{fb}")
    elif e["event"] == "parallel_start":
        print(f"    [{e['t']:>6.2f}s] --- parallel: {', '.join(e['nodes'])} ---")
    elif e["event"] in ("node_retry", "node_failed"):
        print(f"    [{e['t']:>6.2f}s] {e['event'].upper()} {e['node']}: {e.get('error','')[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--ref", action="append", default=[],
                    help="reference as PATH:ROLE where ROLE is "
                         "logo|hero|product|layout|style (repeatable, max "
                         f"{MAX_REFERENCE_IMAGES}). Bare PATH defaults to hero.")
    ap.add_argument("--out", default="armc_v3_runs")
    ap.add_argument("--no-image", action="store_true")
    a = ap.parse_args()
    parsed = []
    for spec in a.ref:
        if ":" in spec and spec.rsplit(":", 1)[1].lower() in refroles.ROLES:
            path, role = spec.rsplit(":", 1)
            parsed.append({"path": path, "role": role.lower()})
        else:
            parsed.append({"path": spec, "role": "hero"})
    r = run(a.prompt, a.goal, a.platform, parsed,
            not a.no_image, Path(a.out), on_event=_live)
    if r["status"] == "needs_input":
        print("NEEDS INPUT:", r["question"]); return
    print(f"\n  calls={r['llm_calls']}  wall={r['wall_s']}s  "
          f"cost=${r['cost_total']:.4f}  thinking_tokens={r['thinking_tokens']}")
    print(f"  subject:  {(r.get('subject') or {}).get('subject','')[:150]}")
    print(f"  rendered: {(r.get('render') or {}).get('rendered_by')}")
    print(f"  image:    {r.get('image','(none)')}")


if __name__ == "__main__":
    main()
