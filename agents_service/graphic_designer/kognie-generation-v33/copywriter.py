"""COPYWRITER node for the v3.2 generation engine.

Sits after GROUND + text hygiene, in parallel with SUBJECT (neither needs the
other), and before the specialists. GROUND copies the user's words into schema
fields; nobody writes copy, so a thin brief gives a thin headline and a reworded
repeat ("25 YEARS" over "25 years of boats") slips past the lexical dedupe.

The engine runs it through `run(..., copywriter=make_hook(...))`; see README.md
at the package root for the wiring into the app.

Decisions (27 Oct 2026):
  * Mandatory fields are still clarified with the user as today; the answers
    are material for the copy, not text to paste.
  * No options are shown to the user and none are written out: one final deck,
    as fast as possible.
  * The copywriter writes from the MEANING of what the user typed (subject,
    intent, offer); only facts (names, numbers, dates, codes, URLs) stay exact.
  * Voice and spelling follow the brand profile when one exists, otherwise the
    user's own spelling and location.

Code checks the deck (repetition between slots, invented numbers, uncovered
fields, unrequested CTA, generic headline, silently dropped facts, word budget)
and gives the model one repair pass if a check fails. `deck_to_locked()` turns
the deck into the `locked` dict the Typographer / Compositor / Director read.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# A flat sibling of generation_pipeline.py, imported the way v32_bridge imports the engine.
_V32_DIR = str(Path(__file__).resolve().parent)
if _V32_DIR not in sys.path:
    sys.path.insert(0, _V32_DIR)
import generation_pipeline as v32  # noqa: E402

COPY_THINKING = os.environ.get("KOGNIE_COPY_THINKING", "low")
COPY_TEMP = float(os.environ.get("KOGNIE_COPY_TEMP", "0.9"))
# One repair at most: speed over polish. The prompt carries the checks as a self-check list,
# so most decks pass first time; a repair costs ~8s.
MAX_REPAIRS = int(os.environ.get("KOGNIE_COPY_MAX_REPAIRS", "1"))

# Total words on the graphic, by platform. Tunable; a starting point from how
# each format is viewed (a thumbnail is read at ~210px wide, signage from metres
# away, a LinkedIn post up close).
WORD_BUDGET = {
    "YouTube Thumbnail": 8,
    "Digital Signage Portrait": 16,
    "Instagram Story": 22,
    "Instagram Square Post": 24,
    "Instagram Portrait Post": 26,
    "Pinterest Pin": 24,
    "X (Twitter) In-Stream Post": 24,
    "Facebook Feed Landscape": 26,
    "LinkedIn Feed Post": 32,
    "Presentation Slide": 32,
}

# Goals whose mandatory facts are rules or logistics (entry steps, eligibility, schedule,
# apply link) get extra words: that copy is small print, not headline weight.
GOAL_EXTRA_WORDS = {
    "Giveaway / Contest": 10, "We're Hiring / Job Opening": 6, "Webinar Invitation": 6,
    "Workshop / Masterclass Promo": 6, "Event Announcement": 6, "Customer Testimonial": 8,
}

# Craft notes per design goal: what the copy for this goal must do, in a sentence or three.
GOAL_NOTES = {
    "App Download Promo": "Lead with what the app does for the reader in their words, not its feature "
        "list. App name exact. Store badges are artwork, not copy: do not write 'Download on the App Store'.",
    "Brand Awareness / Quote Post": "If the user gave a quote, it is the graphic: set it VERBATIM (you may "
        "trim with an ellipsis, never reword) and attribute it exactly. If they gave a message, not a "
        "quote, write one line worth screenshotting; nothing else competes with it.",
    "Company Milestone / Anniversary": "The milestone number is the hook but appears once. Say what it "
        "means (people, work, place) rather than congratulating yourself; thanks only if the user asked.",
    "Customer Testimonial": "The customer's words are evidence: set the quote VERBATIM. You may excerpt "
        "the strongest contiguous passage and mark cuts with an ellipsis, but never reword, merge, tidy "
        "or add a word inside quotation marks. The quote IS the headline slot (slot name 'quote'). "
        "Name, title and company exact. Ratings or badges stay small.",
    "Discount Coupon / Promo Code": "The saving and the code are the two heroes; the code appears once, "
        "exactly as given, with how and where to use it and any expiry.",
    "Event Announcement": "Headline sells the experience, details answer what / when / where in one "
        "scannable line. Dates and times exact.",
    "Festive / Seasonal Sale": "Tie the offer to the occasion without festive clichés; the discount and "
        "dates are exact and appear once.",
    "Flash Sale": "The discount is the hero and the real deadline is the urgency; never invent scarcity "
        "('only 5 left') or a countdown the user did not give.",
    "Giveaway / Contest": "Headline makes the prize desirable. Entry steps are contest RULES: keep every "
        "step the user gave, short and imperative, in their order. If a step refers to a format this "
        "graphic is not (e.g. 'like this post' on a Story), rewrite it so it can be done from where the "
        "graphic is seen, keeping the same action, and record the change in adaptations. Deadline, "
        "eligibility and winner announcement are small print but must be exact.",
    "Grand Opening": "The news (first, second, new location) belongs in the headline; place and date "
        "exact; an opening-day perk is the second hook.",
    "New Collection Drop": "Name the collection exactly; one line that captures its mood or material; "
        "drop date exact.",
    "Product Launch": "Product name exact; the headline is the single most compelling benefit, stated "
        "concretely; features only if they earn their space.",
    "Seasonal Greeting": "Warm and specific to the brand or place; one greeting line, no sales copy "
        "unless the user asked for an offer.",
    "Team Introduction / New Hire": "The person's name and role exact; one human, specific line about "
        "them from what the user told us. Never invent hobbies, quotes or background.",
    "We're Hiring / Job Opening": "Job title exact (it is what candidates search for). The headline is "
        "why someone good would want this job, drawn from what the user told us about the team or "
        "work; never invent perks. Salary, location, work pattern, apply link and deadline exact. "
        "Leave out private remarks about past hires or internal problems.",
    "Webinar Invitation": "Title promises an outcome for this audience; speaker name and role exact; "
        "date, time and time zone exact; length only if it helps.",
    "Workshop / Masterclass Promo": "Headline says what people will be able to do or make; host or "
        "teacher name exact; date, time, place, price and places available exact.",
}

COPYWRITER = """You are a senior brand copywriter. You write every word on ONE graphic. Designers set
your copy exactly as you write it, so it must be final and display-ready.

DESIGN GOAL: {goal}
PLATFORM: {platform} ({aspect_ratio}). WORD BUDGET for the whole graphic: about {budget} words.
WHAT THIS FORMAT IS FOR: {format_character}
PLATFORM CONSTRAINTS: {platform_craft}

FOR THIS GOAL: {goal_note}

BRAND: {brand}
SPELLING: {spelling}

MANDATORY FIELDS (each must be carried by your copy, or reported missing if it is a fact the user
never gave):
{must}
OPTIONAL FIELDS (only if the user gave the content and it earns its space):
{optional}

WHAT THE USER TOLD US
Brief: \"\"\"{raw_prompt}\"\"\"
Answers to our clarifying questions:
{clarifications}
Facts extracted from the above (by field): {locked}
Other stated facts: {other_facts}
Redemption mechanic: {redemption}
Fields the user did not provide: {missing}
Call to action: {cta_rule}

RULES
0. READ FOR INTENT. Separate what goes ON the graphic from what is said ABOUT it: tone requests
   ("not corporate", "don't overdo it"), private asides, jokes, apologies and corrections are
   instructions, not copy. When the user corrects themselves, the correction wins everywhere.
1. Write from MEANING, not phrasing. Read the brief and the answers for the subject, the offer and
   the intent, then write the copy yourself. A headline the user suggested in an answer, or
   "something like ...", is a hint, NEVER text to reuse. Keep their facts, find a better line:
   specific, rhythmic, memorable, right for this reader on this platform. Exception: a QUOTE from
   a named person is set verbatim (trimmed with an ellipsis if needed), never reworded.
1b. THE NEWS HOOK. Name the one fact that makes this graphic worth posting (a second location, a
   25th year, a first-ever, a launch, a record). It MUST be on the graphic, ideally inside the
   headline where it costs no extra words ("Our second oven is ready"). When you trim to fit the
   budget, the news hook is the last thing to go: cut secondary details first.
2. FACTS ARE EXACT. Names, numbers, percentages, prices, dates, times, places, codes, URLs and
   handles are copied exactly as given (you may reformat for display: "Thursday 13 November, 6pm"
   -> "Thu 13 Nov · 6pm"). Never invent one: no dates, prices, discounts, statistics, awards,
   quotes or claims the user did not state. Ground every line in what the user told us.
3. VOICE AND SPELLING. Follow the brand's voice when a brand profile is given, otherwise the user's
   own register. Use the SPELLING variant stated above in every word, including words the user did
   not write (colour/color, jewellery/jewelry, centre/center, favourite/favorite, -ise/-ize).
4. SLOTS. Use only what this graphic needs, from: eyebrow, headline, quote, subheadline, body,
   details, offer, code_line, attribution, signoff, cta. Fewer, stronger lines. Stay within the
   budget.
5. HEADLINE. Weigh several genuinely different angles in your head (the offer with punch, the
   moment or feeling, a turn of phrase from this product, place or people) and keep only the best.
   The goal's category label alone ("Flash Sale", "Grand Opening", "Webinar") is not a headline.
6. ONE FACT, ONE PLACE. No fact, number, name, product, date or key word appears in two slots,
   even when two mandatory fields ask for it: carry overlapping fields in ONE slot and map both to
   it in coverage. A line that restates another line must add something new or not exist.
7. Avoid AI copy clichés: unlock, elevate, supercharge, seamless, effortless, timeless, curated,
   "Not just X. It's Y.", three one-word beats, stacked exclamation marks, vague claims.
8. Every fact the user stated is on the graphic or in facts_left_out with a short reason. Other
   stated facts are numbered F1, F2...: list the ones a slot carries in its "carries" (even when you
   rephrased them into a headline).

SELF-CHECK BEFORE YOU ANSWER (the deck is checked by code; a failure costs a rewrite):
- headline is not the user's suggested wording; the news hook is on the graphic;
- no number, name or key word appears in two slots; no number the user did not give;
- every quote is word-for-word from the user; every stated fact is carried or listed as left out;
- total words within about {budget}; no call to action unless asked; spelling as stated above.

Return ONLY strict JSON, no commentary:
{{
  "news_hook": "<the fact that makes this worth posting, in a few words>",
  "slots": [{{"slot": "<slot name>", "text": "<final text>", "carries": ["<field_id>|F1|redemption"]}}],
  "field_types": {{"<mandatory field_id>": "factual|creative"}},
  "coverage": {{"<mandatory field_id>": "<slot name, or 'missing'>"}},
  "facts_left_out": [{{"fact": "...", "why": "<a few words>"}}],
  "adaptations": [{{"from": "<the user's wording>", "to": "<what the graphic says>", "why": "<a few words>"}}]
}}"""

REPAIR = """Your copy deck failed these checks. Fix every one and return the full corrected deck in the
same JSON shape. Keep everything that passed.

FAILED CHECKS:
{problems}

YOUR PREVIOUS DECK:
{deck}"""


# --------------------------------------------------------------------------- checks

_NUM = re.compile(r"\d+(?:[.,:]\d+)*")
_CODE = re.compile(r"\b(?=[A-Z0-9]*[A-Z]{3})(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b")  # KORA50, HARVEST30


def _tokens(text: str) -> set:
    return v32._content_tokens(text)


# Number words count as their number, so "A Quarter-Century" and "25 years" are caught as
# the same fact twice. Longest phrases first so "quarter-century" wins over "century".
_NUMBER_WORDS = [
    (r"quarter[\s-]*century", "25"), (r"half[\s-]*century", "50"), (r"half[\s-]*price", "50"),
    (r"twenty[\s-]*five", "25"), (r"twenty[\s-]*one", "21"), (r"fifty", "50"), (r"forty", "40"),
    (r"thirty", "30"), (r"twenty", "20"), (r"fifteen", "15"), (r"twelve", "12"), (r"ten", "10"),
    (r"century", "100"), (r"decades?", "10"), (r"dozen", "12"), (r"\bhalf\b", "50"),
    (r"\bhalve[sd]?\b", "50"), (r"hundred", "100"),
]


def _numbers(text: str) -> set:
    """Numbers on the line (number words included), ignoring digits inside promo codes
    (KORA50 is not a second '50')."""
    t = _CODE.sub(" ", str(text))
    words = set()
    low = t.lower()
    for pat, num in _NUMBER_WORDS:
        if re.search(r"\b" + pat + r"\b" if not pat.startswith(r"\b") else pat, low):
            words.add(num)
            low = re.sub(r"\b" + pat + r"\b" if not pat.startswith(r"\b") else pat, " ", low)
    return {n.replace(",", "") for n in _NUM.findall(t)} | words


def _bigrams(text: str) -> set:
    w = [t for t in re.findall(r"[a-z0-9']+", str(text).lower()) if t not in v32._STOPWORDS]
    return {f"{a} {b}" for a, b in zip(w, w[1:])}


# ---- spelling variant -------------------------------------------------------------
# US -> British/Indian/Australian spelling. Words that only differ by -ize/-ise are
# handled by the regex below (with the -ize words that never change listed apart).
US_TO_UK = {
    "color": "colour", "colors": "colours", "colorful": "colourful", "colored": "coloured",
    "flavor": "flavour", "flavors": "flavours", "flavorful": "flavourful",
    "favorite": "favourite", "favorites": "favourites", "favor": "favour",
    "neighbor": "neighbour", "neighbors": "neighbours", "neighborhood": "neighbourhood",
    "honor": "honour", "labor": "labour", "humor": "humour", "harbor": "harbour",
    "savor": "savour", "savory": "savoury", "behavior": "behaviour", "rumor": "rumour",
    "center": "centre", "centers": "centres", "theater": "theatre", "meter": "metre",
    "fiber": "fibre", "jewelry": "jewellery", "catalog": "catalogue", "dialog": "dialogue",
    "gray": "grey", "cozy": "cosy", "mom": "mum", "pajamas": "pyjamas", "tire": "tyre",
    "traveling": "travelling", "traveled": "travelled", "traveler": "traveller",
    "canceled": "cancelled", "modeling": "modelling", "jeweler": "jeweller",
    "enrollment": "enrolment", "fulfill": "fulfil", "defense": "defence", "offense": "offence",
    "aluminum": "aluminium", "donut": "doughnut", "program": "programme", "license": "licence",
}
UK_TO_US = {v: k for k, v in US_TO_UK.items()}
_IZE = re.compile(r"\b([a-z]+)iz(e|es|ed|ing|ation|ations|er|ers)\b")
_IZE_NEVER = {"size", "sizes", "sized", "prize", "prizes", "seize", "seized", "capsize", "citizen",
              "citizens", "maize", "bronze", "freeze", "breeze", "squeeze", "snooze"}
_ISE = re.compile(r"\b([a-z]+)is(e|es|ed|ing|ation|ations)\b")
_ISE_REAL = {"organis", "personalis", "customis", "realis", "recognis", "specialis", "categoris",
             "prioritis", "optimis", "minimis", "maximis", "apologis", "summaris", "visualis",
             "modernis", "energis", "revolutionis", "glamoris", "finalis", "authoris", "memoris"}

UK_PLACES = ["uk", "united kingdom", "england", "scotland", "wales", "london", "manchester",
             "brixham", "bristol", "edinburgh", "glasgow", "leeds", "york", "devon", "cornwall",
             "ireland", "dublin", "£"]
IN_PLACES = ["india", "bengaluru", "bangalore", "mumbai", "delhi", "kolkata", "chennai",
             "hyderabad", "pune", "jaipur", "ahmedabad", "kochi", "indiranagar", "₹", " rs ",
             "rupees", " ist"]
AU_PLACES = ["australia", "sydney", "melbourne", "brisbane", "perth", "new zealand", "auckland"]
US_PLACES = ["usa", "united states", "new york", "nyc", "san francisco", "los angeles", "chicago",
             "austin", "seattle", "boston", "texas", "california"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def _spelling_hits(text: str, british: bool) -> list[str]:
    """Words in `text` spelled the OTHER way from the wanted variant."""
    hits = []
    for w in _words(text):
        if british and w in US_TO_UK:
            hits.append(f"{w} -> {US_TO_UK[w]}")
        elif not british and w in UK_TO_US:
            hits.append(f"{w} -> {UK_TO_US[w]}")
    low = str(text).lower()
    if british:
        for m in _IZE.finditer(low):
            if m.group(0) not in _IZE_NEVER:
                hits.append(f"{m.group(0)} -> {m.group(1)}is{m.group(2)}")
    else:
        for m in _ISE.finditer(low):
            if m.group(1) + "is" in _ISE_REAL:
                hits.append(f"{m.group(0)} -> {m.group(1)}iz{m.group(2)}")
    return hits


def detect_variant(brand: dict | None, user_text: str) -> dict:
    """Spelling variant: brand profile > the user's own spelling > their location > US default."""
    lang = str((brand or {}).get("language", "")).lower()
    if lang:
        if any(k in lang for k in ("british", "uk", "indian", "india", "australian", "en-gb", "en-in", "en-au")):
            return {"variant": "british", "basis": f"brand profile: {brand['language']}"}
        if any(k in lang for k in ("american", "us", "en-us")):
            return {"variant": "american", "basis": f"brand profile: {brand['language']}"}
    uk_words = [w for w in _words(user_text) if w in UK_TO_US] + \
               [m.group(0) for m in _ISE.finditer(user_text.lower()) if m.group(1) + "is" in _ISE_REAL]
    us_words = [w for w in _words(user_text) if w in US_TO_UK]
    if len(uk_words) != len(us_words):
        return ({"variant": "british", "basis": f"the user's own spelling ({', '.join(uk_words)})"}
                if len(uk_words) > len(us_words) else
                {"variant": "american", "basis": f"the user's own spelling ({', '.join(us_words)})"})
    low = f" {user_text.lower()} "
    for places, variant in ((IN_PLACES, "british"), (UK_PLACES, "british"),
                            (AU_PLACES, "british"), (US_PLACES, "american")):
        hit = next((p.strip() for p in places if p in low), None)
        if hit:
            return {"variant": variant, "basis": f"location ({hit})"}
    return {"variant": "american", "basis": "no brand, spelling or location signal (default)"}


def spelling_instruction(v: dict) -> str:
    if v["variant"] == "british":
        return (f"British/Indian English (basis: {v['basis']}): colour, centre, favourite, "
                "jewellery, neighbourhood, organise, programme, grey.")
    return (f"American English (basis: {v['basis']}): color, center, favorite, jewelry, "
            "neighborhood, organize, program, gray.")


def _echo(headline: str, user_lines: list[str], hint_lines: list[str] = ()) -> str | None:
    """The user line the headline merely repeats, if any.

    A suggestion from a clarifying answer ("something like 25 years of building boats") is
    echoed when 80%+ of the headline's words come from it. A sentence of the brief only counts
    when the headline is close to the whole sentence (it also covers 60%+ of the sentence):
    "Win a Year of Bramble Roasters Coffee" carries the prize the brief describes and is fine,
    because that sentence is the news, not a headline the user wrote.
    """
    # Spelling-neutral: "Color Theory" echoes a user's "colour theory" just as much.
    norm = lambda toks: {UK_TO_US.get(t, t) for t in toks}  # noqa: E731
    ht = norm(_tokens(headline))
    if len(ht) < 2:
        return None
    for line in hint_lines:
        lt = norm(_tokens(line))
        if lt and len(ht & lt) / len(ht) >= 0.8:
            return line
    for line in user_lines:
        lt = norm(_tokens(line))
        if lt and len(ht & lt) / len(ht) >= 0.8 and len(ht & lt) / len(lt) >= 0.6:
            return line
    return None


_QUOTED = re.compile(r"[“\"]([^”\"]{8,})[”\"]")


def _norm_quote(text: str) -> str:
    t = str(text).lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^a-z0-9%£$€' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _misquotes(slots: list, quote_source: str) -> list[str]:
    """Quoted text that is not word-for-word in what the user gave. Cuts marked with an
    ellipsis are allowed: each fragment between ellipses must appear verbatim."""
    src = _norm_quote(quote_source)
    out = []
    for s in slots:
        text = str(s["text"])
        segments = _QUOTED.findall(text)
        if s["slot"] == "quote" and not segments:
            segments = [text]
        for seg in segments:
            for frag in re.split(r"…|\.\.\.|\[\.\.\.\]", seg):
                f = _norm_quote(frag)
                if len(f.split()) >= 2 and f not in src:
                    out.append(f"MISQUOTE: {s['slot']} puts \"{frag.strip()}\" in quotation marks, "
                               "but those are not the person's exact words — use a verbatim excerpt "
                               "from the quote the user gave (mark cuts with …), never reword it")
    return out


def check_deck(deck: dict, must_fields: list, source_text: str, cta_requested: bool,
               budget: int, goal: str = "", other_facts: list | None = None,
               user_lines: list[str] | None = None, variant: dict | None = None,
               hint_lines: list[str] | None = None, quote_source: str | None = None) -> list[str]:
    """Deterministic checks on a copy deck. Returns a list of problems (empty = pass)."""
    problems = []
    slots = [s for s in deck.get("slots", []) if str(s.get("text", "")).strip()]

    # 0a. Headline must be the copywriter's line, not the user's words back. A quoted
    # headline is exempt: quotes are meant to be the person's own words (checked in 0d).
    head = next((s for s in slots if s["slot"] == "headline"), None)
    if head and (user_lines or hint_lines) and not _QUOTED.search(head["text"]):
        src = _echo(head["text"], user_lines or [], hint_lines or [])
        if src:
            problems.append(f"ECHOED HEADLINE: \"{head['text']}\" mostly repeats the user's own words "
                            f"(\"{src}\") — keep the facts, write a better line")
    # 0d. Quotes are verbatim.
    if quote_source:
        problems += _misquotes(slots, quote_source)
    # 0b. The news hook the copywriter named must be on the graphic.
    hook = str(deck.get("news_hook") or "").strip()
    on_text = set().union(*[_tokens(s["text"]) for s in slots]) if slots else set()
    if hook:
        ht = _tokens(hook) - _tokens(goal)
        if ht and len(ht & on_text) / len(ht) < 0.5:
            problems.append(f"NEWS HOOK MISSING: you named \"{hook}\" as the news, but it is not on "
                            "the graphic — put it in the headline")
    # 0c. Spelling variant.
    if variant:
        for s in slots:
            hits = _spelling_hits(s["text"], variant["variant"] == "british")
            if hits:
                problems.append(f"SPELLING ({variant['variant']}): {s['slot']} \"{s['text']}\": "
                                + ", ".join(hits))

    # 1. Repetition between slots: shared number, shared content bigram, or heavy word overlap.
    for i, a in enumerate(slots):
        for b in slots[i + 1:]:
            ta, tb = _tokens(a["text"]), _tokens(b["text"])
            shared_num = (_numbers(a["text"]) & _numbers(b["text"]))
            shared_bi = _bigrams(a["text"]) & _bigrams(b["text"])
            overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
            if shared_num or shared_bi or (overlap >= 0.5 and len(ta & tb) >= 1):
                what = (f"number {sorted(shared_num)}" if shared_num else
                        f"phrase {sorted(shared_bi)}" if shared_bi else
                        f"words {sorted(ta & tb)}")
                problems.append(f"REPETITION: {a['slot']} \"{a['text']}\" and {b['slot']} "
                                f"\"{b['text']}\" share {what}")

    # 2. Invented numbers: every number on the graphic must come from the user.
    src_nums = _numbers(source_text)
    for s in slots:
        for n in _numbers(s["text"]) - src_nums:
            # allow reformatting of times/dates already present as digits ("6pm" vs "6 pm")
            if not any(n in x or x in n for x in src_nums):
                problems.append(f"INVENTED NUMBER: \"{n}\" in {s['slot']} \"{s['text']}\" "
                                "is not in the user's brief or answers")

    # 3. Coverage: creative mandatory fields must be written; factual ones may be missing.
    cov = deck.get("coverage") or {}
    types = deck.get("field_types") or {}
    used = {s["slot"] for s in slots}
    for f in must_fields:
        fid = f["field_id"]
        where = cov.get(fid)
        if where in (None, "", "missing"):
            if types.get(fid) == "creative":
                problems.append(f"UNCOVERED: creative mandatory field {fid} ({f['label']}) has no "
                                "copy — write it")
            continue
        if where not in used:
            problems.append(f"COVERAGE: {fid} is mapped to slot '{where}', which has no text")

    # 4. CTA only when the user asked for one.
    if not cta_requested:
        for s in slots:
            if s["slot"] == "cta" or v32._is_cta_phrase(s["text"]):
                problems.append(f"CTA: the user did not ask for a call to action, but "
                                f"{s['slot']} reads \"{s['text']}\"")

    # 4b. Headline must be copy, not the goal's category label.
    if head and goal:
        filler = {"our", "the", "is", "live", "now", "new", "big", "it's", "its", "a"}
        ht = _tokens(head["text"]) - filler
        if ht and ht <= (_tokens(goal) | {"sale", "opening", "webinar", "hiring"}):
            problems.append(f"GENERIC HEADLINE: \"{head['text']}\" is just the category label — "
                            "write a headline that says something")

    # 4c. No silent drops: each other stated fact is on the graphic or listed as left out.
    on_graphic = set().union(*[_tokens(s["text"]) for s in slots]) if slots else set()
    left_out = " ".join(str(f.get("fact", "")) for f in deck.get("facts_left_out") or [])
    # A fact the deck says a slot carries (F1, F2... in "carries") counts as on the graphic,
    # so a fact rephrased into a headline ("we ship every week" -> "Ship code weekly") passes.
    carried = {str(c).strip().upper() for s in slots for c in (s.get("carries") or [])}
    for i, fact in enumerate(other_facts or [], 1):
        ft = _tokens(fact)
        if f"F{i}" in carried:
            continue
        if ft and len(ft & on_graphic) / len(ft) < 0.5 and len(ft & _tokens(left_out)) / len(ft) < 0.5:
            problems.append(f"DROPPED FACT: \"{fact}\" is neither on the graphic nor listed in "
                            "facts_left_out with a reason")

    # 5. Word budget (30% tolerance; mandatory facts can force a little over).
    words = sum(len(str(s["text"]).split()) for s in slots)
    if words > budget * 1.3:
        problems.append(f"OVER BUDGET: {words} words against a budget of about {budget} for this "
                        "platform — cut or merge lines")
    return problems


# --------------------------------------------------------------------------- node

def deck_to_locked(deck: dict) -> dict:
    """The deck as the `locked` dict the downstream nodes already read: slot -> text."""
    return {s["slot"]: s["text"] for s in deck.get("slots", []) if str(s.get("text", "")).strip()}


def brand_block(brand: dict | None) -> str:
    if not brand:
        return ("No brand profile. Take voice and spelling from the user's own words and location; "
                "ground everything in what they told us.")
    return "; ".join(f"{k}: {v}" for k, v in brand.items() if v)


def run_copywriter(client, *, goal: str, platform: str, raw_prompt: str,
                   clarifications: list[dict], ground: dict, locked: dict,
                   other_facts: list, cta_requested: bool, brand: dict | None = None,
                   thinking: str | None = None, emit=lambda e: None) -> dict:
    brief = v32.load_guidance(goal, platform)
    authored = brief.get("authored") or {}
    budget = WORD_BUDGET.get(platform, 24) + GOAL_EXTRA_WORDS.get(goal, 0)
    red = ground.get("redemption") or {}
    clar_s = "\n".join(f"  Q: {c['q']}\n  A: {c['a']}" for c in clarifications) or "  (none)"
    user_text = " ".join([raw_prompt] + [c["a"] for c in clarifications])
    variant = detect_variant(brand, user_text)
    # The user's own lines for the echoed-headline check: brief sentences, and the answers
    # (an answer is where a suggested headline lives, so it is checked more strictly).
    user_lines = [s for s in re.split(r"(?<=[.!?])\s+|\n", raw_prompt) if s.strip()]
    hint_lines = [c["a"] for c in clarifications]
    prompt = COPYWRITER.format(
        goal_note=GOAL_NOTES.get(goal, "Write what this goal needs; nothing more."),
        spelling=spelling_instruction(variant),
        goal=goal, platform=platform, aspect_ratio=brief["aspect_ratio"], budget=budget,
        format_character=authored.get("format_character", f"A {goal} graphic."),
        platform_craft=authored.get("platform_craft") or v32.safe_area(platform),
        brand=brand_block(brand),
        must="\n".join(f"  - {f['field_id']}: {f['label']}" for f in brief["must_communicate"]),
        optional="\n".join(f"  - {f['field_id']}: {f['label']}" for f in brief["may_include"]
                           if f["field_id"] not in v32.CTA_FIELD_IDS),
        raw_prompt=raw_prompt.strip(), clarifications=clar_s,
        locked="; ".join(f'{k}: "{v}"' for k, v in locked.items()) or "(none)",
        other_facts="; ".join(f"F{i}: {f}" for i, f in enumerate(other_facts, 1)) or "none",
        redemption=(f'code {red["token"]}, used {red["how"]}' if red.get("token") else "none"),
        missing=", ".join(ground.get("missing_mandatory") or []) or "none",
        cta_rule=("the user asked for one: write it in a 'cta' slot"
                  if cta_requested else
                  "NONE. No cta slot and no 'shop now', 'register', 'book now' or similar line."))
    source_text = " ".join([raw_prompt, clar_s, json.dumps(ground, ensure_ascii=False)])
    think = thinking or COPY_THINKING

    deck, meta = v32._call(client, prompt, "copywriter", emit, thinking=think, temp=COPY_TEMP)
    problems = check_deck(deck, brief["must_communicate"], source_text, cta_requested, budget,
                          goal, other_facts, user_lines, variant, hint_lines, user_text)
    attempts = [{"deck": deck, "problems": problems, "meta": meta}]
    # Up to MAX_REPAIRS repair passes; each sees the latest deck and its remaining problems.
    # Keep whichever deck has the fewest problems.
    for n in range(1, MAX_REPAIRS + 1):
        if not problems:
            break
        last = attempts[-1]
        fixed, meta2 = v32._call(
            client, prompt + "\n\n" + REPAIR.format(
                problems="\n".join(f"- {p}" for p in last["problems"]),
                deck=json.dumps(last["deck"], ensure_ascii=False)),
            f"copywriter_repair{n}", emit, thinking=think, temp=0.5)
        problems2 = check_deck(fixed, brief["must_communicate"], source_text, cta_requested, budget,
                               goal, other_facts, user_lines, variant, hint_lines, user_text)
        attempts.append({"deck": fixed, "problems": problems2, "meta": meta2})
        if len(problems2) <= len(problems):
            deck, problems = fixed, problems2
    return {"deck": deck, "problems": problems, "budget": budget, "attempts": attempts,
            "thinking": think, "locked": deck_to_locked(deck), "variant": variant,
            "seconds": round(sum(a["meta"]["seconds"] for a in attempts), 1),
            "first_pass_seconds": attempts[0]["meta"]["seconds"],
            "cost": round(sum(a["meta"]["cost"] for a in attempts), 5),
            "metas": [a["meta"] for a in attempts]}


def make_hook(*, goal: str, platform: str, raw_prompt: str, clarifications: list[dict],
              brand: dict | None = None):
    """The callable `run(copywriter=...)` takes, bound to one brief.

    raw_prompt: the user's own words (not an LLM rewrite of them).
    clarifications: [{"q": question or field label, "a": the user's answer}].
    brand: brand profile or None; a "language" key sets the spelling variant.
    run() supplies the rest (GROUND output, locked text, other facts, CTA flag).
    """
    def copywriter(client, *, ground, locked, other_facts, cta_requested, emit):
        return run_copywriter(client, goal=goal, platform=platform, raw_prompt=raw_prompt,
                              clarifications=clarifications, ground=ground, locked=locked,
                              other_facts=other_facts, cta_requested=cta_requested,
                              brand=brand, emit=emit)
    return copywriter
