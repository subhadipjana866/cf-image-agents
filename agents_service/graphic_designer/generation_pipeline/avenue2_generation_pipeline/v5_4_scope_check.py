"""
v5_4_scope_check.py  (Phase 1.0 — pre-flight scope gate)
----------------------------------------------------------------------
Runs BEFORE intent clarification and BEFORE layout assets are loaded.

The user selects a design goal + platform in the UI, then types a prompt.
The prompt may contradict those selections:

  - different DELIVERABLE:  selected "Festive Sale" but prompt says
    "design me a tattoo"            -> goal conflict (maybe unsupported)
  - different PLATFORM(S):  selected "Pinterest Pin" but prompt says
    "make it for instagram vertical and facebook square both"
                                    -> platform conflict + multi-platform

CRITICAL DISTINCTION the extraction must honor: an unusual SUBJECT is not
a different deliverable. "Christmas sale at my tattoo parlor, 20% off"
is still a Festive Sale graphic (the tattoo parlor is the business) ->
NO conflict. Only flag detected_design_goal when the user wants a
different KIND of output.

Flow (implemented in kognie_generation_pipeline.generate_graphic):
  match          -> standard pipeline (clarifier -> router -> ...)
  mismatch       -> blocking `scope_mismatch` interrupt:
                    "You selected X + Y but your prompt asks for Z. Proceed with Z?"
      user: yes, Z is in our catalog   -> switch goal/platform in state
                                          (backend + frontend), standard flow
      user: yes, Z NOT in our catalog  -> BYPASS: no layout, no CoT --
                                          one direct image-gen call with just
                                          the user's context ("freeform")
      user: no                         -> keep selections, standard flow
  multi-platform -> primary platform now, remainder returned as
                    pending_platforms for the session state to queue.

The LLM only EXTRACTS (maps prompt mentions to canonical catalog names);
every match/mismatch DECISION is plain deterministic code -- this is what
closes the silent-failure hole where the router ignored a correct
ambiguity flag whenever clarity_score was high.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
COMBINATIONS_FILE = SCRIPT_DIR / "phase0_combinations.json"

_UNSUPPORTED_PREFIX = "UNSUPPORTED:"


# The catalog is layout_guidance.json -- the same file the generation pipeline
# reasons from. phase0_combinations.json remains only as a fallback for a
# checkout without the v33 package: when the two disagree, a combination the
# guidance cannot serve is not in the catalog at all, so nothing downstream can
# accept a goal/platform pair the renderer has no rules for.
_GUIDANCE_FILE = Path(os.environ.get(
    "KOGNIE_LAYOUT_GUIDANCE",
    SCRIPT_DIR.parent.parent / "kognie-generation-v33" / "layout_guidance.json"))


def _catalog_from_guidance(path: Path = _GUIDANCE_FILE):
    """``(goals, platforms)`` read straight out of layout_guidance.json, or
    ``None`` when the file is missing or unreadable."""
    try:
        guidance = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    goals, platforms = set(), {}
    for rec in guidance.values():
        if not isinstance(rec, dict):
            continue
        goals.add(rec.get("design_goal", ""))
        if rec.get("platform"):
            platforms.setdefault(rec["platform"], rec.get("aspect_ratio", ""))
    if not goals or not platforms:
        return None
    return (sorted(g for g in goals if g),
            [{"name": n, "aspect_ratio": r} for n, r in sorted(platforms.items())])


def load_catalog(combinations_path: Path = COMBINATIONS_FILE) -> tuple[list[str], list[dict]]:
    """Supported (design_goals, platforms).

    layout_guidance.json is the source of truth; the JSON at
    ``combinations_path`` is the fallback. Empty lists if neither is readable.
    """
    from_guidance = _catalog_from_guidance()
    if from_guidance:
        return from_guidance
    try:
        cfg = json.loads(Path(combinations_path).read_text(encoding="utf-8"))
        return cfg.get("design_goals", []), cfg.get("platforms", [])
    except Exception:
        return [], []


_EXTRACTION_PROMPT = """\
You are a scope extraction engine for a marketing-graphic service. The user has ALREADY
selected a design goal and platform in the UI. Your ONLY job is to detect whether their
prompt asks for a DIFFERENT deliverable or DIFFERENT platform(s). You extract; you do not decide.

SUPPORTED DESIGN GOALS (canonical names):
{goals}

SUPPORTED PLATFORMS (canonical names):
{platforms}

USER'S UI SELECTION: design_goal="{selected_goal}", platform="{selected_platform}"

USER PROMPT:
{raw_prompt}

Rules:
1. detected_design_goal:
   - null if the prompt's deliverable is consistent with the selected design goal,
     OR the prompt doesn't imply any deliverable type at all.
   - A canonical name from the supported list if the prompt clearly asks for a DIFFERENT
     deliverable that we DO support (e.g. selected "Festive Sale" but prompt says
     "announce our grand opening" -> "Grand Opening").
   - "{unsupported}<short description>" if the prompt asks for a deliverable we do NOT
     support (e.g. "design me a tattoo" -> "{unsupported}tattoo design"; "make me a
     resume" -> "{unsupported}resume").
   ⚠️ SUBJECT vs DELIVERABLE: an unusual business/subject is NOT a different deliverable.
     "christmas sale at my tattoo parlor" -> null (still a Festive Sale graphic).
     "a tattoo of a dragon" -> "{unsupported}tattoo design" (different deliverable).
2. detected_platforms:
   - [] if the prompt doesn't mention any target platform/format.
   - Otherwise every platform the prompt names, each mapped to a canonical supported name
     when clearly equivalent (e.g. "instagram vertical" -> "Instagram Portrait Post",
     "IG story" -> "Instagram Story"), else "{unsupported}<verbatim mention>"
     (e.g. "facebook square" has no supported equivalent -> "{unsupported}facebook square").
   - Mentioning the SELECTED platform is still a mention: include it.

Output ONLY strict JSON:
{{"detected_design_goal": null, "detected_platforms": []}}
"""


def check_request_scope(
    raw_prompt: str,
    selected_design_goal: str,
    selected_platform: str,
    model: str = "models/gemini-3.1-flash-lite",
    combinations_path: Path = COMBINATIONS_FILE,
    role: str | None = None,
    pin: str | None = None,
) -> dict:
    """
    Extract prompt-implied goal/platforms (LLM) and compare against the UI
    selection (deterministic code). Never raises; on any failure returns
    match=True so the pipeline degrades to current behavior.
    """
    result = {
        "match": True,
        "goal_conflict": False,
        "platform_conflict": False,
        "multi_platform": False,
        "detected_goal": None,
        "detected_goal_supported": None,   # True/False when a conflict exists, else None
        "detected_platforms": [],
        "primary_platform": selected_platform,
        "pending_platforms": [],           # supported platforms beyond the primary
        "unsupported_platforms": [],       # mentions we can't serve
        "_usage": {},
    }

    goals, platforms = load_catalog(combinations_path)
    platform_names = [p["name"] for p in platforms]

    try:
        from llm_fallback import generate_with_fallback

        prompt = _EXTRACTION_PROMPT.format(
            goals="\n".join(f"  - {g}" for g in goals) or "  (catalog unavailable)",
            platforms="\n".join(f"  - {p}" for p in platform_names) or "  (catalog unavailable)",
            selected_goal=selected_design_goal,
            selected_platform=selected_platform,
            raw_prompt=raw_prompt,
            unsupported=_UNSUPPORTED_PREFIX,
        )
        response = generate_with_fallback(
            model=model,
            prompt_text=prompt,
            temperature=0.0,
            response_mime_type="application/json",
            role=role,
            pin=pin,
        )
        raw = response.text.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0) if m else raw)
        if response.usage_metadata:
            result["_usage"] = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
            }
    except Exception as e:
        print(f"  [SCOPE-CHECK] extraction failed ({e}) -- proceeding as match")
        return result

    # ---- Deterministic comparison (the actual gate) ----------------------
    detected_goal = data.get("detected_design_goal")
    detected_platforms = [p for p in (data.get("detected_platforms") or []) if isinstance(p, str)]

    if detected_goal:
        result["detected_goal"] = detected_goal
        if detected_goal.startswith(_UNSUPPORTED_PREFIX):
            result["goal_conflict"] = True
            result["detected_goal_supported"] = False
            result["detected_goal"] = detected_goal[len(_UNSUPPORTED_PREFIX):].strip()
        elif detected_goal.strip().lower() != selected_design_goal.strip().lower():
            result["goal_conflict"] = True
            result["detected_goal_supported"] = detected_goal in goals

    if detected_platforms:
        result["detected_platforms"] = detected_platforms
        supported_mentions = [p for p in detected_platforms if not p.startswith(_UNSUPPORTED_PREFIX)]
        result["unsupported_platforms"] = [
            p[len(_UNSUPPORTED_PREFIX):].strip() for p in detected_platforms if p.startswith(_UNSUPPORTED_PREFIX)
        ]
        result["multi_platform"] = len(detected_platforms) > 1

        mentions_selected = any(
            p.strip().lower() == selected_platform.strip().lower() for p in supported_mentions
        )
        # Conflict when the prompt names platforms and the selection isn't
        # simply "the one platform they asked for".
        if not (len(detected_platforms) == 1 and mentions_selected):
            result["platform_conflict"] = bool(
                result["multi_platform"] or supported_mentions or result["unsupported_platforms"]
            ) and not (len(detected_platforms) == 1 and mentions_selected)

        # Primary: keep the UI selection if the prompt includes it; else the
        # first supported mention; else keep the selection (nothing usable).
        if mentions_selected:
            result["primary_platform"] = selected_platform
        elif supported_mentions:
            result["primary_platform"] = supported_mentions[0]
        result["pending_platforms"] = [
            p for p in supported_mentions
            if p.strip().lower() != result["primary_platform"].strip().lower()
        ]

    result["match"] = not (result["goal_conflict"] or result["platform_conflict"])
    return result


def platform_ratio(platform_name: str, combinations_path: Path = COMBINATIONS_FILE) -> str | None:
    """Canonical aspect ratio for a supported platform name, else None."""
    _, platforms = load_catalog(combinations_path)
    for p in platforms:
        if p["name"].strip().lower() == platform_name.strip().lower():
            return p["aspect_ratio"]
    return None
