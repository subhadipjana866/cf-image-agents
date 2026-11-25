"""Offline regression tests for the critique pipeline — no API keys, no S3, no network.

Every LLM call is stubbed, and the parent package the pipeline imports (``..config``,
``..utils``) is faked, so this runs from the bare folder before AND after it is dropped into
``graphic_designer/``:

    python critiquepipeline/tests/test_offline.py        # plain python
    pytest critiquepipeline/tests/test_offline.py        # or pytest

Each test pins a failure mode seen in the Sep 2026 live runs (DEVELOPER_GUIDE.md §10).
"""
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
PARENT = "_cp_parent"


def _load_package():
    """Import the folder as `_cp_parent.critiquepipeline` with a fake parent package."""
    if f"{PARENT}.critiquepipeline" in sys.modules:
        return sys.modules[f"{PARENT}.critiquepipeline"]
    parent = types.ModuleType(PARENT)
    parent.__path__ = []
    config = types.ModuleType(f"{PARENT}.config")
    config.GEMINI_API_KEY, config.s3_client = "", None
    utils = types.ModuleType(f"{PARENT}.utils")
    utils.parseJson = lambda s: json.loads(s) if isinstance(s, str) else s
    utils.makeApiRequest = lambda *a, **k: None
    utils.calculateCost = lambda *a: 0.0
    utils.apiLog = lambda *a, **k: None
    utils.messageText = lambda r: getattr(r, "content", r)
    utils.profile = lambda f: f
    sys.modules.update({PARENT: parent, f"{PARENT}.config": config, f"{PARENT}.utils": utils})
    if importlib.util.find_spec("langgraph") is None:        # nodes.py imports Command at load
        lg, lgt = types.ModuleType("langgraph"), types.ModuleType("langgraph.types")
        lgt.Command = type("Command", (), {"__init__": lambda self, **k: self.__dict__.update(k)})
        lgt.interrupt = lambda *a, **k: None
        sys.modules.update({"langgraph": lg, "langgraph.types": lgt})
    spec = importlib.util.spec_from_file_location(
        f"{PARENT}.critiquepipeline", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load_package()
A = sys.modules[f"{PARENT}.critiquepipeline.agents"]
N = sys.modules[f"{PARENT}.critiquepipeline.nodes"]
P = sys.modules[f"{PARENT}.critiquepipeline.prompts"]
G = sys.modules[f"{PARENT}.critiquepipeline.grounding"]
Telemetry = sys.modules[f"{PARENT}.critiquepipeline.telemetry"].Telemetry


def rec(fid, topic, nano, conf=0.9, lens="composition", risk="local", ev="generic principle",
        cat=None, uft=None, typ="EDIT"):
    return {"fid": fid, "topic": topic, "nano_prompt": nano, "confidence": conf,
            "source_lens": lens, "edit_risk": risk, "evidence": ev,
            "checklist_category": cat or topic, "type": typ, "user_friendly_text": uft or topic}


def _stub_llm(arbiter_order, judge=lambda fid: (True, "ok")):
    """Replace the text-LLM transport: arbiter keeps `arbiter_order`; judge decides per fid."""
    def fake(phase, model, prompt, tele, tb=None):
        if phase == "arbiter":
            return json.dumps({"images": [{"image_id": "image_1", "keep": [
                {"fid": f, "rank": i + 1} for i, f in enumerate(arbiter_order)]}]})
        if phase == "executability":
            assert "[[" not in prompt, "unfilled placeholder in EXECUTABILITY_PROMPT"
            fids = re.findall(r'"fid": "(f\d+)"', prompt)
            return json.dumps({"verdicts": [
                {"fid": f, "executable": judge(f)[0], "reason": judge(f)[1]} for f in fids]})
        raise AssertionError(f"unexpected LLM phase {phase}")
    A._text_invoke = fake


def _arbiter(recs, order, judge=lambda fid: (True, "ok"), n=3):
    _stub_llm(order, judge)
    merged = [{"image_id": "image_1", "overall_assessment": "",
               "recommendations": [dict(r) for r in recs]}]
    out = json.loads(A.arbiterAgent(merged, None, {}, {"intent": ""}, n, Telemetry()))
    return out["images"][0]["recommendations"]


# --------------------------------------------------------------------------- #
# executability gate
# --------------------------------------------------------------------------- #
def test_gate_drops_and_backfills():
    recs = [rec("f0", "Placeholder", "Replace 'SN_404' with the real product code", lens="content"),
            rec("f1", "Logo", "Add the brand logo in the upper-left corner"),
            rec("f2", "Dark footer", "Darken the band behind the footer text"),
            rec("f3", "Price", "Move the price tag below the headline"),
            rec("f4", "CTA size", "Enlarge the ORDER NOW button"),
            rec("f5", "Relayout", "Move everything left", risk="structural",
                ev="REQUIRES MOVING 4 ELEMENTS: x")]
    got = _arbiter(recs, [f"f{i}" for i in range(6)],
                   judge=lambda f: (f != "f0", "needs unknown value" if f == "f0" else "ok"))
    assert [r["topic"] for r in got] == ["Dark footer", "Price", "CTA size"]
    assert all("fid" not in r and "_reword" not in r for r in got)


def test_gate_zero_when_nothing_executable():
    recs = [rec("f0", "A", "Recolor the CTA"), rec("f1", "B", "Darken the footer")]
    assert _arbiter(recs, ["f0", "f1"], judge=lambda f: (False, "x")) == []


def test_rule_edges():
    R = A._ruleNotExecutable
    assert R({"nano_prompt": "Shrink the logo in the corner"}) is None
    assert R({"nano_prompt": "Remove the logo watermark"}) is None
    assert R({"nano_prompt": "Redraw the brand logo crisply"})
    assert R({"nano_prompt": "Add logo", "type": "UPSCALE"}) is None
    assert R({"nano_prompt": "x", "evidence": "REQUIRES MOVING 2 ELEMENTS"}) is None


# --------------------------------------------------------------------------- #
# user-facing text
# --------------------------------------------------------------------------- #
def test_user_text_prefers_lens_and_rejects_hand_back():
    recs = [rec("f0", "Placeholder", "Remove the text 'Tickets: TBC' from the corner.",
                uft="Remove the leftover 'Tickets: TBC' line.", lens="content"),
            rec("f1", "Spelling", "In the lower-left corner, change 'OCTOBR' to 'OCTOBER'.",
                uft="The word OCTOBR is misspelled.", lens="content")]
    _stub_llm([])
    merged = [{"image_id": "image_1", "overall_assessment": "", "recommendations": recs}]
    A._text_invoke = lambda phase, *a, **k: json.dumps(
        {"images": [{"image_id": "image_1", "keep": [
            {"fid": "f0", "rank": 1, "reword": "Ensure this is updated with booking info."},
            {"fid": "f1", "rank": 2}]}]}) if phase == "arbiter" else json.dumps({"verdicts": []})
    out = json.loads(A.arbiterAgent(merged, None, {}, {"intent": ""}, 3, Telemetry()))
    texts = [r["user_friendly_text"] for r in out["images"][0]["recommendations"]]
    assert texts == ["Remove the leftover 'Tickets: TBC' line.",
                     "Change 'OCTOBR' to 'OCTOBER'."], texts


# --------------------------------------------------------------------------- #
# lens scope (code-enforced)
# --------------------------------------------------------------------------- #
def test_scope_guard():
    drop = [{"source_lens": "composition", "topic": "Missing CTA", "evidence": "no CTA",
             "nano_prompt": "Add a small 'Shop Now' button in the bottom-right corner."},
            {"source_lens": "content", "topic": "Redundant text",
             "evidence": "'THIS WEEKEND ONLY' repeats the timing already established.",
             "nano_prompt": "Remove the text 'THIS WEEKEND ONLY' above the URL."}]
    keep = [{"source_lens": "composition", "topic": "Duplicate text",
             "evidence": "'SHOP AT NORTHFIELD.CO' appears three times.",
             "nano_prompt": "Remove the two 'SHOP AT NORTHFIELD.CO' lines in the upper-left; "
                            "keep the single one at the bottom-left."},
            {"source_lens": "integrity", "topic": "Missing brand name", "evidence": "brief",
             "nano_prompt": "Add the text 'NORTHFIELD OUTFITTERS' in the upper-left corner."}]
    assert all(A._outOfScope(r) for r in drop)
    assert not any(A._outOfScope(r) for r in keep)


def test_nudge_guard():
    drop = [rec("f0", "Title position", "Move the 'SUMMER SALE' title slightly to the left."),
            rec("f1", "Button position", "Move the 'SHOP NOW' button 10% up.",
                uft="Move the button 10% up."),
            rec("f2", "Balance", "In the upper-left, shift the logo a few pixels right.", lens="aesthetic"),
            rec("f3", "Alignment", "Nudge the price down so it lines up with the headline."),
            rec("f4", "Spacing", "Slightly lower the tagline under the headline.")]
    keep = [rec("f5", "Cut-off headline", "Shrink and move the headline 'OF SOUND' slightly left "
                "so it sits fully inside the canvas.", ev="headline clipped by the right edge"),
            rec("f6", "Overlap", "Move the 'SHOP NOW' button 10% down.",
                ev="button overlaps the price text"),
            rec("f7", "Dark corner", "Darken the lower-left corner slightly behind the footer."),
            rec("f8", "Covered footer", "Lift the footer above the bottom edge.",
                ev="sits where the app's own buttons would cover it"),
            rec("f9", "Discount", "Change the badge text to '20% OFF'.", lens="content")]
    assert [r["fid"] for r in drop if not A._isNudge(r)] == []
    assert [r["fid"] for r in keep if A._isNudge(r)] == []
    assert _arbiter(drop, [r["fid"] for r in drop], n=10) == []      # dropped before the arbiter


# --------------------------------------------------------------------------- #
# dedup / ranking
# --------------------------------------------------------------------------- #
def test_cross_category_duplicate_merges_and_keeps_its_slot():
    dup = "Remove the two 'SHOP AT NORTHFIELD.CO' lines in the upper-left; keep the bottom one."
    recs = [rec("f0", "Duplicate text", dup, 0.95, "integrity", cat="Typography",
                ev="'SHOP AT NORTHFIELD.CO' appears three times."),
            rec("f1", "Duplicate text", dup, 0.9, "content", cat="copy quality",
                ev="'SHOP AT NORTHFIELD.CO' appears three times."),
            rec("f2", "Duplicate text", dup, 0.9, "composition", cat="Hierarchy",
                ev="'SHOP AT NORTHFIELD.CO' appears three times."),
            rec("f3", "Cut-off text", "Shrink '40% OFF' so 'OFF' is fully visible.", 1.0,
                ev="'40% OFF' is cut off."),
            rec("f4", "Hard-to-read text", "Recolor 'ALL RAIN JACKETS' to white for legibility.",
                0.7, "aesthetic", ev="Grey 'ALL RAIN JACKETS' lacks contrast.")]
    merged = A._predupe([dict(r) for r in recs])
    dups = [r for r in merged if r["topic"] == "Duplicate text"]
    assert len(dups) == 1 and "corroborated" in dups[0]["evidence"]
    final, _ = A._curate(merged, 3)
    assert "Duplicate text" in [r["topic"] for r in final]


def test_same_string_different_problem_stays_separate():
    a = rec("a", "Spelling error", "Change 'OCTOBR' to 'OCTOBER'.", lens="content",
            ev="'OCTOBR' is misspelled.")
    b = rec("b", "Hard-to-read text", "Brighten 'OCTOBR' to improve legibility.",
            ev="'OCTOBR' is illegible on the photo.")
    assert len(A._predupe([a, b])) == 2


# --------------------------------------------------------------------------- #
# executor helpers
# --------------------------------------------------------------------------- #
def test_edit_quality():
    q = lambda nano: A.editQuality([{"topic": "", "nano_prompt": nano}])
    assert q("In the upper-left corner, change the text 'wirless' to 'wireless'.") == "high"
    assert q("Recolor the text '40-hour battery' to off-white.") == "medium"
    assert q("Remove the two 'SHOP AT NORTHFIELD.CO' lines.") == "medium"
    assert q("Shrink the '40% OFF' headline so OFF fits.") == "medium"


def test_edit_instruction_not_treated_as_brief():
    assert N._isEditInstruction("Using the current image as reference, preserve ... Remove ...")
    assert not N._isEditInstruction("A moody jazz-club poster, headline 'Kirkgate Jazz Nights'")
    assert not N._isEditInstruction(None)


# --------------------------------------------------------------------------- #
# brand grounding
# --------------------------------------------------------------------------- #
# Shape main.fetch_brand_dna actually puts in state['brand_dna'] (flat keys, color_N palettes).
LIVE_BRAND = {
    "brand_name": "Northfield", "brand_context": "Outdoor apparel", "domain": "northfield.co",
    "tone": "calm, confident", "personality_traits": ["rugged", "understated"],
    "font_pairing": {"headline": "condensed bold sans", "body": "clean humanist sans"},
    "visual_style": {"mood": "natural, muted", "lighting": "soft daylight"},
    "color_primary": "#1B2A4A", "color_secondary": ["#C9A227"], "color_accent": "#C9A227",
    "color_background": "#FAF7F0",
    "color_palettes": {"ranked_palettes": [{"rank": 1, "name": "Core", "colors": {
        "color_1": {"hex": "#1B2A4A"}, "color_2": {"hex": "#C9A227"}, "color_3": {"hex": "#3A5A40"}}}]},
    "logo": ["https://x/logo.png"], "templates": [], "color_swatches": [],
}

# BrandDNA dataclass schema (brand_profile/pipeline/dna_models.py).
DNA_BRAND = {
    "brand_name": "Aurora",
    "brand_personality": {"tone": "playful", "keywords": ["bold"]},
    "color_system": {"primary": [["#E63946"]], "secondary": [["#111111", "#FFFFFF"]]},
    "typography_system": {"font_pairing": ["Inter", "Playfair"], "headline_style": "heavy serif"},
    "composition_rules": {"density": "airy"},
    "visual_style": {"mood": "energetic"},
    "constraints": {"max_colors": 3, "max_fonts": 2},
    "confidence_scores": {"color": 0.9},
}


def test_brand_normalizer_reads_live_shape():
    b = G.brandNormalizer(LIVE_BRAND)
    names = [c["name"] for c in b["palette"]["named"]]
    assert "deep navy blue" in names and "warm gold" in names and "deep green" in names
    assert len(names) == len(set(c["hex"].lower() for c in b["palette"]["named"]))   # deduped
    assert b["palette"]["primary"] == ["#1B2A4A"] and b["palette"]["background"] == "#FAF7F0"
    assert b["personality"]["tone"] == "calm, confident" and b["personality"]["traits"]
    assert b["fonts"]["families"]["headline"] == "condensed bold sans"
    assert b["logo_url"] == "https://x/logo.png"


def test_brand_normalizer_keeps_dna_shape():
    b = G.brandNormalizer(DNA_BRAND)
    assert b["palette"]["primary"] == ["#E63946"]
    assert b["personality"] == {"tone": "playful", "keywords": ["bold"]}
    assert b["fonts"]["headline"] == "heavy serif" and b["constraints"]["max_fonts"] == 2
    assert {"red", "near-black", "white"} <= {c["name"] for c in b["palette"]["named"]}


def test_brand_normalizer_no_identity_is_none():
    assert G.brandNormalizer(None) is None
    assert G.brandNormalizer({}) is None
    assert G.brandNormalizer({"logo": ["https://x/l.png"], "templates": []}) is None
    assert G.brandNormalizer({"brand_name": "X", "tone": None, "color_palettes": {}}) is None


def test_color_names():
    C = G._colorName
    assert C("#1B2A4A") == "deep navy blue" and C("#000080") == "navy blue"
    assert C("#C9A227") == "warm gold" and C("#FFD700") == "warm gold"
    assert C("#111111") == "near-black" and C("#FAF7F0") == "off-white"
    assert C("not-a-hex") is None


def test_brand_block():
    none = P.brandBlock(None)
    assert "none attached" in none and "[[" not in none
    out = P.brandBlock(G.brandNormalizer(LIVE_BRAND))
    assert "[[" not in out and "Northfield" in out
    assert "deep navy blue (primary)" in out and "warm gold" in out and "calm, confident" in out
    colours = next(l for l in out.splitlines() if l.startswith("- brand colours"))
    assert "#" not in colours                     # colours named in words, never as hex
    assert "BRAND-SAFE FIX RULES" in out and "brand_check" in out


def test_arbiter_sees_brand_block_and_brand_check():
    seen = {}

    def fake(phase, model, prompt, tele, tb=None):
        if phase == "arbiter":
            seen["prompt"] = prompt
            return json.dumps({"images": [{"image_id": "image_1", "keep": [{"fid": "f0", "rank": 1}]}]})
        return json.dumps({"verdicts": [{"fid": "f0", "executable": True, "reason": "ok"}]})
    A._text_invoke = fake
    r = rec("f0", "Off-brand CTA", "Recolor the orange ORDER NOW button to the brand's warm gold")
    r["brand_check"] = "recolours to brand warm gold"
    merged = [{"image_id": "image_1", "overall_assessment": "", "recommendations": [r]}]
    brand = G.brandNormalizer(LIVE_BRAND)
    eff = G.genericLimits(brand)
    A.arbiterAgent(merged, brand, eff, {"intent": ""}, 3, Telemetry())
    p = seen["prompt"]
    assert "[[" not in p and "BRAND-SAFE FIX RULES" in p and "recolours to brand warm gold" in p
    assert "palette_identity" not in p                     # identity comes via brandBlock only


# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #
def test_templates_fully_render():
    assert "[[" not in P.render(P.VERIFY_EDIT_PROMPT, accepted_changes="[]", pending_changes="[]")
    out = A._output("content", 1)
    assert "[[" not in out and "REMOVING ONE OF SEVERAL" in out and "AI-EXECUTABLE ONLY" in out
    assert '"brand_check"' in out


def test_lens_prompts_fully_render_with_and_without_brand():
    captured = {}
    orig = A._vision_invoke
    A._vision_invoke = lambda phase, model, prompt, *a, **k: captured.__setitem__(phase, prompt) or {}
    try:
        for brand in (None, G.brandNormalizer(LIVE_BRAND)):
            eff = G.genericLimits(brand)
            avail = {"mode": "generic", "brand": brand is not None, "base_prompt": False,
                     "provenance": "uploaded", "intent": "", "platform": "", "goal": ""}
            state = {"purpose": "", "prompt": ""}
            A.integrityEval(["x"], "b64", None, [], avail, state, Telemetry(), brand)
            A.compositionEval(["x"], "b64", None, brand, eff, avail, state, Telemetry())
            A.contentEval(["x"], "b64", None, brand, eff, avail, state, Telemetry())
            A.aestheticEval(["x"], "b64", None, brand, eff, avail, state, Telemetry())
            assert set(captured) == {"integrity", "composition", "content", "aesthetic"}
            for phase, prompt in captured.items():
                assert "[[" not in prompt, f"unfilled placeholder in {phase}"
                marker = "BRAND-SAFE FIX RULES" if brand else "none attached"
                assert marker in prompt, f"{phase} missing brand block"
            captured.clear()
    finally:
        A._vision_invoke = orig


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                       # report every failure, not just the first
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
