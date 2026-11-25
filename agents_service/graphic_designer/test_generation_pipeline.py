"""
Standalone test harness for the decomposed generation-flow graph
(generationManager -> scopeGate -> intentClarifier -> intentRouter ->
manifestFieldsGate -> photoRequiredGate -> clarificationGate ->
imageAnalyzerLayoutSelector -> logoDecisionGate -> briefFinalizerAndBriefReview ->
cotFusionRemainder -> generation -> run*Model -> logoCompositor ->
interruptBeforeFidelitySuggestion), including the three ask-node branches that
own the interrupts for their stage: scopeGate -> scopeMismatchGate ->
scopeAbortGate, manifestFieldsGate -> manifestFieldsAsk, and
briefFinalizerAndBriefReview -> midCotClarificationAsk.

Drives the compiled LangGraph app directly (in-process, no websocket/FastAPI
server needed), automating interrupt/resume the same way main.py's websocket
loop does, and instruments clarify_intent()/route_intent() with call counters
to verify the core fix this refactor was for: no stage upstream of an
interrupt gets re-run just because a LATER gate in the chain paused.

Requires real credentials in .env (GEMINI_API_KEY at minimum -- every scenario
below makes real LLM calls). By default, scenarios stop right after the
reasoning prelude (briefFinalizerAndBriefReview/cotFusionRemainder) and do NOT
proceed into actual
image generation / S3 upload, to keep this cheap and fast to re-run. Set
RUN_FULL_IMAGE_GEN=1 to also run the one full end-to-end scenario that does.

Usage:
    python3 graphic_designer/test_generation_pipeline.py
    RUN_FULL_IMAGE_GEN=1 python3 graphic_designer/test_generation_pipeline.py
"""

import asyncio
import json
import os
import sys
import uuid
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from graphic_designer.graph import graph
import graphic_designer.nodes as nodes

RUN_FULL_IMAGE_GEN = os.environ.get("RUN_FULL_IMAGE_GEN") == "1"

AUTO_CONTINUE_TYPES = (
    "request_classified", "generation_started", "editing_started",
    "design_suggestion_started", "expansion_started",
    "brand_conflict_warning", "edit_executability_warning",
)
# "layering_started" is answered, not auto-continued: it asks whether to
# detect the objects ("auto") or use bboxes the user drew ("manual").
# "critic_intent" (which replaced the auto-continued "critic_started" spinner)
# is answered too: it asks what the user plans to do with the image, and that
# answer is the brief the whole critique is judged against.

checkpointer = InMemorySaver()
# Full app -- lets the graph run all the way through real image generation +
# S3 upload. Only scenario 7 (RUN_FULL_IMAGE_GEN=1) should use this one.
app_full = graph.compile(checkpointer=checkpointer)
# Cheap app -- pauses right before "generation" (the model-dispatch node), so
# scenarios that only exercise the reasoning prelude (everything this session
# actually changed: intentClarifier..cotFusionRemainder) never reach a
# real vendor image-generation call or S3 upload. Scenarios 1-6 use this one.
app = graph.compile(checkpointer=checkpointer, interrupt_before=["generation"])


# ─────────────────────────────────────────────────────────────────────────
# Call-count instrumentation -- proves the decomposition actually stopped
# redundant re-computation across chained interrupts.
# ─────────────────────────────────────────────────────────────────────────
class CallCounter:
    def __init__(self):
        self.counts = {}

    @contextmanager
    def track(self, **fns_by_name):
        originals = {}
        for name, fn in fns_by_name.items():
            originals[name] = fn
            self.counts[name] = 0

            def make_wrapper(name, fn):
                def wrapper(*args, **kwargs):
                    self.counts[name] += 1
                    return fn(*args, **kwargs)
                return wrapper

            setattr(nodes, name, make_wrapper(name, fn))
        try:
            yield self
        finally:
            for name, fn in originals.items():
                setattr(nodes, name, fn)


counter = CallCounter()


# ─────────────────────────────────────────────────────────────────────────
# Base state + conversation-driving harness
# ─────────────────────────────────────────────────────────────────────────
def base_state(prompt, **overrides):
    state = {
        "username": "tester",
        "session_id": f"test-{uuid.uuid4().hex[:8]}",
        "user_id": "test-user",
        "prompt": prompt,
        "purpose": "Instagram post",
        "platform": None,
        "style": None,
        "current_image_url": None,
        "previous_image_url": None,
        "model": "nano-banana-pro",
        "num_images": 1,
        "aspect_ratio": "9:16",
        "attached_images": [],
        "brand_id": None,
        "brand_name": None,
        "brand_dna": {},
        "image_native_width": 0,
        "image_native_height": 0,
        "design_goal": "Festive Sale",
        "platform_canonical": "Instagram Square Post",
    }
    state.update(overrides)
    return state


def show_interrupt(iv):
    print(f"  INTERRUPT type={iv.get('type')!r} skippable={iv.get('skippable')}")
    for k, v in iv.items():
        if k in ("type", "cost"):
            continue
        text = str(v)
        if len(text) > 160:
            text = text[:160] + "..."
        print(f"    {k}: {text}")


def default_responder(iv, occurrence_index):
    """Best-effort generic answer for whatever a text_input interrupt actually
    asks -- content-sniffed from the payload rather than a fixed position, so
    scenarios don't break just because an LLM classifier's exact judgment
    (e.g. whether a prompt counts as "ambiguous") varies run to run."""
    # generation_completed's resume contract (interruptBeforeFidelitySuggestion,
    # nodes.py) does a *direct* dict-key read of model/purpose/aspect_ratio with
    # no default -- omitting them raises KeyError, not a graceful fallback. Real
    # frontends always echo these back, so the harness must too.
    if iv.get("type") == "layering_started":
        return {"response": "auto"}

    if iv.get("type") == "critic_intent":
        return {"response": ("A 30% off Diwali sale post for our Instagram, aimed at "
                             "existing customers of a home decor brand.")}

    if iv.get("type") == "generation_completed":
        return {"response": "no", "model": "nano-banana-pro", "purpose": "Instagram post",
                "aspect_ratio": "9:16"}

    question = iv.get("question")
    reason = str(iv.get("reason", ""))
    q_text = " ".join(question) if isinstance(question, list) else str(question or "")
    blob = (q_text + " " + reason).lower()

    if "logo" in blob:
        return {"response": "no"}  # decline -- keep deterministic/cheap by default
    if "photo" in blob or "image" in blob:
        return {"response": ""}  # skip -- let it AI-generate
    if "clarification_questions" in iv or isinstance(question, list):
        return {"response": (
            "It's a 30% off Diwali sale for a home decor brand called Lumière Home, "
            "ending November 15th. Apply at lumierehome.com/careers if relevant."
        )}
    if "proceed with what you asked for instead" in blob or "detected_goal" in iv:
        return {"response": "yes"}
    return {"response": "yes"}


def types_of(interrupt_log):
    return [iv.get("type") for iv in interrupt_log]


def mentions(interrupt_log, keyword, interrupt_type="text_input"):
    """True if any interrupt of the given type had `keyword` in its question/reason."""
    kw = keyword.lower()
    for iv in interrupt_log:
        if iv.get("type") != interrupt_type:
            continue
        question = iv.get("question")
        q_text = " ".join(question) if isinstance(question, list) else str(question or "")
        blob = (q_text + " " + str(iv.get("reason", ""))).lower()
        if kw in blob:
            return True
    return False


async def drive(thread_id, initial_state, responder=default_responder, canned_responses=None,
                 max_steps=15, app_instance=None):
    """Invoke the graph, auto-continuing progress pings. For each genuine
    blocking interrupt: if `canned_responses` still has entries, pop and use
    the next one (exact positional control when a test needs it); otherwise
    fall back to `responder(interrupt_value, occurrence_index)` to answer
    generically based on the interrupt's actual content. Returns
    (final_result, interrupt_log) where interrupt_log is the list of full
    interrupt *value dicts* actually seen, in order (use types_of()/mentions()
    to inspect it).

    `app_instance` defaults to the cheap `app` (paused before "generation" --
    never reaches real image generation/S3 upload). Pass `app_full` explicitly
    for the one scenario that intentionally runs all the way through."""
    app_instance = app_instance or app
    config = {"configurable": {"thread_id": thread_id}}
    result = await app_instance.ainvoke(initial_state, config=config)
    interrupt_log = []
    responses = list(canned_responses or [])
    occurrence = 0

    for _ in range(max_steps):
        if "__interrupt__" not in result or not result["__interrupt__"]:
            return result, interrupt_log
        iv = result["__interrupt__"][0].value
        interrupt_log.append(iv)
        show_interrupt(iv)

        if iv.get("type") in AUTO_CONTINUE_TYPES:
            result = await app_instance.ainvoke(Command(resume=json.dumps({"response": ""})), config=config)
            continue

        occurrence += 1
        if responses:
            payload = responses.pop(0)
        else:
            payload = responder(iv, occurrence)
        print(f"  -> resuming with: {payload}")
        result = await app_instance.ainvoke(Command(resume=json.dumps(payload)), config=config)

    return result, interrupt_log


async def get_state_snapshot(thread_id, keys, app_instance=None):
    app_instance = app_instance or app
    config = {"configurable": {"thread_id": thread_id}}
    snap = await app_instance.aget_state(config)
    return {k: snap.values.get(k) for k in keys}


PIPELINE_STAGE_KEYS = [
    "clarify_result", "router_flags", "selected_variant_dict", "selected_variant",
    "fused_prompt", "negative_prompt", "merge_directive", "photo_mode",
    "logo_mode", "logo_insertion_mode", "logo_directive", "is_freeform",
]


def print_snapshot(snap):
    for k, v in snap.items():
        text = str(v)
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"    {k}: {text}")


PASS = []
FAIL = []


def check(label, condition, detail=""):
    if condition:
        PASS.append(label)
        print(f"  [PASS] {label}")
    else:
        FAIL.append(label)
        print(f"  [FAIL] {label}  {detail}")


# ─────────────────────────────────────────────────────────────────────────
# Scenario 1: happy path -- clear prompt, no photo, no logo, no clarification
# ─────────────────────────────────────────────────────────────────────────
async def scenario_happy_path():
    print("\n" + "=" * 70)
    print("SCENARIO 1: happy path (clear prompt, no photo, no logo)")
    print("=" * 70)
    thread_id = str(uuid.uuid4())
    state = base_state(
        "A vibrant 50% off Diwali sale banner for a home decor brand, with "
        "warm gold and maroon tones, festive lanterns, and a bold 'SHOP NOW' "
        "call to action. Sale ends November 15th.",
    )
    with counter.track(clarify_intent=nodes.clarify_intent, route_intent=nodes.route_intent):
        result, log = await drive(thread_id, state, max_steps=10)
        print(f"  interrupt sequence: {types_of(log)}")
        print(f"  clarify_intent calls: {counter.counts['clarify_intent']}, "
              f"route_intent calls: {counter.counts['route_intent']}")

        # NOTE: whether clarification/logo interrupts fire at all depends on the
        # LLM classifier's own judgment call (not something we force), so this
        # scenario doesn't assert zero interrupts -- it asserts the flow always
        # reaches a fused prompt, and that however many gates fired, the two
        # expensive stages were still each called at most twice (once + at most
        # one re-ask if clarification happened to trigger).
        check("clarify_intent called at most twice (initial + at most one re-ask)",
              counter.counts["clarify_intent"] <= 2, counter.counts["clarify_intent"])
        check("route_intent called at most once (never re-run by a later gate)",
              counter.counts["route_intent"] <= 1, counter.counts["route_intent"])

    snap = await get_state_snapshot(thread_id, PIPELINE_STAGE_KEYS)
    print("  final state snapshot:")
    print_snapshot(snap)
    check("fused_prompt populated", bool(snap.get("fused_prompt")))
    return thread_id


# ─────────────────────────────────────────────────────────────────────────
# Scenario 2: clarification needed -- vague/ambiguous prompt
# ─────────────────────────────────────────────────────────────────────────
async def scenario_clarification_needed():
    print("\n" + "=" * 70)
    print("SCENARIO 2: clarification needed (vague prompt)")
    print("=" * 70)
    thread_id = str(uuid.uuid4())
    state = base_state("make it look nice and add a discount")

    with counter.track(clarify_intent=nodes.clarify_intent, route_intent=nodes.route_intent):
        result, log = await drive(
            thread_id, state,
            canned_responses=[
                {"response": "It's a 30% off Diwali sale for a home decor brand, ending Nov 15th."},
            ],
            max_steps=12,
        )
        print(f"  interrupt sequence: {types_of(log)}")
        print(f"  clarify_intent calls: {counter.counts['clarify_intent']} "
              "(expect 2: one initial + one re-ask after the user answers -- NOT 3+)")
        print(f"  route_intent calls: {counter.counts['route_intent']} (expect exactly 1)")

        check("clarificationGate fired at least once", "text_input" in types_of(log), types_of(log))
        check("clarify_intent called at most twice (initial + one re-ask, no extra redundant calls)",
              counter.counts["clarify_intent"] <= 2, counter.counts["clarify_intent"])
        check("route_intent called exactly once (not re-run just because clarification paused)",
              counter.counts["route_intent"] == 1, counter.counts["route_intent"])

    snap = await get_state_snapshot(thread_id, PIPELINE_STAGE_KEYS)
    print("  final state snapshot:")
    print_snapshot(snap)
    check("fused_prompt reflects the clarified answer (non-empty)", bool(snap.get("fused_prompt")))
    return thread_id


# ─────────────────────────────────────────────────────────────────────────
# Scenario 3: photo required -- force layout_profile.shared_rules.real_photo_required
# ─────────────────────────────────────────────────────────────────────────
async def scenario_photo_required():
    print("\n" + "=" * 70)
    print("SCENARIO 3: photo required (forced via monkeypatched layout_profile)")
    print("=" * 70)
    thread_id = str(uuid.uuid4())
    state = base_state("A clean, elegant sale banner featuring our signature candle.")

    original_ensure = nodes.ensure_layout_assets

    def forced_ensure_layout_assets(*args, **kwargs):
        profile, wireframes = original_ensure(*args, **kwargs)
        profile = dict(profile)
        profile["shared_rules"] = dict(profile.get("shared_rules", {}))
        profile["shared_rules"]["real_photo_required"] = True
        profile["shared_rules"]["image_requirement_reason"] = (
            "TEST: forced real-photo requirement for photoRequiredGate testing."
        )
        return profile, wireframes

    nodes.ensure_layout_assets = forced_ensure_layout_assets
    try:
        with counter.track(clarify_intent=nodes.clarify_intent, route_intent=nodes.route_intent):
            result, log = await drive(
                thread_id, state,
                canned_responses=[
                    {"response": ""},  # skip -- let it AI-generate the photo zone
                ],
                max_steps=12,
            )
            print(f"  interrupt sequence: {types_of(log)}")
            print(f"  clarify_intent calls: {counter.counts['clarify_intent']}, "
                  f"route_intent calls: {counter.counts['route_intent']}")

            check("photoRequiredGate fired (real photo required, none attached)",
                  "text_input" in types_of(log), types_of(log))
            check("clarify_intent called at most once (photo gate doesn't force a re-clarify)",
                  counter.counts["clarify_intent"] <= 1, counter.counts["clarify_intent"])
            check("route_intent called at most once", counter.counts["route_intent"] <= 1,
                  counter.counts["route_intent"])
    finally:
        nodes.ensure_layout_assets = original_ensure

    snap = await get_state_snapshot(thread_id, PIPELINE_STAGE_KEYS)
    print("  final state snapshot:")
    print_snapshot(snap)
    check("fused_prompt populated despite two chained interrupts (photo + clarification)",
          bool(snap.get("fused_prompt")))
    return thread_id


# ─────────────────────────────────────────────────────────────────────────
# Scenario 4: logo decision -- no brand kit logo, layout has a logo zone
# ─────────────────────────────────────────────────────────────────────────
async def scenario_logo_decision_ask():
    print("\n" + "=" * 70)
    print("SCENARIO 4: logo decision -- ad hoc ask (no brand kit logo)")
    print("=" * 70)
    thread_id = str(uuid.uuid4())
    state = base_state(
        "A festive 40% off sale graphic for our candle shop, warm tones, elegant typography, "
        "brand name Lumière Home, no discount code needed.",
        brand_dna={},  # no logo on file
    )
    # default_responder declines any "logo" ask and answers clarification generically,
    # so we don't need to guess how many (or which order) interrupts will fire.
    result, log = await drive(thread_id, state, max_steps=12)
    print(f"  interrupt sequence: {types_of(log)}")
    check("logoDecisionGate fired (layout has a logo zone, no brand-kit logo)",
          mentions(log, "logo"), types_of(log))

    snap = await get_state_snapshot(thread_id, PIPELINE_STAGE_KEYS)
    print("  final state snapshot:")
    print_snapshot(snap)
    check("logo_mode == 'omitted' after declining", snap.get("logo_mode") == "omitted",
          snap.get("logo_mode"))
    return thread_id


async def scenario_logo_decision_brand_kit():
    print("\n" + "=" * 70)
    print("SCENARIO 5: logo decision -- brand-kit logo present (should auto-skip the ask)")
    print("=" * 70)
    thread_id = str(uuid.uuid4())
    # A small 1x1 red PNG as a stand-in "logo" data URI. In production
    # brand_dna["logo"] always holds real S3 URLs (confirmed in main.py's brand
    # onboarding), and convert_urls_to_b64() correctly only fetches real HTTP(S)
    # URLs -- so for this test we patch it to pass a data URI through unchanged,
    # rather than weakening the real function to accept non-production input.
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    logo_data_uri = f"data:image/png;base64,{tiny_png_b64}"
    state = base_state(
        "A festive 40% off sale graphic for our candle shop, warm tones, elegant typography.",
        brand_dna={"logo": [logo_data_uri]},
    )

    original_convert = nodes.convert_urls_to_b64

    def fake_convert_urls_to_b64(urls):
        return [u if u.startswith("data:") else original_convert([u])[0] for u in urls]

    nodes.convert_urls_to_b64 = fake_convert_urls_to_b64
    try:
        result, log = await drive(thread_id, state, max_steps=10)
    finally:
        nodes.convert_urls_to_b64 = original_convert

    print(f"  interrupt sequence: {types_of(log)}")
    check("logoDecisionGate did NOT fire (brand-kit logo auto-resolved, no ask needed)",
          not mentions(log, "logo"), types_of(log))

    snap = await get_state_snapshot(thread_id, PIPELINE_STAGE_KEYS + ["logo_asset"])
    print("  final state snapshot:")
    print_snapshot({k: v for k, v in snap.items() if k != "logo_asset"})
    print(f"    logo_asset: {'<present, len=' + str(len(snap.get('logo_asset') or '')) + '>'}")
    check("logo_mode == 'provided' via brand kit (no ask fired)",
          snap.get("logo_mode") == "provided", snap.get("logo_mode"))
    return thread_id


# ─────────────────────────────────────────────────────────────────────────
# Scenario 6: scope mismatch -- prompt clearly implies a different goal
# ─────────────────────────────────────────────────────────────────────────
async def scenario_scope_mismatch():
    print("\n" + "=" * 70)
    print("SCENARIO 6: scope mismatch (prompt implies a different design goal)")
    print("=" * 70)
    thread_id = str(uuid.uuid4())
    state = base_state(
        "We're hiring a senior backend engineer! Join our growing team, apply today.",
        design_goal="Festive Sale",
        platform_canonical="Instagram Square Post",
        aspect_ratio="9:16",
    )
    result, log = await drive(
        thread_id, state,
        canned_responses=[{"response": "yes"}],  # accept the detected goal
        max_steps=10,
    )
    print(f"  interrupt sequence: {types_of(log)}")
    check("scopeGate fired a mismatch interrupt", "text_input" in types_of(log), types_of(log))
    return thread_id


# ─────────────────────────────────────────────────────────────────────────
# Scenario 7 (optional, costs real money): full run through actual image gen
# ─────────────────────────────────────────────────────────────────────────
async def scenario_full_image_generation():
    print("\n" + "=" * 70)
    print("SCENARIO 7: FULL end-to-end through real image generation + S3 upload")
    print("=" * 70)
    thread_id = str(uuid.uuid4())
    state = base_state(
        "A vibrant 50% off Diwali sale banner for a home decor brand, warm gold "
        "and maroon tones, festive lanterns, bold 'SHOP NOW' CTA.",
        brand_dna={},
    )
    result, log = await drive(thread_id, state, max_steps=12, app_instance=app_full)
    print(f"  interrupt sequence: {types_of(log)}")
    check("reached generation_completed (image actually generated)",
          "generation_completed" in types_of(log), types_of(log))
    snap = await get_state_snapshot(thread_id, ["current_image_url", "cost"], app_instance=app_full)
    print(f"    current_image_url: {snap.get('current_image_url')}")
    print(f"    cost: {snap.get('cost')}")
    check("current_image_url populated with a real URL", bool(snap.get("current_image_url")))
    return thread_id


async def main():
    print(f"RUN_FULL_IMAGE_GEN={RUN_FULL_IMAGE_GEN}")
    await scenario_happy_path()
    await scenario_clarification_needed()
    await scenario_photo_required()
    await scenario_logo_decision_ask()
    await scenario_logo_decision_brand_kit()
    await scenario_scope_mismatch()
    if RUN_FULL_IMAGE_GEN:
        await scenario_full_image_generation()
    else:
        print("\n(Skipping scenario 7 -- full image generation. Set RUN_FULL_IMAGE_GEN=1 to include it.)")

    print("\n" + "=" * 70)
    print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 70)
    if FAIL:
        print("FAILED CHECKS:")
        for f in FAIL:
            print(f"  - {f}")
    return len(FAIL) == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
