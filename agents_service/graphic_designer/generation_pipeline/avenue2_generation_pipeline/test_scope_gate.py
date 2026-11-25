"""Validates the Phase 1.0 scope gate (v5_4_scope_check + generate_graphic wiring).

Cases:
  1. TATTOO deliverable conflict (unsupported)      -> goal_conflict, not supported
  2. Tattoo PARLOR as subject (same deliverable)    -> match (no conflict)
  3. Wendy's fully-detailed multi-platform prompt   -> platform_conflict + multi_platform
                                                        (this silently PROCEEDed before the gate)
  4. Supported goal switch (grand opening)          -> goal_conflict, supported=True
  5. FULL RUN: tattoo prompt + user says "yes"      -> freeform bypass, real image generated,
                                                        no layout / no CoT
"""
import sys, json
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from v5_4_scope_check import check_request_scope

def show(label, res):
    keys = ("match", "goal_conflict", "detected_goal", "detected_goal_supported",
            "platform_conflict", "multi_platform", "detected_platforms",
            "primary_platform", "pending_platforms", "unsupported_platforms")
    print(f"\n=== {label} ===")
    print(json.dumps({k: res[k] for k in keys}, indent=2))

r1 = check_request_scope(
    "A tattoo of a dragon wrapped around a skull, black and grey realism",
    "Festive Sale", "Instagram Portrait Post")
show("1. TATTOO deliverable (expect goal_conflict, unsupported)", r1)

r2 = check_request_scope(
    "Christmas sale at my tattoo parlor, 20% off all bookings in December. Book now!",
    "Festive Sale", "Instagram Portrait Post")
show("2. Tattoo PARLOR subject (expect match=True)", r2)

r3 = check_request_scope(
    "Wendy's Christmas Sale -- 25% off all combos, valid Dec 1 to Dec 25, use code SANTA25. "
    "Order now at wendys.com. Make it for instagram vertical, facebook square both.",
    "Festive Sale", "Pinterest Pin")
show("3. Multi-platform vs Pinterest (expect platform_conflict + multi)", r3)

r4 = check_request_scope(
    "Announce the grand opening of our new bakery on August 5th, free croissants for the first 50 customers",
    "Festive Sale", "Instagram Portrait Post")
show("4. Grand opening (expect goal_conflict, supported=True)", r4)

# ── 5. Full end-to-end freeform bypass ────────────────────────────────────
print("\n=== 5. FULL RUN: tattoo prompt, user confirms detected intent -> freeform bypass ===")
from kognie_generation_pipeline import generate_graphic

def yes_to_detected(payload):
    print("  [INTERRUPT scope_mismatch] ->", payload["message"])
    print("  [USER] yes, proceed with what I asked for")
    return {"proceed_with_detected": True}

out = generate_graphic(
    design_goal="Festive Sale", platform="Instagram Portrait Post", aspect_ratio="4:5",
    prompt="A tattoo of a dragon wrapped around a skull, black and grey realism",
    output_dir=Path("results/scope_gate_freeform"),
    on_scope_mismatch=yes_to_detected,
)
print(f"  mode={out.mode}  image_ok={out.image_ok}  pipeline={'None' if out.pipeline is None else 'PipelineResult'}")
print(f"  final_goal={out.final_design_goal!r}  final_platform={out.final_platform!r}")
print(f"  timings={out.timings}")
assert out.mode == "freeform" and out.pipeline is None, "expected freeform bypass!"
print("\nALL SCOPE GATE CHECKS COMPLETE")
