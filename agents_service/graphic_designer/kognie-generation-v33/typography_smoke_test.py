"""
typography_smoke_test.py -- end-to-end check of the typography integration on the
real v3.2 pipeline (all nodes, real render, text QA). Writes typography_e2e/.

    python typography_smoke_test.py            # E1-E5
    python typography_smoke_test.py E1 E4      # a subset
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import generation_pipeline as pipeline
from create_more import create_more

OUT = Path(__file__).resolve().parent / "out" / "typography_smoke"
# Any image with clear lettering works as the "font" reference; the bundled Amatic SC
# sample stands in for a user upload here.
FONT_IMG = Path(__file__).resolve().parent / "typography_assets/samples/amatic_sc.png"

CASES = {
    "E1": dict(goal="Event Announcement", platform="Instagram Portrait Post",
               raw_prompt="Summer Jazz Night at Blue Harbor Cafe, Friday 12 July at 8 PM, free entry. "
                          "Use Pacifico for the title."),
    "E2": dict(goal="Product Launch", platform="LinkedIn Feed Post",
               raw_prompt="We just launched Insights Studio, AI analytics for small retail teams. "
                          "Available now. Use Helvetica Neue for all the text."),
    "E3": dict(goal="Brand Awareness / Quote Post", platform="Instagram Story",
               raw_prompt="Just a calm abstract image of soft peach and lavender waves for our story "
                          "background. No text at all."),
    "E4": dict(goal="Grand Opening", platform="Instagram Square Post",
               raw_prompt="Grand opening of Willow Bakery on Saturday 3 May, 9 AM. Free coffee for the "
                          "first 50 guests. Take the lettering style from my attached image.",
               reference_images=[(FONT_IMG, "font")]),
}


def summarise(name, r):
    row = {"case": name, "status": r.get("status"), "image": r.get("image"),
           "rendered_by": (r.get("render") or {}).get("rendered_by"),
           "wall_s": r.get("wall_s"), "cost": r.get("cost_total")}
    t = r.get("typography") or {}
    row["fonts"] = t.get("fonts_used")
    row["notices"] = t.get("user_notices")
    row["policy_log"] = t.get("policy_log")
    tl = r.get("text_layers") or {}
    row["elements"] = [{"id": e["id"], "role": e["role"], "text": e["text"], "font": e["font_family"],
                        "size": e["font_size_px"], "colour": e["color_hex"],
                        "qa": e["qa"].get("status"), "source": e["qa"].get("source"),
                        "reg": {k: e["qa"]["registration"][k] for k in ("dx", "dy", "scale", "score")}
                        if e["qa"].get("registration") else None,
                        "contrast_after": (e["contrast"]["after_render"] or {}).get("ratio"),
                        "contrast_target": (e["contrast"]["plan"] or {}).get("target")}
                       for e in tl.get("elements", [])]
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1:] or ["E1", "E2", "E3", "E4", "E5"]
    rows, results = [], {}
    for name in [w for w in want if w in CASES]:
        t0 = time.time()
        events = []
        r = pipeline.run(out_dir=OUT / name, on_event=events.append, **CASES[name])
        results[name] = r
        (OUT / f"{name}.events.json").write_text(json.dumps(events, indent=1, default=str), encoding="utf-8")
        row = summarise(name, r)
        rows.append(row)
        print(json.dumps(row, indent=1, ensure_ascii=False, default=str), flush=True)
    if "E5" in want:
        base = results.get("E1")
        if base is None:
            prev = sorted((OUT / "E1").glob("*.json"))
            prev = [p for p in prev if not p.name.endswith(".text_layers.json")]
            base = json.loads(prev[-1].read_text(encoding="utf-8")) if prev else None
        if base and base.get("status") == "ok":
            m = create_more(base, out_dir=OUT / "E5",
                            lineage=[{"archetype": "original", "style_signature": "as first generated"}])
            row = summarise("E5", m)
            row["archetype"] = m.get("chosen_archetype")
            rows.append(row)
            print(json.dumps(row, indent=1, ensure_ascii=False, default=str), flush=True)
    (OUT / f"summary_{int(time.time())}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
