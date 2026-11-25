"""
example_run.py — first generation, then one "create more" on top of it.

    python example_run.py

Needs GEMINI_API_KEY and OPENAI_API_KEY in the environment. Writes into ./out.
"""

from pathlib import Path

import generation_pipeline as pipeline
from create_more import create_more


def main():
    first = pipeline.run(
        raw_prompt=(
            "Hone & Whetstone is running a Japanese knife sharpening workshop "
            "on Sunday 9 November, 10am to 1pm, at the Kirkgate workshop. "
            "GBP 85 including materials, twelve places only."),
        goal="Workshop / Masterclass Promo",
        platform="Instagram Portrait Post",
        out_dir=Path("out/first"),
        on_event=pipeline._live,
    )
    print(f"\n  first    {first['wall_s']}s  ${first['cost_total']:.4f}  "
          f"{first.get('image')}")

    # One variation. The lineage starts empty; entry 0 is the original.
    lineage = [{"archetype": "original",
                "style_signature": "as first generated"}]
    more = create_more(first, out_dir=Path("out/more1"), lineage=lineage)
    print(f"  more #1  {more['wall_s']}s  ${more['cost_total']:.4f}  "
          f"{more.get('image')}")
    print(f"           composition {more['chosen_archetype']}  "
          f"exhausted={more['exhausted']}")


if __name__ == "__main__":
    main()
