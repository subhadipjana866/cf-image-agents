# generation_pipeline — retired (helper modules only)

The v5.4 pipeline this folder documented (Avenue 1's offline layout factory,
Avenue 2's 7-stage multicall CoT fusion, the wireframe store and matcher, and
the per-model render dispatch) is gone. Generation now runs in
`graphic_designer/kognie-generation-v33/` through `graphic_designer/nodes.py`'s
`v32Generation` node: GROUND → SUBJECT ∥ COPYWRITER → TYPOGRAPHER / ART
DIRECTOR / COMPOSITOR → DIRECTOR → render, with the design goal, platform,
aspect ratio, content fields and composition menu all read from that engine's
`layout_guidance.json` (see `graphic_designer/v33_bridge.py`).

What is left here are the stage functions the surviving interactive gates still
import, plus the LLM fallback helper they call:

| File | Still used by |
|---|---|
| `avenue2_generation_pipeline/v5_4_scope_check.py` | `scopeGate` (`check_request_scope`), the goal/platform catalog (`load_catalog`, `platform_ratio`) |
| `avenue2_generation_pipeline/v5_4_intent_clarifier.py` | `intentClarifier` (`clarify_intent`) |
| `avenue2_generation_pipeline/v5_3_intent_router.py` | `intentRouter` (`route_intent`) |
| `avenue2_generation_pipeline/v5_4_fabrication_check.py` | `clarificationGate` (`at_risk_missing_fields`, `build_fabrication_clarification_questions`) |
| `avenue2_generation_pipeline/v5_4_logo_handler.py` | `logoDecisionGate` (`variant_logo_zone`, `build_logo_directive`) |
| `avenue2_generation_pipeline/llm_fallback.py` | the four modules above, for their provider fallback |
| `avenue2_generation_pipeline/phase0_combinations.json` | `v5_4_scope_check`'s catalog fallback |
| `avenue2_generation_pipeline/style_profiles/` | the three legacy prose style profiles, read only by `graphic_designer/style_experiments/`. Production style direction comes from `graphic_designer/styles.json` instead (see `graphic_designer/style_presets.py`). |

`nodes.py` puts `avenue2_generation_pipeline/` on `sys.path` at import time,
because these modules import each other as flat siblings.

`COT_COMPARISON_RESULTS.md` is kept as the record of the single-call vs
multicall CoT comparison that the retired fusion came out of.
