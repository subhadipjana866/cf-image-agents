# Avenue 2 — retired

The live generation path described here (`kognie_generation_pipeline.py` →
`v5_4_pipeline_api.run_generation_pipeline()` → CoT fusion → render) has been
replaced by the v3.3 engine in `graphic_designer/kognie-generation-v33/`. See
`../README.md` for what survives in this folder and which node still imports
each file.

Still here:

| File | Role |
|---|---|
| `v5_4_scope_check.py` | Pre-flight scope gate (deliverable/platform mismatch) and the goal/platform catalog helpers. |
| `v5_4_intent_clarifier.py` | Structures the raw prompt and extracts `missing_fields` against the combination's field schema. |
| `v5_3_intent_router.py` | Routing action, ambiguity flags, photo/layout intent detection. |
| `v5_4_fabrication_check.py` | Deterministic filter over `missing_fields` for fabrication-prone gaps (price/date/address). No LLM call. |
| `v5_4_logo_handler.py` | Logo zone resolution and the `single_call` logo directive. |
| `llm_fallback.py` | Provider fallback chain used by the modules above. |
| `phase0_combinations.json` | Catalog fallback for `v5_4_scope_check`. |
| `style_profiles/` | Three legacy prose style profiles, read only by `graphic_designer/style_experiments/`. |
| `test_scope_gate.py` | Offline tests for the scope gate. |

Deleted with the pipeline: `kognie_generation_pipeline.py`,
`v5_4_pipeline_api.py`, `v5_4_fusion_augmenter.py`, `v5_4_cot_multicall.py`,
`v5_4_image_analyzer.py`, `v5_4_layout_matcher.py`, `v5_3_layout_profile_generator.py`,
`v5_3_wireframe_renderer.py`, `layout_store.py`, `layout_requirements.json`,
`image_generation.py`, `pipeline_stages.py`, `cot_comparison_experiment.py`, and
the `assets/`, `brand_profiles/` and `layout_store/` folders.
