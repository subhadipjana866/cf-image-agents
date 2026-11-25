# critiquepipeline: overview and integration guide

**Version:** v2, Sep 2026. It replaces the `critiquepipeline/` currently in production.
**Read first:** this file. Read `DEVELOPER_GUIDE.md` for depth, especially §10: every change
below is listed there together with the live-test failure that caused it.

---

## 1. What it does

It looks at a generated or uploaded graphic and returns **0 to 3 suggestions** for the user to
accept or reject. Every suggestion has to meet three bars:

- it is a real problem (not taste, not filler),
- it makes a meaningful difference to the graphic, and
- it is **AI-executable**: the image-edit model (GPT-Image-2.5 Flare) can actually carry it out.

A clean graphic correctly gets zero suggestions. The list is never padded to reach 3.

## 2. What changed since the production version

| Area | Change | Why (live evidence) |
|---|---|---|
| Executability | New `executabilityGate`: code rules plus one text-only LLM judge. It runs on the ranked shortlist **before** the top-3 pick, so a dropped item is replaced by the next-best one. | Suggestions the edit model can't do were reaching users. |
| Lens scope | Composition and aesthetic can no longer *add* content. Deadline and offer lines are never removed as "redundant". Both are enforced in code (`_outOfScope`). | "Add a Shop Now button" when the URL was already there; "remove THIS WEEKEND ONLY" on a flash sale. |
| User text | The text the user reads must be an instruction that matches the edit (`_userText`). The lens's own text wins; the arbiter's rewrite is only a fallback. | The user was told "ensure this is updated with booking info", but the edit deleted the line. |
| Duplicates | Removal instructions must name the copy to **keep**, and the compiler carries that through word for word. The same text fix flagged under different labels now merges and gets its corroboration boost. | "Remove the duplicate CTAs" erased every CTA; a finding flagged by four lenses lost its slot. |
| Hallucinations | Hand and face claims need a countable defect. "Missing" claims must list the visible text first. | Invented "malformed hands"; false "missing CTA". |
| Re-critique | A compiled edit prompt is never treated as the image's brief (`nodes._isEditInstruction`). | Re-critiques re-reported defects the edit had already fixed. |
| `verifyEdit` | New `pending_units=` argument for step-by-step editing. It only blames an edit for problems that edit introduced. | It was blaming step 1 for issues step 2 was about to fix. |
| Edit quality | New `editQuality(units)`: `"high"` for edits that re-render letters, `"medium"` otherwise. | Correcting "wirless": 1 of 6 at medium ("wiroless", "wirıless", …), 2 of 2 at high. |
| Brand | `brandNormalizer` reads the live `fetch_brand_dna` shape; one `BRAND_BLOCK` goes into every lens and the arbiter; each recommendation carries `brand_check`. Lenses also flag clear brand drift (colour, voice, mood). | Live brand kits normalized to empty, so critiques were brand-blind and could suggest off-brand colours. |
| Tests | `tests/test_offline.py`: 18 regression tests, no keys or network needed. | Each pins a real failure mode. |

**Unchanged:**
- the `suggestionCreator(state)` signature
- the `s_stack` shape the frontend receives
- `MAX_SUGGESTIONS = 3`
- the models already in use

Everything downstream keeps working without changes. The executor changes in §5 are opt-in
improvements.

## 3. Flow

```
suggestionCreator(state)                                        [nodes.py]
  ├─ (drop base_prompt if it is a compiled edit instruction)
  ├─ resolveGrounding()        layout + brand, or "generic" mode       [grounding.py]
  ├─ 4 lenses in parallel      integrity · composition · content · aesthetic
  ├─ arbiterAgent()                                                     [agents.py]
  │    ├─ _applyScope          code: lens scope violations out
  │    ├─ _predupe             merge the same defect across lenses
  │    ├─ LLM arbiter          ranks a shortlist of up to 7
  │    ├─ _fixUserText         user text = instruction matching the edit
  │    ├─ executabilityGate    rules + one LLM judge: can Flare do this?
  │    └─ _curate → top 3      caps, quality floor, text fixes first, backfill
  ├─ techAgent()               existing agent (subgraphs/suggestion.py)
  └─ structureSuggestions()    s_stack for the frontend

accept ─► executor (suggestionExecutors, outside this package)
            compilerAgent → Flare edit (quality = editQuality) → verifyEdit
```

## 4. Files

| File | What it holds |
|---|---|
| `nodes.py` | Entry point `suggestionCreator`, `MAX_SUGGESTIONS`, `structureSuggestions` |
| `agents.py` | Every LLM call and every code guard: lenses, arbiter, gate, scope, user text, compiler, `editQuality`, `verifyEdit`. Model config is at the top. |
| `prompts.py` | Every prompt; most behaviour tuning happens here. `EXECUTABILITY_PROMPT` is the Flare rubric. |
| `grounding.py` | Layout and brand loading (reads the live `fetch_brand_dna` shape), generic vs grounded mode |
| `telemetry.py` | Cost, time, and notes (unchanged) |
| `tests/test_offline.py` | Offline regression tests |
| `DEVELOPER_GUIDE.md` | Deep reference; §10 covers the live test and its fixes |

## 5. Integration checklist

**Step 1: swap the package.**
1. Back up the current `graphic_designer/critiquepipeline/`.
2. Replace it with this folder, and delete any `__pycache__/`.
3. The relative imports (`..config`, `..utils`, `..subgraphs.suggestion`) are the same as before.

**Step 2: run the offline tests.** Run from anywhere; no keys or network needed:
```
python graphic_designer/critiquepipeline/tests/test_offline.py
```
All 18 should pass.

**Step 3: check `techAgent`.** It lives in `subgraphs/suggestion.py`, outside this package, and it
runs **after** the top-3 cut. If it drops or rewrites suggestions, nothing replaces them. Confirm
that it only adds metadata and passes the recommendations through unchanged.

**Step 4: executor changes in `suggestionExecutors`** (outside this package).
- **a. Model.** Use `gpt-image-2.5-flare` for edits, if it isn't already.
- **b. Quality.** Pass `quality=editQuality(accepted_units)` to the edit call:
  ```python
  from .critiquepipeline import compilerAgent, editQuality, verifyEdit
  ```
- **c. Wire `verifyEdit` after every edit.** It is still not a graph node. On `"regressed"`,
  flag the problem to the user; never auto-revert silently.
- **d. Optional: sequential execution.** It applies one accepted suggestion per edit call.
  It is **not** more accurate in our tests (see §7), but it makes every failure traceable to one
  step and lets you stop or retry just that step. It doubles edit cost and time, so this is a
  product decision:
  ```python
  cur = image
  for i, unit in enumerate(accepted):
      comp = compilerAgent(purpose, [unit], tele)
      after = run_flare_edit(cur, comp["execution_prompt"], quality=editQuality([unit]))
      ver = verifyEdit(cur, after, [unit], tele, pending_units=accepted[i + 1:])
      if ver.get("overall") == "regressed":
          ...                      # surface to the user; keep going or stop, per product call
      cur = after
  ```
  The current fused path (`compilerAgent(purpose, accepted, tele)`, then one edit) still works
  unchanged.

**Step 5: check in staging.** Run 2–3 real graphics, then check these telemetry notes in
`critique_telemetry["notes"]`:

| Note | What it tells you |
|---|---|
| `not_executable` | What the gate dropped, and why |
| `out_of_scope` | What the lens-scope guard dropped |
| `curation_dropped` | What the caps, floor, and top-3 cut dropped |
| `user_text_source` | Where the user text came from: `original`, `reword`, `nano_prompt`, or `unfixed`. `unfixed` should be rare. |
| `suggestion_counts` | Raw findings vs. shown. If "shown" is always 3, the quality floor is too loose. |

## 6. Config knobs

| Where | Knob | Default |
|---|---|---|
| `nodes.py` | `MAX_SUGGESTIONS` | `3` (a ceiling, not a target) |
| `agents.py` | `LENS_MODEL`, `ARBITER_MODEL`, `EXEC_MODEL` | `gemini-3-flash-preview` |
| `agents.py` | `COMPILER_MODEL` | `gemini-3.1-flash-lite` |
| `agents.py` | `MIN_DISPLAY_CONFIDENCE` / `MIN_AESTHETIC_CONFIDENCE` | `0.55` / `0.7` |
| `agents.py` | `_RETYPE_RE` | Which edits get `quality="high"` |
| `prompts.py` | `EXECUTABILITY_PROMPT` | The Flare capability rubric. **Update it when the edit model changes.** |

## 7. Measured (Sep 2026 live runs: Kognie → critique → Flare)

**Detection.** All planted defects were found and ranked #1–#2 in every run (3 graphics, 6 defects).

**Execution, final run (G2 + G3).** Every step was clean on both the sequential and fused paths,
and every final image was checked by eye.

**Sequential vs fused, across all 9 graphic runs:**

| Path | Graphics fully clean | Cost |
|---|---|---|
| Sequential | 6 of 9 | About 2× the edit time and money |
| Fused | 5 of 9 | Baseline |

The only difference was a one-word spelling fix, the small-text problem `quality="high"` now
addresses. Accuracy is effectively the same. What sequential buys is traceability.

**Cost and time:**

| Step | Time | Cost |
|---|---|---|
| Critique | ≈13–21 s | ≈$0.02 |
| Flare edit, medium | ≈15 s | ≈$0.07 |
| Flare edit, high | ≈21 s | ≈$0.25 |

## 8. Known limitations

- **The layout rulebook needs S3.** The live tests ran in generic mode, with no S3 layout. In
  production, grounded mode adds the layout rules on top.
- **Re-critiquing a finished image still produces polish notes** (margins, spacing). As a
  "done?" check, `verifyEdit` is more reliable.
- **Some code guards were tuned on few examples.** The scope guard and "don't do this" examples
  are regex and prompt rules fitted to observed failures. Watch `out_of_scope` in staging for
  false drops.
- **The integrity lens can still hallucinate geometry defects on product renders.**
- **The arbiter-skip path doesn't apply `_applyScope` or `_fixUserText`.** That path in
  `nodes.py` only runs when fewer than 2 lenses run, which never happens today. Add both if
  lens gating changes.
