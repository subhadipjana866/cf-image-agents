# critiquepipeline

Grounded-lenses + arbiter critique subsystem — drop-in replacement for the old single
`creatorAgent` critique path. Spec: `../kognie_critique_v2.md`.

## Files
| File | Contents |
|---|---|
| `grounding.py` | `brandNormalizer`, `loadLayout`, `mergeConstraints`, `genericLimits`, `resolveGrounding` |
| `prompts.py` | lens + arbiter prompt templates (`[[placeholder]]` delimiters) + `render()` |
| `agents.py` | `caseAgent`, `integrityEval`, `compositionEval`, `contentEval`, `mergeFindings`, `arbiterAgent` + vision helpers |
| `telemetry.py` | `Telemetry` — per-phase cost + time, printed at end, returned to state |
| `nodes.py` | `suggestionCreator` — robust, parallel, instrumented graph node |
| `__init__.py` | public exports |

## Placement
Drop this folder in as a sub-package next to `subgraphs/`, i.e.
`graphic_designer/critiquepipeline/`. It uses `..config`, `..utils`, `..services`,
`..subgraphs.suggestion` exactly like the existing `subgraphs/*` packages.

## Wiring into the main graph
Replace the old import + node:
```python
# old:  from .subgraphs.suggestion import caseAgent, creatorAgent, techAgent
#       (+ the big suggestionCreator defined inline in nodes.py)
from .critiquepipeline import suggestionCreator          # new node
```
Point the `suggestionCreator` graph node at this function. Everything downstream
(`suggestionHandler`, SRC, `suggestionExecutors`) is unchanged — `s_stack` has the same
shape. `techAgent` is REUSED from your existing `subgraphs/suggestion.py`.

## What it reuses (nothing re-implemented)
`config.s3_client` / `config.GEMINI_API_KEY`, `utils.parseJson` / `makeApiRequest` /
`calculateCost` / `profile`, and your existing `subgraphs.suggestion.techAgent`.

## Flow (see spec §0)
```
resolveGrounding (main thread; interrupts an UPLOAD that lacks goal/platform)
  -> layout via loadLayout (None = unsupported -> generic mode)
  -> brand via brandNormalizer (None = not attached)
lens gating: Integrity + Composition always; Content if layout|brand|platform else caseAgent
Phase B: integrityEval || compositionEval || contentEval   (ThreadPoolExecutor)
mergeFindings -> arbiterAgent (skipped if 1 lens & <= N findings)
  -> executabilityGate (rules + LLM judge, on the ranked pool; also runs when arbiter is skipped)
  -> curate to top N=3 (dropped slots backfilled from the pool, never padded)
-> existing techAgent -> flatten -> s_stack
Telemetry.summary() prints cost + time; also returned in state['critique_telemetry']
```

## Integration TODOs (resolved during the critic_pipeline -> critiquepipeline swap-in)
1. `grounding.py`: fixed to the real live-store S3 key format — a *folder* per combo,
   `layout_store/{goal}__{platform}/layout_profile.json` (the flat
   `..._layout_profile.json` form never existed and 404'd into generic mode). Also reads
   `platform_canonical` (what generation actually sets) before falling back to `platform`.
2. State must carry, for the critique: `purpose`, `aspect_ratio`, and for UPLOADS
   `clarification={design_goal, platform}` (or `design_goal`/`platform` directly); optional
   `brand_dna` (raw payload). Generated images already have `design_goal`/`platform`.
3. `agents.LENS_MODEL` / `CASE_MODEL` / `ARBITER_MODEL` — pinned to `gemini-3-flash-preview`
   (the model this deployment already prices in `config.GEN_PRICES` and uses elsewhere).
   The vendored drop's `gemini-3.6-flash` has no `GEN_PRICES` entry (silent $0 billing) and
   isn't used anywhere else in this codebase — swap it in only after confirming pricing and
   availability on this account.
4. Confirm your `techAgent(x)` returns `(json_string, cost)`; `_run_tech_agent` expects that.

## Validated locally
`brandNormalizer` against the real nested-array brand_dna (flattening, synthesized
`accent`/`background`, `tertiary`+`tartiary` merge, logo/confidence), `mergeConstraints`
precedence (`min(3,5)=3`), `genericLimits`, `_slug`, `mergeFindings`, and `render` (no
unfilled placeholders). All modules byte-compile. Live LLM/S3 calls need real creds.
```
