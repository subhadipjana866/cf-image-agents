# Generation flow — interrupts and expected response types

Scope: only the generation flow —

```
generationManager → scopeGate → intentClarifier → intentRouter → manifestFieldsGate
→ photoRequiredGate → clarificationGate → imageAnalyzerLayoutSelector → logoDecisionGate
→ briefFinalizerAndBriefReview → cotFusionRemainder → generation → run*Model
→ logoCompositor → interruptBeforeFidelitySuggestion
```

Three of those stages branch into a dedicated **ask node** when — and only when —
they actually need to interrupt (see "Ask-node split" below):

```
scopeGate ─(mismatch)→ scopeMismatchGate ─(declined)→ scopeAbortGate ─(rephrased)→ scopeGate
manifestFieldsGate ─(unresolved fields/assets)→ manifestFieldsAsk
briefFinalizerAndBriefReview ─([NEEDS INPUT])→ midCotClarificationAsk → cotFusionRemainder
```

(`graphic_designer/nodes.py`, `graphic_designer/graph.py`). Editing, suggestion,
layering, and expand flows have their own interrupts and are not covered here.

**Update note**: the reasoning pipeline used to be one monolithic node
(`generationPipeline`) that called `run_generation_pipeline()` and fired up to 4
interrupts from inside it via plain Python callbacks. It's since been decomposed
into separate `Command`-routed nodes (`intentClarifier` through
`cotFusionRemainder`) so that each interrupt point is a durable graph
checkpoint on its own — a later stage pausing no longer forces an earlier,
already-completed stage's LLM calls to re-run. The interrupt *types and response
contracts* below are unchanged from before; only *which node* each one now lives in
has changed.

## Ask-node split

Node decomposition fixed *cross-stage* replay, but `interrupt()` still re-runs its
**own** node from the top on resume — so any expensive work sitting above the
`interrupt()` call inside a node was paid for twice. Three nodes had that problem,
and each is now split so the interrupt lives in a node whose pre-interrupt body is
nothing but state reads:

| Stage | Expensive pre-interrupt work | Now lives in | Interrupt moved to |
|---|---|---|---|
| Scope check | `check_request_scope()` (Flash call) | `scopeGate` → result checkpointed in `state["scope_result"]` | `scopeMismatchGate`, then `scopeAbortGate` |
| Manifest fields | `invoke_with_fallback()` field extraction | `manifestFieldsGate` → ask payload checkpointed in `state["manifest_pending_fields"]` / `["manifest_pending_assets"]` | `manifestFieldsAsk` |
| Mid-CoT `[NEEDS INPUT]` | `run_multicall_fusion()` stage 1 of 7 (flash-lite call, wireframe image attached) | `briefFinalizerAndBriefReview` → stage output checkpointed in `state["cot_prior_stages_text"]`, questions in `state["cot_needs_input"]`; stages 2–7 then run in `cotFusionRemainder` | `midCotClarificationAsk` |

Measured effect (same drive, before vs after): scope mismatch → accept **2 → 1**
Flash calls; mismatch → decline → rephrase **3 → 2**; manifest ask → structured
reply **2 → 1** extraction calls; mid-CoT `[NEEDS INPUT]` → answered **8 → 7**
flash-lite calls (the re-run one carried the wireframe image).

That last row previously used a process-local memo (`_COT_STAGE_CACHE`) to skip
the replayed provider call instead of splitting the node. The memo is gone: it
was bounded (32 entries, LRU) and process-local, so a resume served by another
worker — or after a restart — paid for stage 1 again, and the node body above the
interrupt (layout-asset fetch, manifest + routing-instruction assembly) re-ran
regardless. The split makes it durable, because completed work now rides in the
checkpoint like every other stage's.

The ask nodes are entered *only* when there is something to ask — when the check
matches, or extraction resolved every field, the owning gate routes straight to the
next stage and the ask node never runs. The wire-level interrupt payloads and resume
contracts are byte-identical to before the split; the frontend sees no change.

The remaining gates (`photoRequiredGate`, `clarificationGate`, `logoDecisionGate`)
are deliberately *not* split — the only thing above their `interrupt()` is dict and
tier lookups, so their replay is already free.

Every interrupt is a dict with at least `"type"` and `"cost"`. The frontend resumes
with `Command(resume=json.dumps({...}))`; the resumed dict is parsed on our side via
`parseJson(...)`. `main.py`'s websocket loop auto-continues (no real user input, just
a spinner ping) only for interrupt types in this tuple:

```python
("request_classified", "generation_started", "editing_started",
 "design_suggestion_started", "expansion_started")
```

`"request_classified"` is the first frame of every turn: `{"type": "request_classified",
"request_type": "<generation|generate_more|regenerate|editing|suggestion|layering|expand|question|chit-chat|unclear>",
"cost": 0}`. It is sent by `requestTypeAnnounce`, the node right after `navigator`
(which now classifies as a real node and stores `request_type`), before the handler
for that type runs. Nothing is asked of the user; the frontend can use it to show the
right UI immediately.

`"critic_started"` used to be in that tuple and no longer exists at all. A general
critique now opens with `"critic_intent"`, a real question asking what the user plans
to do with the image (goal, audience, where it runs). The answer is kept in
`critique_intent`, parsed into `critique_goal` / `critique_platform` against the same
catalog the generation pipeline uses, and rendered into all four critique lenses plus
the arbiter — a design can only be judged against what it is meant to do. It is asked
once per image, so a re-critique of the same artwork reuses the answer.

`"layering_started"` used to be in that tuple and no longer is: it now asks how to
layer and waits for the answer — `response`/`mode` of `"auto"` (detect the objects,
then confirm them at `bboxes_generated` as before) or `"manual"`, in which case the
same message carries `bboxes: [...]` (the boxes the user drew, same shape as the ones
`bboxes_generated` sends out). Boxes drawn over text select which detected texts stay
layered; every other detected text is pasted back onto the background after the
objects are peeled.

Everything else (in this flow: uniformly `"text_input"`, plus `"generation_completed"`)
is sent to the frontend and the loop genuinely waits for the user's next message.

## Table

| # | Node | `type` | Fires when | Response type expected | Skippable |
|---|---|---|---|---|---|
| 1 | `generationManager` (`nodes.py:561`) | `generation_started` | Always, on entering the generation flow | **None** — auto-continued by `main.py` immediately with an empty `response`; purely a "spinner started" ping | n/a |
| 2 | `scopeMismatchGate` (`nodes.py:1266`) | `text_input` | `scopeGate` found the prompt's implied design goal/platform doesn't match what the user selected. Carries `detected_goal`, `detected_goal_supported`, `primary_platform`, `pending_platforms` | **Yes/No** — `response` ∈ `"yes"/"y"/"proceed"/"true"` (or `proceed_with_detected: true`) to accept the detected goal instead; anything else declines and falls through to row 2b | No |
| 2b | `scopeAbortGate` (`nodes.py:1345`) | `text_input` (+ `generation_aborted: true`) | The user declined row 2 — this generation is cancelled and we ask what they want instead | **Text** — a rephrased/new request in `response` (or `prompt`) becomes the new prompt and re-enters `scopeGate` for a fresh scope check; an empty reply keeps the original catalog selection and continues | No |
| 2c | `manifestFieldsAsk` (`nodes.py:1751`) | `text_input` | The combo's `frontend_manifest` declares mandatory content fields (or required-tier assets beyond photo/logo) that neither the launch payload nor `manifestFieldsGate`'s extraction pass could resolve. **Node is not entered at all otherwise** | **Structured** — `field_answers: {field_id: value}` and/or `assets: {asset_key: <src>\|"skipped"}`; a plain `response` string is also accepted (extracted against, then appended to the brief verbatim). Anything left unanswered is recorded as `"skipped"` for the life of the project | **Yes** |
| 3 | `photoRequiredGate` (`nodes.py:1839`) | `text_input` | The selected layout needs a real product/hero photo (`shared_rules.real_photo_required`) but none is attached. **No-op pass-through otherwise** — no interrupt fires at all | **Image** — a photo under `photo` / `photo_url` / `current_image_url` / `attached_images[0]` (URL or data URI); can be skipped (empty `response`) to let the model AI-generate that zone instead | **Yes** |
| 4 | `clarificationGate` (`nodes.py:1925`) | `text_input` | The router (or either of `intentRouter`'s two deterministic hard gates — structural ambiguity, fabrication-risk) flagged genuine ambiguity. **No-op pass-through otherwise**. **Can also fire a second time** mid-CoT, from `midCotClarificationAsk` (row 6) — same contract, different node | **Text** — free-form answer in `response` (read as `updated_prompt`); may optionally also include a photo (`photo`/`photo_url`/`current_image_url`) if the clarification was photo-related | No |
| 5 | `logoDecisionGate` (`nodes.py:2119`) | `text_input` | Either (a) the selected layout has a logo zone and no logo has been resolved yet (a brand-kit logo, `brand_dna["logo"]`, already resolved in `intentClarifier` counts as resolved and skips this ask entirely), or (b) **the user asked for a logo in the prompt itself** — see below. **No-op pass-through otherwise** | **Yes/No + Image (conditional)** — `response` ∈ `"yes"/"y"/"true"` (or `want_logo: true`) to proceed; if yes, a logo image is expected under `logo` / `logo_url` / `current_image_url` / `attached_images[0]`; "no"/skip proceeds with the logo zone omitted. When the ask was triggered by an explicit request the payload carries `logo_requested: true` and an attachment alone counts as yes | **Yes** |

An explicit logo request in the prompt ("put our logo top-right") overrides all three
conditions that normally suppress this ask: a layout with no logo zone, a manifest tier
below `required`, and a project-level `asset_prefs["logo"] = skipped` (which is what an
un-uploaded logo at project creation leaves behind — the user never actually declined
there). Declining that request records the request text in
`asset_prefs["logo"]["explicit_decline"]`, so the same prompt riding into a regeneration
does not nag, while a freshly typed request later does ask again. The request also biases
wireframe selection in `imageAnalyzerLayoutSelector` toward variants that have a logo zone.

**The request-driven ask is an asset request, not a yes/no question** — the user already
said they want a logo, so the only thing left to collect is the file. It uses the same
structured contract as `manifestFieldsAsk` (row 2c), so the frontend renders the same
upload control it uses for any other missing asset:

```jsonc
// out
{ "type": "text_input", "skippable": true, "logo_requested": true,
  "question": "<plain-text fallback>",
  "assets": [ { "asset_key": "logo", "tier": "required", "reason": "…" } ] }
// back
{ "assets": { "logo": "<url|data-uri>" } }        // or "skipped" to carry on without it
```

`logo` / `logo_url` / `current_image_url` / `attached_images[0]` are still accepted on the
reply, so older frontends keep working. The plain yes/no ask (row 5's original wording) is
unchanged and still fires for the layout-has-a-logo-zone case, where the user has not said
whether they want one at all.

The **same asset request** fires from two more places, so a logo the user asks for is never
dropped just because their request took a different route into an image model:

| Route | Node | Notes |
|---|---|---|
| Layout/CoT generation | `logoDecisionGate` | The rows above. |
| **Freeform generation** (custom design goal, or an accepted off-catalog detected goal — both skip the whole gate chain) | `logoDecisionGate`, reached via `_freeformNext` | Only diverts when the prompt names a logo; a freeform prompt that doesn't is gate-free exactly as before. Guidance rides on `fused_prompt` (there is no layout or routing instruction to put it in), and `logoCompositor` places the file into the default top-center band. |
| **Editing** ("add a logo", "put our logo bottom-right" on an existing image) | `editLogoGate`, between `editingManager` and `editIntentClassifier` | Pure pass-through unless the edit names a logo. A logo already on file is used silently. The collected file is handed to the Gemini edit executors as a **second reference image** so the model honours the placement the user asked for; Flux/GPT-image fallbacks take one image only, so there the logo reference is dropped and only the "don't invent one" guidance applies. Outcome rides on `edit_logo_mode`, rewritten every edit so it cannot leak into the next one. |

In every route, asking for a logo and supplying nothing appends an explicit *do not invent a
brand mark* instruction to whatever text reaches the model — otherwise it letters its own
version to satisfy the words in the prompt.
| 6 | `midCotClarificationAsk` (`nodes.py:2429`) | `text_input` | The multicall CoT fusion's own `[NEEDS INPUT]` escape hatch fires, between stage 1 (`stage_brief_review`) and stage 2 of 7 — a genuinely blocking gap the upstream gates (rows 4/5) missed. `briefFinalizerAndBriefReview` runs stage 1 with `defer_clarification=True`, checkpoints its output, and routes here only if the stage raised something. Same contract as row 4, reused verbatim | **Text** — free-form answer in `response` (read as `updated_prompt`); no answer injects an explicit anti-fabrication note instead of a guess | No |
| 7 | `interruptBeforeFidelitySuggestion` (`nodes.py:3164`) | `generation_completed` | After the image is generated (and, if applicable, `logoCompositor` has run) | **Yes/No** — `response == "yes"` routes to auto-suggestions; otherwise `response` is treated as the user's next free-form prompt/message. Also requires `model`, `purpose`, `aspect_ratio` to be echoed back in the resume payload (read directly off the response dict, not defaulted) | No |

## Notes

- Rows 3–5 are each their own dedicated graph node now (`photoRequiredGate`,
  `clarificationGate`, `logoDecisionGate`) — every one of them self-no-ops
  (`Command(goto=<next stage>, update=state)`, no interrupt fired) when its trigger
  condition is false, and only re-runs *itself* on resume, not `intentClarifier`/
  `intentRouter`/`imageAnalyzerLayoutSelector`'s LLM calls.
- Rows 2/2b/2c/6 go one step further: their owning stage (`scopeGate`,
  `manifestFieldsGate`, `briefFinalizerAndBriefReview`) keeps the LLM call and hands
  the *dialog* to a separate ask node, so re-running the ask node on resume re-runs
  nothing but state reads — see "Ask-node split" above. The ask nodes are only ever
  routed to when there is something to ask (each still guards defensively against a
  stale/empty payload rather than firing a blank interrupt).
- `scopeAbortGate` (row 2b) is the only interrupt in this flow that can *cancel* the
  run: a non-empty reply is routed back into `scopeGate` as a brand-new prompt, so a
  rephrase that still conflicts with the catalog selection is caught again rather
  than generated blindly.
- Row 6's fusion is split across two nodes rather than one call: stage 1 runs in
  `briefFinalizerAndBriefReview` with `defer_clarification=True` (the pause comes
  back as data — `_needs_input` — instead of interrupting from inside the fusion),
  and stages 2–7 run in `cotFusionRemainder`, seeded with
  `state["cot_prior_stages_text"]` plus the answer folded in by
  `build_clarification_note()`. `cotFusionRemainder` fires no interrupt at all, so
  nothing downstream of the pause can replay it either. Both nodes rebuild their
  `run_multicall_fusion` kwargs from state via `_cotFusionInputs()` — pure string
  assembly over a memoized layout-asset fetch, no LLM call — so the two halves
  always see the same brief.
- None of `intentClarifier`, `intentRouter`, `imageAnalyzerLayoutSelector`, the 7
  `run*Model` nodes, or `logoCompositor` fire interrupts directly — they run
  straight through (or silently fall back to another model on exception).
- Every blocking interrupt in this flow uses the generic `"type": "text_input"` —
  there's no dedicated `"scope_mismatch"`/`"clarification_needed"`/`"logo_decision"`
  wire-level type; they're only distinguished by which extra payload keys are
  present (`detected_goal`, `clarification_questions`, `reason`, etc.).
- `"cost"` is always included and is deducted from the user's wallet on every
  single interrupt round-trip (`main.py`), including auto-continued ones — rows
  2–6 explicitly set `"cost": 0` (real reasoning-stage costs are billed on the
  next node's own `state["cost"]` update, not on the interrupt itself), while
  rows 1 and 7 report the actual accumulated cost so far.
