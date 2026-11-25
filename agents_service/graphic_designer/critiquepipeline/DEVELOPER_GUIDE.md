# critiquepipeline — Developer Guide

Onboarding doc for the grounded image-critique subsystem. Read this top-to-bottom once,
then keep the "Files" and "Control flow" sections handy.

---

## 1. What this is (in one paragraph)

When Kognie generates or a user uploads a graphic, this subsystem produces a short list of
high-quality improvement **suggestions** (accept / reject / modify in the UI). It replaces the
old single-evaluator critique with **four focused lenses run in parallel**, an **arbiter** that
curates them down to the best few, and the existing **techAgent** for model metadata. Every
suggestion is grounded in the design's real spec — the `layout_profile.json` (always) and the
`brand_dna` (when attached) — so the critique cites concrete rules instead of inventing them.
The mission is narrow on purpose: improve **textual content, readability, and graphic-design
principles** so the graphic is functional and usable — not to decorate or maximise completeness.

---

## 2. Where it plugs into the main graph

- It **replaces** the old inline `suggestionCreator` + `creatorAgent` path.
- One-line swap in the main graph: `from .critiquepipeline import suggestionCreator`, point the
  `suggestionCreator` node at it.
- It **reuses** your existing `subgraphs.suggestion.techAgent` and produces the same-shaped
  `s_stack`, so `suggestionHandler`, the SRC classifier, and `suggestionExecutors` are unchanged.
- Place the folder as a sub-package next to `subgraphs/` (i.e. `graphic_designer/critiquepipeline/`)
  so the relative imports (`..config`, `..utils`, `..services`) resolve.

---

## 3. Files — what each one does

| File | Responsibility | Key names |
|---|---|---|
| `nodes.py` | The graph node. Orchestrates the whole job and returns the final `s_stack` + telemetry. | `suggestionCreator`, `structureSuggestions`, `_run_tech_agent` |
| `grounding.py` | Deterministic (no LLM) grounding. Loads/normalizes the spec + brand, decides mode. | `resolveGrounding`, `loadLayout`, `brandNormalizer`, `mergeConstraints`, `genericLimits` |
| `agents.py` | All LLM calls: the 4 lenses, the arbiter, the executability gate, the compiler, plus transport + model config. | `integrityEval`, `compositionEval`, `contentEval`, `aestheticEval`, `arbiterAgent`, `executabilityGate`, `compilerAgent`, `mergeFindings`, `caseAgent` |
| `prompts.py` | Every prompt template + the shared rule blocks. Uses `[[placeholder]]` + `render()`. | `INTEGRITY/COMPOSITION/CONTENT_..._PROMPT`, `ARBITER_PROMPT`, `SUGGESTION_COMPILATION_PROMPT`, `NANO_PROMPT_RULES`, `SHARED_OUTPUT_NOTE` |
| `telemetry.py` | Thread-safe cost + time tracker. Prints a table and returns a DB-ready record. | `Telemetry`, `.span()`, `.summary()` |
| `__init__.py` | Public exports (import surface). | — |
| `README.md` | Short integration cheatsheet + TODOs. | — |

### File deep-dives

**`grounding.py`** — pure functions, no network unless `loadLayout` runs.
- `brandNormalizer(raw)` → flat, prompt-safe brand profile. The raw `brand_dna` does **not**
  match old assumptions: colors arrive as nested `[[...]]`, `accent`/`background` don't exist
  (derived from the recommended palette), and there's a `tertiary`/`tartiary` typo — this
  function fixes all of that. It also reads the flat shape production actually stores in
  `state['brand_dna']` (`main.fetch_brand_dna`: `tone`, `personality_traits`, `font_pairing`,
  `color_primary/secondary/accent/background`, palettes keyed `color_1..N`) — before that, every
  live critique ran brand-blind. Adds `palette.named` (`[{name, hex, role}]`, names from
  `_colorName`) because nano_prompts must name colours in words. Returns `None` when there is
  no brand identity (no brand, or a kit with only asset URLs).
- `loadLayout(goal, platform)` → fetches `layout_store/{goal}__{platform}__layout_profile.json`
  from S3. Returns `None` if the combo is **unsupported** (→ generic mode, not an error).
- `mergeConstraints(layout, brand)` → precedence rule: **layout wins on limits, brand on
  identity**; effective limit = the tighter (e.g. `max_colors = min(3, brand)`).
- `resolveGrounding(state)` → the single entry point. Uploads take their goal/platform from
  `critique_goal`/`critique_platform` — parsed by `suggestionManager` from the user's answer to
  "what are you planning to do with this image?" (`critic_intent`) — then from
  `state['clarification']`. It **interrupts** to ask only when all of those are empty *and* no
  `critique_intent` was ever captured; once the user has answered the intent question in their
  own words, asking a second, narrower version of it reads as if we did not listen, and an
  answer that named no supported combo simply runs generic. Returns
  `{goal, platform, layout, brand, eff, avail}` where `avail.mode` is `"grounded"` or
  `"generic"` and `avail.intent` is the user's verbatim intent (rendered into every lens and
  the arbiter via `prompts.intentBlock`).

**`agents.py`** — model config lives at the top as constants (see §6). Each `*Eval`:
1. builds a grounded prompt via `render(...)`, 2. runs one vision call, 3. returns tagged
findings. `mergeFindings` combines the 4 lens outputs and assigns a stable `fid` per finding.
`arbiterAgent` sends **compact** findings (id + category + evidence, no verbose text) and gets
back only `{fid, rank, reword}`, then **reassembles** the full recs in code (small output =
fast). `compilerAgent` fuses accepted nano_prompts into one execution prompt.

**`prompts.py`** — the IP. Shared rule blocks injected into every lens:
- `MISSION` + `SELECTIVITY` (in `SHARED_OUTPUT_NOTE`): only major flaws or missing **critical
  text**; 0 suggestions is valid; never add a logo or optional/decorative copy; prefer fixes.
- `NANO_PROMPT_RULES`: the **two-audience** rule (`user_friendly_text` = plain for the user;
  `nano_prompt` = executable for the edit model) + concrete-move / text-copy / glyph rules.

---

## 4. Control flow

```
suggestionCreator(state)                                   [nodes.py]
  │
  ├─ Telemetry() starts the clock
  │
  ├─ resolveGrounding(state)                               [grounding.py]
  │     • intent: what the user said this image is FOR (asked by suggestionManager
  │       BEFORE this node runs) -> avail.intent, rendered into every lens + arbiter
  │     • provenance: generated (goal/platform known) vs uploaded (intent-parsed
  │       goal/platform; interrupt only if nobody ever asked)
  │     • loadLayout() -> layout or None      brandNormalizer() -> brand or None
  │     • mergeConstraints/genericLimits -> eff      avail.mode = grounded | generic
  │
  ├─ lens gating: Integrity + Composition + Aesthetic always on;
  │               Content on if layout|brand|platform else caseAgent() decides
  │
  ├─ PHASE B — all four lenses IN PARALLEL  [agents.py]
  │     integrityEval   -> defects / artifacts / glyphs / (fidelity if base_prompt)
  │     contentEval     -> brief alignment (generated only) / copy quality / placeholder strings /
  │                        platform / AI-tell / logo
  │     compositionEval -> hierarchy / color / readability (overflow, density, alignment) / typography
  │     aestheticEval   -> senior-designer polish WITHIN brand+text rules (spacing, balance)
  │     (every lens can return [] freely; one failing lens does not kill the job)
  │
  │     NO WIREFRAME GROUNDING: the critique never sees layout_variants[].zones[]. A wireframe is
  │     one candidate arrangement, not a contract the finished artwork owes anything to; grounding
  │     on it produced suggestions phrased against an invisible template ("move it to the
  │     designated CTA zone") and pushed the arbiter toward whole-canvas relayouts. This is why
  │     B1/B2 collapsed into one phase — matchLayoutVariant (a paid vision call whose only consumer
  │     was zone selection) is gone. shared_rules ARE still used: typography_scales, negative_rules,
  │     required_content_fields, reading_order_description, max_colors/contrast_floor/safe_zone.
  │
  ├─ mergeFindings(...) -> one list, stable fids, each tagged edit_risk: "local"|"structural"
  │     (_tag() deterministically forces type="GENERATION" when edit_risk="structural" —
  │      routing does not depend on the LLM remembering to set `type` correctly)
  │
  ├─ PHASE C — arbiter (skipped if a single lens returned <= N)
  │     _predupe(...)   -> deterministic pre-arbiter dedup (category + keyword overlap), IN CODE
  │     arbiterAgent    -> dedupe, must-keep multi-lens defects, anti-clutter, rank by
  │                        EXPECTED VALUE (severity x edit-success-likelihood — local outranks
  │                        structural at equal severity), cap POOL=N+4, NO padding
  │     arbiter ranks a POOL of N+4 candidates (not just N) so there's material to backfill from
  │     executabilityGate(...) -> drops every pool item GPT-Image-2.5 Flare can't reliably
  │                        carry out: deterministic rules (_ruleNotExecutable: logo add/redraw,
  │                        "REQUIRES MOVING N>=3", transparent bg, pixel-exact) then ONE text-only
  │                        LLM judge call (EXECUTABILITY_PROMPT). Runs BEFORE top-N so a dropped
  │                        item is backfilled. Text repairs get NO exemption here. Judge fails OPEN.
  │                        Drops -> telemetry notes.not_executable {fid, lens, topic, reason}
  │     _applyStructuralCap(...) -> deterministic: at most 1 structural-risk suggestion kept,
  │                        and only if no local suggestion already covers its category
  │     _selectDiverseTopN(...) -> deterministic: walks the pool in rank order, skips any
  │                        candidate too similar to one already kept, backfills from the next
  │                        candidate by score — final list may be < N, never padded
  │
  ├─ techAgent(arbiter_json)   (EXISTING agent, reused)    [subgraphs.suggestion]
  │
  ├─ structureSuggestions(tech_json) -> final s_stack in the FRONTEND contract
  │
  └─ tele.summary() -> job record {cost, wall_s, tokens, phases}
        Command(goto="suggestionHandler", update={ s_stack, cost, critique_telemetry })
```

Downstream (unchanged except where noted): `suggestionHandler` shows the stack → user
accepts/rejects/modifies (SRC classifier) → `suggestionExecutors` runs `compilerAgent` + the
image edit → **new: `verifyEdit(before, after, accepted_units, tele)`** should be called here
(not yet wired into a graph node — it's a function in `agents.py`, ready to hook into a new
`postEditVerifier` node). It compares before/after against what was accepted and flags any NEW
regression the edit introduced (e.g. duplicated text from a risky structural change) — per
product decision, it **flags, does not auto-revert**.

`compilerAgent` now picks between two mode blocks based on whether any accepted unit is
structural-risk: **EDIT lane** (strict/surgical, all-local) or **REGEN lane** (permissive,
still image-referenced, grants explicit permission to reflow text layout for the structural
unit only) — still one compiled prompt / one edit call, no sequential batching.

---

## 5. Data contracts

**s_stack item** (what the frontend receives — produced by `structureSuggestions`):
```json
{ "image_id":"image_1", "s_id":0, "status":"pending", "focus":false, "user_response":null,
  "edits":2, "type":"EDIT", "tier":4, "topic":"Text rendering",
  "user_friendly_text":"plain text for the user", "nano_prompt":"executable edit instruction" }
```
`edits` = total number of suggestions in the stack. Internal fields (source_lens, confidence,
evidence, checklist_category) are intentionally dropped here.

**critique_telemetry** (persist to the DB after generation):
```json
{ "cost":0.0039, "wall_s":37.5, "tokens":19614,
  "phases":[ {"phase":"integrity","model":"gemini-2.5-flash","tokens":3248,"cost":0.0006,"sec":9.36}, ... ] }
```

**Grounding inputs from `state`:** `current_image_url` or `attached_images`, `base_prompt`
(present for generated), `purpose`, `aspect_ratio`, `design_goal`/`platform` (generated),
`critique_intent` + `critique_goal`/`critique_platform` (the `critic_intent` answer, uploads
and generated alike) or `clarification:{design_goal, platform}` (uploads), optional raw
`brand_dna`.

---

## 6. Config knobs (top of `agents.py`)

| Constant | Default | Notes |
|---|---|---|
| `LENS_MODEL` / `LENS_THINKING` | `gemini-3-flash-preview` / `512` | 4 vision lenses + `verifyEdit`. Thinking capped to kill latency, kept >0 to protect judgment. |
| `CASE_MODEL` / `CASE_THINKING` | `gemini-3-flash-preview` / `0` | mechanical text/logo pre-check (generic mode only). |
| `ARBITER_MODEL` / `ARBITER_THINKING` | `gemini-3-flash-preview` / `512` | curation. **Kept on full flash — flash-lite was unstable (0-vs-4 swings).** |
| `COMPILER_MODEL` / `COMPILER_THINKING` | `gemini-3.1-flash-lite` / `0` | mechanical fusion, fast tier. |
| `EXEC_MODEL` / `EXEC_THINKING` | `gemini-3-flash-preview` / `256` | executability gate judge (text-only). Rubric in `prompts.EXECUTABILITY_PROMPT`, written for **GPT-Image-2.5 Flare**. Revisit when the edit model changes. |
| `MAX_SUGGESTIONS` (nodes.py) | `3` | **ceiling, not a target.** `MIN_DISPLAY_CONFIDENCE` (0.55) / `MIN_AESTHETIC_CONFIDENCE` (0.7) / `MAX_AESTHETIC_SUGGESTIONS` (1) in agents.py enforce the quality floor in code, so a clean design correctly returns 0-2. The user sees the top 0-3 **AI-executable** suggestions. |
| `LAYOUT_STORE_BUCKET`/`_PREFIX` (grounding.py) | `kha-documents` / `layout_store/` | confirm against your env. |

Pinned to `gemini-3-flash-preview` during the critic_pipeline -> critiquepipeline swap-in: it's
the model this deployment already prices in `config.GEN_PRICES` and uses elsewhere (e.g.
`CLASSIFY_MODEL_GOOGLE`). The vendored drop originally specified `gemini-3.6-flash`, which has
no `GEN_PRICES` entry (silent $0 billing via the UNPRICED fallback) and appears nowhere else in
this codebase — treat it as unverified for this account. Confirm availability + add pricing
before switching. More generally, model names depend on which models your API key's account can
call — verify with `client.models.list()` before changing any of these.

---

## 7. Modes & robustness (know these before debugging)

- **Generic mode** (`layout is None`): unsupported goal×platform → lenses judge on principles,
  never cite rules they don't have. Happens silently.
- **No brand**: brand identity/logo checks skip; layout checks still run.
- **Brand attached**: `prompts.brandBlock(brand)` renders one `BRAND_BLOCK` into all four lenses
  and the arbiter. It binds every nano_prompt the lens writes (brand colours or neutrals only,
  same type style family, brand voice, brand mood) and requires a `brand_check` field per
  recommendation — the check happens at generation time because the executor sends the
  nano_prompt to the edit model unchanged. Composition owns colour drift, content owns voice
  drift, aesthetic owns mood drift; the arbiter drops fids whose `brand_check` shows an
  off-brand fix. Telemetry note `brand` shows whether grounding reached the lenses
  (`attached` vs `present`).
- **No base_prompt** (upload): fidelity tier skips; upload triggers the clarification interrupt.
- A lens can return `[]` — that's normal, not a bug. So can the whole job (clean graphic).

---

## 7b. Compiled-prompt language quality (fusion rules)

The compiler (`compilerAgent` / `SUGGESTION_COMPILATION_PROMPT`) fuses every accepted suggestion
into ONE prompt for the image-edit model — still one call, no sequential batching. Because
fusion is the step most likely to bury or garble an individual instruction, it follows explicit
image-edit prompting conventions, not just "write a paragraph":

- **Priority bookending.** `suggestion_units` arrive in rank order (`s_id 0` = most important).
  The compiler is told this explicitly and must name that unit's goal in BOTH the opening anchor
  and the closing sentence — primacy + recency — so the top suggestion can't get diluted by
  sitting in the middle of a longer instruction list (sequencing is still by operation-type
  safety order — removals first — not by rank, so bookending is how priority survives that).
- **Consistent naming.** Once an element is named ("the ORDER NOW button"), every later mention
  in the compiled prompt must reuse that exact name — prevents naming drift when two suggestions
  from different lenses described the same element differently.
- **Contradiction check** (STEP 0.5): if two accepted units would visibly conflict, only the
  higher-priority one survives; the drop is logged in `compilation_notes`.
- **No silent drops**: every surviving unit must appear in exactly one instruction block, or be
  explicitly merged (both s_ids logged) — a self-check step re-verifies this before output.
- **Positive framing paired with negation** (`NANO_PROMPT_RULES` rule 8): a "don't do X" is
  always paired with the positive replacement in the same sentence.
- **Relational grounding** (`NANO_PROMPT_RULES` rule 2): elements can be named relative to a
  neighbor ("the button below the headline") when that pins the target down more reliably than a
  bare label — a known image-edit-model accuracy technique.
- **Length discipline**: soft cap around 150 words on the full compiled prompt — padding dilutes
  every instruction, including the bookended top-priority one.

Verified live: a compiled prompt for a single-unit case read *"The most important change is
recoloring the subheadline..."* in the opening AND *"...to ensure visual harmony..."* echoing the
same goal in the closing sentence — the bookend renders correctly even at N=1.

---

## 8. Running / testing

- The package **byte-compiles** with no third-party deps installed (langchain/langgraph/s3 are
  lazy-imported). `python -m py_compile *.py`.
- Live runs need a `GEMINI_API_KEY` and (for real layouts) S3 access.
- Pure logic (`brandNormalizer`, `mergeConstraints`, `structureSuggestions`, `mergeFindings`)
  is unit-testable without any network — stub the parent `config`/`utils` and langgraph.

---

## 9. Known gaps / next steps

- **Wire `verifyEdit` into a real graph node.** The function exists and is validated on live
  runs, but there's no `postEditVerifier` node in the actual graph yet — someone needs to call
  it after `suggestionExecutors` and decide what the UI does with a `"regressed"` result (today:
  flag and surface to the user, never silently auto-revert — see §4 diagram).
- **Live finding, updated after a second round of live testing (important — read before tuning
  further):** structural relayouts are NOT broadly unreliable — across 8 live case-runs across
  two sessions, structural suggestions applied cleanly in 6 (`grand_opening`, `hiring` x2), came
  back "partial" harmlessly in 1 (`menu_promo`), and only genuinely regressed on ONE specific
  case (`product_launch`'s 2-column relayout moving a photo panel + 4 text elements at once) —
  and even that case regressed once, then applied cleanly on a third attempt with the same class
  of instruction. This points to real STOCHASTIC variance in the image-edit model on harder,
  multi-element relayouts, not a deterministic prompt gap. Two concrete mitigations are now in
  place: (1) `NO_DUPLICATION_RULE` (prompts.py) — baked into both `EDIT_MODE_BLOCK` and
  `REGEN_MODE_BLOCK` — explicitly instructs "move, don't copy" and requires exactly one instance
  of the CTA/every text block; verified via inspection that the compiler correctly includes this
  in the compiled prompt every time. (2) `EDIT_RISK_RULE` now tells lenses to prefer the simplest
  possible structural fix (1-2 elements) and to name in "evidence" when a fix genuinely requires
  moving 3+ elements, so the arbiter (STEP 7) can rank multi-element structural fixes as even
  lower expected-value. **Neither mitigation makes the underlying image model deterministic** —
  `verifyEdit` remains the essential backstop regardless of how well-instructed the edit is, and
  should stay mandatory for any accepted structural suggestion. Do not conclude "this is fixed
  now" from a single clean re-run; conclude "the safety net (flag, don't silently ship) is doing
  real work, and the failure rate on genuinely complex relayouts is real but non-trivial to drive
  to zero via prompting alone."
- **Case study: 3 consecutive attempts on one hard relayout (`event_announcement`), fully traced.**
  A user report ("this didn't work when the instructions were simple enough") led to a precise
  root-cause chain worth keeping as a reference:
  1. Attempt 1: the Composition lens's nano_prompt named only 3 of 5 real text elements and used
     a HALLUCINATED category label ("date") that didn't correspond to any visible string. Fix:
     `EDIT_RISK_RULE` now requires COMPLETE ENUMERATION — every text element named by its literal
     visible content, never a generic category word.
  2. Attempt 2 (re-tested live after the fix): the compiled prompt now named all 5 elements
     explicitly, even with colors — confirmed the fix worked exactly as designed at the lens
     level — but the edit still regressed, this time with a NEW artifact (a visible seam through
     the photo subject) instead of the earlier failure mode. This proved the remaining variance
     was NOT a wording problem: it isolated to genuine image-model difficulty on a 5-element
     relayout of visually complex, layered photography (a garnished glass in front of other
     blurred drinks) — a harder task than the 1-2-element moves that succeeded elsewhere
     (`hiring`, `grand_opening`). Fix: `EDIT_RISK_RULE` strengthened to (a) prefer a SMALLER
     PARTIAL fix over a full 3+-element relayout when the partial fix would meaningfully help,
     and (b) require a literal `"REQUIRES MOVING N ELEMENTS: ..."` marker at the start of
     `evidence` when a full relayout can't be reduced — plus a NEW deterministic code enforcement,
     `_capHighRiskConfidence` (agents.py), which regexes for that exact marker and caps
     confidence to 0.5 regardless of what the lens set it to, so the arbiter's own confidence
     filter has real teeth even if the LLM's free-text ranking misses the risk.
  3. Attempt 3 (re-tested live again): the lens correctly emitted the literal marker
     (`"REQUIRES MOVING 5 ELEMENTS: ..."`), confirmed the deterministic cap fired (confidence
     landed at exactly 0.5), and the edit still didn't relocate the text — but this time
     `verifyEdit` reported **zero regressions**: no duplication, no seam, no garbling. The model
     conservatively left the correct original text in place rather than corrupting it while
     trying. **This is the outcome that matters**: the failure mode moved from dangerous
     (duplicate/garble/artifact) to safe (no-op on the risky part, real gains on the safe part —
     a background artifact was also accepted in this batch and landed cleanly).
  **Conclusion**: full multi-element (3+) text relayouts on visually complex photography remain
  genuinely unreliable at the image-model layer — no amount of prompt precision fully solved it
  across 3 attempts on the same image. But the mitigations built across this investigation
  (complete enumeration, partial-fix preference, explicit risk marker + deterministic confidence
  cap, and `verifyEdit` as the mandatory backstop) collectively turned an unreliable *and
  dangerous* edit into an unreliable *but safe* one. Given this now-repeated pattern (this case,
  plus `product_launch` needing 3 attempts earlier), **revisit whether 3+-element structural
  suggestions should be auto-applied in a single-pass flow at all**, vs. surfaced as a distinct
  "try this — may take a few attempts" action separate from normal accept/reject. This is a
  product decision, not a prompt-tuning one, and it's now backed by real repeated evidence.
- **Suggestion-level similarity dedup + backfill** (new): `_selectDiverseTopN` + `_tooSimilar` in
  `agents.py` prevent two near-duplicate suggestions from both appearing in the final top-N. The
  arbiter is asked to rank a POOL of `N+4` candidates (not just N); the code walks that pool in
  rank order and skips any candidate too similar (by category-agnostic keyword overlap across
  topic/evidence/nano_prompt) to one already kept — which is exactly "drop the lower-scored
  duplicate, backfill with the next-closest-score candidate." If the pool empties or everything
  left is a near-duplicate of what's already kept, fewer than N are shown — never padded. Unit-
  tested offline (no API cost) in `_tooSimilar`/`_selectDiverseTopN`.
- **Image duplication is also now flagged defensively by two lenses**, independent of the
  compiler fix: `INTEGRITY_EVALUATOR` Tier 1 flags duplicated text/UI elements as MAJOR;
  `COMPOSITION_EVALUATOR` Tier 1 flags a duplicate/competing CTA or repeated core message as a
  MAJOR hierarchy violation. These catch a duplication that slips through on the *next* critique
  pass, even if `verifyEdit` isn't wired into a graph node yet in your deployment.
- **Lens coverage**: the integrity lens can miss a *subtle* artifact; off-brand accent colors
  aren't always flagged. The calibration examples (§ prompts.py `CALIBRATION:` lines) help but
  don't eliminate this — expect ongoing prompt tuning here.
- **Global constraints**: image-edit models honor discrete local edits well but only partially
  honor "exact N colors / strict %" even after the compiler rewrites specs into concrete moves —
  `verifyEdit` is the real backstop for this, not prompt wording alone.
- **`matchLayoutVariant` (REMOVED)**: this used to pick which `layout_variant`'s zones to ground
  composition/aesthetic against. It was removed along with all wireframe grounding — see
  "NO WIREFRAME GROUNDING" in agents.py. The historical observation that it selected variant 0
  on every live test case is now moot, but it is the reason the removal is low-risk: grounding
  was almost always against variant 0 anyway, so dropping zones changes far less than it appears.
- **Provenance gating**: `avail["provenance"]` is `"generated"` only when `base_prompt` OR
  `design_goal` is set. Uploads never get missing-mandatory-field / missing-CTA findings, even in
  grounded mode, because a layout is loaded for them *after the fact* from the goal/platform the
  user supplies — the asset was never built to it. If a generated design ever loses BOTH signals
  it will degrade to upload treatment (lose brief checks), which is the safe direction to fail.

---

## 10. Live end-to-end test (Sep 2026): Kognie generation → critique → GPT-Image-2.5 Flare

**Setup.** Three Kognie-generated graphics, each with two known defects planted in advance:
- G1: a typo and a "TBC" placeholder, both from the brief.
- G2: a clipped "40% OFF" and a CTA shown 3×, planted with a Flare edit.
- G3: a misspelled "wirless" and a near-invisible spec line, planted with a Flare edit.

The critique ran in generic mode (no S3 layout), the top 2 suggestions were accepted, and they
were executed both **sequentially** (one Flare edit per suggestion, `verifyEdit` after each) and
**fused** (one compiled call). The finished image was then critiqued again. The harness itself
lives outside the repo.

**Finding the planted defects.** All 6 were found and ranked #1/#2 in every run. In run 1,
all three #3 slots were noise:
- hallucinated "malformed hands"
- "remove THIS WEEKEND ONLY as redundant"
- "add a CTA" when the URL was already visible

After the fixes below, G3 shows exactly its 2 real issues. G2's #3 is a reasonable contrast note.

**Fixes that came out of the test, each proven necessary live:**

| Symptom | Fix |
|---|---|
| "Remove the duplicate CTAs" erased **every** copy, including the one to keep | `NANO_PROMPT_RULES` rule 9 (name the copy to KEEP) and compiler KEEP CLAUSES |
| The user read "ensure this is updated with booking info", but the edit deleted the line | `_userText`: prefer the lens's text, which it wrote with the nano_prompt; the arbiter's rewrite is only a repair and must be an instruction |
| Composition asked for a "Shop Now" button on a poster that already showed the URL | `_outOfScope`: composition and aesthetic may not add content (code, not prompt) |
| "Remove 'THIS WEEKEND ONLY' as redundant", repeated even after a prompt example | `_outOfScope`: never remove deadline or offer copy as redundant |
| A duplicate flagged by four lenses lost its slot to a polish note | `_predupe` merges the same text repair across categories; "duplicate" added to `_TEXT_PROBLEMS` |
| Hand and face hallucinations | Integrity lens: "ANATOMY CLAIMS NEED PROOF" (a countable defect, prominent subjects only) |
| Re-critique re-reported already-fixed defects from a stale or edit-instruction `base_prompt` | `nodes._isEditInstruction` drops a compiled edit prompt; text claims are read off the image |
| Sequential `verifyEdit` blamed existing or still-to-come issues on the current step | `verifyEdit(..., pending_units=...)` plus the "A REGRESSION IS ONLY SOMETHING THIS EDIT CAUSED" rule |
| Flare garbled a one-word spelling fix: 1 of 6 at medium ("wiroless", "wirıless", "wirbless") | `editQuality(units)` returns `"high"` for letter re-rendering (2 of 2 clean at high; ≈4× output tokens, +6 s) |

**Sequential vs fused.**

| Run | Sequential | Fused |
|---|---|---|
| Final (G2 + G3) | Every step clean, verified by eye | Also clean |
| All 9 graphic runs | 6 of 9 fully clean | 5 of 9 fully clean |

The only difference across all runs was one garbled word (the small-text problem `editQuality`
now addresses), so accuracy is effectively the same. Sequential isolates failures to one step
and makes `verifyEdit` attributable, at twice the edit cost and time.

**Still open:**
- The re-critique of a finished image still produces subjective polish notes (margins, spacing).
  It is a weaker "is this done?" signal than `verifyEdit`.
- The integrity lens can still hallucinate geometry defects on product renders.
- The executor (`suggestionExecutors`, outside this package) must adopt `editQuality` and, if
  editing sequentially, `pending_units`.
