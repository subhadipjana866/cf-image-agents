# Graphic Designer — Conversational Supervisor Upgrade

## Context

The `graphic_designer` LangGraph agent works as a set of powerful pipelines (generation, editing, expand, layering, suggestion, Q&A) but behaves like a rigid state machine, not a conversational bot. Three concrete failures reported, traced to their mechanisms:

1. **"Can't understand stupid/free-form conversation."** `navigator` (`nodes.py:196`) is a two-phase `parseJson` classifier that maps input into a fixed bucket set and, on *any* exception, silently returns `"unclear"` (`nodes.py:280-281`). There is no graceful "I'm not sure what you mean — did you want X or Y?" behavior; ambiguity is a dead end.

2. **"Can't continue if anything goes wrong."** Errors dead-end at `extremeFallback` or loop silently back to `navigator` with no conversational recovery. Worse, every `interrupt()` site (≈28 of them) parses the resume payload rigidly — e.g. `scopeGate` coerces anything not in `{"yes","y","proceed","true"}` to "no" (`nodes.py:1204-1206`). Off-contract input is misread, not understood.

3. **"Can't move between operations conversationally."** The root structural flaw: **`navigator` only runs at the top of a *fresh* turn.** Once any flow pauses on `interrupt()`, `main.py` detects `_snap.next` and resumes *directly into the paused node* (`main.py:1596`), bypassing all routing. Mid-generation, the user is trapped answering one hardcoded question and cannot ask a question, change their mind, switch to editing, or bail out. Each interrupt handler hardcodes its own `goto`.

**Intended outcome:** a supervisor + universal input-broker overlay that mediates *every* user turn — including turns that arrive while a flow is suspended — so the bot can interpret free-form input, answer side-questions without losing its place, switch/abandon/resume operations conversationally, and recover from errors, all while **reusing the existing operation pipelines unchanged** and **keeping the frontend JSON contract stable**.

**Chosen scope (confirmed with user):** Incremental overlay. `main.py`'s resume loop may be redesigned; frontend payload shape stays stable. Durable-checkpointer swap is **out of scope** (keep `InMemorySaver` + MySQL/S3 snapshot).

---

## Target architecture (overlay, not rewrite)

Four new pieces layered on top of today's graph. Existing operation flows (`generationManager → …`, `editingManager → …`, subgraphs) are untouched internally.

```
                     ┌─────────────────────────────────────────────┐
   fresh turn ─────► │  supervisor  (tool-calling router + fallback)│
                     └───────┬─────────────────────────────────────┘
                             │ start_* / answer / chitchat / recover
        ┌────────────────────┼───────────────────────────┐
        ▼                    ▼                            ▼
  generationManager     editingManager   ...   helperAgent / chitChatHandler
        │  (unchanged pipelines, but every interrupt() call now goes through)
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  await_user()  ← UNIVERSAL INPUT BROKER (utils)               │
   │  interprets resume payload: answer? question? cancel? switch? │
   └──────────────────────────────────────────────────────────────┘
```

### 1. Supervisor node (replaces `navigator` as the hub)
A single conversational router that owns top-of-turn decisions, using **tool/function-calling** instead of `parseJson` (more robust than JSON-in-text; the one existing example is `helperAgent`'s `create_tool_calling_agent` at `common_nodes.py:103` — model it on that, reusing `TEXT_MODEL_FALLBACK_CHAIN`).

- **Tools exposed** (thin wrappers that just return a routing decision + extracted args): `start_generation`, `start_generate_more`, `start_editing`, `start_expand`, `start_layering`, `start_suggestion`, `answer_question`, `small_talk`, `resume_current_task`, `switch_task`, `cancel_task`, `ask_clarification`.
- **Graceful fallback:** when confidence is low or input is genuinely ambiguous, it calls `ask_clarification` (a real conversational turn offering concrete options) instead of the current `"unclear"` dead end.
- Keeps the existing deterministic guard (`generate_more` degrades to `generation` when no `current_image_url`, `nodes.py:242-244`).
- Emits an assistant turn into history on *every* pass so context stays coherent.
- **Wiring:** `navigator` conditional-edge dict in `graph.py:97-108` is retargeted to the supervisor's decision keys; the `lambda state: state` placeholder node at `graph.py:28` is replaced by the real supervisor function. The existing category→node map is preserved (question→helperAgent, generation→generationManager, …) so no downstream flow changes.

### 2. Universal input broker — `await_user()` (new, in `utils.py`)
The single most important change. **All** `interrupt()`+`parseJson()` blocks are replaced by one helper so every pause point becomes conversational. Signature roughly:

```python
def await_user(state, *, question, expected, resume_hint, extra_payload=None):
    """Interrupt, then interpret the raw resume payload against `expected`.
    Returns a dict: {"kind": "answer"|"meta"|"escape", "value": ..., ...}."""
```

Behavior on resume (frontend payload shape unchanged — broker interprets whatever the frontend already sends, e.g. `{"response": "..."}`):
1. `interrupt({...})` fires with the same wire `type`/keys as today (keeps `main.py` auto-continue tuple + billing working — see `GENERATION_FLOW_INTERRUPTS.md`).
2. A cheap classification (reuse `invoke_with_fallback`, `temperature=0`) decides whether the reply **answers** the pending question, is a **meta** turn (question / chit-chat), or is an **escape** (cancel / switch operation / "start over").
3. **answer** → parsed into `expected`'s shape and returned to the calling node → flow proceeds exactly as today.
4. **meta** → answer inline (route through `helperAgent`/`chitChatHandler` logic), then re-present the original `question` via another `interrupt()` — the user gets their side-question answered *without losing their place in the pipeline*.
5. **escape** → return `{"kind":"escape", intent, args}`; the calling node returns `Command(goto="supervisor", update={... push current task to task_stack ...})`. This is what finally lets a user bail out of / switch operations mid-flow.

Adoption is incremental: convert the shared handlers first (`common_nodes.py` — `helperAgent`, `chitChatHandler`, `questionHandler`, `unclearHandler`, `nextStep`), then the generation gates (`scopeGate`, `photoRequiredGate`, `clarificationGate`, `logoDecisionGate`, `interruptBeforeFidelitySuggestion`), then editing/expand/layering/suggestion. Every converted site drops its bespoke `parseJson(...)` + hardcoded-`goto` block.

### 3. Task context + stack in state (`states.py`)
Add to `GraphicDesigner` (`states.py:9-123`) — same `Annotated[..., lambda x, y: y]` last-write-wins convention:
- `active_task: Dict` — `{type, entry_node, resume_node, status, started_at}`.
- `task_stack: List[Dict]` — suspended tasks, so "switch to editing, then go back" resumes the generation brief (its `clarified_intent`/`layout_profile`/etc. already survive in state).
- `pending_input: Dict` — `{question, expected, resume_node}` describing what's currently awaited (lets the supervisor re-present a pending ask after a side-conversation).
- `last_error: Dict` — for the recovery handler.
- Keep `conversation_history` but make appends consistent (always record the assistant turn; several nodes currently skip it).

### 4. Conversational error recovery (`recoveryHandler`, new node)
Wrap each operation manager's body in try/except that, on failure, sets `last_error` and routes to `recoveryHandler` instead of the silent `extremeFallback` dead-end. `recoveryHandler` explains what failed in plain language and offers concrete next steps (retry / try a different model / switch operation) via `await_user()`, then routes accordingly. `extremeFallback` (`nodes.py:3384`) becomes a thin caller of this.

### 5. `main.py` resume-loop redesign (`main.py:1570-1851`)
Because the broker now handles escapes/meta inside the graph, the resume driver simplifies and stops trapping users:
- Keep launch-vs-resume detection via `aget_state().next`.
- Keep the auto-continue tuple + wallet billing (unchanged wire types mean this still works).
- The blind `Command(resume=json.dumps(data))` path stays, but resume payloads now flow into `await_user()`, which can redirect to the supervisor — so a "switch"/"cancel"/off-topic message mid-interrupt is finally honored instead of coerced. No frontend change required.

---

## Files to modify

| File | Change |
|---|---|
| `graphic_designer/nodes.py` | Replace `navigator` with `supervisor` (tool-calling); add `recoveryHandler`; convert generation-gate interrupt sites to `await_user()`; wrap managers in try/except→recovery. |
| `graphic_designer/common_nodes.py` | Convert `helperAgent`/`chitChatHandler`/`questionHandler`/`unclearHandler`/`nextStep` to `await_user()`; make history appends consistent. |
| `graphic_designer/utils.py` | Add `await_user()` broker + its resume-intent classifier (reuse `invoke_with_fallback`, `parseJson`, `calculateCost`). |
| `graphic_designer/states.py` | Add `active_task`, `task_stack`, `pending_input`, `last_error`. |
| `graphic_designer/graph.py` | Point `navigator` node/edges at `supervisor`; register `recoveryHandler`; add its edges. |
| `graphic_designer/prompts.py` | Add supervisor system/tool prompt, resume-intent-classifier prompt, recovery prompt. Retire `NAVIGATOR_PHASE1/2` once supervisor lands. |
| `main.py` | Simplify resume loop around the broker (§5); keep billing + auto-continue + frontend contract. |
| `graphic_designer/GENERATION_FLOW_INTERRUPTS.md` | Document the broker-mediated interrupt contract. |

Subgraphs (`subgraphs/*.py`) inherit the benefit automatically where they call the converted `common_nodes` handlers; their own `interrupt()` sites can be migrated to `await_user()` in a later pass.

## Reuse (don't reinvent)
- Tool-calling agent pattern: `helperAgent` / `create_tool_calling_agent` (`common_nodes.py:56-104`).
- LLM client + fallback: `invoke_with_fallback` + `TEXT_MODEL_FALLBACK_CHAIN` (`utils.py:248-323`).
- JSON parse / cost: `parseJson` (`utils.py:98`), `calculateCost` (`utils.py:138`).
- `Command(goto=..., update=...)` routing convention throughout `nodes.py`.
- Interrupt wire types + auto-continue tuple already defined in `GENERATION_FLOW_INTERRUPTS.md`.

## Staged rollout
1. **State + broker foundation** — add `states.py` fields and `await_user()`; convert the 5 `common_nodes.py` handlers. (Immediately makes side-questions/chitchat non-trapping in the most common pauses.)
2. **Supervisor** — build `supervisor` + tool prompts; retarget `graph.py` edges; keep old navigator behavior reachable behind a flag until parity is confirmed.
3. **Escape/switch + task stack** — wire `switch_task`/`cancel_task`/`resume_current_task`, push/pop `task_stack`.
4. **Recovery** — add `recoveryHandler`, route managers' exceptions to it, thin out `extremeFallback`.
5. **main.py resume-loop** cleanup + convert remaining generation-gate interrupts.

## Verification
- **Unit/local:** drive the compiled `graphic_app` (as `test_generation_pipeline.py` does) through scripted turns; assert routing decisions and that `Command(goto=...)` targets match intent.
- **Trapped-flow test (the key regression):** start a generation, and at a mid-pipeline interrupt (`scopeGate`/`logoDecisionGate`) send (a) an off-topic question, (b) "actually switch to editing my other image", (c) gibberish. Assert: (a) answered then original question re-presented, (b) generation pushed to `task_stack` and editing started, (c) a real clarification offered — never a silent "unclear".
- **Error recovery test:** force an image-model failure; assert `recoveryHandler` fires with a human-readable explanation and offered options, not a dead end.
- **End-to-end:** run the `/ws` websocket path (`main.py`) with a scripted client; confirm wallet billing, auto-continue frames, and frontend payload shape are unchanged.
- **Regression:** run existing pipeline tests to confirm the untouched generation/editing pipelines still pass.
