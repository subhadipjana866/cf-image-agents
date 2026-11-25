# new_arch.pdf → Message-Passing Architecture: Gap Analysis & Migration Plan

## Context

`new_arch.pdf` ("Chatbot Architecture") specifies a **message-passing model** for the
`graphic_designer` graph: every state-graph entry carries a metadata envelope
(From / Receiver / Reply-to / Processor / Previous-processor / Status / Context / …),
routing is driven by that metadata (not by re-classifying content every turn), and a
**"virtual interrupt"** makes free-form (non-interrupt) input behave like an interrupt
reply — it returns to the **processor that last interacted with the user**, bypassing the
Navigator. It also adds display-gating (Private/Public/On-demand) and per-receiver history.

Today the graph does none of this: `navigator` (an LLM `parseJson` classifier) re-derives the
route on every fresh turn, terminal nodes hardcode `Command(goto="navigator")`, and
`conversation_history` is a flat `[{role, content}]` list with no routing/display metadata.

**Important prior art:** `graphic_designer/CONVERSATIONAL_UPGRADE_PLAN.md` already designs a
**supervisor + `await_user()` broker + task stack + recovery** overlay (still *unimplemented* —
`navigator` is untouched). The `new_arch.pdf` envelope is the data model that formalizes that
same behavior. **This plan builds the envelope and adopts the supervisor together**, reusing that
plan's pieces rather than duplicating them.

**Scope (confirmed with user):** incremental in-place refactor of the existing graph; plan the
full doc end-to-end. Frontend JSON contract stays stable; `InMemorySaver` stays.

## Doc concept → implementation mapping

| Doc term | Implementation |
|---|---|
| Processor | A graph node / subgraph (managers, handlers, subgraphs, User). Named via a `Processor` enum in `config.py`. |
| Navigator | Kept only as the **TBD-resolver** (becomes the `supervisor`, tool-calling). Never sender/receiver. |
| Real interrupt reply | Already handled — LangGraph resumes the node that called `interrupt()`. No change to that mechanism. |
| Virtual interrupt | Fresh/free-form input when not paused → routed to **Previous Processor**, not the supervisor. |
| Previous Processor | Global store **outside** the state graph, keyed by `thread_id`, in `main.py`; updated whenever a processor emits a user-facing message. |
| Reply-to set | Route the entry straight to that processor/subgraph, skipping the supervisor. |
| Reply-to empty | Route to Previous Processor. |
| Status Private/Public/On-demand | Per-message display gating enforced in `main.py`'s send path. |
| Context | Last two message bodies from the sending processor (helper derives it). |
| Receiver-scoped history | `history_for(receiver)` replaces hand-sliced `conversation_history`. |

## Design

### 1. Message envelope (`states.py`)
Add a `MessageEnvelope` TypedDict and a `messages: Annotated[List[MessageEnvelope], last-write-wins]`
field to `GraphicDesigner`. Fields per doc: `id, purpose, model, from_, receiver, reply_to,
processor, messages:[{body,status}], context`. Keep `conversation_history` during the transition
(derive it from `messages` for LLM prompts) so nothing breaks mid-migration.
Reuse the vestigial `pending_node` → repurpose as `reply_to`; reuse `name` → `receiver`.

### 2. Emit helper (`utils.py`)
Central `emit(state, *, body, status, receiver, reply_to, processor, purpose=None, ...)` that appends
an envelope, sets the wire `type`/`question`/`cost` keys the frontend + billing already expect
(see `GENERATION_FLOW_INTERRUPTS.md`), and records the emitting processor as Previous Processor.
Every raw `interrupt({...})` site is migrated to go through this. Pair with the existing plan's
`await_user()` broker (interprets the resume payload: answer / meta / escape).

### 3. Previous-Processor store (`main.py`)
Module-level `PREVIOUS_PROCESSOR: dict[thread_id, processor]` (mirrors how `checkpointer` is keyed),
read when constructing the launch/resume input and to resolve virtual-interrupt routing; cleared on
thread delete alongside the checkpointer.

### 4. Routing (`graph.py`, `nodes.py`)
- Replace `navigator` (the `parseJson` classifier at `nodes.py:196`) with the `supervisor`
  (tool-calling, TBD-resolver) from CONVERSATIONAL_UPGRADE_PLAN §1; retarget the conditional-edge
  dict at `graph.py:97-108` to its decision keys (category→node map preserved).
- Add a **dispatch** entry step: on each turn decide route by envelope —
  `reply_to` set → that node; else virtual interrupt → Previous Processor; else Processor==TBD →
  supervisor. Real-interrupt resumes still land in the paused node automatically.
- Convert the unconditional `Command(goto="navigator")` returns (in
  `interruptBeforeFidelitySuggestion`, `successfulEditHandler`, `nextStep`, `chitChatHandler`,
  `questionHandler`, `unclearHandler`, `extremeFallback`, …) to set `reply_to`/Previous-Processor
  and route via dispatch, so a follow-up stays with the conversation owner.

### 5. Display gating (`main.py` send path, ~`1717-1895`)
Only emit frames where `reply_to == User` and `status == Public`. Keep the existing auto-continue
tuple for transient `*_started`/`cot_progress` frames. `Private` never sent. `On-demand` stored and
sent only when a follow-up asks (e.g. "why?" / "show cost") — a small intent handled by the broker.
Preserve current billing (reads interrupt `cost`) — carry `cost` as a Public/On-demand Second body.

### 6. Receiver-scoped history (`utils.py`, subgraph invokes)
`history_for(receiver)` filters `messages` by `receiver`. Replace the hand-sliced
`conversation_history` passed into `valGraph`/`sugGraph`/`layGraph` invokes and into the
supervisor/LLM prompts.

### 7. Direct inter-processor / subgraph messaging
Where a node calls another subgraph directly today (e.g. `suggestionHandler` → `valGraph`), tag the
request with `reply_to = <caller>` so the reply returns to the caller without the supervisor,
matching the doc's Edit-Manager ↔ Suggestion-Manager example.

## Critical files

- `graphic_designer/states.py` — `MessageEnvelope` + `messages`; repurpose `pending_node`→`reply_to`, `name`→`receiver`.
- `graphic_designer/utils.py` — `emit()`, `history_for()`, `await_user()` broker (reuse `invoke_with_fallback`, `parseJson`, `calculateCost`).
- `graphic_designer/nodes.py` — `supervisor` (replaces `navigator`); migrate interrupt sites to `emit()`/`await_user()`; drop hardcoded `goto="navigator"` bounces.
- `graphic_designer/common_nodes.py` — convert `helperAgent`/`chitChatHandler`/`questionHandler`/`unclearHandler`/`nextStep` first (highest reuse; they run in every subgraph).
- `graphic_designer/subgraphs/*.py` — inherit converted handlers; tag inter-subgraph requests with `reply_to`.
- `graphic_designer/graph.py` — dispatch node + retargeted edges; supervisor wiring.
- `graphic_designer/config.py` — `Processor` name constants; message `Status` constants.
- `main.py` — `PREVIOUS_PROCESSOR` store; envelope construction for launch/resume input; display-gating on send; on-demand retrieval; keep billing + auto-continue + payload shape.

## Reuse (don't reinvent)
- Tool-calling supervisor pattern: `helperAgent` / `create_tool_calling_agent` (`common_nodes.py:56-104`).
- LLM + fallback: `invoke_with_fallback` + `TEXT_MODEL_FALLBACK_CHAIN` (`utils.py`).
- `parseJson` / `calculateCost` (`utils.py`); `Command(goto=, update=)` convention.
- Interrupt wire types + auto-continue tuple + billing (`GENERATION_FLOW_INTERRUPTS.md`, `main.py:1817`).
- The entire CONVERSATIONAL_UPGRADE_PLAN.md (supervisor, broker, task stack, recovery) — this plan
  is its metadata-layer completion, not a competing design.

## Staged rollout (app stays runnable each step)
1. **Data model** — add `MessageEnvelope`/`messages`, `Processor`/`Status` constants, `PREVIOUS_PROCESSOR` store, `emit()`/`history_for()` helpers. No behavior change (envelopes populated but not yet routed on).
2. **Emit everywhere** — route all `interrupt()` sites through `emit()`; every processor stamps From/Receiver/Reply-to/Processor and updates Previous Processor.
3. **Supervisor + dispatch routing** — replace `navigator`, add dispatch conditional edge (Reply-to → Previous Processor → TBD/supervisor); convert the `goto="navigator"` bounces. Keep old navigator reachable behind a flag until parity confirmed.
4. **Display gating** — enforce Reply-to==User & Public on send; wire On-demand retrieval; Private suppressed.
5. **Receiver-scoped history + direct inter-processor messaging** — swap in `history_for()`; tag cross-subgraph calls with `reply_to`.

## Verification
- **Local drive:** compile `graphic_app` and run scripted turns (as `test_generation_pipeline.py` does); assert dispatch targets match the envelope (log each routing decision).
- **Virtual-interrupt test (key):** after a `generation_completed`, send a free-form follow-up; assert it goes to the Previous Processor **without** a supervisor LLM call; then send an off-topic question and assert Reply-to routing / re-presentation.
- **Display test:** emit Private / On-demand / Public bodies; assert only Reply-to==User+Public reach the socket, On-demand appears only when requested.
- **Routing harness:** extend `supervisor_routing_test.py` (already the routing eyeball harness) with envelope-driven cases.
- **End-to-end `/ws`:** scripted client through launch → generation → create-more → suggestion → edit; confirm wallet billing, auto-continue frames, and frontend payload shape unchanged.
- **Regression:** existing generation/editing pipeline tests still pass; per-subgraph history correct after the `history_for()` swap.

## Open risks
- Previous-Processor "outside the state graph" means it is **not checkpointed** — after a process
  restart/reconnect it must be re-derived (fallback: last `receiver` in `messages`, or supervisor).
- Every node's return contract changes (envelope-aware) — do it behind helpers so the diff per node
  is one-line, and keep `conversation_history` alive until step 5 to de-risk.
