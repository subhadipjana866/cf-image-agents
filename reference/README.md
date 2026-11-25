# image-agents

A task-agent orchestrator for image creation and editing, built on LangGraph.
It implements the "Image Task-Agent System: Architecture v2" design: one
orchestrator that puts every input through four steps (interpret, select,
check readiness, dispatch) and then reacts, one step at a time, to the
outcomes agents return.

## Install

    pip install -e .              # orchestrator only
    pip install -e ".[llm]"       # plus LangChain chat models
    pip install -e ".[ocr]"       # plus pytesseract, one option for text detection
    pip install -e ".[dev]"       # plus pytest

## Run

    python examples/demo.py       # offline walk-through, no model access needed
    pytest                        # all offline

With a real model:

```python
from image_agents import Assistant, ChatModelLLM

llm = ChatModelLLM({"default": "openai:gpt-5.4-mini"})   # optionally one model per step
assistant = Assistant(llm, workdir="./assets")

ref = assistant.add_image("session-1", "banner.png", "coffee brand banner")
reply = assistant.send("session-1", "Make the headline bigger and the button orange")
# reply = {"text": ..., "assets": [...], "client_actions": [...], "request": {...} | None}
```

When `reply["request"]` is set, the system is waiting on the user; send their
next message with `assistant.send` as usual. When `reply["client_actions"]`
contains an interface to open, report what the user did with
`assistant.ui_event(session, "action_completed" | "interface_closed", task_id=..., image=...)`.

## Layout

| Path | What it holds |
| --- | --- |
| `orchestrator.py` | The LangGraph graph: entry, interpret, schedule, select, readiness, dispatch, outcome, finalize, reply, ui_event |
| `steps.py` | The three model-backed steps and prompt rendering |
| `prompts/` | `interpretation.txt`, `selection.txt`, `readiness.txt`, `editor_plan.txt`; the last three take several tasks per call |
| `cards/` | One YAML card per agent (ten) |
| `registry.py` | Card schema; code-side filtering before selection |
| `schemas.py` | Interpretation, Selection, Readiness, Task, TaskEnvelope, Outcome |
| `assets.py` | Versioned asset store: versions, layers, text blocks, reports |
| `agents/base.py` | The agent contract and `SubgraphAdapter` |
| `agents/editor.py` | Image Editor: a planning call plus edit tools |
| `agents/manual_ops.py` | ManualOps: opens an interface, returns a client action |
| `agents/external.py` | External-agent wrapper around a transport you supply |
| `agents/stubs.py` | Placeholder agents for the other eight |
| `vision/text_detection.py` | Text lines, blocks, rotated boxes, glyph masks; the OCR engine is pluggable |
| `app.py` | `Assistant`: `send`, `add_image`, `ui_event`, `tasks` |
| `config.py` | Limits and operator policy |

## Connecting your existing sub-graphs

Eight agents are stubs. Replace each by wrapping your sub-graph in the
adapter, keeping the agent id from its card:

```python
from image_agents import Assistant
from image_agents.agents import SubgraphAdapter
from image_agents.schemas import Outcome

def to_state(env, ctx):                 # task envelope -> your sub-graph's input state
    return {"brief": env.goal, "constraints": env.constraints}

def from_state(final, env, ctx):        # your final state -> one of four outcomes
    if final.get("question"):
        return Outcome(kind="input_required", request=final["question"])
    ref = ctx.store.new_asset(env.session_id, final["image_path"], description=env.goal[:80],
                              made_by="image-generator", instruction=env.goal)
    return Outcome(kind="completed", assets=[ref], generative=True,
                   summary="Created the design.")

generator = SubgraphAdapter("image-generator", your_generator_graph, to_state, from_state)
assistant = Assistant(llm, "./assets", agents=[generator])
```

The Image Editor needs a generative backend for content changes (object
removal, background replacement, pixel text):

```python
from image_agents.agents import ImageEditorAgent
editor = ImageEditorAgent(generative_backend=YourInpaintingModel())   # .edit(image, instruction, box)
```

## How the code maps to the design

- **One task per target and performer, chained.** `interpret` creates the
  tasks; `_effective_assets` makes each chained task work on the result of
  the one before.
- **Pending, supporting, parked.** One task at a time may wait on the user.
  A supporting task leaves it pending and its results are shown to the next
  interpretation call. A new unrelated task parks it; `finalize` brings the
  parked request back when nothing else is waiting.
- **Readiness.** The target asset is resolved in code; other inputs go
  through `readiness.txt` only for cards with `readiness_model: true` (the
  Image Editor). Missing inputs are asked for once, before the agent runs.
- **Assistance.** An agent's need is interpreted and an agent selected
  exactly as for user input, with the requester and its ancestors excluded.
- **Hard rules in code.** User-only result kinds are forced to
  `performed_by: user`; a user-performed task is only ever offered agents
  with `opens_interface: true`; an agent's sub-task is always
  system-performed; a contradiction empties the task list.
- **UI events** enter at `ui_event` and never reach interpretation.
- **Quality gate** runs in `finalize`, once per plan, on the final version,
  when any step in the plan was generative.

## Model calls per input

Calls are batched where the tasks allow it:

- **Selection:** one call assigns an agent to every task formed from one
  input. Each task carries its own `candidate_agents`, so exclusions and the
  user-performed rule still apply per task.
- **Readiness and planning:** for an agent whose card has
  `batchable: true` (the Image Editor), tasks chained one after another are
  checked in one readiness call and planned in one planning call, up to
  `Limits.max_batch_size` (default 4). They stay separate tasks, so each is
  tracked, versioned and reported on its own.

Four edits to one image therefore take four calls (interpret, select,
readiness, plan) instead of thirteen.

Fallbacks keep a weak batched reply from producing a wrong result:

- A selection or readiness reply that leaves a task out is followed by a
  call for that task alone.
- A plan that does not cover the first task is discarded, and the batch is
  planned again one task per call.
- If readiness finds a gap in a later task, the earlier tasks still run and
  the user is asked about the later one when its turn comes, without a
  second check.
- If planning stops at a task that needs input or help, or fails, the tasks
  after it are planned later, on the result they depend on.

Tasks still run one at a time; batching reduces calls, not waiting on the
image tools.

## One deviation from the design document

The document maps a pending task to LangGraph's `interrupt()`. This code does
not use `interrupt()` at the orchestrator level. Each user message is one
graph run, and the pending task is kept in checkpointed state. The reason:
the user's next message must pass through interpretation before anyone knows
whether it supplies the input, asks for help, changes the subject or
cancels. Resuming an interrupt directly would skip that. Your own sub-graphs
can still use `interrupt()` internally.

## Text detection

Grouping lines into blocks, rotated bounding rectangles and glyph masks are
plain Python and NumPy. Reading the text needs an OCR engine, and that is
pluggable: `detect_text_blocks(image, detector)` takes any callable that
returns `(quad, text, confidence)` per line. `tesseract_detector()` is
included as one option. Tesseract is built for documents and is weaker on
stylised text over images, so for generated designs a scene-text engine or a
cloud OCR service wrapped in the same shape will do better.

## What is not implemented

- **Eight agents are stubs** returning placeholder results.
- **A2A transport.** `ExternalAgent` enforces what may be shared and maps the
  reply, but you write the transport against your A2A client.
- **Quality-gate retries.** The gate reports problems; it does not send the
  work back to the producing agent.
- **Parallel execution** of independent tasks. Tasks run one at a time.
- **Spend budget, tracing and observed metrics.** Step, depth and attempt
  limits are enforced; cost accounting and trace export are not included.
- **Durable storage.** The default checkpointer and the asset metadata are
  in memory. Pass a Postgres checkpointer to `Assistant` and back
  `AssetStore` with a database for production.
- **Tension policy and low confidence** both proceed and record an
  assumption; neither asks the user first.

## What has and has not been run

Tested offline with scripted model replies (`tests/`): the multi-target
chain, pending task with supporting task and resolved reply, contradiction,
manual step with parked dependents and a UI event, account actions, agent
assistance, parking and return of a request, cancellation, an amendment, an
external agent with consent, the sub-graph adapter, a missing design, batched
calls with their cap and each fallback,
the registry filters, block grouping, rotated rectangles and the glyph mask.

Not run, because they need services unavailable where this was written:

- `ChatModelLLM` against a real model. The prompts have therefore not been
  exercised by this code with any model, including GPT 5.4-mini.
- `tesseract_detector`, the Tesseract adapter for text detection. The rest of
  the text pipeline is tested with a stand-in engine.
