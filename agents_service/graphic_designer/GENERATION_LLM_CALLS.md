# Which AI calls happen when a user makes a design

This explains, in plain terms, every time we send a request to an AI model while
a user is generating one image — what asks, why, and how many.

## First, three words you need

**Node** — one step in the pipeline. The app moves through steps in order, and
each step either does some work or asks the user a question. Node names in this
doc are the function names in `graphic_designer/nodes.py`.

**Call** — one request to an AI provider (Google, OpenAI, Anthropic) and its
answer. Calls cost money and take time. Fewer calls = cheaper and faster.

**Role** — what *kind* of AI work a step needs, rather than which model does it.
We use four:

| Role | Means | Used by |
|---|---|---|
| `classify_extract` | "read this text and tell me what it is / pull facts out of it" | The early steps |
| `cot_reasoning` | "think through the design and write the final instructions" | The 7 thinking steps |
| `copywriting` | "write the words that appear on the poster" | One thinking step |
| `image` | "draw the picture" | The final step |

The point of roles: **no step names a model any more.** A step says "I need
classification work", and one config file (`config.py`, `MODEL_ROLES`) decides
which company and which model actually handles it. To switch models, you edit
that one file — you don't touch any pipeline code.

## The steps, in order

Reading the table: "Calls" is how many AI requests that step makes. Many steps
make **zero** — they're ordinary code, not AI.

| # | Step | Calls | What it's doing |
|---|---|---|---|
| 1 | `navigator` | 1 (sometimes 2) | Reads the user's message and decides which flow this is: new design, edit, resize, suggestions, or just a question. The 2nd call only happens for non-design messages, to tell a question from small talk |
| 2 | `generationManager` | 0 | Just shows the "generating…" spinner |
| 3 | `scopeGate` | 1 | Checks the user asked for the kind of design they picked. Someone who selected "Festive Sale" but typed "make me a menu card" gets caught here |
| 3b | `scopeMismatchGate`, `scopeAbortGate` | 0 | Only ask the user about the mismatch above. They reuse the answer step 3 already got, so nothing is re-checked |
| 4 | `intentClarifier` | 1 | Turns the messy user prompt into a clean, structured design brief |
| 5 | `intentRouter` | 1 | Checks the brief for brand-rule violations and whether it's too vague to design from |
| 6 | `manifestFieldsGate` | 0 or 1 | Some design types require certain details (a date, a discount %). This pulls them out of what the user already typed, so we don't ask for things they already told us |
| 6b | `manifestFieldsAsk` | 0 or 1 | If the user replies in a sentence instead of filling boxes, this reads the sentence |
| 7 | `photoRequiredGate` | 0 | Checks if a photo is needed. Plain code |
| 8 | `clarificationGate` | 0 or 1 | If step 5 said the request was too vague, we ask the user, then rebuild the brief with their answer |
| 9 | `imageAnalyzerLayoutSelector` | 0 | Picks the layout template and measures the photo. **No AI** — this is image maths |
| 10 | `logoDecisionGate` | 0 | Decides whether a logo goes in. Plain code |
| 11 | `briefFinalizerAndBriefReview` | 1 | Thinking step 1 of 7: reviews the brief and can stop to ask the user one blocking question |
| 12 | `midCotClarificationAsk` | 0 | Asks that question, if there was one |
| 13 | `cotFusionRemainder` | 6 (or 5) | Thinking steps 2–7: writes the copy, plans the layout, picks the styling, and compiles the final instructions |
| 14 | `generation` | 0 | Picks which image model to use |
| 15 | `run*Model` | 1 image | Actually draws the picture |
| 16 | `logoCompositor` | 0 | Pastes the logo on afterwards. Plain image editing |
| 17 | `interruptBeforeFidelitySuggestion` | 0 | Shows the finished image and asks what's next |

### Steps that use no AI at all

`generationManager`, `scopeMismatchGate`, `scopeAbortGate`, `photoRequiredGate`,
`imageAnalyzerLayoutSelector`, `logoDecisionGate`, `midCotClarificationAsk`,
`generation`, `logoCompositor`, `interruptBeforeFidelitySuggestion`.

Worth knowing: **choosing the layout and placing the logo are not AI decisions.**
They read a template file and measure pixels. The name
`imageAnalyzerLayoutSelector` makes it sound like AI, but it isn't.

## The 7 thinking steps (steps 11 and 13)

These are the heart of it — where the design actually gets decided. They run one
after another, each one seeing everything the previous ones decided.

| # | Stage | Sends the layout image? | What it decides |
|---|---|---|---|
| 1 | `stage_brief_review` | yes | Reads the brief, spots gaps. Can pause and ask the user |
| 2 | `stage_0_text_generation` | no | Writes the actual words: headline, offer, button text. These get locked — no later step may change them |
| 3 | `stage_1_element_mapping` | yes | Decides what goes in each area of the layout |
| 4 | `stage_2_style_execution` | yes | Colours, fonts, lighting, mood |
| 5 | `stage_3_negative_contract` | no | Lists what must NOT appear (placeholder boxes, wireframe lines) |
| 6 | `compliance_review` | no | Double-checks the plan against the rules before committing |
| 7 | `final_prompt_and_negative` | yes | Compiles everything into the final instruction sent to the image model |

**Why they're split across two steps (11 and 13):** stage 1 can stop and ask the
user a question. If all 7 ran in one step, answering that question would make the
app re-run everything from the beginning. Splitting them means stage 1's result
is saved before the question is asked, so answering only costs the remaining
stages.

**Why the later stages are the expensive ones:** every stage resends everything
the earlier stages wrote. So stage 7's request is much bigger than stage 1's.
That's why cutting a *late* stage saves far more than cutting an early one.

## How many calls, really

These are measured numbers from real runs, not estimates
(`eval/baselines/<version>.jsonl`):

| Situation | AI calls | Notes |
|---|---|---|
| Normal design, clear prompt | **15** | The typical case |
| Vague prompt (we ask a question) | **16** | One extra to rebuild the brief |
| With a photo attached | **15** | |
| With a logo | **14** | |
| User picked the wrong design type | **3** | Stops early |
| Custom design goal | **2** | Skips the whole pipeline, goes straight to drawing |
| **"Give me another version"** | **2** | Reuses everything from the first design |
| A plain question (no design) | **11** | |

Plus one image call whenever a picture is actually drawn.

**Two things to read carefully in those numbers:**

1. **Failed attempts are counted.** If Google is out of quota and OpenAI answers
   instead, that's 2 calls, not 1. We count it that way on purpose — a failed
   attempt still costs time. So don't read the Google/OpenAI split as a
   load-balance ratio; right now most Google rows are quota errors followed by an
   OpenAI row that worked.

2. **"Give me another version" costs 2 calls instead of 15.** It reuses the brief
   and the design instructions from the first image and only picks a new layout.
   Making the other flows (editing, resizing) work this way is where the biggest
   remaining saving is.

To re-measure: `python3 graphic_designer/eval/harness.py --record`

## Switches you can flip

All off/safe by default. Each is an environment variable — no code change.

| Switch | Turn it on to… | Saves |
|---|---|---|
| `KOGNIE_CONSOLIDATED_FRONT_CHAIN=on` | Let step 1 answer both its questions in one call | 1 call on non-design messages |
| `COT_STAGES_PROFILE=lean` | Build the final thinking step with code instead of AI | 1 call per design, and it's the biggest one |
| `KOGNIE_REUSE_SESSION_STATE=off` | Go back to rebuilding everything when a user reconnects | costs ~13 extra calls |
| `KOGNIE_ROUTER_STRATEGIES=off` | Stop spreading load across providers | no change to count |

The first two are off by default because they change what the AI produces, even
if only slightly. Test them with the harness before turning them on for real users.

## Where models are configured

Everything lives in `graphic_designer/config.py`:

- **`MODEL_ROLES`** — for each role: which provider and model to use first, and
  which to try if that one fails. Each entry has two model names: the one we send
  to the provider, and the one we bill the customer against. They're kept
  separate on purpose, so changing which model does the work never changes what
  the customer pays.

- **`PROVIDER_LIMITS`** — how many requests per minute each provider allows.
  **Currently all blank.** Fill these in from each provider's dashboard; leaving
  them blank simply turns the throttling off, which is safer than guessing a
  number.

- **`model_router.preflight()`** — set `KOGNIE_PREFLIGHT=on` and the app tests
  every configured model once at startup. If a model can't be used (wrong plan,
  no quota), you find out at boot instead of when a customer hits it.
