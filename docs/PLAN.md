# Image Chat Agents on Cloudflare: Build Plan

Goal: a chat app where users create and edit images in conversation. It keeps
the orchestration loop of the original Python project (`reference/`), which
puts every input through **interpret → select → readiness → dispatch**, and
rebuilds it on Cloudflare's Agents SDK.

## Required components and where each one lives

| Requirement | Implementation |
| --- | --- |
| LLM | OpenAI `gpt-5.4-mini` (external LLM, `OPENAI_API_KEY` secret, model set by `CHAT_MODEL`) for interpretation, selection, readiness, edit planning and chitchat, with JSON-schema output for the structured steps. Falls back to Workers AI `@cf/meta/llama-3.3-70b-instruct-fp8-fast` when no key is set. Optional Cloudflare AI Gateway via `OPENAI_BASE_URL`. |
| Workflow / coordination | `ImageChatAgent` (an `AIChatAgent` Durable Object, one per chat) runs the orchestrator loop. Image jobs run as a Cloudflare **Workflow** (`ImageJobWorkflow`) with durable, retried steps; progress and completion are reported back to the agent. |
| User input (chat or voice) | React UI served as Worker static assets. Text through `useAgentChat`; voice recorded in the browser and transcribed with `@cf/openai/whisper-large-v3-turbo`, then fed into the same loop. |
| Memory / state | Durable Object SQLite: chat messages (built into `AIChatAgent`), tasks, plans, the pending request and asset versions. Image bytes in R2. Orchestrator state is synced live to the UI through agent state. |

## Agents (three)

| Agent | Does | Runs |
| --- | --- | --- |
| `image-generator` | Creates a new image from a brief. Optional reference images. | Workflow step: `@cf/black-forest-labs/flux-2-klein-9b` |
| `image-editor` | Changes an existing image: content, style, colours, text, add or remove objects. Needs readiness (target image + clear instruction) and a planning call that turns the request into an edit prompt. | Workflow step: FLUX.2 klein with the current version as `input_image_0` (resized to 512 px or less) |
| `chitchat` | Greetings, questions about the app, small talk, and anything that is not an image task. | Inline, streamed LLM reply |

Dropped from the original: manual-ops (UI interfaces), external/A2A agents
and consent, layering, critic, evaluator and quality gate, suggestion, slop
terminator, question handler (folded into chitchat), and OCR text detection.

## What is kept from the original design

- Four steps per input: interpretation, selection, readiness, dispatch, then
  one step at a time reacting to the agent's outcome.
- The input types: `new_task`, `input_for_pending_task`, `supporting_task`,
  `amendment`, `cancellation`.
- Tasks with statuses, chaining (`depends_on`), and `_effective_assets` so a
  chained edit works on the previous result.
- Four outcome kinds: `completed`, `input_required`, `assistance_required`,
  `failed`.
- One pending request at a time; parking and returning a request.
- Hard rules enforced in code, not left to the model.
- Versioned assets: every edit makes a new version; earlier versions stay.

## Steps

0. **Scaffold.** Move the Python project to `reference/`. Worker + Vite +
   React; `wrangler.jsonc` with AI, R2, Images, Durable Object and Workflow
   bindings. A first chat that streams a Llama 3.3 reply.
1. **Contracts and state.** TypeScript schemas (zod) ported from
   `schemas.py`; three agent cards; SQLite tables for tasks and assets.
2. **Orchestrator core.** Port the loop from `orchestrator.py` as plain
   TypeScript with an injected LLM, so it can be tested offline. Port the
   flow tests with scripted model replies.
3. **Prompts.** Port and trim `interpretation`, `selection`, `readiness` and
   `editor_plan` prompts for three agents and Llama 3.3 JSON output.
4. **Agents and Workflow.** Chitchat inline; generator and editor through
   `ImageJobWorkflow`, writing new asset versions to R2.
5. **UI.** Chat, image upload, image cards with versions, pending-request
   options as buttons, live job progress.
6. **Voice input.** Mic button, Whisper transcription.
7. **Deploy and document.** `wrangler deploy`, README, final prompt history.

## Prompt history

Every turn is exported to `prompt_history/` by a Stop hook
(`.claude/hooks/save_chat_history.py`). See `prompt_history/PROMPTS.md`.

## Change of plan (2026-11-25)

The agents are not rebuilt in TypeScript. On the user's request the generator, editor
and chitchat agents are Kognie's Python agents (`agents_service/`), run in a Cloudflare
Container and called by the Durable Object's dispatch step. Images use OpenAI
`gpt-image-2.5-flare`, not FLUX. The loop runs in the Durable Object directly, so
Cloudflare Workflows and voice input are not used. The project is pushed to GitHub,
not deployed. See README.md for the current design.
