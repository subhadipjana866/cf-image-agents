# Image Chat Agents on Cloudflare

A chat app where users create and edit images in conversation. Every message goes
through the orchestration loop of the original `image_agents` project
(**interpret → select → readiness → dispatch**), rebuilt on Cloudflare's Agents SDK.
The work is done by three agents taken from Kognie's Python code: an image
generator, an image editor and a chitchat agent.

```
Browser (React, useAgentChat)
   │  WebSocket: chat messages, live state      POST upload, GET /images/*
   ▼
Worker ── ImageChatAgent  (AIChatAgent Durable Object, one per chat)
            │  SQLite: messages, tasks, pending request, assets and versions
            │  loop: interpret → select → readiness → dispatch   (src/orchestrator)
            │  LLM:  OpenAI (gpt-5.4-mini) / Workers AI Llama 3.3 fallback
            ├── R2: image bytes (chats/<chat>/<asset>/v<n>.<ext>)
            └── AgentsContainer (Cloudflare Container) ── agents_service/ (Python)
                    image-generator   gpt-image-2.5-flare, freeform / more / regenerate
                    image-editor      15 generative edit tools, gpt-image-2.5-flare
                    chitchat-handler  small talk, capabilities
```

## Required components

| Requirement | Where |
| --- | --- |
| LLM | OpenAI `gpt-5.4-mini` for interpretation and selection (`src/llm.ts`, `ROUTER_MODEL`); the agents run on `gpt-5.4` (`AGENT_MODEL`). Without `OPENAI_API_KEY` the orchestrator falls back to Workers AI `@cf/meta/llama-3.3-70b-instruct-fp8-fast`. |
| Workflow / coordination | `ImageChatAgent` Durable Object runs the loop and coordinates tasks, chaining and questions; the agents run in a Cloudflare Container (`AgentsContainer`). |
| User input | Chat UI served as Worker static assets (`src/client`), connected with `useAgentChat`; image upload. |
| Memory / state | Durable Object SQLite (messages via `AIChatAgent`, orchestrator state via `setState`, synced live to the UI) and R2 for images. |

## How a message is handled

1. **Interpret** (`src/orchestrator/steps.ts`, prompt `prompts/interpretation.txt`, the
   original): classifies the input as `new_task`, `input_for_pending_task`,
   `supporting_task`, `amendment` or `cancellation`, and splits it into tasks
   ("make the background navy and change the badge to 70% off" becomes two chained edits).
2. **Select** (`prompts/selection.txt` with Kognie's three agent cards,
   `src/orchestrator/cards.json`): one call assigns an agent to every new task.
3. **Readiness** (in code): an editor task needs an image. It uses the result of the task
   it depends on, its target, or the image in focus; with none, the user is asked.
4. **Dispatch** (`src/orchestrator/loop.ts`): calls the Python agent with the task and the
   current image. The outcome is one of:
   - completed: new images are stored in R2 as new assets (generator) or new versions (editor)
   - input_required: the agent's `ask_user` question becomes the pending request; the
     user's answer resumes the same LangGraph run
   - failed

One request at a time waits on the user. A new unrelated request parks it, and it comes
back when nothing else is waiting.

## The agents (`agents_service/`)

`graphic_designer/`, `image_agents/` and `agent_navigation/` are copied from
Kognie-python (working tree of 2026-11-25) without changes. `service.py` wraps them in
FastAPI with the same build code as `graphic_designer/agents/build.py`, limited to:

- **image-generator**: `generate_freeform_design`, `create_more_variations`,
  `regenerate_design` (catalog pipeline left out). Variations use Gemini, so they need
  `GEMINI_API_KEY`.
- **image-editor**: the 15 generative edit tools (exact Pillow tools, upscale, restore,
  export left out).
- **chitchat-handler**: unchanged.

`container_shims.py` adapts the environment, not the code. It stubs the modules these
agents never reach that would pull in torch (RAG pipeline, OCR layers, FAISS), and
replaces S3 with an in-memory store whose bytes go back to the Worker and into R2.

## Run locally

Needs Node 22 (`nvm use`), Docker, and a Cloudflare login (`npx wrangler login`;
Workers AI runs remotely even in dev).

```bash
npm install
cp .dev.vars.example .dev.vars   # add OPENAI_API_KEY
npm run agents                   # terminal 1: the Python agents on :8090 (Docker)
npm run dev                      # terminal 2: http://localhost:5173
```

`AGENTS_SERVICE_URL` in `.dev.vars` points the Worker at that Docker container. Without
it the Worker uses the Container binding. That is the production path, but under
`vite dev` the container's outbound internet hangs in Cloudflare's local egress proxy,
so the agents cannot reach OpenAI.

Tests (offline, scripted model replies):

```bash
npm test
npm run typecheck
```

## Layout

| Path | What |
| --- | --- |
| `src/server.ts` | Worker entry, `ImageChatAgent` Durable Object, image and upload routes |
| `src/orchestrator/` | The loop, steps, types, prompts and cards |
| `src/container.ts` | `AgentsContainer` and the call to `/invoke` |
| `src/llm.ts` | OpenAI / Workers AI access |
| `src/client/` | React chat UI |
| `agents_service/` | Kognie's Python agents and their HTTP wrapper (Dockerfile) |
| `reference/` | The original Python orchestrator this loop is ported from |
| `prompt_history/` | Every prompt and reply of the build session |
| `docs/PLAN.md` | Build plan |
