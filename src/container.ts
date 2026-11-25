import { Container } from "@cloudflare/containers";
import type { InvokeRequest, InvokeResponse } from "./orchestrator/types";

/**
 * Kognie's Python agents (agents_service/) in a Cloudflare Container.
 * One shared instance: the service keeps no per-chat state beyond a run that
 * is waiting on the user, keyed by thread id.
 */
export class AgentsContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "15m";

  constructor(ctx: DurableObjectState<{}>, env: Env) {
    super(ctx, env);
    const vars: Record<string, string> = {};
    if (env.OPENAI_API_KEY) vars.OPENAI_API_KEY = env.OPENAI_API_KEY;
    const gemini = (env as { GEMINI_API_KEY?: string }).GEMINI_API_KEY;
    if (gemini) vars.GEMINI_API_KEY = gemini; // only create-more variations use Gemini
    if (env.AGENT_MODEL) vars.AGENT_MODEL = env.AGENT_MODEL;
    this.envVars = vars;
  }
}

/**
 * POST /invoke on the agents service.
 *
 * AGENTS_SERVICE_URL, when set, sends the call to that URL instead of the
 * Container binding. Local dev uses it (`npm run agents`, the same image under
 * plain Docker) because outbound internet from containers under `vite dev`
 * hangs in Cloudflare's local egress proxy, so the agents cannot reach OpenAI.
 */
export async function invokeAgent(env: Env, req: InvokeRequest): Promise<InvokeResponse> {
  const started = Date.now();
  const direct = (env as { AGENTS_SERVICE_URL?: string }).AGENTS_SERVICE_URL;
  console.log(`[agents] invoke ${req.agent_id} thread=${req.thread_id} via ${direct ? direct : "container"}`);
  const init: RequestInit = {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req)
  };
  const res = direct
    ? await fetch(`${direct.replace(/\/$/, "")}/invoke`, init)
    : await env.AGENTS_CONTAINER.getByName("kognie-agents").fetch("http://agents/invoke", init);
  console.log(`[agents] ${req.agent_id} -> ${res.status} in ${Date.now() - started}ms`);
  if (!res.ok) throw new Error(`agents service ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return (await res.json()) as InvokeResponse;
}
