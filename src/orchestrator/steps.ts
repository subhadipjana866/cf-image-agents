/**
 * The model-backed orchestrator steps: interpret and select.
 * Ported from reference/src/image_agents/steps.py; the prompt files are the
 * originals (src/orchestrator/prompts), the cards are Kognie's three cards.
 */
import interpretationPrompt from "./prompts/interpretation.txt?raw";
import selectionPrompt from "./prompts/selection.txt?raw";
import cards from "./cards.json";
import { Interpretation, Selection, type AgentId } from "./types";

export type JsonLLM = (step: "interpret" | "select", prompt: string) => Promise<unknown>;

export type Card = (typeof cards)[number];
export const CARDS = cards as Card[];
export const cardFor = (id: string): Card | undefined => CARDS.find((c) => c.view.id === id);

const POLICY = {
  external_agents_allowed: false,
  consent_required_before_sharing: true,
  user_only_results: ["account_action"]
};

/** Fill {{placeholders}}. Non-string values are rendered as JSON. */
export function render(template: string, values: Record<string, unknown>): string {
  let text = template;
  for (const [key, value] of Object.entries(values)) {
    const v = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    text = text.split(`{{${key}}}`).join(v);
  }
  return text;
}

/** Parse the first JSON object in a model reply, tolerating code fences. */
export function extractJson(text: string): unknown {
  const fenced = text.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
  const body = fenced ? fenced[1] : text;
  const start = body.indexOf("{");
  if (start < 0) throw new Error("model reply contains no JSON object");
  // Trim trailing prose after the last closing brace.
  const end = body.lastIndexOf("}");
  return JSON.parse(body.slice(start, end + 1));
}

export async function interpret(
  llm: JsonLLM,
  args: {
    text: string;
    history: { role: string; content: string }[];
    assets: unknown[];
    pendingTask: unknown | null;
    activeTasks: unknown[];
  }
): Promise<Interpretation> {
  const prompt = render(interpretationPrompt, {
    source: "user",
    input: args.text,
    conversation_history:
      args.history
        .slice(-20)
        .map((m) => `${m.role}: ${m.content}`)
        .join("\n") || "(none)",
    assets: args.assets.length ? args.assets : "(none)",
    pending_task: args.pendingTask ?? "(none)",
    active_tasks: args.activeTasks.length ? args.activeTasks : "(none)"
  });
  const interp = Interpretation.parse(await llm("interpret", prompt));
  // Rule enforced in code, whatever the model returned: a contradiction holds the input.
  if (interp.conflicts.some((c) => c.kind === "contradiction") && interp.clarification) {
    interp.tasks = [];
  }
  return interp;
}

export type TaskView = {
  id: string;
  goal: string;
  desired_result: string;
  target_assets: unknown[];
  constraints: string[];
  assumptions: string[];
  depends_on: string[];
};

/** Choose an agent for each task in one call; a task the reply leaves out is asked again alone. */
export async function select(
  llm: JsonLLM,
  tasks: TaskView[],
  assets: unknown[]
): Promise<Record<string, Selection>> {
  const allowed = new Set<string>(CARDS.map((c) => c.view.id));
  const views = tasks.map((t) => ({
    ...t,
    performed_by: "system",
    supports_task_id: null,
    origin: "user",
    candidate_agents: [...allowed]
  }));
  const prompt = render(selectionPrompt, {
    registry: CARDS.map((c) => c.view),
    policy: POLICY,
    assets: assets.length ? assets : "(none)",
    tasks: views
  });
  const raw = (await llm("select", prompt)) as Record<string, unknown>;
  let items = raw.assignments as unknown[] | undefined;
  if (!items && "decision" in raw && tasks.length === 1) items = [raw];
  if (tasks.length === 1 && items?.length === 1) items = [{ ...(items[0] as object), task_id: tasks[0].id }];

  const out: Record<string, Selection> = {};
  for (const item of items ?? []) {
    const parsed = Selection.safeParse(item);
    if (!parsed.success || !parsed.data.task_id || out[parsed.data.task_id]) continue;
    const sel = parsed.data;
    if (!tasks.some((t) => t.id === sel.task_id)) continue;
    if (sel.decision !== "no_capable_agent" && !allowed.has(sel.agent_id ?? "")) {
      out[sel.task_id!] = {
        ...sel,
        decision: "no_capable_agent",
        capability_gap: `selection chose an agent not allowed for this task: ${sel.agent_id}`
      };
    } else out[sel.task_id!] = sel;
  }
  const missing = tasks.filter((t) => !out[t.id]);
  if (missing.length && tasks.length > 1) {
    for (const t of missing) Object.assign(out, await select(llm, [t], assets));
  }
  for (const t of tasks) {
    out[t.id] ??= {
      task_id: t.id,
      analysis: "",
      decision: "no_capable_agent",
      confidence: "low",
      missing_inputs: [],
      capability_gap: "selection returned no assignment for this task"
    };
  }
  return out;
}

export const isAgentId = (id: string | null | undefined): id is AgentId =>
  id === "image-generator" || id === "image-editor" || id === "chitchat-handler";
