/**
 * The orchestrator loop: every user input goes through interpret, then each task
 * through select, readiness and dispatch, one at a time, reacting to the outcome
 * the agent returns. Ported from reference/src/image_agents/orchestrator.py and
 * trimmed to three agents; the agents themselves are Kognie's Python agents,
 * reached through `deps.invoke` (the Cloudflare Container).
 *
 * Plain TypeScript with injected dependencies, so it runs in the Durable Object
 * and in tests with scripted replies.
 */
import { cardFor, interpret, isAgentId, select, type JsonLLM, type TaskView } from "./steps";
import {
  FINAL,
  type AgentId,
  type Asset,
  type AssetRef,
  type ChatState,
  type InvokeRequest,
  type InvokeResponse,
  type Task,
  type TaskIntent
} from "./types";

export type OutEvent =
  | { type: "status"; text: string }
  | { type: "text"; text: string; from: string }
  | { type: "image"; asset_id: string; version: number; key: string; mime: string; caption: string }
  | { type: "ask"; text: string; options: string[] };

export type Deps = {
  chatId: string;
  llm: JsonLLM;
  invoke: (req: InvokeRequest) => Promise<InvokeResponse>;
  /** Data URI of a stored image. */
  loadImage: (key: string) => Promise<string>;
  saveImage: (key: string, bytes: Uint8Array, mime: string) => Promise<void>;
  emit: (event: OutEvent) => void;
  /** Persist and broadcast the state mid-turn (after each task), so the UI keeps up. */
  saveState?: (state: ChatState) => void;
  history: { role: "user" | "assistant"; content: string }[];
};

export const LIMITS = { maxStepsPerTurn: 12, maxAttempts: 2 };

const EXT: Record<string, string> = { "image/webp": "webp", "image/png": "png", "image/jpeg": "jpg" };

// ------------------------------------------------------------ views for prompts
export function describeAssets(state: ChatState): unknown[] {
  return state.assets.map((a) => {
    const v = a.versions[a.versions.length - 1];
    return {
      asset_id: a.asset_id,
      latest_version: v.version,
      description: a.description,
      produced_by: v.made_by,
      method: v.made_by === "upload" ? "upload" : "generative",
      in_focus: state.focus?.asset_id === a.asset_id
    };
  });
}

function pendingView(state: ChatState) {
  const t = state.tasks.find((x) => x.id === state.pending_id);
  if (!t) return null;
  return { id: t.id, goal: t.goal, requested: t.request?.text ?? "", options: t.request?.options ?? [] };
}

function activeView(state: ChatState) {
  return state.tasks
    .filter((t) => t.id !== state.pending_id && !FINAL.includes(t.status))
    .map((t) => ({ id: t.id, goal: t.goal, status: t.status === "parked" ? "parked" : "in progress" }));
}

const taskView = (t: Task): TaskView => ({
  id: t.id,
  goal: t.goal,
  desired_result: t.desired_result,
  target_assets: t.target_assets,
  constraints: t.constraints,
  assumptions: t.assumptions,
  depends_on: t.depends_on
});

// ------------------------------------------------------------------ helpers
function addTasks(state: ChatState, intents: TaskIntent[]): string[] {
  const plan = `P${state.counter + 1}`;
  const mapping = new Map(intents.map((it, i) => [it.id, `T${state.counter + i + 1}`]));
  const ids: string[] = [];
  for (const it of intents) {
    state.counter += 1;
    const id = mapping.get(it.id)!;
    state.tasks.push({
      id,
      plan_id: plan,
      goal: it.goal,
      desired_result: it.desired_result,
      target_assets: it.target_assets.length ? it.target_assets : state.focus ? [state.focus] : [],
      constraints: it.constraints,
      assumptions: it.assumptions,
      depends_on: it.depends_on
        .map((d) => mapping.get(d) ?? d)
        .filter((d) => [...mapping.values()].includes(d) || state.tasks.some((t) => t.id === d)),
      supports_task_id: it.supports_task_id ?? null,
      status: "ready",
      agent_id: null,
      attempt: 0,
      request: null,
      answer: null,
      result_assets: [],
      summary: ""
    });
    ids.push(id);
  }
  return ids;
}

function cancelTree(state: ChatState, id: string) {
  for (const t of state.tasks) {
    if ((t.id === id || t.depends_on.includes(id)) && !FINAL.includes(t.status)) {
      t.status = "cancelled";
      if (t.id !== id) cancelTree(state, t.id);
    }
  }
  if (state.pending_id === id) state.pending_id = null;
}

const asset = (state: ChatState, id: string): Asset | undefined => state.assets.find((a) => a.asset_id === id);

/** The asset a task works on: the result of the task it depends on, else its target, else the focus. */
function effectiveAsset(state: ChatState, task: Task): AssetRef | null {
  for (const dep of [...task.depends_on].reverse()) {
    const d = state.tasks.find((t) => t.id === dep);
    const produced = d?.result_assets[d.result_assets.length - 1];
    if (produced) return produced;
  }
  const ref = task.target_assets[0] ?? state.focus;
  return ref && asset(state, ref.asset_id) ? ref : null;
}

function runnable(state: ChatState): Task | undefined {
  return state.tasks.find(
    (t) =>
      t.status === "ready" &&
      t.depends_on.every((d) => {
        const dep = state.tasks.find((x) => x.id === d);
        return !dep || dep.status === "completed" || dep.status === "failed";
      })
  );
}

function ask(state: ChatState, task: Task, text: string, options: string[], from: "agent" | "readiness", deps: Deps) {
  // One task at a time may wait on the user; an older pending one is parked.
  const old = state.tasks.find((t) => t.id === state.pending_id);
  if (old && old.id !== task.id) old.status = "parked";
  task.status = "pending";
  task.request = { text, options, from };
  state.pending_id = task.id;
  deps.emit({ type: "ask", text, options });
}

// ------------------------------------------------------------------ the loop
export async function handleInput(state: ChatState, text: string, deps: Deps): Promise<ChatState> {
  deps.emit({ type: "status", text: "Understanding your request…" });
  const interp = await interpret(deps.llm, {
    text,
    history: deps.history,
    assets: describeAssets(state),
    pendingTask: pendingView(state),
    activeTasks: activeView(state)
  });

  const pending = state.tasks.find((t) => t.id === state.pending_id);
  switch (interp.input_type) {
    case "cancellation": {
      const id = interp.affected_task_id ?? pending?.id;
      if (id) cancelTree(state, id);
      deps.emit({ type: "text", text: "Okay, I've stopped that.", from: "orchestrator" });
      return finalize(state, deps);
    }
    case "input_for_pending_task": {
      const target =
        state.tasks.find((t) => t.id === interp.pending_task_input?.task_id) ?? pending;
      if (target) {
        target.answer = interp.pending_task_input?.input || text;
        target.status = "ready";
        if (state.pending_id === target.id) state.pending_id = null;
      }
      break;
    }
    case "amendment": {
      const affected = state.tasks.find((t) => t.id === interp.affected_task_id);
      if (affected && !FINAL.includes(affected.status)) cancelTree(state, affected.id);
      addTasks(state, interp.tasks);
      break;
    }
    case "new_task":
    case "supporting_task": {
      if (!interp.tasks.length && interp.clarification) {
        deps.emit({ type: "ask", text: interp.clarification.request, options: interp.clarification.options });
        return state;
      }
      if (pending && interp.input_type === "new_task") pending.status = "parked";
      if (pending && interp.input_type === "new_task") state.pending_id = null;
      addTasks(state, interp.tasks);
      break;
    }
  }
  if (!interp.tasks.length && interp.clarification && interp.input_type !== "input_for_pending_task") {
    deps.emit({ type: "ask", text: interp.clarification.request, options: interp.clarification.options });
  }
  await schedule(state, deps);
  return finalize(state, deps);
}

async function schedule(state: ChatState, deps: Deps) {
  for (let step = 0; step < LIMITS.maxStepsPerTurn; step++) {
    if (state.pending_id) return; // waiting on the user
    const task = runnable(state);
    if (!task) return;

    // ---- select (batched over every unassigned ready task)
    if (!task.agent_id) {
      deps.emit({ type: "status", text: "Choosing who should do it…" });
      const unassigned = state.tasks.filter((t) => t.status === "ready" && !t.agent_id);
      const picks = await select(deps.llm, unassigned.map(taskView), describeAssets(state));
      for (const t of unassigned) {
        const sel = picks[t.id];
        if (sel?.decision === "assign" && isAgentId(sel.agent_id)) t.agent_id = sel.agent_id;
        else if (sel?.decision === "missing_inputs") {
          t.agent_id = isAgentId(sel.agent_id) ? sel.agent_id : null;
          if (!t.agent_id) {
            ask(state, t, `To do that I need: ${sel.missing_inputs.join(", ")}.`, [], "readiness", deps);
            return;
          }
        } else {
          t.status = "failed";
          t.summary = sel?.capability_gap ?? "no agent can do this";
          deps.emit({
            type: "text",
            text: `I can't do that here (${t.summary}). I can create new designs, edit an image, or chat.`,
            from: "orchestrator"
          });
        }
      }
      if (task.status !== "ready") continue;
    }

    // ---- readiness: the target image is resolved in code
    const card = cardFor(task.agent_id!);
    const target = effectiveAsset(state, task);
    if (card?.needs_asset && !target && !task.answer) {
      ask(
        state,
        task,
        "Which image should I work on? Upload one, or ask me to create it first.",
        [],
        "readiness",
        deps
      );
      return;
    }

    // ---- dispatch
    await dispatch(state, task, task.agent_id as AgentId, target, deps);
    deps.saveState?.(state);
  }
}

async function dispatch(state: ChatState, task: Task, agentId: AgentId, target: AssetRef | null, deps: Deps) {
  task.status = "in_progress";
  task.attempt += task.answer ? 0 : 1;
  deps.emit({
    type: "status",
    text:
      agentId === "image-generator"
        ? "Designing your image… (this takes about 30 seconds)"
        : agentId === "image-editor"
          ? "Editing the image…"
          : "Thinking…"
  });

  const a = target ? asset(state, target.asset_id) : undefined;
  const base = a ? (a.versions.find((v) => v.version === target?.version) ?? a.versions[a.versions.length - 1]) : undefined;
  const versions = base
    ? [
        {
          number: 1,
          image: await deps.loadImage(base.key),
          made_by: base.made_by,
          tool: base.tool,
          instruction: base.instruction,
          parent: null,
          model: base.model
        }
      ]
    : [];
  const goal = [task.goal, ...task.constraints.map((c) => `- ${c}`)].join("\n");
  const req: InvokeRequest = {
    agent_id: agentId,
    thread_id: `${deps.chatId}:${task.id}:${task.attempt}`,
    message: goal,
    resume: task.answer,
    question: task.request?.text ?? null,
    history: deps.history.slice(-8),
    session: {
      session_id: deps.chatId,
      brief: agentId === "image-generator" ? { prompt: goal, num_images: 1 } : null,
      versions
    }
  };
  const answered = task.answer;
  task.answer = null;
  task.request = null;

  let out: InvokeResponse;
  try {
    out = await deps.invoke(req);
  } catch (err) {
    out = { status: "failed", reply: "", ask: null, new_versions: [], cost: 0, error: String(err) };
  }

  if (out.status === "input_required" && out.ask) {
    ask(state, task, out.ask.question, out.ask.options, "agent", deps);
    return;
  }

  // New images: the generator makes new assets; the editor makes new versions of its target.
  for (const nv of out.new_versions) {
    const bytes = Uint8Array.from(atob(nv.image_b64), (c) => c.charCodeAt(0));
    let owner: Asset;
    if (agentId === "image-editor" && a) owner = a;
    else {
      owner = { asset_id: `a${state.assets.length + 1}`, description: task.goal.slice(0, 80), versions: [] };
      state.assets.push(owner);
    }
    const version = owner.versions.length + 1;
    const key = `chats/${deps.chatId}/${owner.asset_id}/v${version}.${EXT[nv.mime] ?? "bin"}`;
    await deps.saveImage(key, bytes, nv.mime);
    owner.versions.push({
      version,
      key,
      mime: nv.mime,
      made_by: agentId,
      tool: nv.tool,
      instruction: nv.instruction,
      parent: agentId === "image-editor" ? (base?.version ?? null) : null,
      model: nv.model,
      created: Date.now()
    });
    const ref = { asset_id: owner.asset_id, version };
    task.result_assets.push(ref);
    state.focus = ref;
    deps.emit({ type: "image", ...ref, key, mime: nv.mime, caption: `${owner.asset_id} v${version}` });
  }

  if (out.reply) deps.emit({ type: "text", text: out.reply, from: agentId });
  if (out.status === "failed") {
    if (!out.reply) {
      deps.emit({
        type: "text",
        text: `That didn't work${out.error ? ` (${out.error.slice(0, 160)})` : ""}. Nothing was changed.`,
        from: "orchestrator"
      });
    }
    task.status = "failed";
    task.summary = out.error ?? "failed";
    return;
  }
  task.status = "completed";
  task.summary = out.reply.slice(0, 300);
  void answered;
}

/** Bring a parked request back when nothing else is waiting. */
function finalize(state: ChatState, deps: Deps): ChatState {
  if (!state.pending_id) {
    const parked = state.tasks.find((t) => t.status === "parked" && t.request);
    if (parked) {
      parked.status = "pending";
      state.pending_id = parked.id;
      deps.emit({ type: "ask", text: `Back to your earlier request: ${parked.request!.text}`, options: parked.request!.options });
    }
  }
  state.status = null;
  return state;
}
