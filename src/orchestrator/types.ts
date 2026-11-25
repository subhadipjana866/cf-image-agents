/** Data contracts of the orchestrator, ported from reference/src/image_agents/schemas.py. */
import { z } from "zod";

export const AssetRef = z.object({ asset_id: z.string(), version: z.number().int().optional() });
export type AssetRef = z.infer<typeof AssetRef>;

// ------------------------------------------------------------ interpretation
const TaskIntent = z.object({
  id: z.string(),
  goal: z.string(),
  desired_result: z.string().default("other"),
  performed_by: z.enum(["system", "user"]).default("system"),
  target_assets: z.array(AssetRef).default([]),
  constraints: z.array(z.string()).default([]),
  assumptions: z.array(z.string()).default([]),
  depends_on: z.array(z.string()).default([]),
  supports_task_id: z.string().nullish()
});
export type TaskIntent = z.infer<typeof TaskIntent>;

export const Interpretation = z.object({
  analysis: z.string().default(""),
  input_type: z.enum(["new_task", "input_for_pending_task", "supporting_task", "amendment", "cancellation"]),
  pending_task_input: z.object({ task_id: z.string(), input: z.string() }).nullish(),
  affected_task_id: z.string().nullish(),
  conflicts: z
    .array(
      z.object({
        kind: z.enum(["tension", "contradiction"]),
        explanation: z.string().default(""),
        resolutions: z.array(z.string()).default([])
      })
    )
    .default([]),
  tasks: z.array(TaskIntent).default([]),
  clarification: z
    .object({ request: z.string(), options: z.array(z.string()).default([]), why: z.string().default("") })
    .nullish()
});
export type Interpretation = z.infer<typeof Interpretation>;

// ----------------------------------------------------------------- selection
export const Selection = z.object({
  task_id: z.string().nullish(),
  analysis: z.string().default(""),
  decision: z.enum(["assign", "missing_inputs", "no_capable_agent"]),
  agent_id: z.string().nullish(),
  confidence: z.enum(["high", "medium", "low"]).default("medium"),
  missing_inputs: z.array(z.string()).default([]),
  capability_gap: z.string().nullish()
});
export type Selection = z.infer<typeof Selection>;

// -------------------------------------------------------------- state (synced)
export type AssetVersion = {
  version: number;
  key: string; // R2 object key
  mime: string;
  made_by: string; // agent id, or "upload"
  tool: string;
  instruction: string;
  parent: number | null;
  model: string | null;
  created: number;
};

export type Asset = { asset_id: string; description: string; versions: AssetVersion[] };

export type TaskStatus =
  | "ready"
  | "in_progress"
  | "pending" // waiting for the user
  | "parked" // was pending; the user moved on
  | "completed"
  | "failed"
  | "cancelled";
export const FINAL: TaskStatus[] = ["completed", "failed", "cancelled"];

export type Task = {
  id: string;
  plan_id: string;
  goal: string;
  desired_result: string;
  target_assets: AssetRef[];
  constraints: string[];
  assumptions: string[];
  depends_on: string[];
  supports_task_id: string | null;
  status: TaskStatus;
  agent_id: string | null;
  attempt: number;
  /** What the task asked the user, while pending or parked. */
  request: { text: string; options: string[]; from: "agent" | "readiness" } | null;
  /** The user's answer to `request`, consumed by the next dispatch. */
  answer: string | null;
  result_assets: AssetRef[];
  summary: string;
};

export type ChatState = {
  assets: Asset[];
  tasks: Task[];
  pending_id: string | null;
  focus: AssetRef | null;
  counter: number;
  /** Shown in the UI while a turn runs. */
  status: string | null;
};

export const initialChatState: ChatState = {
  assets: [],
  tasks: [],
  pending_id: null,
  focus: null,
  counter: 0,
  status: null
};

// ------------------------------------------------------ agents service contract
export type AgentId = "image-generator" | "image-editor" | "chitchat-handler";
export const AGENT_IDS: AgentId[] = ["image-generator", "image-editor", "chitchat-handler"];

export type InvokeRequest = {
  agent_id: AgentId;
  thread_id: string;
  message: string;
  resume?: string | null;
  question?: string | null;
  history: { role: "user" | "assistant"; content: string }[];
  session: {
    session_id: string;
    brief?: { prompt: string; aspect_ratio?: string | null; num_images?: number } | null;
    versions: {
      number: number;
      image: string;
      made_by: string;
      tool: string;
      instruction: string;
      parent: number | null;
      model: string | null;
    }[];
  };
};

export type InvokeResponse = {
  status: "completed" | "input_required" | "failed";
  reply: string;
  ask: { question: string; options: string[] } | null;
  new_versions: {
    number: number;
    parent: number | null;
    tool: string;
    instruction: string;
    model: string | null;
    mime: string;
    image_b64: string;
  }[];
  cost: number;
  error: string | null;
};
