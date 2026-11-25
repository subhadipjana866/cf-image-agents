import { describe, expect, it } from "vitest";
import { handleInput, type Deps, type OutEvent } from "../src/orchestrator/loop";
import { initialChatState, type ChatState, type InvokeRequest, type InvokeResponse } from "../src/orchestrator/types";

const PNG = btoa("fake-image-bytes");

function harness(script: { interpret: unknown[]; select: unknown[]; invoke: InvokeResponse[] }) {
  const events: OutEvent[] = [];
  const calls: InvokeRequest[] = [];
  const store = new Map<string, Uint8Array>();
  const deps: Deps = {
    chatId: "c1",
    history: [],
    emit: (e) => events.push(e),
    llm: async (step) => {
      const next = script[step].shift();
      if (!next) throw new Error(`unexpected ${step} call`);
      return next;
    },
    invoke: async (req) => {
      calls.push(structuredClone(req));
      const next = script.invoke.shift();
      if (!next) throw new Error("unexpected invoke");
      return next;
    },
    loadImage: async (key) => {
      if (!store.has(key)) throw new Error(`missing ${key}`);
      return `data:image/png;base64,${PNG}`;
    },
    saveImage: async (key, bytes) => void store.set(key, bytes)
  };
  return { deps, events, calls, store };
}

const task = (id: string, goal: string, extra: object = {}) => ({ id, goal, desired_result: "new_design", ...extra });
const assign = (task_id: string, agent_id: string) => ({ task_id, decision: "assign", agent_id });
const done = (reply: string, images = 1): InvokeResponse => ({
  status: "completed",
  reply,
  ask: null,
  cost: 0.06,
  error: null,
  new_versions: Array.from({ length: images }, (_, i) => ({
    number: i + 2,
    parent: 1,
    tool: "t",
    instruction: "i",
    model: "gpt-image-2.5-flare",
    mime: "image/png",
    image_b64: PNG
  }))
});

describe("orchestrator loop", () => {
  it("generates, then a chained edit works on the result of the task before it", async () => {
    const h = harness({
      interpret: [
        {
          input_type: "new_task",
          tasks: [task("t1", "Create a flash sale post"), task("t2", "Make the background navy", { depends_on: ["t1"], desired_result: "edited_design" })]
        }
      ],
      select: [{ assignments: [assign("T1", "image-generator"), assign("T2", "image-editor")] }],
      invoke: [done("Made it."), done("Background is navy.")]
    });
    const state = await handleInput(structuredClone(initialChatState), "make a sale post, navy background", h.deps);

    expect(h.calls.map((c) => c.agent_id)).toEqual(["image-generator", "image-editor"]);
    expect(h.calls[0].session.brief?.prompt).toContain("flash sale");
    expect(h.calls[1].session.versions).toHaveLength(1); // the generator's image, sent inline
    expect(state.assets).toHaveLength(1);
    expect(state.assets[0].versions.map((v) => v.made_by)).toEqual(["image-generator", "image-editor"]);
    expect(state.focus).toEqual({ asset_id: "a1", version: 2 });
    expect(state.tasks.map((t) => t.status)).toEqual(["completed", "completed"]);
    expect(h.store.size).toBe(2);
  });

  it("asks for an image before running the editor when there is none", async () => {
    const h = harness({
      interpret: [{ input_type: "new_task", tasks: [task("t1", "Remove the background", { desired_result: "edited_design" })] }],
      select: [{ assignments: [assign("T1", "image-editor")] }],
      invoke: []
    });
    const state = await handleInput(structuredClone(initialChatState), "remove the background", h.deps);
    expect(h.calls).toHaveLength(0);
    expect(state.pending_id).toBe("T1");
    expect(h.events.some((e) => e.type === "ask")).toBe(true);
  });

  it("an agent question becomes the pending request and the reply resumes the same run", async () => {
    const h = harness({
      interpret: [
        { input_type: "new_task", tasks: [task("t1", "Design a poster")] },
        { input_type: "input_for_pending_task", pending_task_input: { task_id: "T1", input: "Jazz night, Friday" } }
      ],
      select: [{ assignments: [assign("T1", "image-generator")] }],
      invoke: [
        { status: "input_required", reply: "", ask: { question: "What is the event?", options: [] }, new_versions: [], cost: 0, error: null },
        done("Poster made.")
      ]
    });
    let state: ChatState = await handleInput(structuredClone(initialChatState), "design a poster", h.deps);
    expect(state.pending_id).toBe("T1");
    expect(state.tasks[0].request?.text).toBe("What is the event?");

    state = await handleInput(state, "Jazz night, Friday", h.deps);
    expect(h.calls[1].resume).toBe("Jazz night, Friday");
    expect(h.calls[1].question).toBe("What is the event?");
    expect(h.calls[1].thread_id).toBe(h.calls[0].thread_id); // same checkpoint thread
    expect(state.pending_id).toBeNull();
    expect(state.tasks[0].status).toBe("completed");
  });

  it("a new request parks the pending one and brings it back afterwards", async () => {
    const h = harness({
      interpret: [
        { input_type: "new_task", tasks: [task("t1", "Design a poster")] },
        { input_type: "new_task", tasks: [task("t1", "Say hi", { desired_result: "conversation" })] }
      ],
      select: [{ assignments: [assign("T1", "image-generator")] }, { assignments: [assign("T2", "chitchat-handler")] }],
      invoke: [
        { status: "input_required", reply: "", ask: { question: "What is the event?", options: [] }, new_versions: [], cost: 0, error: null },
        done("Hi there!", 0)
      ]
    });
    let state = await handleInput(structuredClone(initialChatState), "design a poster", h.deps);
    state = await handleInput(state, "hello!", h.deps);
    expect(h.calls[1].agent_id).toBe("chitchat-handler");
    expect(state.pending_id).toBe("T1"); // brought back
    const asks = h.events.filter((e) => e.type === "ask");
    expect(asks[asks.length - 1]).toMatchObject({ text: expect.stringContaining("What is the event?") });
  });

  it("cancellation stops the pending task", async () => {
    const h = harness({
      interpret: [
        { input_type: "new_task", tasks: [task("t1", "Remove the background", { desired_result: "edited_design" })] },
        { input_type: "cancellation", affected_task_id: "T1" }
      ],
      select: [{ assignments: [assign("T1", "image-editor")] }],
      invoke: []
    });
    let state = await handleInput(structuredClone(initialChatState), "remove the background", h.deps);
    state = await handleInput(state, "never mind", h.deps);
    expect(state.tasks[0].status).toBe("cancelled");
    expect(state.pending_id).toBeNull();
  });
});
