import { AIChatAgent } from "@cloudflare/ai-chat";
import { routeAgentRequest } from "agents";
import { createUIMessageStream, createUIMessageStreamResponse, type UIMessage } from "ai";
import { AgentsContainer, invokeAgent } from "./container";
import { completeJson } from "./llm";
import { describeAssets, handleInput, type OutEvent } from "./orchestrator/loop";
import { extractJson } from "./orchestrator/steps";
import { initialChatState, type ChatState } from "./orchestrator/types";

export { AgentsContainer };

/** Plain-text view of the chat for prompts; images become "[image a1 v2]". */
function toHistory(messages: UIMessage[]) {
  return messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.parts
        .map((p) => {
          if (p.type === "text") return p.text;
          if (p.type === "data-image") {
            const d = p.data as { asset_id: string; version: number };
            return `[image ${d.asset_id} v${d.version}]`;
          }
          if (p.type === "data-ask") return (p.data as { text: string }).text;
          return "";
        })
        .join("\n")
    }))
    .filter((m) => m.content.trim());
}

function toDataUri(bytes: ArrayBuffer, mime: string): string {
  const u8 = new Uint8Array(bytes);
  let bin = "";
  for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return `data:${mime};base64,${btoa(bin)}`;
}

/**
 * One Durable Object per chat. Holds the chat (AIChatAgent's SQLite), the
 * orchestrator state (tasks, pending request, assets; synced to the UI) and
 * runs the interpret → select → readiness → dispatch loop on each message.
 */
export class ImageChatAgent extends AIChatAgent<Env, ChatState> {
  initialState = initialChatState;

  async onChatMessage() {
    const last = this.messages[this.messages.length - 1];
    const text = last?.parts.map((p) => (p.type === "text" ? p.text : "")).join("").trim() ?? "";
    const history = toHistory(this.messages.slice(0, -1));
    const env = this.env;
    const chatId = this.name;

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        let n = 0;
        const emit = (e: OutEvent) => {
          if (e.type === "status") {
            this.setState({ ...this.state, status: e.text });
            writer.write({ type: "data-status", data: { text: e.text }, transient: true });
            return;
          }
          const id = `p${n++}`;
          if (e.type === "text") {
            writer.write({ type: "text-start", id });
            writer.write({ type: "text-delta", id, delta: e.text });
            writer.write({ type: "text-end", id });
          } else if (e.type === "image") {
            writer.write({ type: "data-image", id, data: e });
          } else {
            writer.write({ type: "data-ask", id, data: e });
          }
        };

        let state = structuredClone(this.state);
        try {
          state = await handleInput(state, text, {
            chatId,
            history,
            emit,
            saveState: (s) => this.setState({ ...structuredClone(s), status: this.state.status }),
            llm: async (_step, prompt) => extractJson(await completeJson(env, prompt)),
            invoke: (req) => invokeAgent(env, req),
            loadImage: async (key) => {
              const obj = await env.IMAGES_BUCKET.get(key);
              if (!obj) throw new Error(`image ${key} not found`);
              return toDataUri(await obj.arrayBuffer(), obj.httpMetadata?.contentType ?? "image/png");
            },
            saveImage: async (key, bytes, mime) => {
              await env.IMAGES_BUCKET.put(key, bytes, { httpMetadata: { contentType: mime } });
            }
          });
        } catch (err) {
          console.error(err);
          emit({ type: "text", text: `Something went wrong: ${String(err).slice(0, 200)}`, from: "orchestrator" });
        }
        this.setState({ ...state, status: null });
      }
    });
    return createUIMessageStreamResponse({ stream });
  }

  /** POST .../upload (multipart "file"): the image becomes a new asset in focus. */
  async onRequest(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "POST" && url.pathname.endsWith("/upload")) {
      const form = await request.formData();
      const file = form.get("file");
      if (!(file instanceof File) || !file.type.startsWith("image/")) {
        return Response.json({ error: "expected an image file" }, { status: 400 });
      }
      const state = structuredClone(this.state);
      const asset_id = `a${state.assets.length + 1}`;
      const ext = file.type.split("/")[1]?.replace("jpeg", "jpg") ?? "bin";
      const key = `chats/${this.name}/${asset_id}/v1.${ext}`;
      await this.env.IMAGES_BUCKET.put(key, await file.arrayBuffer(), { httpMetadata: { contentType: file.type } });
      state.assets.push({
        asset_id,
        description: String(form.get("description") || file.name || "uploaded image").slice(0, 80),
        versions: [
          { version: 1, key, mime: file.type, made_by: "upload", tool: "upload", instruction: "uploaded by the user", parent: null, model: null, created: Date.now() }
        ]
      });
      state.focus = { asset_id, version: 1 };
      this.setState(state);
      return Response.json({ asset_id, version: 1, key, assets: describeAssets(state) });
    }
    return super.onRequest(request);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    // Images are stored under chats/<chat>/<asset>/v<n>.<ext>.
    if (request.method === "GET" && url.pathname.startsWith("/images/chats/")) {
      const obj = await env.IMAGES_BUCKET.get(decodeURIComponent(url.pathname.slice("/images/".length)));
      if (!obj) return new Response("Not found", { status: 404 });
      return new Response(obj.body, {
        headers: {
          "content-type": obj.httpMetadata?.contentType ?? "application/octet-stream",
          "cache-control": "private, max-age=31536000, immutable"
        }
      });
    }
    return (await routeAgentRequest(request, env)) ?? new Response("Not found", { status: 404 });
  }
} satisfies ExportedHandler<Env>;
