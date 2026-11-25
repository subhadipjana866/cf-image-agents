/**
 * Text LLM access.
 *
 * Default: OpenAI (OPENAI_API_KEY secret, model from CHAT_MODEL, default
 * gpt-5.4-mini) through the AI SDK. Without a key: Workers AI Llama 3.3.
 *
 * Llama goes through a thin client instead of workers-ai-provider: Llama 3.3
 * streams each token in both `response` and `choices[0].delta.content`, and
 * workers-ai-provider 4.0.1 emits both, so every token appears twice.
 */
import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";

export const LLAMA_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
export const DEFAULT_OPENAI_MODEL = "gpt-5.4-mini";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

type LlmEnv = { AI: Ai; OPENAI_API_KEY?: string; CHAT_MODEL?: string; OPENAI_BASE_URL?: string };

export function modelName(env: LlmEnv): string {
  return env.OPENAI_API_KEY ? env.CHAT_MODEL || DEFAULT_OPENAI_MODEL : LLAMA_MODEL;
}

/** Yields text deltas of a chat completion. */
export async function* streamChat(
  env: LlmEnv,
  messages: ChatMessage[],
  opts: { maxTokens?: number; signal?: AbortSignal } = {}
): AsyncGenerator<string> {
  if (env.OPENAI_API_KEY) {
    const openai = createOpenAI({ apiKey: env.OPENAI_API_KEY, baseURL: env.OPENAI_BASE_URL || undefined });
    // AI SDK v7 takes system text as `instructions`, not as messages.
    const instructions = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
    const result = streamText({
      model: openai(modelName(env)),
      instructions: instructions || undefined,
      messages: messages.filter((m) => m.role !== "system") as { role: "user" | "assistant"; content: string }[],
      maxOutputTokens: opts.maxTokens,
      abortSignal: opts.signal
    });
    yield* result.textStream;
    return;
  }
  yield* streamLlama(env.AI, messages, opts);
}

async function* streamLlama(
  ai: Ai,
  messages: ChatMessage[],
  opts: { maxTokens?: number; signal?: AbortSignal }
): AsyncGenerator<string> {
  const stream = (await ai.run(LLAMA_MODEL, {
    messages,
    stream: true,
    max_tokens: opts.maxTokens ?? 1024
  })) as ReadableStream<Uint8Array>;

  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      if (opts.signal?.aborted) return;
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        const text = llamaDelta(data);
        if (text) yield text;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function llamaDelta(data: string): string {
  try {
    const chunk = JSON.parse(data);
    // Prefer the OpenAI-style field; fall back to the native one. Never both.
    const choice = chunk.choices?.[0]?.delta?.content;
    if (typeof choice === "string") return choice;
    return typeof chunk.response === "string" ? chunk.response : "";
  } catch {
    return "";
  }
}

/**
 * One JSON-object completion, for the orchestrator's structured steps.
 * OpenAI JSON mode when a key is set; otherwise Llama 3.3 with JSON output.
 */
export async function completeJson(env: LlmEnv & { ROUTER_MODEL?: string }, prompt: string): Promise<string> {
  if (env.OPENAI_API_KEY) {
    const base = (env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
    const res = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${env.OPENAI_API_KEY}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: env.ROUTER_MODEL || modelName(env),
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" }
      })
    });
    if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).slice(0, 300)}`);
    const body = (await res.json()) as { choices: { message: { content: string } }[] };
    return body.choices[0].message.content;
  }
  const out = (await env.AI.run(LLAMA_MODEL, {
    messages: [{ role: "user", content: prompt }],
    response_format: { type: "json_object" },
    max_tokens: 4096
  })) as { response?: unknown };
  return typeof out.response === "string" ? out.response : JSON.stringify(out.response ?? {});
}
