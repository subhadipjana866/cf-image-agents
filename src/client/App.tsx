import { useAgentChat } from "@cloudflare/ai-chat/react";
import { useAgent } from "agents/react";
import type { UIMessage } from "ai";
import { useEffect, useRef, useState } from "react";
import type { ChatState } from "../orchestrator/types";

type ImagePart = { asset_id: string; version: number; key: string; caption: string };
type AskPart = { text: string; options: string[] };
type ChatMessage = UIMessage<unknown, { image: ImagePart; ask: AskPart; status: { text: string } }>;

function chatId(): string {
  const url = new URL(window.location.href);
  let id = url.searchParams.get("chat");
  if (!id) {
    id = crypto.randomUUID().slice(0, 8);
    url.searchParams.set("chat", id);
    window.history.replaceState(null, "", url);
  }
  return id;
}

const imageUrl = (key: string) => `/images/${key}`;

export function App() {
  const [id] = useState(chatId);
  const [state, setState] = useState<ChatState | null>(null);
  const agent = useAgent<ChatState>({ agent: "ImageChatAgent", name: id, onStateUpdate: (s) => setState(s) });
  const { messages, sendMessage, status, error } = useAgentChat<unknown, ChatMessage>({ agent });
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = status === "submitted" || status === "streaming";

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView.
    void endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, state?.status]);

  const send = (text: string) => {
    if (!text.trim() || busy) return;
    sendMessage({ role: "user", parts: [{ type: "text", text }] });
    setInput("");
  };

  const upload = async (file: File) => {
    setUploading(true);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("description", file.name);
      const res = await fetch(`/agents/image-chat-agent/${id}/upload`, { method: "POST", body });
      if (!res.ok) throw new Error(await res.text());
    } catch (err) {
      alert(`Upload failed: ${err}`);
    } finally {
      setUploading(false);
    }
  };

  const lastAsk = (() => {
    const m = messages[messages.length - 1];
    if (m?.role !== "assistant") return null;
    const parts = m.parts.filter((p) => p.type === "data-ask");
    return parts.length ? (parts[parts.length - 1].data as AskPart) : null;
  })();
  const focus = state?.focus
    ? state.assets.find((a) => a.asset_id === state.focus!.asset_id)?.versions.find((v) => v.version === state.focus!.version)
    : undefined;

  return (
    <div className="layout">
      <main className="chat">
        <header>
          <span>Image Chat</span>
          <span className="muted">chat {id}</span>
        </header>
        <ol className="messages">
          {messages.length === 0 && (
            <li className="hint">
              Try “Create a flash sale post for an antique flower vase”, then “make the background navy and change the
              badge to 70% off”. You can also upload an image to edit.
            </li>
          )}
          {messages.map((m) => (
            <li key={m.id} className={m.role}>
              {m.parts.map((p, i) => {
                if (p.type === "text") return <p key={i}>{p.text}</p>;
                if (p.type === "data-image") {
                  const d = p.data as ImagePart;
                  return (
                    <figure key={i}>
                      <a href={imageUrl(d.key)} target="_blank" rel="noreferrer">
                        <img src={imageUrl(d.key)} alt={d.caption} />
                      </a>
                      <figcaption>{d.caption}</figcaption>
                    </figure>
                  );
                }
                if (p.type === "data-ask") return <p key={i} className="ask">{(p.data as AskPart).text}</p>;
                return null;
              })}
            </li>
          ))}
          {busy && <li className="assistant status">{state?.status ?? "Working…"}</li>}
          {error && <li className="assistant error">Error: {error.message}</li>}
          <div ref={endRef} />
        </ol>
        {lastAsk && lastAsk.options.length > 0 && !busy && (
          <div className="options">
            {lastAsk.options.map((o) => (
              <button key={o} onClick={() => send(o)}>
                {o}
              </button>
            ))}
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void upload(f);
              e.target.value = "";
            }}
          />
          <button type="button" className="secondary" disabled={uploading} onClick={() => fileRef.current?.click()}>
            {uploading ? "Uploading…" : "Upload"}
          </button>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Describe an image or an edit…" />
          <button disabled={busy || !input.trim()}>Send</button>
        </form>
      </main>
      <aside className="canvas">
        <h2>Current image</h2>
        {focus ? (
          <>
            <img src={imageUrl(focus.key)} alt="current" />
            <p className="muted">
              {state!.focus!.asset_id} v{focus.version} · {focus.made_by} · {focus.tool}
            </p>
          </>
        ) : (
          <p className="muted">No image yet.</p>
        )}
        {state && state.assets.length > 0 && (
          <>
            <h2>All images</h2>
            <div className="thumbs">
              {state.assets.flatMap((a) =>
                a.versions.map((v) => (
                  <a key={v.key} href={imageUrl(v.key)} target="_blank" rel="noreferrer" title={`${a.asset_id} v${v.version}: ${v.instruction}`}>
                    <img src={imageUrl(v.key)} alt={`${a.asset_id} v${v.version}`} />
                    <span>
                      {a.asset_id} v{v.version}
                    </span>
                  </a>
                ))
              )}
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
