#!/usr/bin/env python3
"""Export the Claude Code chat transcript into prompt_history/ after every turn.

Runs as a Stop hook. Reads the hook payload (JSON) from stdin, takes the
session transcript (JSONL) it points to, and writes:

  prompt_history/sessions/<start>_<session>.jsonl  raw transcript copy (lossless)
  prompt_history/sessions/<start>_<session>.md     readable transcript
  prompt_history/CHAT_HISTORY.md                   all sessions, oldest first
  prompt_history/PROMPTS.md                        user prompts only, numbered

Every file is regenerated from the full transcript each turn, so nothing is
lost if one run is skipped. Can also be run by hand:

  python3 .claude/hooks/save_chat_history.py <transcript.jsonl> [...]
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
OUT_DIR = PROJECT_DIR / "prompt_history"
SESSIONS_DIR = OUT_DIR / "sessions"

REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def clean(text):
    return REMINDER_RE.sub("", text).strip()


def load_entries(path):
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize_tool(block):
    name = block.get("name", "tool")
    inp = block.get("input") or {}
    detail = (
        inp.get("description")
        or inp.get("file_path")
        or inp.get("command")
        or inp.get("skill")
        or inp.get("prompt")
        or inp.get("query")
        or ""
    )
    detail = " ".join(str(detail).split())
    if len(detail) > 160:
        detail = detail[:157] + "..."
    return f"`{name}`" + (f" — {detail}" if detail else "")


def user_prompt_text(entry):
    """Return the human-typed prompt text, or None for tool results / meta."""
    if entry.get("isMeta") or entry.get("isSidechain"):
        return None
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        text = clean(content)
        return text or None
    if isinstance(content, list):
        if any(b.get("type") == "tool_result" for b in content if isinstance(b, dict)):
            return None
        parts = [clean(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text"]
        text = "\n\n".join(p for p in parts if p)
        return text or None
    return None


def build_turns(entries):
    """Group transcript into turns: one user prompt + Claude's replies/tool calls."""
    turns = []
    current = None
    for e in entries:
        etype = e.get("type")
        if etype == "user":
            prompt = user_prompt_text(e)
            if prompt is not None:
                current = {"timestamp": e.get("timestamp", ""), "prompt": prompt, "items": []}
                turns.append(current)
        elif etype == "assistant" and current is not None and not e.get("isSidechain"):
            for block in e.get("message", {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    current["items"].append(("text", block["text"].strip()))
                elif block.get("type") == "tool_use":
                    current["items"].append(("tool", summarize_tool(block)))
    return turns


def render_session(session_id, turns, title=None):
    lines = [f"# Session {session_id}"]
    if title:
        lines.append(f"\n**Title:** {title}")
    if turns:
        lines.append(f"\n**Started:** {turns[0]['timestamp']}  ")
        lines.append(f"**Turns:** {len(turns)}")
    for i, turn in enumerate(turns, 1):
        lines.append(f"\n---\n\n## Turn {i} — {turn['timestamp']}\n")
        lines.append("### User\n")
        lines.append(turn["prompt"])
        lines.append("\n### Claude\n")
        tools = []
        for kind, value in turn["items"]:
            if kind == "tool":
                tools.append(value)
                continue
            if tools:
                lines.append("<details><summary>Tool calls</summary>\n")
                lines.extend(f"- {t}" for t in tools)
                lines.append("\n</details>\n")
                tools = []
            lines.append(value + "\n")
        if tools:
            lines.append("<details><summary>Tool calls</summary>\n")
            lines.extend(f"- {t}" for t in tools)
            lines.append("\n</details>\n")
    return "\n".join(lines) + "\n"


def export_transcript(transcript_path):
    entries = load_entries(transcript_path)
    session_id = Path(transcript_path).stem
    title = None
    for e in entries:
        if e.get("type") == "custom-title" and e.get("customTitle"):
            title = e["customTitle"]
    turns = build_turns(entries)
    if not turns:
        return
    start = turns[0]["timestamp"][:19].replace(":", "-") or "unknown"
    base = f"{start}_{session_id[:8]}"
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_path, SESSIONS_DIR / f"{base}.jsonl")
    meta = {"session_id": session_id, "title": title, "start": turns[0]["timestamp"], "turns": turns}
    (SESSIONS_DIR / f"{base}.md").write_text(render_session(session_id, turns, title), encoding="utf-8")
    (SESSIONS_DIR / f"{base}.turns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def rebuild_indexes():
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.turns.json")):
        try:
            sessions.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    sessions.sort(key=lambda s: s.get("start", ""))

    combined = ["# Chat History\n", "Full prompt/response history for this project, oldest first.",
                "Raw lossless transcripts are in `sessions/*.jsonl`.\n"]
    prompts = ["# Prompt History\n", "Every prompt sent to Claude for this project, in order.\n"]
    n = 0
    for s in sessions:
        body = render_session(s["session_id"], s["turns"], s.get("title"))
        combined.append("\n".join("#" + l if l.startswith("#") else l for l in body.split("\n")))
        prompts.append(f"\n## Session {s['session_id']}" + (f" — {s['title']}" if s.get("title") else "") + "\n")
        for t in s["turns"]:
            n += 1
            prompts.append(f"### {n}. {t['timestamp']}\n\n{t['prompt']}\n")
    (OUT_DIR / "CHAT_HISTORY.md").write_text("\n".join(combined), encoding="utf-8")
    (OUT_DIR / "PROMPTS.md").write_text("\n".join(prompts), encoding="utf-8")


def main():
    paths = sys.argv[1:]
    if not paths:
        payload = json.load(sys.stdin)
        if payload.get("transcript_path"):
            paths = [payload["transcript_path"]]
    paths = [os.path.expanduser(p) for p in paths]
    # Re-sync sessions exported earlier, so turns missed by a skipped hook run
    # (e.g. a session where the hook was not loaded yet) are picked up later.
    if paths and SESSIONS_DIR.is_dir():
        transcript_dir = Path(paths[0]).parent
        for f in SESSIONS_DIR.glob("*.turns.json"):
            try:
                sid = json.loads(f.read_text(encoding="utf-8"))["session_id"]
            except (json.JSONDecodeError, OSError, KeyError):
                continue
            candidate = str(transcript_dir / f"{sid}.jsonl")
            if candidate not in paths:
                paths.append(candidate)
    for p in paths:
        if os.path.isfile(p):
            export_transcript(p)
    if SESSIONS_DIR.is_dir():
        rebuild_indexes()


if __name__ == "__main__":
    main()
