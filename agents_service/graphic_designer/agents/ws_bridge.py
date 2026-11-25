"""Websocket bridge: today's frontend frames in, today's frame types out, chat engine inside.

``main.py`` calls :func:`engine_turn` (in a worker thread) instead of invoking the old
LangGraph graph when ``KOGNIE_CHAT_ENGINE`` is on. One :class:`Chat` lives per websocket
thread id. The returned frames use the exact shapes the frontend already renders
(``generation_completed``, ``editing_successful``, ``expand_successful``,
``suggestions_executed``, ``manual_edit``, ``text_input`` with ``fields`` /
``kind: reselect_image`` / ``suggestions``), and ``gstate`` carries the keys main.py's
persistence (design_projects, S3 project state, newimages, generations) reads.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from .chat import Chat
from .session import Brief, DesignSession, Version

_ENGINES: dict[str, Chat] = {}
_LOCK = threading.Lock()

# ManualOps interfaces that are controls in manual_operations.json, by their json id.
_INTERFACE_TO_OP = {"crop_tool": "crop", "corner_radius_control": "corner_radius"}

_SPINNER = {
    "image-generator": ("generation_started", "Creating your graphic..."),
    "image-editor": ("editing_started", "Editing your graphic..."),
    "critic-agent": ("design_suggestion_started", "Our Critic Agent is reviewing your graphic..."),
}


# --------------------------------------------------------------------- store
def has_engine(thread_id: str) -> bool:
    return thread_id in _ENGINES


def evict(thread_id: str) -> None:
    """Thread retention expired (main._expire_thread)."""
    with _LOCK:
        _ENGINES.pop(thread_id, None)


def cancel(thread_id: str) -> None:
    chat = _ENGINES.get(thread_id)
    if chat:
        chat.cancel_running()


def _engine(thread_id: str, ctx: dict) -> Chat:
    with _LOCK:
        chat = _ENGINES.get(thread_id)
        if chat is None:
            session = DesignSession(user_id=str(ctx.get("user_id") or "anonymous"),
                                    session_id=str(ctx.get("session_id") or thread_id))
            chat = Chat(session, verbose=True, defer_auto_critique=True)
            chat.reply_style = ("Reply to the person directly, as 'you', in plain words. The app "
                                "shows images itself: never paste image links or URLs, and never "
                                "mention internal ids like 'asset a1' or version numbers.")
            _restore(chat, ctx)
            _ENGINES[thread_id] = chat
        return chat


def _restore(chat: Chat, ctx: dict) -> None:
    """A cold thread (new connection, server restart): pick up the project and the chat."""
    s = chat.session
    try:
        state = (ctx.get("load_state") or (lambda: {}))() or {}
    except Exception as err:  # noqa: BLE001 - restore is best effort
        print(f"[chat engine] project state restore skipped: {err}")
        state = {}
    if state:
        s.brief = Brief(prompt=state.get("prompt") or state.get("clarified_intent") or "",
                        design_goal=state.get("design_goal"),
                        platform=state.get("platform_canonical"),
                        aspect_ratio=state.get("aspect_ratio"),
                        field_answers=dict(state.get("field_answers") or {}))
        s.v33_prev = state.get("v32_prev")
        s.lineage = list(state.get("v32_lineage") or [])
        s.lineage_urls = list(state.get("v32_lineage_urls") or [])
        url = (state.get("current_image_url") or "").split("|")[0]
        if url:
            s.start_with_image(url, "restored from the project")
    try:
        history = (ctx.get("load_history") or (lambda: []))() or []
    except Exception as err:  # noqa: BLE001
        print(f"[chat engine] chat history restore skipped: {err}")
        history = []
    chat.nav_history = [{"role": "assistant" if h.get("role") != "user" else "user",
                         "content": str(h.get("content", ""))[:500]} for h in history][-20:]


def _moderate(s: DesignSession, data: dict, text: str, selected: Optional[str],
              attachments: list[str]) -> Optional[dict]:
    """OpenAI moderation of everything the user sent this turn: the message, form answers,
    edited suggestion text, attached images, uploaded logo / photo assets, and an image
    newly put on the canvas that this session did not make. None when it passes; otherwise
    the frame that tells the user why."""
    from . import moderation

    texts = [text] + [str(v) for v in (data.get("field_answers") or {}).values()
                      if isinstance(v, (str, int, float)) and str(v).strip()]
    texts += [str(c.get("user_response")) for c in (data.get("suggestions") or [])
              if isinstance(c, dict) and c.get("user_response")]
    images = list(attachments)
    assets = data.get("assets") if isinstance(data.get("assets"), dict) else {}
    images += [v for v in assets.values()
               if isinstance(v, str) and v.startswith(("http://", "https://", "data:image"))]
    if selected and selected not in {v.url for v in s.versions}:
        images.append(selected)
    verdict = moderation.check("\n".join(t for t in texts if t.strip()), images)
    s.record("moderation", ok=not verdict.flagged, detail=", ".join(verdict.categories) or "clean",
             checks=verdict.checked)
    if not verdict.flagged:
        return None
    message = verdict.message()
    print(f"[chat engine] message blocked by moderation: {verdict.categories} "
          f"(image: {verdict.in_image})")
    return {"type": "text_input", "question": message, "current_image_url": _url(s),
            "cost": 0.0, "show_cost": False, "sender": "orchestrator", "task_id": None,
            "recipient": "user",
            "messages": [{"sender": "orchestrator", "task_id": None, "recipient": "user",
                          "text": message}],
            "moderation": {"flagged": True, "categories": verdict.categories,
                           "in_image": verdict.in_image}}


def _drawn_boxes(data: dict) -> list[dict]:
    """Boxes the user drew on the image, in the shapes the old main.py accepted
    (bboxes / boxes / manual_bboxes), normalised by the same nodes._normalize_bboxes."""
    raw = data.get("bboxes") or data.get("boxes") or data.get("manual_bboxes")
    if not raw:
        return []
    from ..nodes import _normalize_bboxes
    try:
        return _normalize_bboxes(raw)
    except Exception as err:  # noqa: BLE001 - bad boxes must not break the turn
        print(f"[chat engine] drawn boxes ignored: {err}")
        return []


# ---------------------------------------------------------------------- turn
def engine_turn(thread_id: str, data: dict, ctx: dict) -> dict[str, Any]:
    """Run one frontend frame. Returns {"frames": [...], "gstate": {...}}."""
    from ..utils import SpendLimitExceeded

    chat = _engine(thread_id, ctx)
    s = chat.session
    emit: Optional[Callable[[dict], None]] = ctx.get("emit")
    chat.on_progress = (lambda e: emit(_spinner_frame(e, s))) if emit else None
    cost_before = s.cost
    _apply_context(s, data, ctx)

    text = str(data.get("prompt") or data.get("response") or "").strip()
    if ctx.get("is_regenerate") and not text:
        text = "Regenerate this design from the same brief."
    selected = _selected_image(s, data, ctx)
    attachments = [a for a in (data.get("attached_images") or []) if isinstance(a, str) and a]

    # Moderation first: nothing reaches the engine (or the session) until it passes.
    blocked = _moderate(s, data, text, selected, attachments)
    if blocked is not None:
        return {"frames": [blocked], "gstate": _gstate(chat, data, ctx)}
    _apply_assets(s, data)

    try:
        if chat.pending_auto_critique and text.lower() in ("", "yes", "y", "ok", "okay", "sure"):
            chat.continue_auto_critique()
        else:
            chat.pending_auto_critique = False
            chat.turn(text, selected, field_answers=data.get("field_answers") or None,
                      suggestions=data.get("suggestions") or None, attachments=attachments,
                      bboxes=_drawn_boxes(data),
                      ask_reply=data.get("ask_reply") if isinstance(data.get("ask_reply"), dict)
                      else None)
    except SpendLimitExceeded as err:
        chat.events.append({"type": "message", "text": "We've paused here: this session hit "
                                                      f"its spending safety limit ({err})."})
    except Exception as err:  # noqa: BLE001 - never crash the socket; tell the user
        import traceback
        traceback.print_exc()
        chat.events.append({"type": "message",
                            "text": "Sorry, something went wrong on our side while handling "
                                    "that. Please try again."})
        print(f"[chat engine] turn failed: {type(err).__name__}: {err}")
    frames = _frames(chat, s.cost - cost_before)
    return {"frames": frames, "gstate": _gstate(chat, data, ctx)}


def _apply_context(s: DesignSession, data: dict, ctx: dict) -> None:
    if ctx.get("user_id"):
        s.user_id = str(ctx["user_id"])
    if ctx.get("session_id"):
        s.session_id = str(ctx["session_id"])
    if ctx.get("brand_dna"):
        s.brand_dna = ctx["brand_dna"]
        s.brand_name = (ctx["brand_dna"] or {}).get("brand_name") or data.get("brand_name")
    # The app's pickers: a design goal / platform / ratio / style chosen in the editor.
    picks = {k: data.get(k) for k in ("design_goal", "platform", "purpose", "aspect_ratio",
                                      "style", "num_images") if data.get(k)}
    if picks:
        brief = s.brief or Brief(prompt="")
        brief.design_goal = picks.get("design_goal") or brief.design_goal
        brief.platform = picks.get("platform") or picks.get("purpose") or brief.platform
        brief.aspect_ratio = picks.get("aspect_ratio") or brief.aspect_ratio
        brief.style = picks.get("style") or brief.style
        if picks.get("num_images"):
            brief.num_images = int(picks["num_images"])
        s.brief = brief


def _selected_image(s: DesignSession, data: dict, ctx: dict) -> Optional[str]:
    """The canvas selection the frontend sends: only a change of image counts as selecting."""
    graphic = ctx.get("frame_graphic") or {}
    url = (graphic.get("image_url") or (data.get("current_image_url") or "")).split("|")[0]
    s.canvas_selected = bool(url)
    last = ctx.get("session_last_graphic") or {}
    # Not loaded now: only a request that works on an image falls back to it (chat).
    s.fallback_image = last.get("image_url") or None
    if url and (not s.current or s.current.url != url):
        return url
    return None


def _apply_assets(s: DesignSession, data: dict) -> None:
    assets = data.get("assets") or {}
    logo = assets.get("logo") if isinstance(assets, dict) else None
    if isinstance(logo, str) and logo and logo != "skipped":
        brief = s.brief or Brief(prompt="")
        brief.logo_url, brief.logo_mode = logo, "provided"
        s.brief = brief


# -------------------------------------------------------------------- frames
def _url(s: DesignSession) -> str:
    return s.current.url if s.current else ""


def _env(event: dict) -> dict:
    """Who sent a message, for which task, and who it is for (from the chat event)."""
    return {"sender": event.get("sender") or "orchestrator", "task_id": event.get("task_id"),
            "recipient": event.get("recipient") or "user"}


def _part(event: dict, text: str) -> dict:
    return {**_env(event), "text": text}


def _join(parts: list[dict]) -> str:
    return "\n\n".join(p["text"] for p in parts)


def _stamp(frame: dict, event: dict, parts: list[dict], own_text: str = "") -> dict:
    """Put the sender / task / recipient on a frame, and list every message folded into
    its question (earlier notes first, then the frame's own) under `messages`."""
    frame.update(_env(event))
    frame["messages"] = list(parts) + ([_part(event, own_text)] if own_text else [])
    return frame


def _spinner_frame(event: dict, s: DesignSession) -> Optional[dict]:
    spin = _SPINNER.get(event.get("agent_id"))
    if not spin:
        return None
    return {"type": spin[0], "question": spin[1], "current_image_url": _url(s), "cost": 0.0,
            "show_cost": False, **_env(event)}


def _result_frame(event: dict, s: DesignSession, prefix: str) -> Optional[dict]:
    if event.get("new_layers"):
        # Today's common_nodes.nextStep frame; main.py registers each layer graphic from
        # segmentation_result, exactly as for the old graph.
        from ..config import LAYERING_EXECUTED
        made = event["new_layers"][-1]
        reply = (prefix + "\n\n" if prefix else "") + LAYERING_EXECUTED
        return {"type": "layering_executed", "question": reply.strip(),
                "prompt": event.get("goal", ""),
                "current_image_url": made.get("image_url") or _url(s),
                "segmentation_result": made.get("segmentation_result") or [],
                "bboxes": made.get("bboxes") or [],
                "extracted_texts": made.get("extracted_texts") or []}
    versions: list[Version] = event["new_versions"]
    if not versions:
        return None
    urls = "|".join(v.url for v in versions)
    previous = event.get("previous")
    reply = (prefix + "\n\n" if prefix else "") + (event.get("reply") or "")
    if event["agent_id"] == "image-generator":
        frame = {"type": "generation_completed", "question": reply.strip(),
                 "prompt": event.get("goal", ""), "enhanced_prompt": event.get("goal", ""),
                 "current_image_url": urls}
        layers = (versions[-1].meta.get("text_layers") or {})
        if layers.get("layers"):
            frame["text_layers"] = layers["layers"]
            if layers.get("notices"):
                frame["typography_notices"] = layers["notices"]
        return frame
    kind = ("expand_successful" if versions[-1].tool == "resize_for_platforms"
            else "suggestions_executed" if event.get("payload", {}).get("from_suggestions")
            else "editing_successful")
    return {"type": kind, "question": reply.strip(), "prompt": event.get("goal", ""),
            "current_image_url": urls, "previous_image_url": previous.url if previous else ""}


def _ask_frame(event: dict, s: DesignSession, prefix: str) -> dict:
    question = (prefix + "\n\n" if prefix else "") + event["question"]
    if event.get("kind") == "suggestions":
        items = event.get("suggestions") or []
        stack = [{"image_id": "image_1", "s_id": x["n"], "status": "pending", "focus": i == 0,
                  "user_response": None, "edits": len(items), "type": x.get("type", "EDIT"),
                  "tier": None, "topic": x.get("text", ""), "user_friendly_text": x.get("text", ""),
                  "nano_prompt": x.get("instruction", "")} for i, x in enumerate(items)]
        return {"type": "text_input", "question": question, "suggestions": stack,
                "suggestion_type": "auto", "current_image_url": _url(s)}
    if event.get("ask_kind") == "form":
        fields = [{"field_id": f["field_id"], "label": f["label"], "input_type": "text",
                   "example": f.get("example", ""), "question": f["question"]}
                  for f in event.get("fields") or []]
        return {"type": "text_input", "skippable": True, "question": question,
                "fields": fields, "assets": []}
    if event.get("ask_kind") == "layering_targets":
        # Today's layeringManager frame: name the elements, or draw boxes on the image.
        extra = event.get("extra") or {}
        return {"type": "layering_started", "question": question, "current_image_url": _url(s),
                "accepts": extra.get("accepts") or ["object_names", "bboxes"],
                "auto_text": extra.get("auto_text", False), "note": extra.get("note", "")}
    if event.get("ask_kind") == "select_image":
        return {"type": "text_input", "kind": "reselect_image", "question": question,
                "accepts": ["image"], "current_image_url": _url(s)}
    if event.get("ask_spec"):
        # kognie.ask.v1: the frontend draws options / inputs from `ask`; the text no longer
        # lists the options.
        return {"type": "text_input", "question": question, "current_image_url": _url(s),
                "ask": event["ask_spec"]}
    frame = {"type": "text_input", "question": question, "current_image_url": _url(s)}
    if event.get("options"):
        frame["options"] = event["options"]
        frame["question"] += "\n" + "\n".join(f"- {o}" for o in event["options"])
    return frame


def _frames(chat: Chat, turn_cost: float) -> list[dict]:
    s = chat.session
    live_spinners = getattr(chat, "on_progress", None) is not None
    frames: list[dict] = []
    texts: list[dict] = []          # messages waiting to be folded into the next frame
    final: Optional[dict] = None
    for e in chat.events:
        if e["type"] == "progress":
            spin = _SPINNER.get(e["agent_id"])
            if spin and not live_spinners:
                frames.append({"type": spin[0], "question": spin[1], "current_image_url": _url(s),
                               "cost": 0.0, "_spinner": True, **_env(e)})
        elif e["type"] == "message":
            texts.append(_part(e, e["text"]))
        elif e["type"] == "client_action":
            a = e["action"]
            if a.get("type") == "manual_edit":
                # Today's manualEditHandler frame: manual_op is the id from
                # manual_operations.json (None + multiple=True for several controls).
                final = _stamp({"type": "manual_edit", "question": a.get("question", ""),
                                "manual_op": a.get("manual_op"), "multiple": bool(a.get("multiple")),
                                "prompt": a.get("prompt", ""), "current_image_url": _url(s),
                                "cost": 0.0}, e, [], a.get("question", ""))
                continue
            prefill = a.get("prefill") or {}
            interface = a.get("interface")
            op = prefill.get("control") or _INTERFACE_TO_OP.get(interface) or interface
            final = _stamp({"type": "manual_edit", "question": "", "interface": interface,
                            "manual_op": op, "multiple": False, "prefill": prefill, "prompt": "",
                            "current_image_url": _url(s), "cost": 0.0}, e, [])
        elif e["type"] == "result":
            frame = _result_frame(e, s, _join(texts))
            if frame:
                _stamp(frame, e, texts, e.get("reply") or "")
                texts = []
                frame["cost"] = round(float(e.get("cost") or 0.0), 6)
                frames.append(frame)
            elif e.get("reply"):
                if final is not None and final["type"] == "manual_edit" and not final["question"]:
                    final["question"] = e["reply"]
                    final["messages"].append(_part(e, e["reply"]))
                else:
                    texts.append(_part(e, e["reply"]))
        elif e["type"] == "ask":
            final = _stamp(_ask_frame(e, s, _join(texts)), e, texts, e["question"])
            texts = []
    results = [f for f in frames if not f.get("_spinner")]
    if final is None and texts and results:
        # e.g. GENERATION_COMPLETED after the design: it is the result frame's own question,
        # as today, so the frontend's automatic "yes" follows it.
        results[-1]["question"] = (results[-1]["question"] + "\n\n" + _join(texts)).strip()
        results[-1].setdefault("messages", []).extend(texts)
        texts = []
    if final is None and texts:
        last = texts[-1]
        final = {"type": "text_input", "question": _join(texts), "current_image_url": _url(s),
                 "sender": last["sender"], "task_id": last["task_id"],
                 "recipient": last["recipient"], "messages": list(texts)}
    elif final is not None and texts:
        final["question"] = (final.get("question", "") + "\n\n" + _join(texts)).strip()
        final.setdefault("messages", []).extend(texts)
    if final is not None:
        frames.append(final)
    if not any(not f.get("_spinner") for f in frames):
        question = "What would you like to do next?"
        frames.append({"type": "text_input", "question": question, "current_image_url": _url(s),
                       "sender": "orchestrator", "task_id": None, "recipient": "user",
                       "messages": [{"sender": "orchestrator", "task_id": None,
                                     "recipient": "user", "text": question}]})
    # Cost not carried by a result frame (navigation, collector, agents' own model calls)
    # rides on the turn's last frame, so every cent is billed exactly once.
    billed = sum(f.get("cost", 0.0) for f in frames)
    rest = max(0.0, round(turn_cost - billed, 6))
    frames[-1]["cost"] = round(frames[-1].get("cost", 0.0) + rest, 6)
    for f in frames:
        f.setdefault("cost", 0.0)
        f["show_cost"] = (not f.get("_spinner")) and (
            f["type"] in ("generation_completed", "editing_successful", "expand_successful",
                          "suggestions_executed", "layering_executed") or f["cost"] > 0)
    return frames


# -------------------------------------------------------------------- gstate
def _gstate(chat: Chat, data: dict, ctx: dict) -> dict:
    """The state keys main.py's persistence and graphic registration read."""
    s = chat.session
    b = s.brief
    image_models = {v.url: v.model for v in s.versions if v.model}
    last_gen = next((v for v in reversed(s.versions) if v.made_by == "image-generator"), None)
    last_result_tool = s.current.tool if s.current else ""
    canvas = ""
    if b and b.design_goal and b.platform:
        try:
            from .. import v33_bridge
            facts = v33_bridge.layout_facts(b.design_goal, b.platform) or {}
            px = facts.get("canvas_px") or []
            if isinstance(px, (list, tuple)) and len(px) == 2:
                canvas = f"{px[0]}x{px[1]}"
        except Exception:  # noqa: BLE001 - target size is informational
            canvas = ""
    return {
        "session_id": s.session_id, "user_id": s.user_id,
        "prompt": b.prompt if b else "", "clarified_intent": b.prompt if b else "",
        "base_prompt": b.prompt if b else "",
        "design_goal": (b.design_goal if b else None) or "",
        "platform_canonical": (b.platform if b else None) or "",
        "aspect_ratio": (b.aspect_ratio if b else None) or data.get("aspect_ratio") or "",
        "target_size": canvas, "field_answers": dict(b.field_answers) if b else {},
        "is_freeform": bool(b and not b.design_goal), "asset_prefs": {},
        "generation_count": sum(1 for v in s.versions if v.made_by == "image-generator"),
        "image_models": image_models,
        "current_image_url": s.current.url if s.current else "",
        "last_generated_image_url": last_gen.url if last_gen else "",
        "active_graphic": ctx.get("frame_graphic"),
        "request_type": {"create_more_variations": "generate_more",
                         "regenerate_design": "regenerate"}.get(last_result_tool, ""),
        "brand_id": ctx.get("brand_id"), "brand_name": s.brand_name,
        "v32_prev": s.v33_prev, "v32_lineage": s.lineage, "v32_lineage_urls": s.lineage_urls,
        "critique_provenance": "generated" if last_gen else None,
    }
