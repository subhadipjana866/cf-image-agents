"""The fixed structure of a question asked from inside a flow: ``kognie.ask.v1``.

Every question an agent asks mid-task (``ask_user``), and every question the bot itself
asks (continue or drop?, try again?, more suggestions?, the layering offer, the router's
"did you mean"), reaches the frontend as an ``ask`` object on the frame, always with the
same keys, so the frontend can draw buttons, checkboxes or inputs without reading the
text. The input collector's form, the image picker, the suggestion cards and the layering
question keep their existing frames.

The frontend answers with ``ask_reply`` (see :func:`reply_text`). A reply whose ``id``
matches the open question answers it directly; free text still works as before.
"""
from __future__ import annotations

import re
from typing import Any, Optional

SCHEMA = "kognie.ask.v1"
KINDS = ("single_choice", "multi_choice", "confirm", "text", "form")
FIELD_TYPES = ("text", "textarea", "number", "color", "url", "date", "select")

_YES_NO = {"yes", "no"}


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:40] or fallback


def _field(f: Any, i: int) -> dict:
    if isinstance(f, str):
        f = {"label": f}
    ftype = str(f.get("type") or "text").lower()
    opts = [str(o) for o in f.get("options") or []]
    return {
        "id": str(f.get("id") or _slug(f.get("label", ""), f"field_{i + 1}")),
        "label": str(f.get("label") or f.get("id") or f"Field {i + 1}"),
        "type": ftype if ftype in FIELD_TYPES else "text",
        "placeholder": str(f.get("placeholder") or ""),
        "required": bool(f.get("required", False)),
        "default": f.get("default"),
        "options": [{"id": _slug(o, f"opt_{j + 1}"), "label": o, "value": o}
                    for j, o in enumerate(opts)],
    }


def build(*, ask_id: str, task_id: Optional[str], asked_by: str, purpose: str, prompt: str,
          options: Optional[list] = None, multiple: bool = False,
          fields: Optional[list] = None, allow_free_text: bool = True,
          allow_skip: bool = False, default_option_id: Optional[str] = None) -> dict:
    """One ``kognie.ask.v1`` object. ``kind`` follows from what is given: fields -> form,
    options -> confirm (exactly Yes / No) / multi_choice / single_choice, else text."""
    opts = []
    for i, o in enumerate(options or []):
        label = o.get("label") if isinstance(o, dict) else str(o)
        value = o.get("value", label) if isinstance(o, dict) else str(o)
        oid = (o.get("id") if isinstance(o, dict) else None) or _slug(label, f"opt_{i + 1}")
        opts.append({"id": str(oid), "label": str(label), "value": str(value)})
    flds = [_field(f, i) for i, f in enumerate(fields or [])]
    if flds:
        kind = "form"
    elif opts and {o["label"].strip().lower() for o in opts} == _YES_NO:
        kind = "confirm"
    elif opts:
        kind = "multi_choice" if multiple else "single_choice"
    else:
        kind = "text"
    return {
        "schema": SCHEMA,
        "id": ask_id,
        "task_id": task_id,
        "asked_by": asked_by,
        "purpose": purpose,
        "kind": kind,
        "prompt": prompt,
        "options": opts,
        "fields": flds,
        "allow_free_text": bool(allow_free_text),
        "allow_skip": bool(allow_skip),
        "default_option_id": default_option_id,
    }


def reply_text(spec: dict, reply: dict, free_text: str = "") -> str:
    """The user's answer as plain words for the agent or offer that asked: chosen option
    labels, then ``label: value`` for each filled field, then anything typed."""
    by_id = {o["id"]: o for o in spec.get("options") or []}
    chosen = [by_id[i]["value"] for i in (reply.get("option_ids") or []) if i in by_id]
    fields = {f["id"]: f for f in spec.get("fields") or []}
    values = reply.get("values") or {}
    parts = []
    if chosen:
        parts.append(", ".join(chosen))
    for fid, v in values.items():
        if str(v).strip():
            label = fields.get(fid, {}).get("label", fid)
            parts.append(f"{label}: {v}")
    if reply.get("skipped"):
        parts.append("skip")
    if free_text.strip():
        parts.append(free_text.strip())
    return "; ".join(parts)
