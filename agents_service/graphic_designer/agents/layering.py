"""The layering agent: today's layering code, run unchanged behind two tools.

Nothing in the layering pipeline is reimplemented. The tools call the same pieces
the old graph walked through, in the same order:

  named targets  : nodes._layeringTargetsFromPrompt (the subgraph's own classifier)
                   -> bboxGenerator -> segmentGenerator -> pasteBackTexts
  drawn boxes    : manualLayering -> segmentGenerator -> pasteBackTexts

The old layeringManager asked "which elements?" with a ``layering_started``
interrupt; here ask_layer_targets raises the same question (same frame, same
fields) as an agent interrupt, and boxes the frontend draws arrive on the reply
exactly as before (``bboxes`` / ``boxes`` / ``manual_bboxes``, normalised by
nodes._normalize_bboxes). The result is the old ``layering_executed`` frame, so
main.py bills and registers the layer graphics as it always did.
"""
from __future__ import annotations

import time

from langchain.agents.middleware import dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from .build import DEFAULT_MODEL, one_tool_at_a_time
from .common import base_state, make_agent, safe_tool
from .runner import run_chain
from .session import DesignSession

AGENT_ID = "layering-agent"


def _layering_state(session: DesignSession, request: str) -> dict:
    v = session.current
    return base_state(session) | {
        "prompt": request,
        "current_image_url": v.url if v else "",
        "conversation_history": [],
    }


def _record_result(session: DesignSession, chain, tool_name: str, started: float,
                   mode: str) -> str:
    st = chain.state
    cost = float(st.get("cost") or 0.0)
    layers = [u for u in (st.get("segmentation_result") or []) if u]
    if chain.outcome == "blocked" and chain.stopped_at == "layDoNotLayer":
        reply = ("NOTHING_FOUND: none of the requested elements could be found in the image, so "
                 "no layers were made.")
        session.record(tool_name, ok=True, detail=reply, cost=cost, started=started,
                       path=chain.path)
        return reply
    if chain.outcome != "done" or not layers:
        reply = (f"FAILED: layering did not produce any layers (stopped at "
                 f"{chain.stopped_at or 'segmentation'}). Nothing was changed.")
        session.record(tool_name, ok=False, detail=reply, cost=cost, started=started,
                       path=chain.path)
        return reply
    source = session.current
    session.layers.append({
        "image_url": st.get("current_image_url") or (source.url if source else ""),
        "version": source.number if source else None,
        "segmentation_result": layers,
        "bboxes": st.get("bboxes") or [],
        "extracted_texts": st.get("extracted_texts") or [],
        "mode": mode,
    })
    session.record(tool_name, ok=True, detail=f"{len(layers)} layer(s)", cost=cost,
                   started=started, path=chain.path)
    texts = [t.get("text") for t in st.get("extracted_texts") or [] if t.get("text")]
    return (f"OK: made {len(layers)} layer(s) (the first is the background)"
            + (f"; text layers: {', '.join(texts[:8])}" if texts else "") + ".")


@tool
@safe_tool
def separate_layers(runtime: ToolRuntime[DesignSession], request: str) -> str:
    """Separate elements of the current design into their own layers.

    `request`: the user's own words about what to separate ("the chair and the lamp",
    "the headline", "everything", "the marked regions"). If the user drew boxes on
    the image, those boxes are used. Otherwise the request is read by the layering
    classifier; when it names nothing that can be layered the tool replies
    NEEDS_TARGETS and you should call ask_layer_targets.
    """
    session = runtime.context
    if not session.current:
        return "FAILED: there is no image to separate into layers."
    from .. import nodes as n

    started = time.monotonic()
    state = _layering_state(session, request)
    boxes = list(session.manual_bboxes or [])
    if boxes:
        session.manual_bboxes = []            # used once, as the old resume update was
        state |= {"manual_bboxes": boxes, "layering_mode": "manual",
                  "layering_targets_resolved": False}
        chain = run_chain("manualLayering", state)
        return _record_result(session, chain, "separate_layers", started, "manual")

    objects, texts, detect_all, cost = n._layeringTargetsFromPrompt(state)
    if not objects and not texts and not detect_all:
        session.record("separate_layers", ok=True, detail="no targets named", cost=cost,
                       started=started)
        return ("NEEDS_TARGETS: the request does not say what to separate. Call "
                "ask_layer_targets.")
    state |= {"layering_mode": "auto", "manual_bboxes": [],
              "requested_objects": objects, "requested_texts": texts,
              "layering_targets_resolved": True, "cost": cost}
    chain = run_chain("bboxGenerator", state)
    return _record_result(session, chain, "separate_layers", started, "auto")


@tool
def ask_layer_targets(runtime: ToolRuntime[DesignSession]) -> str:
    """Ask the user which elements to turn into layers (they can name them or draw
    boxes on the image). Returns their reply; then call separate_layers with it."""
    import settings
    from ..config import LAYERING_OBJECT_ASK, LAYERING_TEXT_AUTO_NOTE

    session = runtime.context
    reply = interrupt({
        "question": LAYERING_OBJECT_ASK,
        "ask_kind": "layering_targets",
        "accepts": ["object_names", "bboxes"],
        "auto_text": False if getattr(settings, "TEXT_DEMAND_DRIVEN", True) else True,
        "note": LAYERING_TEXT_AUTO_NOTE,
    })
    drawn = len(session.manual_bboxes or [])
    return (f"USER REPLIED: {reply}"
            + (f" | The user drew {drawn} box(es) on the image." if drawn else ""))


LAYERING_PROMPT = """\
You are the layering agent of Kognie, an assistant that designs marketing graphics.
You separate elements of the current design (objects, people, products, text) into their
own layers so the user can move and edit them individually.

How to work:
- If the user drew boxes, or the request says what to separate (including "everything" or
  "all of it"), call separate_layers with the user's own words.
- If separate_layers replies NEEDS_TARGETS, or the request does not say what to separate,
  call ask_layer_targets, then call separate_layers with the reply.
- If the user says they do not want layers after all, stop and say so in one line.

Results:
- OK: say how many layers you made, briefly. The app shows the layers itself.
- NOTHING_FOUND: say that those elements could not be found and suggest drawing boxes.
- FAILED: say plainly that layering did not work and nothing changed.

Outside your job (say so kindly in one sentence): changing a design, creating new designs,
critique, removing an element from the design (the edit agent does that).
Write like a helpful designer: short, plain, no internal tool names or jargon.
"""


@dynamic_prompt
def layering_prompt(request) -> str:
    session = request.runtime.context
    drawn = len(session.manual_bboxes or []) if session else 0
    note = (f"\nThe user has drawn {drawn} box(es) on the image for this request."
            if drawn else "\nThe user has not drawn any boxes.")
    return LAYERING_PROMPT + note


def build_layering_agent(model: str = DEFAULT_MODEL):
    return make_agent(init_chat_model(model) if isinstance(model, str) else model,
                      [separate_layers, ask_layer_targets],
                      middleware=[layering_prompt, one_tool_at_a_time],
                      context_schema=DesignSession,
                      name=AGENT_ID)
