"""AI slop remover subgraph.

Mounted directly as a node on the main graph (``slopManager`` ->
``slopSubgraph`` -> ``slopResultRouter``, see ``graph.py``), like the layering
subgraph, so a resume continues where it paused instead of re-running the
removal from scratch.

Flow::

    START -> slopExecute -> slopReview -> slopClassifier -> END
                                              |  ^
                      question / chit-chat /  v  |  reply comes back here
                      unclear  ->  slopQuestion / slopChitchat / slopUnclear

``slopClassifier`` leaves ``slop_verdict`` for ``slopResultRouter``:

* ``create_more`` -- the reply was exactly "create more"; it goes to the
  navigator as-is, on the cleaned image.
* ``new_request`` -- an edit, a new graphic or another operation; it goes to
  the navigator too.
* ``continue`` -- anything else; the cleaned image goes on to the Critic Agent.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from ..states import SlopSubGraph
from ..utils import parseJson, profile
from ..common_nodes import chitChatHandler, questionHandler, unclearHandler, _replyAttachments
from ..config import SLOP_CREATE_MORE_ASK
from ..slop_decision import classify_slop_reply

# The exact reply that sends the cleaned image straight to the navigator.
CREATE_MORE_REPLY = "create more"


@profile
def slopExecute(state: SlopSubGraph) -> Command:
    """Remove the AI slop from ``current_image_url``.

    Placeholder: the removal mechanism is not implemented yet, so the image
    passes through unchanged. Whatever replaces this body must leave the
    cleaned image in ``current_image_url`` (uploaded to S3), the image it was
    made from in ``previous_image_url``, the model that rendered it in
    ``image_models`` (see ``nodes._with_image_models``), and add its spend to
    ``cost``.
    """
    # TODO(senior dev): AI slop removal mechanism goes here.
    print(f"[slopExecute] removal mechanism not implemented yet -- passing "
          f"{state.get('current_image_url', '')!r} through unchanged")
    return Command(goto="slopReview", update={"cost": state.get('cost', 0.0)})


@profile
def slopReview(state: SlopSubGraph) -> Command:
    """Show the cleaned image and ask whether the user wants to create more."""
    print("[slopReview] presenting the cleaned image")
    user_response = parseJson(interrupt({
        "type": "slop_removed",
        "question": SLOP_CREATE_MORE_ASK,
        "prompt": state.get('prompt', ''),
        "cost": state.get('cost', 0.0),
        "current_image_url": state.get('current_image_url', ''),
        "previous_image_url": state.get('previous_image_url', ''),
    }))
    reply = user_response.get('response', '')
    # The interrupt above billed everything spent so far.
    _reply_refs = _replyAttachments(user_response)
    return Command(goto="slopClassifier", update={
        "prompt": reply,
        "cost": 0.0,
        # Honour the image selected on the canvas at reply time, as
        # interruptBeforeFidelitySuggestion does.
        "current_image_url": user_response.get('current_image_url') or state.get('current_image_url', ''),
        "attached_images": _reply_refs or state.get('attached_images', []),
        "edit_reference_images": _reply_refs,
        "model": user_response.get('model', state.get('model', '')),
        "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')),
        "style": user_response.get('style', state.get('style', '')),
        "conversation_history": state.get('conversation_history', []) + [
            {"role": "system", "content": SLOP_CREATE_MORE_ASK},
            {"role": "user", "content": reply},
        ],
    })


@profile
def slopClassifier(state: SlopSubGraph) -> Command:
    """Decide where the latest reply goes.

    The question/chit-chat/unclear handlers already appended their exchange
    to ``conversation_history``, as ``slopReview`` did for its own.
    """
    reply = state.get('prompt', '') or ''
    if reply.strip().lower() == CREATE_MORE_REPLY:
        print("[slopClassifier] 'create more' -> navigator")
        return Command(goto=END, update={"slop_verdict": "create_more"})

    verdict, cost = classify_slop_reply(reply, state.get('conversation_history', []))
    cost = state.get('cost', 0.0) + cost
    print(f"[slopClassifier] verdict={verdict}")
    if verdict == "question":
        return Command(goto="slopQuestion", update={"cost": cost})
    if verdict == "chit-chat":
        return Command(goto="slopChitchat", update={"cost": cost})
    if verdict == "unclear":
        return Command(goto="slopUnclear", update={"cost": cost})
    if verdict == "new_request":
        return Command(goto=END, update={"slop_verdict": "new_request", "cost": cost})
    # "continue", or the Decisions API failed: carry on to the Critic Agent,
    # which is where the flow goes when the user does not ask for anything else.
    return Command(goto=END, update={"slop_verdict": "continue", "cost": cost})


slopGraph = StateGraph(SlopSubGraph)

slopGraph.add_node("slopExecute", slopExecute)
slopGraph.add_node("slopReview", slopReview)
slopGraph.add_node("slopClassifier", slopClassifier)
slopGraph.add_node("slopQuestion", questionHandler)
slopGraph.add_node("slopChitchat", chitChatHandler)
slopGraph.add_node("slopUnclear", unclearHandler)

slopGraph.add_edge(START, "slopExecute")

slg = slopGraph.compile()
