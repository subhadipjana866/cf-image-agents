"""Run today's graph nodes in their existing order, outside the graph.

Each generation/edit node returns ``Command(goto=next, update=state)``. The
runner calls the node, merges the update and follows ``goto`` until it reaches
the node where the old graph showed the user a result. The pipelines are not
reimplemented or reordered; this only replaces LangGraph's edge walking for the
nodes below.

A node that is not in ``_NODES`` is one that would talk to the user (an
interrupt gate). The run stops there and reports ``blocked`` with that node's
name, so a tool can hand the gap back instead of pausing mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.types import Command

# Where the old graph showed the user a finished result. Layering ends the subgraph
# after pasteBackTexts (static edge to END, then layeringResultRouter -> nextStep).
_DONE = {"interruptBeforeFidelitySuggestion", "successfulEditHandler", "layeringDone"}
_FAILED = {"extremeFallback"}
# Nodes with a static edge instead of a Command (subgraphs/layering.py: segmentGenerator
# -> pasteBackTexts -> END).
_STATIC_NEXT = {"aspectRatioHandler": "gptImageExecutor",
                "segmentGenerator": "pasteBackTexts", "pasteBackTexts": "layeringDone"}


def _nodes() -> dict[str, Callable[[dict], Any]]:
    from .. import nodes as n   # heavy import, deferred to the first run
    from ..subgraphs import layering as lay

    return {
        "generation": n.v32Generation,
        "runGptImageModel": n.runGptImageModel,
        "v32CreateMore": n.v32CreateMore,
        "logoCompositor": n.logoCompositor,
        "aspectRatioHandler": n.aspectRatioHandler,
        "gptImageExecutor": n.gptImageExecutor,
        "bananaProExecutor": n.bananaProExecutor,
        "bananaExecutor": n.bananaExecutor,
        "fluxExecutor": n.fluxExecutor,
        # Layering (subgraphs/layering.py), called as the subgraph calls them.
        "bboxGenerator": lay.bboxGenerator,
        "manualLayering": lay.manualLayering,
        "segmentGenerator": lay.segmentGenerator,
        "pasteBackTexts": lay.pasteBackTexts,
    }


@dataclass
class ChainResult:
    outcome: str                     # "done" | "failed" | "blocked"
    state: dict
    path: list[str] = field(default_factory=list)
    stopped_at: str = ""


def run_chain(start: str, state: dict, *, max_steps: int = 12) -> ChainResult:
    nodes = _nodes()
    node, path = start, []
    state = dict(state)
    for _ in range(max_steps):
        if node in _DONE:
            return ChainResult("done", state, path, node)
        if node in _FAILED:
            return ChainResult("failed", state, path, node)
        fn = nodes.get(node)
        if fn is None:
            return ChainResult("blocked", state, path, node)
        path.append(node)
        out = fn(state)
        if isinstance(out, Command):
            update = out.update or {}
            state = {**state, **(update if isinstance(update, dict) else dict(update))}
            node = out.goto if isinstance(out.goto, str) else out.goto[0]
        else:                                       # node returned the state itself
            if isinstance(out, dict):
                state = {**state, **out}
            node = _STATIC_NEXT.get(node, "")
            if not node:
                return ChainResult("failed", state, path, path[-1])
    return ChainResult("failed", state, path, f"step limit after {node}")
