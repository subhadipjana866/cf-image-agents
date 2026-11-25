"""critiquepipeline — grounded lenses + arbiter critique subsystem.

Public API mirrors the old ``subgraphs.suggestion`` imports so wiring is a one-line swap:

    from .critiquepipeline import suggestionCreator

Individual pieces are exported for testing / reuse.
"""

from .telemetry import Telemetry
from .grounding import (
    brandNormalizer, loadLayout, mergeConstraints, genericLimits, resolveGrounding,
)
from .agents import (
    caseAgent, integrityEval, compositionEval, contentEval, aestheticEval,
    mergeFindings, arbiterAgent, executabilityGate, compilerAgent, editQuality, verifyEdit,
)
from .nodes import suggestionCreator, structureSuggestions

__all__ = [
    "Telemetry",
    "brandNormalizer", "loadLayout", "mergeConstraints", "genericLimits", "resolveGrounding",
    "caseAgent", "integrityEval", "compositionEval", "contentEval", "aestheticEval",
    "mergeFindings", "arbiterAgent", "executabilityGate", "compilerAgent", "editQuality", "verifyEdit",
    "suggestionCreator", "structureSuggestions",
]
