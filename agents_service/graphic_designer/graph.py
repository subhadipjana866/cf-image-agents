"""Graphic Designer agent graph definition.

This module assembles the LangGraph :class:`~langgraph.graph.StateGraph` that
drives the Graphic Designer agent.  It imports every node function from
:mod:`graphic_designer.nodes` and :mod:`graphic_designer.common_nodes`, registers
them as graph nodes, and wires the edges (including conditional routing) that
define the agent's execution flow.

"""

from langgraph.graph import StateGraph, START, END
from .states import GraphicDesigner
from .nodes import promptCollector, navigatorClassify, requestTypeAnnounce, generationManager, generateMoreManager, regenerateManager, editingManager, editModeClassifier, manualEditHandler, suggestionManager, scopeGate, scopeMismatchGate, scopeAbortGate, intentClarifier, intentRouter, manifestFieldsGate, manifestFieldsAsk, photoRequiredGate, clarificationGate, imageAnalyzerLayoutSelector, logoDecisionGate, runGptImageModel, logoCompositor, editAssetGate, editStyleSwitch, editIntentClassifier, editBrandWarning, editExecutabilityWarning, knownCategoryPromptHandler, bananaExecutor, bananaProExecutor, fluxExecutor, gptImageExecutor, successfulEditHandler, editWrongImageHandler, extremeFallback, aspectRatioHandler, successfulExpandHandler, suggestionHandler, suggestionExecutors, suggestionMoreAsk, interruptBeforeFidelitySuggestion, layeringManager, layeringResultRouter, interruptBeforeLayering, slopManager, slopResultRouter, expandManager, criticismHandler, v32Generation, v32NeedsInputAsk, v32CreateMore
# The layering subgraph is mounted directly as a node (rather than invoked
# from inside a node function) so it inherits the parent graph's checkpointer
# and LangGraph resumes it exactly where it paused -- see layeringManager's
# docstring in nodes.py for why that matters.
from .subgraphs.layering import lgb as layeringSubgraph
# Mounted the same way, for the same reason -- see slopManager.
from .subgraphs.slop import slg as slopSubgraph
from .common_nodes import chitChatHandler, questionHandler, unclearHandler, nextStep, helperAgent, specificSuggestionCreator, specificSuggestionHandler
# General (whole-design) critique. Replaces suggestionFidelityCreator and
# suggestionDesignCreator: grounded lenses (integrity/composition/content/aesthetic) run
# in parallel, an arbiter curates them, and the result is the same s_stack shape the
# rest of the suggestion flow already consumes. The specific-suggestion route
# (specificSuggestionCreator) is untouched.
from .critiquepipeline import suggestionCreator
from langgraph.checkpoint.memory import InMemorySaver



graph = StateGraph(GraphicDesigner)
"""StateGraph: The main state-graph instance for the Graphic Designer agent.

All node registrations and edge wiring in this module are applied to this object.
"""

# graph.add_node("receptionist", receptionist)
# graph.add_node("aspectRatioDeterminer", aspectRatioDeterminer)
graph.add_node("promptCollector", promptCollector)
# navigator = classify the request (stores request_type); requestTypeAnnounce then
# tells the frontend the type via an auto-continued interrupt and routes to the
# handler (Command-routed, so no conditional edge). See nodes.NAVIGATOR_ROUTES.
graph.add_node("navigator", navigatorClassify)
graph.add_node("requestTypeAnnounce", requestTypeAnnounce)
graph.add_node('chitChatHandler', chitChatHandler)
graph.add_node("questionHandler", questionHandler)
graph.add_node("generationManager", generationManager)
graph.add_node("generateMoreManager", generateMoreManager)  # Command-routed: no static out-edge
# Regenerate: same brief, SAME wireframe, real generation re-run. Distinct from
# generateMoreManager, which routes straight to v32CreateMore: same content as
# the selected graphic, a different composition, and never a question. Command-routed straight to
# intentRouter (the interactive gates are skipped -- see the node's docstring),
# so no static out-edge.
graph.add_node("regenerateManager", regenerateManager)
# Every edit enters through editModeClassifier: one a frontend manual control
# does exactly ("brightness to 46%", "increase the corner radius") goes to
# manualEditHandler, which hands it back as manual_op; the rest go on to
# editingManager. Both Command-routed, so no static out-edges.
graph.add_node("editModeClassifier", editModeClassifier)
graph.add_node("manualEditHandler", manualEditHandler)
graph.add_node("editingManager", editingManager)
graph.add_node("suggestionManager", suggestionManager)
# graph.add_node("decisionHandler", decisionHandler)
graph.add_node("unclearHandler", unclearHandler)
# Global landing point for every per-process criticism detector. Command-routed
# (suggestion origin -> suggestionManager, everything else -> navigator), so no
# static out-edge.
graph.add_node("criticismHandler", criticismHandler)
graph.add_node("nextStep", nextStep)
graph.add_node("interruptBeforeLayering", interruptBeforeLayering)
graph.add_node("layeringManager", layeringManager)
graph.add_node("layeringSubgraph", layeringSubgraph)
graph.add_node("layeringResultRouter", layeringResultRouter)
# AI slop remover: slopManager is Command-routed (no image -> navigator, else
# -> the mounted subgraph), so its only static edge is subgraph -> router.
graph.add_node("slopManager", slopManager)
graph.add_node("slopSubgraph", slopSubgraph)
graph.add_node("slopResultRouter", slopResultRouter)
graph.add_node("expandManager", expandManager)
graph.add_node("successfulExpandHandler", successfulExpandHandler)

graph.add_node("helperAgent", helperAgent)

# Generation nodes
# graph.add_node("bestModel", bestModel)
graph.add_node("scopeGate", scopeGate)
# scopeGate's two dialogs live in their own nodes so resuming either one never
# re-runs the check_request_scope call (its result rides in state.scope_result).
graph.add_node("scopeMismatchGate", scopeMismatchGate)
graph.add_node("scopeAbortGate", scopeAbortGate)
graph.add_node("intentClarifier", intentClarifier)
graph.add_node("intentRouter", intentRouter)
graph.add_node("manifestFieldsGate", manifestFieldsGate)
# Same split: the field/asset interrupt is its own node, so resuming it never
# re-runs manifestFieldsGate's LLM extraction pass.
graph.add_node("manifestFieldsAsk", manifestFieldsAsk)
graph.add_node("photoRequiredGate", photoRequiredGate)
graph.add_node("clarificationGate", clarificationGate)
graph.add_node("imageAnalyzerLayoutSelector", imageAnalyzerLayoutSelector)
graph.add_node("logoDecisionGate", logoDecisionGate)
# "generation" is v3.3: one Command-routed node runs the whole reasoning +
# render pipeline (GROUND -> SUBJECT -> TYPOGRAPHER/ART DIRECTOR/COMPOSITOR ->
# DIRECTOR -> gpt-image-2.5-flare) and hands off to logoCompositor itself -- no
# conditional edge, unlike the retired per-model dispatch below.
graph.add_node("generation", v32Generation)
graph.add_node("v32NeedsInputAsk", v32NeedsInputAsk)
graph.add_node("v32CreateMore", v32CreateMore)
# runGptImageModel is the only run*Model executor still wired in: the freeform
# / custom-goal bypass (_freeformNext) has no catalog combination for v33 to
# design from, so it renders directly off the raw prompt instead.
graph.add_node("runGptImageModel", runGptImageModel)
graph.add_node("logoCompositor", logoCompositor)

# Editing nodes
# Runs on every entry into editing: resolves the images the edit needs
# (this turn's attached references, plus a logo the request asked for).
graph.add_node("editAssetGate", editAssetGate)
# "change the style to bauhaus": a style this service has is expanded into a
# full restyle prompt and goes straight to the executors; one it does not have
# goes to the executors as the user wrote it; anything else falls through to
# the classifier. Command-routed, so no static out-edge.
graph.add_node("editStyleSwitch", editStyleSwitch)
graph.add_node("editIntentClassifier", editIntentClassifier)
# Disclaimer-only detour taken when the edit contradicts the attached brand;
# auto-continued by main.py, then on to where editIntentClassifier would have
# gone. Command-routed, so no static out-edge.
graph.add_node("editBrandWarning", editBrandWarning)
# Same kind of detour for an edit the image model is unlikely to carry out
# well ("slightly left", "10% up", named fonts...); runs after
# editBrandWarning when both fire.
graph.add_node("editExecutabilityWarning", editExecutabilityWarning)
graph.add_node("knownCategoryPromptHandler", knownCategoryPromptHandler)
graph.add_node("bananaExecutor", bananaExecutor)
graph.add_node("bananaProExecutor", bananaProExecutor)
graph.add_node("fluxExecutor", fluxExecutor)
graph.add_node("gptImageExecutor", gptImageExecutor)
graph.add_node("successfulEditHandler", successfulEditHandler)
# "Not this one, I meant another image" after an edit: apologise and ask the
# user to select the right image and resend. Command-routed to the navigator.
graph.add_node("editWrongImageHandler", editWrongImageHandler)
graph.add_node("extremeFallback", extremeFallback)

# Common edit actions
graph.add_node("aspectRatioHandler", aspectRatioHandler)


# Suggestion nodes
# General critique: suggestionManager -> suggestionCreator -> suggestionHandler.
# Command-routed, so no static out-edge.
graph.add_node("suggestionCreator", suggestionCreator)
graph.add_node("suggestionHandler", suggestionHandler)
# Command-routed from suggestionHandler: one accepted suggestion per pass,
# executed the moment it is accepted, then back to suggestionHandler with the
# new image and whatever is still undecided.
graph.add_node("suggestionExecutors", suggestionExecutors)
# Offers another automatic critique pass once the suggestion loop actually
# executed something and the stack is empty; "no" falls through to the
# existing interruptBeforeLayering "what next" step.
graph.add_node("suggestionMoreAsk", suggestionMoreAsk)
graph.add_node("interruptBeforeFidelitySuggestion", interruptBeforeFidelitySuggestion)
graph.add_node("specificSuggestionCreator", specificSuggestionCreator)
graph.add_node("specificSuggestionHandler", specificSuggestionHandler)

# Edges

graph.add_edge(START, "promptCollector") # receptionist is using commands to transfer controll to the next nodes, we don't need any edges.
# graph.add_edge("receptionist", "aspectRatioDeterminer")
# graph.add_edge("aspectRatioDeterminer", "promptCollector")
graph.add_edge("promptCollector", "navigator")
graph.add_edge("navigator", "requestTypeAnnounce")

#generation flow
# generationManager -> scopeGate -> intentClarifier -> intentRouter ->
# manifestFieldsGate -> photoRequiredGate -> clarificationGate ->
# imageAnalyzerLayoutSelector -> logoDecisionGate -> generation (v33) ->
# logoCompositor. Each of these (plus scopeGate) is Command-routed, so no
# static edges out of them (freeform bypass goes scopeGate -> runGptImageModel
# directly; each gate node self-no-ops and advances to the next stage when its
# trigger condition is false -- see GENERATION_FLOW_INTERRUPTS.md).
#
# Two of those stages are further split so their interrupt sits in a node of
# its own and a resume can't re-run the LLM call(s) that preceded it:
#   scopeGate -> scopeMismatchGate -> scopeAbortGate (-> back to scopeGate)
#   manifestFieldsGate -> manifestFieldsAsk
# The ask nodes are entered only when there is something to ask; otherwise the
# owning gate routes straight on to the next stage.
#
# "generation" and "v32NeedsInputAsk"/"v32CreateMore"/"runGptImageModel" are
# all Command-routed (see v32Generation, v32NeedsInputAsk, v32CreateMore,
# _freeformNext) -- no static or conditional edges needed. Superseded by v33
# and now deleted from nodes.py: the per-model conditional dispatch
# (dall-e/flux/nano-banana*/stable-diffusion), the 7-stage CoT fusion, and
# generateMoreFinalizer's restructure-the-previous-image fast path.
graph.add_edge("generationManager", "scopeGate")

#editing flow
graph.add_edge("editingManager", "editAssetGate")
# Edit chain: gpt-image first, then gemini-3-pro-image -> nano-banana -> flux.
graph.add_edge("aspectRatioHandler", "gptImageExecutor")

# suggestionManager is Command-routed (see nodes.py): general -> suggestionCreator,
# specific -> specificSuggestionCreator. No static out-edge.

#expand flow
graph.add_edge("successfulExpandHandler", "navigator")

#layering flow: entry gate -> mounted subgraph (its own internal interrupts
#resume in place, not from the top -- see layeringManager's docstring) -> router
graph.add_edge("layeringManager", "layeringSubgraph")
graph.add_edge("layeringSubgraph", "layeringResultRouter")

#slop removal flow: mounted subgraph -> router (navigator or Critic Agent)
graph.add_edge("slopSubgraph", "slopResultRouter")


# checkpointer = InMemorySaver()
# app = graph.compile(checkpointer=checkpointer)