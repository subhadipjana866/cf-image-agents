from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated

# MAIN GRAPH STATE DEFINITION

class GraphicDesigner(TypedDict):
    """Main graph state for the Graphic Designer workflow.

    Holds all mutable state fields that are passed between nodes in the
    primary LangGraph :class:`StateGraph`.  Each field uses an
    ``Annotated[..., lambda x, y: y]`` reducer so that the latest value
    always wins on state merges.

    :cvar user_id: Unique identifier for the user.
    :cvar session_id: Unique identifier for the session.
    :cvar attached_images: List of base-64 data URIs of images attached by
        the user.
    :cvar is_uploaded: Whether the user has uploaded an image.
    :cvar current_image_url: S3 URL of the current image being edited.
    :cvar b64_image_url: Base-64 encoded image data for editing executors.
    :cvar previous_image_url: S3 URL of the image before the latest edit.
    :cvar prompt: The current prompt for generation or editing.
    :cvar base_prompt: The original/base prompt before enhancement.
    :cvar question_for_user: A pending clarification question for the user.
    :cvar model: The selected image-generation/editing model name.
    :cvar aspect_ratio: The target aspect ratio (e.g. ``'16:9'``).
    :cvar style: The art style to apply (e.g. ``'cinematic'``).
    :cvar num_images: Number of images to generate in a batch.
    :cvar purpose: The declared output purpose (e.g. ``'Instagram post'``).
    :cvar cost: Accumulated monetary cost of API calls.
    :cvar conversation_history: List of ``{role, content}`` message dicts.
    :cvar pending_node: The next node to be executed after a decision.
    :cvar edit_intent: The classified editing intent category.
    :cvar name: Name identifier for the current graph execution context.
    :cvar s_stack: Suggestion stack — list of suggestion dicts.
    :cvar s_id: ID of the suggestion currently being processed.
    :cvar suggestion_type: Type of suggestion flow (``'auto'`` or
        ``'user_driven'``).
    """
    user_id: Annotated[str, lambda x, y: y] # Unique identifier for the user
    username: Annotated[str, lambda x, y: y] # Username for personalized messages
    session_id: Annotated[str, lambda x, y: y] # Unique identifier for the session
    attached_images: Annotated[List[str], lambda x, y: y] # List of URLs of images attached by the user
    is_uploaded: Annotated[bool, lambda x, y: y] # Whether the user has uploaded an image
    critique_provenance: Annotated[str, lambda x, y: y] # 'generated' | 'uploaded' — where the image under critique ORIGINALLY came from. Sticky across edits (an edit rewrites base_prompt, so it can't be re-derived); cleared by v32Generation/v32CreateMore when a new image is generated. Gates brief-compliance checks in the critic.
    current_image_url: Annotated[str, lambda x, y: y] # URL of the current image being edited
    critique_intent: Annotated[str, lambda x, y: y] # What the user said they PLAN to do with this image (goal, audience, where it runs), captured by suggestionManager before every general critique. Every critique lens judges against this instead of guessing the brief.
    critique_intent_image: Annotated[str, lambda x, y: y] # The current_image_url the intent above was captured for, so a critique of a DIFFERENT image asks again while a re-critique of the same one does not
    critique_goal: Annotated[str, lambda x, y: y] # Canonical design goal parsed out of critique_intent (may be ''), used ONLY to ground the critique -- never written to design_goal, which is the sticky project-launch catalog setting main.py persists
    critique_platform: Annotated[str, lambda x, y: y] # Canonical platform parsed out of critique_intent (may be ''); same read-only-for-critique rule as critique_goal
    b64_image_url: Annotated[str, lambda x, y: y] # Base64 encoded image data
    previous_image_url: Annotated[str, lambda x, y: y] # URL of the previous image
    prompt: Annotated[str, lambda x, y: y] # Prompt for image generation or editing
    enhanced_prompt: Annotated[str, lambda x, y: y] # Enhanced prompt after processing
    base_prompt: Annotated[str, lambda x, y: y] # Response received from the user
    question_for_user: Annotated[str, lambda x, y: y] # Question posed to the user
    model: Annotated[str, lambda x, y: y] # Model used for image editing
    aspect_ratio: Annotated[str, lambda x, y: y] # Aspect ratio of the images to be edited
    style : Annotated[str, lambda x, y: y] # Style applied to the image
    num_images : Annotated[int, lambda x, y: y] # Number of images to generate
    purpose: Annotated[str, lambda x, y: y] # Purpose of the image editing/generation
    platform: Annotated[str, lambda x, y: y] # Target platform for the image (e.g. 'Instagram', 'Print', etc.)
    cost: Annotated[float, lambda x, y: y] # Cost associated with the image editing/generation
    conversation_history: Annotated[List[Dict[str, str]], lambda x, y: y] # History of the conversation with the user
    pending_node: Annotated[str, lambda x, y: y] # The next node to be executed in the graph
    edit_intent: Annotated[str, lambda x, y: y] # The intent of the edit operation
    name: Annotated[str, lambda x, y: y] # Name associated with the image generation session
    s_stack : Annotated[List[str], lambda x, y: y] # Suggestion stack for managing user feedback
    s_id : Annotated[int, lambda x, y: y] # Suggestion ID for tracking current suggestion being processed
    suggestion_type: Annotated[str, lambda x, y: y] # Type of suggestion (auto or user_driven)
    suggestion_target_confirmed: Annotated[bool, lambda x, y: y]  # The user has picked WHICH graphic an automatic critique is about (asked only when a design has several -- see suggestionManager); cleared on every new generation
    suggestion_more_round: Annotated[int, lambda x, y: y]  # How many times suggestionMoreAsk's "want more suggestions?" has already been said yes to for the CURRENT suggestion chain. Capped at 2 -- the 2nd round executes without a 3rd ask. Reset to 0 at every fresh entry into the suggestion flow (post-generation, a typed "give me suggestions", or criticismHandler's rebuild); only suggestionMoreAsk's own "yes" branch increments it.
    critique_telemetry: Annotated[Dict[str, Any], lambda x, y: y]  # Per-phase cost/time/token record from the critic pipeline: {cost, wall_s, tokens, phases[], notes{}}. notes carries the critique diagnostics: pre_arbiter_findings (every lens's raw findings, before curation), curation_dropped (what the quality floor/caps removed), suggestion_counts (raw vs final, for tuning the floor)
    suggestion_request: Annotated[str, lambda x, y: y] # Stable original text of a specific-suggestion request (specificSuggestionCreator/Handler), kept unchanged across "give me other ideas" loops
    composite_prompt: Annotated[Optional[str], lambda x, y: y] # Unused: suggestions are executed one at a time by suggestionExecutors, never merged into one prompt
    conflicts: Annotated[List[Dict[str, Any]], lambda x, y: y] # Unused: nothing is merged any more, so there is no compile-time conflict to report
    segmentation_result: Annotated[List[str], lambda x, y: y]
    bboxes: Annotated[List[Dict[str, Any]], lambda x, y: y]
    extracted_texts: Annotated[List[Dict[str, Any]], lambda x, y: y]
    layering_mode: Annotated[str, lambda x, y: y]  # How the user answered the layering_started interrupt: 'auto' (detect the objects) | 'manual' (they sent their own bboxes) | '' (they said something else -- layeringHandler classifies it)
    manual_bboxes: Annotated[List[Dict[str, Any]], lambda x, y: y]  # The bboxes sent alongside a 'manual' reply; consumed by the layering subgraph's manualLayering node
    requested_objects: Annotated[List[str], lambda x, y: y]  # Objects the user named for scoped ('auto') layering detection; [] means detect everything
    requested_texts: Annotated[List[str], lambda x, y: y]  # Wording of texts the user named for layering; kept apart from requested_objects so Gemini never hunts for an object by a headline's wording
    layering_targets_resolved: Annotated[bool, lambda x, y: y]  # True when layeringManager already read the objects/texts out of the prompt and skipped the layering_started ask, so layeringHandler must not re-classify
    brand_id: Annotated[str, lambda x, y: y] # The brand associated with the image editing session (e.g. 'spotify', 'apple', etc.)
    brand_name: Annotated[str, lambda x, y: y] # The name of the brand associated with the image editing session (e.g. 'Spotify', 'Apple', etc.)
    brand_dna: Annotated[Dict[str, Any], lambda x, y: y] # Brand DNA details fetched from the brands table
    image_native_width: Annotated[int, lambda x, y: y]   # Native pixel width of the current selected image (sent from frontend)
    image_native_height: Annotated[int, lambda x, y: y]  # Native pixel height of the current selected image (sent from frontend)

    # ── Generation-pipeline fields (avenue2 reasoning pipeline integration) ──
    design_goal: Annotated[str, lambda x, y: y]              # Canonical catalog design goal (e.g. 'Festive Sale')
    platform_canonical: Annotated[str, lambda x, y: y]       # Canonical catalog platform (e.g. 'Instagram Square Post')
    target_size: Annotated[List[int], lambda x, y: y]        # [width, height] exact output pixels requested at project launch (frontend 'size': '1080x1920')
    generation_count: Annotated[int, lambda x, y: y]         # Number of generation runs entered in this project/session
    asset_prefs: Annotated[Dict[str, Any], lambda x, y: y]   # Per-asset project memory: {'product_image'|'logo': {'status': 'provided'|'skipped', 'asset': <src>}}
    clarified_intent: Annotated[str, lambda x, y: y]         # Structured creative brief from clarify_intent()
    layout_profile: Annotated[Dict[str, Any], lambda x, y: y]  # Loaded layout profile for the design_goal/platform
    frontend_manifest: Annotated[Dict[str, Any], lambda x, y: y]  # frontend_manifest.json for the combo (asset tiers + content fields); {} if the combo has none
    generation_subject: Annotated[Dict[str, Any], lambda x, y: y]  # {prompt, design_goal} of the last fresh generation; generationManager compares the next one against it to decide whether field_answers/asset_prefs/chat still apply
    turn_editor_inputs: Annotated[Dict[str, Any], lambda x, y: y]  # {field_answers, asset_prefs} the editor sent with THIS turn (main.py); survive the new-design flush (fresh_generation_reset). Rewritten by main.py every turn
    field_answers: Annotated[Dict[str, Any], lambda x, y: y]   # {field_id: value} for the current design's manifest content fields (launch payload, extraction, or in-COT asks); "skipped" marks a declined field. Flushed when a new design starts (design goal change, or generationManager's Decisions API check)
    extra_assets: Annotated[Dict[str, str], lambda x, y: y]    # {asset_key: src} manifest assets beyond photo/logo, resolved for this generation; appended to the model call as reference images
    manifest_pending_fields: Annotated[List[Dict[str, Any]], lambda x, y: y]  # Ask payload handed from manifestFieldsGate to manifestFieldsAsk: content fields still unresolved after extraction
    manifest_pending_assets: Annotated[List[Dict[str, Any]], lambda x, y: y]  # Ask payload handed from manifestFieldsGate to manifestFieldsAsk: required-tier combo assets still unresolved
    selected_variant: Annotated[str, lambda x, y: y]         # Name of the layout variant selected by the matcher
    image_profile: Annotated[Dict[str, Any], lambda x, y: y]   # PIL shape analysis of an attached photo
    merge_directive: Annotated[str, lambda x, y: y]          # How to blend the photo into the chosen layout zone
    fused_prompt: Annotated[str, lambda x, y: y]             # Finished CoT-fused image prompt
    negative_prompt: Annotated[str, lambda x, y: y]          # Negative prompt from the CoT fusion
    router_flags: Annotated[Dict[str, Any], lambda x, y: y]    # Full route_intent() output (brand/layout/routing flags)
    pending_platforms: Annotated[List[str], lambda x, y: y]  # Extra supported platforms the user named (queued follow-ups)
    photo_mode: Annotated[str, lambda x, y: y]               # 'as_is' | 'reference'
    photo_zone_fullcanvas: Annotated[bool, lambda x, y: y]   # Whether the photo fills the full canvas (full-bleed hero)
    is_freeform: Annotated[bool, lambda x, y: y]             # Scope-gate freeform bypass (no layout/CoT, direct image gen)
    scope_result: Annotated[Dict[str, Any], lambda x, y: y]  # check_request_scope() output, checkpointed by scopeGate so scopeMismatchGate/scopeAbortGate can run their dialogs without re-running the check

    # ── Logo insertion fields ──
    logo_mode: Annotated[str, lambda x, y: y]                # 'provided' | 'omitted' | 'none'
    logo_asset: Annotated[str, lambda x, y: y]               # Resolved logo image as a base-64 data URI
    logo_directive: Annotated[str, lambda x, y: y]           # Instruction text for single_call logo insertion
    logo_zone_pct: Annotated[Dict[str, Any], lambda x, y: y]   # {"x_pct": [...], "y_pct": [...]} logo zone rect
    logo_insertion_mode: Annotated[str, lambda x, y: y]      # 'single_call' | 'strict_paste'
    edit_reference_images: Annotated[List[str], lambda x, y: y]  # Images the user attached with THIS message, forwarded by whichever interrupt handler took the reply and resolved to data URIs by editAssetGate. Handed to the Gemini edit executors as reference images so "add this to the design" works on the real attachment. Only ever holds one turn's attachments -- never the project's launch photo
    edit_brand_conflict: Annotated[str, lambda x, y: y]      # Per-EDIT: why this edit request contradicts the attached brand ('' = no conflict / no brand). Set by editIntentClassifier, read by editBrandWarning for the brand_conflict_warning disclaimer
    edit_not_executable: Annotated[str, lambda x, y: y]      # Per-EDIT: why the image-edit model is unlikely to carry out this request well ('' = fine). Set by editIntentClassifier, read by editExecutabilityWarning for the edit_executability_warning disclaimer
    edit_logo_mode: Annotated[str, lambda x, y: y]           # Per-EDIT logo outcome from editAssetGate: 'provided' (logo_asset rides along as a reference image) | 'omitted' (asked for, none supplied -> tell the model not to invent one) | '' (this edit never mentioned a logo). Rewritten on every edit, so it cannot leak into the next one
    edit_style_switch: Annotated[bool, lambda x, y: y]   # Per-EDIT: this request IS a change of art style (editStyleSwitch matched it), so the prompt already carries its own style instruction and _edit_prompt_guidance must not append the project style on top. Rewritten on every edit
    edit_manual_decision: Annotated[Dict[str, Any], lambda x, y: y]  # Per-EDIT: editModeClassifier's verdict {"mode": "ai" | "manual" | "manual_multiple", "op": manual_operations.json id | None, "ops": [...]}, read by manualEditHandler. Rewritten on every edit
    session_last_graphic: Annotated[Dict[str, Any], lambda x, y: y]  # Per-TURN, from main.py: the project's latest generated/edited newimages row ({} when the frame selected or attached an image). navigatorClassify makes it current_image_url (and active_graphic) for an image request made with nothing selected

    # ── Decomposed reasoning-pipeline stage fields (avoid re-running earlier
    # stages when a later gate/interrupt pauses -- see GENERATION_FLOW_INTERRUPTS.md) ──
    clarify_result: Annotated[Dict[str, Any], lambda x, y: y]  # Full clarify_intent() output
    selected_variant_dict: Annotated[Dict[str, Any], lambda x, y: y]  # Full (possibly re-selected/logo-stripped) variant dict
    canvas_px: Annotated[List[int], lambda x, y: y]          # [width, height] pixel canvas size for the selected layout
    fabrication_directive: Annotated[str, lambda x, y: y]    # Anti-fabrication instruction appended to routing_instruction

    # ── "Generate more" (same brief, different wireframe) fields ──
    used_variants: Annotated[List[str], lambda x, y: y]      # Ordered variant names already generated this session
    last_generation_prompt: Annotated[str, lambda x, y: y]   # Prompt of the last completed full-pipeline generation
    force_new_variant: Annotated[bool, lambda x, y: y]       # One-shot: layout selector must avoid used_variants
    reuse_fused_prompt: Annotated[bool, lambda x, y: y]      # One-shot: "generate more" reuses the previous CoT-fused prompt and skips the CoT fusion nodes entirely; the newly selected wireframe is attached to the image model instead
    tried_models: Annotated[List[str], lambda x, y: y]       # Image models already attempted this generation run (loop guard for the fallback chain)

    # ── "Regenerate" (same brief, SAME wireframe, fresh generation) fields ──
    # Distinct from "generate more" above: that one rotates the layout onto an
    # unused variant and restructures the previous output image into it. A
    # regenerate keeps the layout pinned and re-runs the real generation, so a
    # newly attached reference image or a newly picked style actually changes
    # the brief the image model receives.
    is_regenerate: Annotated[bool, lambda x, y: y]           # One-shot: this turn is a regenerate. Set from the frontend's explicit flag (main.py) or by the navigator's classifier; consumed by regenerateManager and cleared at the end of the CoT fusion
    regen_base_prompt: Annotated[str, lambda x, y: y]        # One-shot: the clean saved brief, before this regenerate's per-run instruction was appended to `prompt`. Keeps `last_generation_prompt` from accumulating one-off notes across repeated regenerates

    # ── Request routing (navigatorClassify -> requestTypeAnnounce) ──
    request_type: Annotated[str, lambda x, y: y]             # What the navigator classified this turn as ('generation'|'editing'|'suggestion'|'question'|'chit-chat'|'unclear'|...). Sent to the frontend by requestTypeAnnounce, then used to pick the next node

    # ── Criticism handling (per-process detector -> global criticismHandler) ──
    criticism_origin: Annotated[str, lambda x, y: y]         # Flow the criticism came from: 'generation'|'editing'|'expand'|'suggestion'|'layering'
    criticism_text: Annotated[str, lambda x, y: y]           # The user's raw criticism message, kept verbatim for the handler
    regenerate_suggestions: Annotated[bool, lambda x, y: y]  # One-shot: suggestionManager must rebuild the stack, skipping its specific/general classifier

    # ── Per-graphic metadata (newimages rows; main.py loads + saves via graphic_store) ──
    active_graphic: Annotated[Optional[Dict[str, Any]], lambda x, y: y]          # The graphic this turn is about, loaded by main.py from newimages via the frame's graphic_id (None = no graphic / unknown). Authoritative for provenance + grounding. Never re-seeded from S3
    graphic_metadata_answer: Annotated[Optional[Dict[str, Any]], lambda x, y: y] # {graphic_id, design_goal, platform, intent} captured by suggestionManager's ask-once; main.py writes it to the DB after the turn
    result_graphics: Annotated[Optional[List[Dict[str, Any]]], lambda x, y: y]   # [{url, platform}] per output image, for results whose platform differs per URL (expandManager)
    image_models: Annotated[Dict[str, str], lambda x, y: y]  # {S3 url: model that actually rendered it}, written by every node that uploads an image (after any fallback), read by main.py when it stores newimages/generations rows. Never the frontend picker value (`model`). Capped by nodes._with_image_models

    # ── Generation pipeline v3.3 (v32Generation / v32CreateMore) ──
    v32_prev: Annotated[Dict[str, Any], lambda x, y: y]      # The FIRST v33.run() result for this design (ground/subject/locked text intact), trimmed of events/thought_summary. Never overwritten by a create_more() result -- that result carries no ground/subject, so overwriting would silently drop the content lock. Cleared when a fresh (non-create-more) generation runs
    v32_lineage: Annotated[List[Dict[str, Any]], lambda x, y: y]  # Ordered {"archetype","style_signature"} entries for every graphic shown so far this design, passed to create_more() so it never repeats a composition
    v32_question: Annotated[str, lambda x, y: y]             # GROUND's "brief unusable" question, when result["status"] == "needs_input"
    v32_needs_input_attempts: Annotated[int, lambda x, y: y] # How many times v32NeedsInputAsk has looped this generation; capped so an unanswerable brief can't loop forever
    v32_exhausted: Annotated[bool, lambda x, y: y]           # create_more()'s composition menu is used up for this lineage -- generateMoreManager shows this instead of paying for a repeat
    v32_menu_size: Annotated[int, lambda x, y: y]            # create_more()'s composition_menu size for this goal x platform, for a "3 of 4" UI hint
    v32_lineage_urls: Annotated[List[str], lambda x, y: y]   # Every image URL produced from v32_prev's lineage (the run() image, its create_more() variations, and their logo-composited copies). generateMoreManager reuses v32_prev only when the SELECTED image is one of these; any other selected image is read back with a vision call instead, so "create more" always varies the graphic the user actually picked. None (legacy checkpoints) means "unknown -- trust v32_prev"
    last_generated_image_url: Annotated[str, lambda x, y: y] # Final URL of the most recent generation (set by logoCompositor, the chokepoint every generation executor passes). "Create more" falls back to it when no image is selected on the canvas
    text_layers: Annotated[Optional[Dict[str, Any]], lambda x, y: y]  # {"image_url", "layers", "notices"}: v33 TYPESET's editable-text metadata (schema kognie.text_layers.v2) for the graphic at image_url, and the font notices for the user ("Helvetica isn't available on Kognie yet ..."). Written by v32Generation/v32CreateMore, image_url moved along by logoCompositor, sent on generation_completed only while image_url is the current image. None when the graphic has no text or typography is off
    v33_render_attempts: Annotated[Optional[List[Dict[str, Any]]], lambda x, y: y]  # Set only when v33's render failed: each renderer's {"model", "ok", "detail"}, so the caller can tell a safety refusal from a technical failure
    v32_used_before: Annotated[List[str], lambda x, y: y]    # create_more()'s "used_before" -- archetype labels rejected as duplicates during this call's own retries (NOT a history across calls; that's v32_lineage), for a "N of menu_size tried" UI hint

    # ── AI slop remover (slopManager -> slopSubgraph -> slopResultRouter) ──
    slop_origin: Annotated[str, lambda x, y: y]              # How this slop removal started: 'post_generation' (accepted on the generation_completed offer) or 'navigator' (typed request)
    slop_verdict: Annotated[str, lambda x, y: y]             # How the slop subgraph ended, read by slopResultRouter: 'create_more' | 'new_request' | 'continue' ('' while it runs)


# PROMPT VALIDATOR SUBGRAPH STATE DEFINITION

class PromptValidatorSubGraph(TypedDict):
    """State for the prompt-validation subgraph.

    Used by the validation subgraph
    (:mod:`graphic_designer.subgraphs.validation`) to extract entities,
    validate them against the image, ask clarification questions, and
    produce a refined editing prompt.

    :cvar original_prompt: The user's original unmodified prompt.
    :cvar prompt: The current (possibly refined) prompt.
    :cvar conversation_history: Subgraph-local conversation history.
    :cvar current_image_url: URL of the image being validated against.
    :cvar intent: The classified editing intent / class ID.
    :cvar extracted_fields: Entity fields extracted from the prompt.
    :cvar status: Current validation status (``'missing'``, ``'completed'``,
        ``'invalid'``).
    :cvar name: Subgraph name identifier (default ``'validation_subgraph'``).
    :cvar count: Clarification loop counter.
    :cvar cost: Accumulated monetary cost of API calls.
    :cvar clarification_question: The pending clarification question, set at
        the source by ``knownCategoryPromptValidator`` and kept stable
        through the specific-suggestion loop (``specificSuggestionCreator``
        / ``specificSuggestionHandler``, in ``common_nodes.py``) so it can be
        re-derived or re-shown correctly regardless of how many side turns
        (suggestions, chit-chat, questions) happen in between.
    :cvar suggestion_request: The user's original specific-suggestion request
        text, kept stable across "give me other ideas" loops.
    """
    original_prompt: Annotated[str, lambda x, y: y]
    prompt: Annotated[str, lambda x, y: y]
    conversation_history: Annotated[List[Dict[str, str]], lambda x, y: y]
    current_image_url: Annotated[str, lambda x, y: y]
    intent: Annotated[str, lambda x, y: y]
    extracted_fields: Annotated[Dict[str, Any], lambda x, y: y]
    status: Annotated[str, lambda x, y: y]
    name: Annotated[str, lambda x, y: y] = 'validation_subgraph'
    count: Annotated[int, lambda x, y: y] = 0
    cost: Annotated[float, lambda x, y: y]
    clarification_question: Annotated[str, lambda x, y: y]
    suggestion_request: Annotated[str, lambda x, y: y]


# SUGGESTION HANDLER SUBGRAPH STATE DEFINITION

class SuggestionSubGraph(TypedDict):
    """State for the suggestion-handling subgraph.

    Used by the suggestion subgraph
    (:mod:`graphic_designer.subgraphs.suggestion`) to classify user responses
    to pending suggestions, update the suggestion stack, and route to the
    appropriate handler.

    :cvar name: Subgraph name identifier (default ``'suggestion_subgraph'``).
    :cvar prompt: The user's latest input/response.
    :cvar conversation_history: Subgraph-local conversation history.
    :cvar s_stack: The current suggestion stack with status metadata.
    :cvar s_id: ID of the suggestion currently in focus.
    :cvar model: The model to use for classification.
    :cvar classification_result: The parsed result from the classifier node.
    :cvar cost: Accumulated monetary cost of API calls.
    :cvar current_image_url: URL of the image under evaluation.
    """
    name: Annotated[str, lambda x, y: y] = 'suggestion_subgraph'
    prompt: Annotated[str, lambda x, y: y]
    conversation_history: Annotated[List[Dict[str, str]], lambda x, y: y]
    s_stack: Annotated[List[Dict[str, Any]], lambda x, y: y]
    s_id: Annotated[int, lambda x, y: y]
    model: Annotated[str, lambda x, y: y]
    classification_result: Annotated[Dict[str, Any], lambda x, y: y]
    cost: Annotated[float, lambda x, y: y]
    current_image_url: Annotated[str, lambda x, y: y]





# AI SLOP REMOVER SUBGRAPH STATE DEFINITION

class SlopSubGraph(GraphicDesigner):
    """State for the AI slop remover subgraph.

    Used by the slop subgraph (:mod:`graphic_designer.subgraphs.slop`), which
    is mounted directly as a node on the main graph. A mounted subgraph only
    sees -- and only writes back -- the keys its own schema declares, so this
    inherits every :class:`GraphicDesigner` key instead of picking a subset:
    the shared question/chit-chat/unclear handlers, the removal step and the
    exits back to the navigator all get the same context (user, session,
    model, style, attachments, image lineage, brand answers...) as every
    other branch.

    ``name`` is ``'slop_subgraph'`` while the subgraph runs, which is what the
    shared handlers in :mod:`graphic_designer.common_nodes` branch on to
    return here instead of to the navigator.
    """


# LAYERING SUBGRAPH STATE DEFINITION

class LayeringSubGraph(TypedDict):
    """State for the layering subgraph.

    Used by the layering subgraph
    (:mod:`graphic_designer.subgraphs.layering`) to handle the layering of the image.

    :cvar name: Subgraph name identifier (default ``'layering_subgraph'``).
    :cvar prompt: The user's latest input/response.
    :cvar conversation_history: Subgraph-local conversation history.
    :cvar s_stack: The current suggestion stack with status metadata.
    :cvar s_id: ID of the suggestion currently in focus.
    :cvar model: The model to use for classification.
    :cvar classification_result: The parsed result from the classifier node.
    :cvar cost: Accumulated monetary cost of API calls.
    :cvar current_image_url: URL of the image under evaluation.
    """
    name: Annotated[str, lambda x, y: y] = 'layering_subgraph'
    prompt: Annotated[str, lambda x, y: y]
    conversation_history: Annotated[List[Dict[str, str]], lambda x, y: y]
    cost: Annotated[float, lambda x, y: y]
    current_image_url: Annotated[str, lambda x, y: y]
    # detection_result deliberately NOT a state field: it holds raw numpy
    # image/depth arrays that can't be checkpoint-serialized. It's kept in an
    # out-of-band cache instead -- see _detection_cache in
    # graphic_designer/subgraphs/layering.py.
    segmentation_result: Annotated[List[str], lambda x, y: y]
    bboxes: Annotated[List[Dict[str, Any]], lambda x, y: y]
    extracted_texts: Annotated[List[Dict[str, Any]], lambda x, y: y]
    layering_mode: Annotated[str, lambda x, y: y]  # 'auto' | 'manual' | '' -- set by layeringManager from the layering_started reply
    manual_bboxes: Annotated[List[Dict[str, Any]], lambda x, y: y]  # User-drawn bboxes accompanying a 'manual' reply
    requested_objects: Annotated[List[str], lambda x, y: y]  # Objects the user named for scoped ('auto') layering detection; [] means detect everything
    requested_texts: Annotated[List[str], lambda x, y: y]  # Wording of texts the user named for layering; kept apart from requested_objects so Gemini never hunts for an object by a headline's wording
    layering_targets_resolved: Annotated[bool, lambda x, y: y]  # True when layeringManager already read the objects/texts out of the prompt and skipped the layering_started ask, so layeringHandler must not re-classify
    user_id: Annotated[str, lambda x, y: y]
    session_id: Annotated[str, lambda x, y: y]
    choice: Annotated[str, lambda x, y: y]