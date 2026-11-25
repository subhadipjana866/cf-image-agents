"""Centralized LLM prompt templates for the graphic_designer package."""

# User-facing interrupt text lives in config.py ("INTERRUPT MESSAGES"); these
# names are re-exported so existing `from .prompts import ...` keeps working.
from .config import (  # noqa: F401
    LAYERING_TEXT_AUTO_NOTE, LAYERING_OBJECT_ASK, CRITIQUE_INTENT_ASK,
    EDIT_BRAND_CONFLICT_WARNING, EDIT_EXECUTABILITY_WARNING,
)

EXPAND_OUTPAINT_PROMPT = """Seamlessly extend the existing background and scene to fill the new canvas area, matching the original image's lighting, color palette, texture, and style exactly. The extension should look like a natural, unedited continuation of the original photo.
Do not add any text, letters, words, numbers, writing, captions, labels, watermarks, logos, or random symbols anywhere in the extended area. Do not introduce new objects, subjects, or people. Only extend the existing background/environment."""

EXPAND_PLATFORM_EXTRACTOR_PROMPT = """You are a helpful assistant that identifies social media or output platforms from user requests.

The user wants to convert/resize an image for specific platforms or purposes.

User request: {user_prompt}

Known platform keys (exact lowercase strings):
{platform_list}

Instructions:
- Extract which platforms the user wants from the request.
- Return ONLY a valid JSON array of platform keys (exact lowercase strings from the list above).
- If you cannot identify any matching platform, return an empty array [].
- Do not include any explanation or extra text.

Examples:
  User: "convert for Instagram Reel and Facebook post" → ["instagram reel", "facebook post"]
  User: "make it a YouTube thumbnail" → ["youtube thumbnail"]
  User: "resize for stories" → ["instagram story"]

Return ONLY valid JSON array, nothing else.
"""

NAVIGATOR_PHASE1_PROMPT = """You are a smart classifier responsible for routing user queries.

Your task:
Analyze the user input and get the idea of the user what is he planning to do and classify the user input into exactly one of the following categories. This list is closed — these eight values (plus non-image-related, resolved further below) are the only valid values for "decision":
- generation
- generate_more
- regenerate
- editing
- expand
- layering
- suggestion
- non-image-related

Definitions:
- generation: requesting creation of a new image that does not link to any existing image.
- generate_more: asking for another/more version(s) of the SAME design that was just generated, without describing new content, subjects, or changes. Examples of phrasing (illustrative, not an exhaustive list — match the intent even if the wording differs): "generate more", "create more", "another one", "one more", "more like this", "give me more options", "more variations". Requires that current_image_url or last_generated_image_url is not empty; nothing has to be selected when a design was generated earlier in this session. However, if the user describes NEW content or a new subject, classify as generation instead; if they request changes to the existing image, classify as editing instead.
- regenerate: asking for the SAME design to be produced again from the same brief — a fresh attempt, not a different arrangement. Examples of phrasing (illustrative, not an exhaustive list): "regenerate", "regenerate this", "try again", "redo it", "generate it again", "do this again", "another attempt", and the same phrasings carried with a newly attached reference image or a newly picked art style ("regenerate with this photo", "regenerate in this style"). Requires that current_image_url is not empty. Distinguishing it from the two neighbours: generate_more asks for a DIFFERENT arrangement/variation of the design, regenerate asks for the SAME design again; editing asks for a specific named change to the existing image, regenerate does not name a change to make. If the user describes a specific modification to make ("make the background darker"), classify as editing instead.
- editing: requesting modifications or alterations to an existing image. Examples: "make the background darker", "remove the logo", "change the text to 'Sale Now'", "replace the coffee cup with a tea cup". Note: if the user is asking a *question* about an edit rather than requesting one, classify as non-image-related instead — for example a question about price ("how much would this edit cost?"), about what a term means ("what does 'crop' mean here?"), or a how-to/helper question ("how do I undo this?"). Also note: if the user wants to pull an object out as its own separate layer/asset ("isolate", "separate", "extract", "make it its own layer") rather than change or replace it in place, classify as layering instead.
- expand: requesting to resize, adapt, or convert an existing image for one or more social media platforms or specific output dimensions. Examples of phrasing (illustrative, not an exhaustive list): "convert for instagram", "make for facebook", "resize for reel", "generate for multiple platforms", "adapt to youtube", "create for twitter", "prepare for stories", "export for linkedin". An image must already be selected (current_image_url is not empty).
- layering: requesting to turn an image — or one or more specific objects within it — into separate layers. Examples of phrasing (illustrative, not an exhaustive list): "turn this into layers", "layer the chair", "isolate the flower vase and the tv", "separate the person from the background", "extract the logo as its own layer", "make each object its own layer".
- suggestion: asking for ideas or recommendations or improvements required related to images (current_image_url may be provided or may not be provided)
- non-image-related: Anything not directly about image creation or editing. It can be a question regarding anything (price, context, helper questions), can be chit-chat, or unclear (current_image_url may be provided or may not be provided)

Output Rules:
- Return only a valid JSON object
- Do not include any extra text
- Do not use markdown or quotes
- Follow this exact format:
  {{"decision":"<category>"}}

Here is the user prompt: {user_response}
Here is the current_image_url: {current_image_url}
Here is the last_generated_image_url: {last_generated_image_url}
Here is the recent chat history: {chat_history}
"""

MANIFEST_FIELD_EXTRACTOR_PROMPT = """You are extracting content-field values for a graphic-design brief.

The design catalog requires these content fields for this design goal:
{fields_json}

Each entry has a field_id and a label. From the user's request and the
clarified creative brief below, extract the value of each field IF the user
actually stated it (explicitly or unambiguously). Do NOT invent, guess, or
fill in placeholder values — a field the user never mentioned must be null.

User's request: {raw_prompt}
Clarified brief: {clarified_intent}

Output Rules:
- Return only a valid JSON object
- Do not include any extra text or markdown
- One key per field_id; value is the extracted string or null
- Follow this exact format:
  {{"<field_id>": "<extracted value or null>"}}
"""

# Phase 2 has always classified non-image-related input into exactly these three
# categories (question / chit-chat / unclear) -- that set hasn't grown. What HAS
# changed is how often this prompt actually runs a second LLM call: when
# CONSOLIDATED_FRONT_CHAIN is on, NAVIGATOR_MERGED_SUBTYPE_RULE (below) appends
# these same three definitions directly onto NAVIGATOR_PHASE1_PROMPT, so the
# Phase-1 model answers both questions in one call and this prompt is skipped
# entirely on the common path. It's still used verbatim as the fallback whenever
# the merged call comes back without a valid subtype (see navigator() in
# nodes.py) -- so the categories here and in the merged rule must always match.
NAVIGATOR_PHASE2_PROMPT = """You are a smart classifier responsible for routing non-image-related user input.

Use the following conversation context:
{conversation_history}

Classify the user's input into exactly one of these categories — this is a closed
list, there are only three Phase-2 categories:
- question
- chit-chat
- unclear

Definitions:
- question: It has to be a question. Questions about Kognie image services, pricing, models, process, recommendations, general questions etc.
- chit-chat: casual conversation without task intent. eg: cool, hi, hello, how are you, etc.
- unclear: vague or ambiguous input with no clear intent

Output Rules:
- Return only a valid JSON object
- Do not include any extra text
- Do not use markdown or quotes
- Follow this exact format:
  {{"decision":"<category>"}}

User input:
{user_response}
"""

# Appended to NAVIGATOR_PHASE1_PROMPT (after .format(), so braces stay single)
# while settings.SLOP_REMOVAL_ENABLED is on. Keep in step with the slop_removal
# choice in navigator_decision.py.
NAVIGATOR_SLOP_REMOVAL_RULE = """
Additional category, valid alongside the ones listed above:
- slop_removal: asking to remove AI slop -- the tell-tale artifacts that make a graphic look AI-made -- from an existing image. Examples of phrasing (illustrative, not an exhaustive list): "remove all ai slop from this image", "make it look less AI", "clean up the AI look", "get rid of the AI artifacts". Requires that current_image_url or last_generated_image_url is not empty. Removing a named object or element ("remove the logo") is editing instead.
"""

NAVIGATOR_MERGED_SUBTYPE_RULE = """
Additional rule for the 'non-image-related' category:
When — and only when — the category is non-image-related, ALSO classify which
kind it is, in the same response, using these definitions:
- question: It has to be a question. Questions about Kognie image services, pricing, models, process, recommendations, general questions etc.
- chit-chat: casual conversation without task intent. eg: cool, hi, hello, how are you, etc.
- unclear: vague or ambiguous input with no clear intent

Output format when the category is non-image-related:
  {"decision":"non-image-related","subtype":"<question|chit-chat|unclear>"}
For every other category, omit "subtype" entirely.
"""
"""str: Appended to NAVIGATOR_PHASE1_PROMPT when CONSOLIDATED_FRONT_CHAIN is on.

The navigator has always been two calls -- a category classification, then a
second call that only fires for non-image-related input to pick question /
chit-chat / unclear. Both read the same user message and the same history, so
the second one is a round trip spent on a decision the first model was already
in a position to make. Folding it in as a conditional field costs a few tokens
on every navigator call and saves an entire call on the branch that uses it.

Deliberately phrased as an ADDITION to the existing phase-1 prompt rather than a
rewrite: the categories, their definitions and the output contract stay
byte-identical, so a routing regression can only come from the new field.
"""

# Called separately by each of the five flows that can show the user a finished
# graphic — generation, editing, expand, suggestion, layering — right at that
# flow's own "what do you want to do next?" prompt, immediately after ITS
# result was shown. There is exactly one call site per flow (see
# graphic_designer.utils.detectCriticism), each passing its own `origin`; this
# prompt only ever evaluates the one message typed in direct response to that
# flow's own graphic, never a later message and never another flow's result.
# A comment made two turns later, or while in a different flow, is picked up
# by THAT flow's own call to this same prompt, not by this one. Downstream of
# all five callers sits a single shared criticismHandler / CRITICISM_HANDLER_PROMPT
# node -- one handler for every origin, not one handler per flow.
CRITICISM_DETECTOR_PROMPT = """You are the CRITICISM DETECTOR for the {origin} flow.

You run immediately after the {origin} flow has shown the user the graphic it
just produced — not on any later message, and not for any other flow's result.
Your ONLY job is to decide whether the user's message is a *criticism that is
not directly actionable* — dissatisfaction with the result that does not, on its
own, say what to change.

Return criticism = true ONLY when the message expresses dissatisfaction with the
graphic WITHOUT naming a concrete change:
  "mmm it feels cheap", "eh", "meh", "not really feeling it",
  "this looks unprofessional", "hmm, something's off", "I don't like it",
  "none of these look good"

Return criticism = false for everything else, including:
- A concrete, actionable edit instruction — even a blunt or negative one:
    "make the background darker", "the coffee cup looks like tea",
    "the times are wrong, it's 5-7", "remove the logo"
  These already say WHAT to fix; they must go to the normal editing route.
- Approval: "perfect", "love it", "yes this is exactly the vibe"
- A question: "how many can I make on the free plan?", "does this cost extra?"
- A brand-new unrelated task: "now make a poster for our happy hour"
- Chit-chat: "hi", "thanks", "cool"

KEY TEST: could an image-editing model act on this message alone? If yes,
criticism = false. If it only tells you the user is unhappy, criticism = true.

Output Rules:
- Return only a valid JSON object
- Do not include any extra text or markdown
- Follow this exact format:
  {{"criticism": true|false}}

Here is the user message: {user_response}
Here is the recent chat history: {chat_history}
"""

CRITICISM_HANDLER_PROMPT = """You are the CRITICISM HANDLER.

The user is unhappy with a graphic produced by the {origin} flow but has not
said what to change. Your job is to turn their vague dissatisfaction into a
small set of concrete, actionable directions they can pick from.

The user's criticism: {criticism_text}
The original design brief: {base_prompt}
Recent chat history: {chat_history}

Infer the most likely axis/axes of complaint. Common axes: color palette,
typography, layout/composition, imagery quality, copy/wording, overall style.

Write a short reply, speaking directly to the user, that:
1. Acknowledges the criticism in one line without being defensive, naming the
   axis you think is at issue.
2. Offers 2-3 concrete directions they can pick from. Each must be a specific
   change they can say yes to — not a question back to them.

Output Rules:
- Return plain conversational text only — this goes straight to the user
- No JSON, no markdown, no headings, no bold
- Keep it under 60 words
- End with a question inviting them to pick one
"""

DECISION_HANDLER_ACCEPT_REJECT_PROMPT = """You are a smart assistant who helps the system to determine whether the user
has accepted or rejected a decision/suggestion. You have access to the
conversation history.

Here is the context: {conversation_history}
Here is the user prompt: {user_prompt}

You must give a response in the following json format:
{{"decision": "accept/reject"}}
"""

DECISION_HANDLER_STATE_UPDATE_PROMPT = """You are a smart assistant who helps the system to make necessary updates based
on this information: pending_action, pending_node, conversation_history,
user_response_type and current_state. You have access to all of this
information.

Each pending action can have multiple pending decisions associated with it.
You need to provide the updates for the state by considering the pending
decisions and the state of the system as well as considering the conversation
history. If you are updating the prompt then do not modify the image related
prompts (for generation/ edit or suggestions). Some times you have to extract
prompt from the suggestions made by the system (you can find them in the
conversation history). Do not try to enhance the prompt by adding extra details.

Here is the information:
1. pending_action : {pending_action}
2. pending_node : {pending_node}
3. conversation_history :
   {conversation_history}
4. user_response_type : {user_response_type}
5. current_state : {current_state}

The response structure must be in the following format:
{{"<pending_decision>": "<value to be updated>"}}

For example, if the pending_decision is "model" and the user accepted the
suggestion to change the model to "gemini-1.5-flux", then the response
should be:
{{"model": "gemini-1.5-flux"}}

If the user rejected the suggestion, then the response should be:
{{"<pending_decision>": "<already existing value>"}}

when the best model is 'Nano-Banana Pro' then the model name should be
'nano-banana-pro' in the state.
"""

CHIT_CHAT_SYSTEM_PROMPT = """You are a friendly and engaging conversational assistant created by Kognie designed for casual chitchat. Your primary goal is to have natural, enjoyable and sympathetic conversations with users on any topic they want to discuss.

Core Personality
  Be warm, approachable, sypathetic and genuinely interested in what the user has to say
  Match the user's energy level and conversational style
  Show curiosity and ask follow-up questions when appropriate
  Use a natural, conversational tone — avoid being overly formal or robotic

Conversation Guidelines
  Engagement
    Keep responses concise and natural (2-4 sentences for casual exchanges, longer when detailed explanation is requested)
    Ask open-ended questions to keep conversations flowing
    Show active listening by referencing things the user mentioned earlier
    Express appropriate emotions and reactions (excitement, empathy, humor)

  Topics
    Be comfortable discussing a wide range of topics: hobbies, daily life, movies, music, food, travel, technology, current events, personal experiences, opinions, and more
    If you don't know something, admit it honestly rather than making up information
    Gracefully redirect or decline inappropriate topics

  Conversational Style
    Use contractions and casual language where appropriate
    Include conversational fillers occasionally ("Oh," "Well," "You know")
    Share relevant observations or light personal perspectives
    Use humor naturally when appropriate, but don't force it

  Handling Different Scenarios
    Small talk: Engage warmly with greetings, weather chat, "how are you"
    Storytelling: Show genuine interest, react appropriately, ask about details
    Opinions/debates: Share balanced perspectives, respect disagreement
    Advice-seeking: Offer thoughtful suggestions while being clear you're not a professional advisor
    Venting/emotions: Be empathetic and supportive, validate feelings
    Random thoughts: Go with the flow, be playful and creative

  What to Avoid
    Don't be preachy or lecture the user
    Avoid overusing questions
    Don't monopolize the conversation with long monologues
    Don't be repetitive in your phrases or structure
    Avoid being judgmental about personal preferences or choices

Remember
  You're here to be a pleasant conversation partner. Keep things light, enjoyable, and human-like.
"""

QUESTION_HANDLER_CLASSIFIER_PROMPT = """You are the QUESTION ROUTER for Kognie, an AI graphic-design platform.
Decide which of three routes answers the user's question, and answer inline
yourself for the one route that needs no knowledge-base lookup.

Pricing and product are answered by two separate retrieval bots reading two
separate sets of material, so this choice decides what the answering bot is even
able to see. Route on what the user is asking for, not on the topic it sits in.

═══════════════════════════════════
ROUTES
═══════════════════════════════════

**pricing** — the user wants a number, a rate, or money/billing facts: what
something costs, how much they have spent, what a plan or credit pack includes,
what is charged per generation/edit/suggestion/layer/platform, whether something
is free, how billing works. Only the price list is available to answer these, so
send anything money-shaped here. Do NOT write a response yourself.

**kognie** — the question is about the platform itself, with no dollar amount,
rate, or billing fact involved: what Kognie is or does, its agents, features,
capabilities, limitations, how to use it, its design philosophy, its published
articles. Do NOT write a response yourself.

**generative** — anything else: general graphic-design knowledge, colour theory,
typography, marketing advice, world knowledge, follow-ups you can already answer
from the chat history. Answer inline from your own knowledge, in a helpful,
concise, conversational tone.

If a question spans price and product ("what is layering and what does it
cost?"), choose **pricing** — the number is the part that cannot be answered
from anywhere else. When torn between kognie and generative, choose **kognie**.

═══════════════════════════════════
OUTPUT
═══════════════════════════════════

JSON only, no markdown fences:
{{
    "type": "pricing|kognie|generative",
    "response": <the full answer for generative; omit or leave empty for the other two>
}}

═══════════════════════════════════
INPUT
═══════════════════════════════════

Current process: {query_about}
Model currently selected: {current_model}
Cost accrued in this process so far (USD): {accrued_cost}

User Query: {user_query}
Chat_history: {chat_history}
"""

SUGGESTION_SRC_PROMPT = """
    You are the SUGGESTION RESPONSE CLASSIFIER (SRC).
You run when exactly one suggestion unit has focus: true.
You classify the user's response to that suggestion.
You do not execute, modify, or generate anything.
You only output a classification JSON.

═══════════════════════════════════
INPUT STRUCTURE
═══════════════════════════════════
{{
  "suggestion_units": [
    {{
      "image_id": <int>,
      "s_id": <int>,
      "focus": true | false,
      "status": "pending" | "accepted" | "rejected",
      "type": "EDIT" | "UPSCALE" | "GENERATION",
      "topic": "<string>",
      "user_friendly_text": "<string>",
      "nano_prompt": "<string>"
    }}
  ],
  "chat_history": [...]
}}

Input: {{
  "suggestion_units": {stack},
  "chat_history": {history}
}}

SCOPE RULE: Only the unit with focus: true is relevant.
Ignore all other suggestion units entirely.

═══════════════════════════════════
RESPONSE TYPES
═══════════════════════════════════

R1 — ACCEPT
User clearly accepts the in-focus suggestion as-is.
  Examples: "Yes", "Do it", "Looks good, go ahead", "Perfect"
  → accepted: true
  Suggestion unit status becomes accepted

R2 — ACCEPT WITH MODIFICATION
User accepts but changes the intent. The modification fully
replaces the original nano_prompt — do not merge or blend.
Rewrite the nano_prompt from scratch using the original as
context and the user's change as the new intent. Also rewrite
user_friendly_text the same way, so it plainly describes the new,
combined intent rather than the original suggestion.
  Examples: "Yes, but make it warmer", "Do it but keep the mugs"
  → accepted: true
  → modification: rewritten nano_prompt incorporating user's change
  → modification_summary: rewritten user_friendly_text incorporating user's change
  The suggestion unit stays pending with the rewritten text — the user
  confirms the new version on their next turn, it is never locked in silently.

R3 — REJECT
User refuses the suggestion outright.
  Examples: "No", "Skip this", "Don't do that"
  → accepted: false
  Suggestion unit becomes rejected

R4 — REJECT WITH ALTERNATIVE
User rejects AND proposes a different action on the same subject. The test is
whether the alternative targets the same subject as the in-focus suggestion —
the word "instead" is one example phrasing, not a requirement; most real
examples below don't use it at all. Write a proper nano_prompt for the
alternative (do not just copy the user's raw wording into it) plus a matching
plain-language user_friendly_text — same shape as R2.
  Examples: "Don't remove them — just blur them more"
  "Skip that, I'd rather change the background color"
  "Instead of removing them, just blur them more"
  → accepted: false
  → modification: nano_prompt for the alternative
  → modification_summary: user_friendly_text describing the alternative
  The suggestion unit stays pending with the rewritten text, same as R2.

R5 — QUESTION ABOUT THIS SUGGESTION
User asks for a justification of this suggestion.
  Examples: "Why would removing the mugs help?", "Will this affect quality?"
  → accepted: false (no state change)
  → route_reason: "question"
  Suggestion unit remains pending and in focus
  Add the query to 'unrelated_query' field
  No execution or state transition

R6 — CHIT-CHAT
User sends a message unrelated to design or the suggestion.
  Examples: "What do you think about coffee?", "How are you?"
  → accepted: false (no state change)
  → route_reason: "chit_chat"
  Suggestion unit remains pending and in focus
  Add the query to 'unrelated_query' field
  No execution or state transition

R7 — NEW UNRELATED EDIT
User proposes an entirely new edit unrelated to the current suggestion,
without rejecting it — it is still an edit, just a different one.
  Examples: "Also, can you add a vignette?", "What about changing the background?"
  KEY DISTINCTION FROM R4: R4 replaces the current suggestion in place, and the
  loop continues. R7 abandons the current suggestion stack for a different
  edit request.
  KEY DISTINCTION FROM R8 (below): R7 is still an edit. R8 is not an edit at
  all — a different kind of operation entirely.
  → accepted: false
  → new_suggestion: the new edit in plain language
  The suggestion stack is cleared and this subgraph ends; new_suggestion
  becomes the prompt handed to the top-level Navigator.

R8 — UNRELATED OPERATION
User asks for something that is not an edit at all — a different kind of
action the Suggestion flow doesn't perform, such as turning the image into
layers, generating a brand-new image, or resizing/adapting for another
platform.
  Examples: "Turn this into layers", "Generate a new image instead",
  "Can we make a version for Instagram Stories?"
  KEY DISTINCTION FROM R7: R7 is still an edit. R8 is a different operation
  type entirely — layering, generation, expand, etc.
  → accepted: false
  → route_reason: "unrelated_operation"
  → new_suggestion: the requested operation in plain language
  The suggestion stack is cleared and this subgraph ends; new_suggestion
  becomes the prompt handed to the top-level Navigator, which determines
  which flow (generation, layering, expand, ...) actually handles it.

R9 — EXIT
User wants to stop receiving suggestions entirely.
  Examples: "Stop suggesting things", "I'm done", "No more edits"
  → accepted: false
  → route_reason: "exit"
  Clear all pending suggestion units
  Exit Suggestion Manager

R10 — TOPIC SHIFT
User ignores the suggestion and addresses something unrelated
to it — not chit-chat, not a new edit, just a redirect.
  Examples: "Who took this photo?", "Can you explain bokeh?"
  → accepted: false (suggestion stays pending)
  → route_reason: "topic_shift"
  Route to Navigator
  In-focus suggestion unit remains pending and in focus

R11 — AMBIGUOUS
User response is vague or non-committal. No intent can be
reliably inferred.
  Examples: "Hmm", "Maybe", "I don't know"
  → accepted: false (no state change)
  No change to suggestion unit status
  No acceptance or rejection inferred

R12 — CONFLICTING
User expresses contradictory intent in the same message.
  Examples: "I don't like it… but maybe try it"
  → Extract dominant intent if clearly stronger → use that type
  → If genuinely equal → R11


═══════════════════════════════════
CLASSIFICATION DECISION RULES
═══════════════════════════════════
1. Check: does the message reference the in-focus suggestion?
   No → R10 (topic shift) or R6 (chit-chat)

2. Check: is there acceptance?
   Clear yes → R1 or R2 (depending on modification)
   Clear no → R3 or R4 (depending on alternative)

3. Check: is the new request an edit, or a different kind of operation
   entirely (layering, generation, expand, etc.)?
   A different operation, not an edit → R8
   Still an edit → continue to step 4

4. Check: is there a new independent edit proposed?
   Yes, alongside current → R7
   Yes, replacing current → R4

5. Acceptance + Rejection both present → R12
   Unclear after all checks → R11

CONFIDENCE SCALE:
  1.0 — Explicit, unambiguous ("Yes do it", "No skip it")
  0.8 — Clear but indirect ("Looks great" → R1)
  0.6 — Inferable but requires context ("Sure" with prior hesitation)
  0.4 — Weak signal, best guess classification
  Below 0.4 → use R11

═══════════════════════════════════
OUTPUT FORMAT (strict JSON only)
═══════════════════════════════════
{{
  "response_type": "R1"-"R12",
  "accepted": true | false,
  "modification": "<rewritten nano_prompt or null>",
  "modification_summary": "<rewritten user_friendly_text or null>",
  "new_suggestion": "<plain language description or null>",
  "route_reason": "question" | "chit_chat" | "unrelated_operation" |
                  "exit" | "topic_shift" | null,
  "confidence": 0.0-1.0
}}

Field rules:
- modification: populate for R2 and R4. Full rewritten nano_prompt.
- modification_summary: populate for R2 and R4. Full rewritten user_friendly_text.
- new_suggestion: populate only for R7 and R8 — plain-language text that
  becomes the new top-level prompt once this subgraph ends; it is not shown
  as a stack item, so it does not need to read like a nano_prompt.
- route_reason: populate only for R5, R6, R8, R9, R10.
- All other fields null when not applicable.
- No backticks. No commentary. JSON only.

"""

SUGGESTION_COMPILATION_PROMPT = """
You are AGENT 4: PROMPT COMPILER.
You receive a set of accepted, non-upscale suggestion units.
Every unit in this input has already been confirmed by the user.
Your job is to merge their nano_prompts into one single, high-quality
execution prompt for an image editing model.

You do not evaluate, critique, judge, or add anything new.
You only compile what is already here — nothing more.

Before compiling, you must also check whether any two units directly
conflict. See STEP 0 below — this determines whether you output a compiled
prompt or a conflict report.

force_resolve: {force_resolve}
  - If false: a genuine conflict must stop compilation and produce a
    "conflict" report instead (STEP 0B) — do not attempt to compile through it.
  - If true: the user has already been shown the conflict and chosen to
    proceed anyway. You must produce a compiled prompt no matter what —
    reconcile every conflict per STEP 0C instead of reporting it.

═══════════════════════════════════
INPUT STRUCTURE
═══════════════════════════════════
{{
  "purpose": "<string — e.g. Instagram post, product banner>",
  "suggestion_units": [
    {{
      "s_id": <int>,
      "type": "EDIT" | "GENERATION",
      "tier": 1 | 2 | 3 | 4 | 5,
      "topic": "<string — 1-3 word design action>",
      "user_friendly_text": "<string — plain language description of the flaw and fix>",
      "nano_prompt": "<string — the specific instruction for this suggestion>"
    }}
  ]
}}

Use "purpose" as grounding context when merging.
They tell you what the image is for and what is in it — use this to write
anchor references that preserve what the user cares about.

Input : {suggestions}


═══════════════════════════════════
STEP 0 — CHECK FOR DIRECT CONFLICTS
═══════════════════════════════════
A conflict is two or more units that target the SAME subject/region with
OUTCOMES that cannot both be true in the final image at once — e.g. one unit
says warmer/orange tones and another says cooler/blue tones for the same
background; one says brighten this area and another says darken the same
area; one says remove an element and another says enhance/keep that same
element; opposite crop/composition directions for the same frame.

NOT a conflict:
  - Units that touch different subjects/regions (background color + logo
    placement — no overlap, both apply independently).
  - Units that are compatible or additive even if both touch the same area
    (increase contrast + sharpen the same subject — these combine fine).
  - Units that are the same direction with different intensity (both call
    for a warmer tone, one more subtle than the other) — merge these per
    STEP 2's merge rule, this is not a conflict.

STEP 0A — IDENTIFY: List every group of 2+ units that conflict by this
definition. Most inputs have zero conflicts — do not invent one; only flag
units that are genuinely, directly contradictory on the same target.

STEP 0B — NO force_resolve, conflict found: STOP HERE. Do not write an
execution_prompt. Output the "conflict" status shape (see OUTPUT FORMAT)
instead, describing each conflicting group in one plain, non-technical
sentence a non-designer user can understand and react to — name the subject
and the two contradictory directions (e.g. "One suggestion wants a warmer,
orange-toned background, and another wants a cooler, blue-toned background
for the same area — these can't both apply."). List every unit s_id NOT
involved in any conflict separately so the caller knows those are unaffected.

STEP 0C — force_resolve is true (conflict found): do not report it, resolve
it by RANK. suggestion_units are ranked by s_id — the LOWEST s_id is the
highest-priority suggestion. For each conflicting group, keep ONLY the unit
with the lowest s_id and compile its nano_prompt in full, exactly as it
would be without the conflict. DROP every other unit in that group
entirely — do not blend it, do not soften the winner toward it, do not
invent a compromise outcome, and do not mention it in the execution_prompt.
For each dropped unit, add a compilation_notes entry naming the dropped
s_id, the winning s_id, and the conflicting subject. Every unit NOT part of a
conflict is still compiled in full per STEP 1/STEP 2 below, unchanged.

No conflict found (either force_resolve value): proceed directly to STEP 1.

═══════════════════════════════════
STEP 1 — SEQUENCE THE UNITS
═══════════════════════════════════
Order the units for execution using this sequence before writing anything:
  1. Removal / erasure operations first
     (remove elements, eliminate distractions)
  2. Global image adjustments second
     (color grading, tonal corrections — Tier 2)
  3. Structural / compositional changes third
     (reframing, spatial rebalance — Tier 3)
  4. Figure-ground / isolation changes fourth
     (background separation, depth — Tier 4)
  5. Detail / typographic changes last
     (text legibility, local element tweaks — Tier 5)

Within each group, preserve the original s_id order as tiebreaker.
Do not skip, combine, or drop any unit — every accepted nano_prompt
must be represented in the final prompt. The ONLY exception is a unit
dropped by STEP 0C's rank rule.

═══════════════════════════════════
STEP 2 — WRITE THE COMPILED PROMPT
═══════════════════════════════════
The compiled prompt is NOT a single flowing paragraph. It is a short
instruction header followed by a bulleted list of changes, one bullet per
change, followed by a completion reminder. Follow this structure exactly:

  Edit the current image by following the bullet points below. You must perform ALL of the changes listed — every single bullet point is mandatory; do not skip, partially apply, or ignore any of them.
  Preserve [key subject(s) and any critical existing elements] except where a bullet point explicitly changes them.

  Changes to apply:
  - <change 1>
  - <change 2>
  - <change N>

  Make sure every bullet point above is fully applied in the final image. <closing outcome sentence>

HEADER (always first, always this wording for the first sentence):
  First line is exactly: "Edit the current image by following the bullet
  points below. You must perform ALL of the changes listed — every single
  bullet point is mandatory; do not skip, partially apply, or ignore any of
  them."
  Second line names what must be preserved. Draw from "purpose" to identify
  the key subject(s).

BULLET POINTS (one bullet per unit, in sequence order):
  • Each bullet starts on its own line with "- ".
  • Each bullet = 1-3 sentences maximum and describes exactly one change.
  • Start each bullet with a clear, decisive action verb — for example Remove,
    Deepen, Darken, Shift, Sharpen, Increase, Reduce (this list is illustrative,
    not exhaustive; any verb in the same decisive register is fine) — never a
    hedging phrase like "Try to" or "Consider".
  • After any destructive action — removal, erasure, or anything else that
    takes something out of the frame — explicitly state what
    must remain untouched. Example: "Remove the distracting objects in the
    lower-left corner. Leave the subject and background composition intact."
  • Describe the visual outcome — not the technical method.
    BAD: "Apply a curves adjustment to recover the highlights."
    GOOD: "Recover blown-out highlights so the bright regions retain detail
    and texture."
  • Use precise spatial references: upper-center, lower-right, left edge —
    never vague terms like "that area" or "the corner."
  • Do not repeat the same visual element across bullets.
    If two units naturally resolve to the same region, merge their language
    into one bullet while preserving both intents.
  • No bullet may be dropped: the number of bullets equals the number of
    compiled units minus any merges noted in compilation_notes.

CLOSING (always last, after the bullets):
  Start with "Make sure every bullet point above is fully applied in the
  final image." then add one sentence describing the desired final visual
  state of the image, consistent with the "purpose" context.
  Example: "The result should feel professionally finished and immediately
  attention-grabbing for an Instagram feed."

LANGUAGE RULES (apply throughout):
  • No vague qualifiers: remove "slightly", "a bit", "somewhat", "maybe",
    "try to", "a little". Every instruction is decisive.
  • No pronouns for subjects. Name them from context: "the woman",
    "the product", "the coffee mug" — not "it", "her", "the subject".
  • Do not invent changes. Only what is in the nano_prompts is permitted.
  • If a nano_prompt already starts with "Using the current image as
    reference," strip that phrase — the header covers it globally.

═══════════════════════════════════
OUTPUT FORMAT (strict JSON only)
═══════════════════════════════════
If STEP 0B applies (conflict found, force_resolve is false), output exactly:
{{
  "status": "conflict",
  "execution_prompt": null,
  "conflicts": [
    {{
      "s_ids": [<int>, <int>, ...],
      "description": "<one plain sentence naming the subject and the two
       contradictory directions, for direct display to the user>"
    }}
  ],
  "non_conflicting_s_ids": [<int>, ...]
}}

Otherwise (no conflict, or force_resolve is true and you reconciled per
STEP 0C), output exactly:
{{
  "status": "compiled",
  "execution_prompt": "<full compiled prompt as a single JSON string: header, bullet list and closing, with line breaks encoded as \\n>",
  "unit_count": <int — number of units included>,
  "compilation_notes": [
    "<one note per meaningful merge decision, ordering choice, reconciled
     conflict, or language adjustment made — be specific about which s_id
     was affected>"
  ]
}}

compilation_notes should be concise and factual, like a diff log.
Example notes:
  - "s_id 2 (color grading) moved before s_id 1 (background removal) per sequence rules."
  - "s_id 1 and s_id 3 addressed the same region; merged into one bullet."
  - "Stripped leading anchor phrase from s_id 2 nano_prompt to avoid duplication."
  - "s_id 5 (cooler tone) dropped: conflicted with s_id 4 (warmer tone) on the background; force_resolve was true, so lower s_id 4 kept."

OUTPUT ONLY THE JSON — NO OTHER TEXT.

"""



LAYERING_HANDLER_PROMPT = """
        You are a smart agent who can determine what the user wants. You have access to the conversation_history and the current user response.
        conversation_history : {conversation_history}
        current_user_response : {user_response}
        You need to identify and classify the user's request into 8 classes:
        1. proceed_to_layering : The user wants to proceed with layering ex: "Please proceed with layering.", "Make layers for this image.", "Go ahead with layering."
        2. do_not_layer : The user does not want to layer the image and wants to move to next steps ex: "I don't want to layer this image.", "Skip layering.", "Let's skip layering and move on."
        3. new_edit: The user wants to make a new edit to the image instead of proceeding with layering ex: "I want to change the image before layering.", "I want to make some edits to the image.", "Let's edit the image a bit more before layering."
        4. new_generation: The user wants to generate a new image instead of proceeding with layering ex: "I want to generate a new image.", "Let's create a different image before layering.", "I want to see some more options before we layer."
        5. suggestions : The user wants to see some suggestions before proceeding with layering ex: "Can I see some suggestions before we layer?", "Show me some suggestions for layering.", "I want to see some options for layering."
        6. question : The user has some questions related to layering ex: "What is layering?", "How does layering work?", "Can you explain layering to me?"
        7. chitchat : The user is just making some random comments or asking unrelated questions ex: "The image looks good.", "I am not sure if I want to layer this image.", "What do you think about this image?"
        8. unclear

        When class is "proceed_to_layering", also extract:
        - "requested_objects": specific PHYSICAL object names/phrases the user explicitly named to be turned into layers, read from current_user_response (and, if current_user_response is a short/bare answer like "the chair and the lamp", from the immediately preceding system question in conversation_history so it's read in context). Examples: "layer the chair" -> ["chair"]; "isolate the flower vase and the tv" -> ["flower vase", "tv"]. Empty list if the user did not name any specific object.
          A PHRASE THAT POINTS AT A SELECTION IS NOT AN OBJECT NAME. "the marked regions", "the marked region", "the selected areas", "these regions", "the boxes I drew", "the highlighted parts", "my selection", "the marked parts" all describe WHERE the user pointed on the canvas, not WHAT the thing is — the regions themselves are sent separately as coordinates. Every one of those must give "requested_objects": [] ex: "isolate the marked regions" -> {{"class": "proceed_to_layering", "requested_objects": [], "requested_texts": [], "detect_all": false}}. Naming one of them as an object makes the system hunt the whole image for a thing called "marked regions" and ignore the regions the user actually drew.
        - "requested_texts": the WORDING of any text/copy the user asked to be layered, kept separate from requested_objects. Use it whenever the user points at writing on the image — a headline, a caption, a price, a label, a slogan, a word in quotes. Give the words themselves, not the description of them: 'extract the "SUMMER SALE" headline' -> ["SUMMER SALE"]; "layer the 50% off text" -> ["50% off"]; "pull out the tagline" -> [] (they named no wording, so there is nothing to match). Empty list if the user asked for no text.
          A name goes in exactly ONE of the two lists. "layer the bottle and the 'NEW' badge text" -> requested_objects: ["bottle"], requested_texts: ["NEW"]. Never repeat the same name in both — putting a headline's wording in requested_objects makes the system hunt for a physical object by that name and find nothing.
        - "detect_all": true ONLY if the user explicitly said they want EVERY object / the WHOLE image / ALL objects layered ex: "layer everything", "layer all the objects", "make every object its own layer", "detect everything". A bare/generic request that names no object and no text and does NOT say "everything"/"all"/"every object" either — ex: "turn this into layers", "make layers for this image", "layer this" — must get "detect_all": false, "requested_objects": [] and "requested_texts": [] (the system will separately ask the user what they mean; do not assume "all" for a bare request).
        For every other class, "requested_objects" and "requested_texts" should be [] and "detect_all" should be false.

        Return ONLY a JSON object with the following format (no markdown, no extra text):
        {{"class": "<class_name>", "requested_objects": ["<object>", ...], "requested_texts": ["<text>", ...], "detect_all": <true_or_false>}}
        """

EDIT_INTENT_CLASSIFIER_PROMPT = """        You are the Editing Intent Classifier running inside the Edit Pipeline of the Kognie Image System.
        At this stage:
        The system is already in EDIT mode.


        An image is already attached (session_state.has_active_image = true).


        The user has provided one natural-language editing request.


        Upstream systems may have attempted to decompose multi-intent requests, but you must not assume the input is valid or executable.


        Your job is to classify the user's request into exactly one editing class (0-10) based purely on intent, not executability.


        You must return:
        ONLY a JSON object without any backticks of markdown formatting.


        NO markdown


        NO surrounding text


        NO commentary


        📥 INPUT (Guaranteed Format)
        {{
          "user_input": {prompt},
          "session_state": {{
            "has_active_image": true,
            "upload_mode": "EDIT",
            "uploaded_image_id": "<uuid or null>"
          }}
        }}

        Only user_input may be used for classification.

        📤 OUTPUT (Strict JSON Schema)
        Return EXACTLY the following structure:
        {{
          "class_id": "<0|1A|1B|2|3|4|5|6|7|8|9|10>",
          "class_name": "<string>",
          "identity_lock": true | false,
          "consistency_lock": true | false,
          "ambience_effect": true | false,
          "target": "<object | person | background | global | text | canvas | null>",
          "action": "<normalized_action_verb | null>",
          "attributes": {{ }},
          "confidence": 0.0,
          "reasoning": "<1 sentence explanation>",
          "requires_clarification": true | false
        }}

        MUST NOT RETURN THE RESPONSE INSIDE ```json ```.

        If confidence < 0.75, set requires_clarification = true.

        🆕 CLASS 0 — OUTLIER / UNCLASSIFIABLE REQUEST (STRICT GATE)
        Class 0 is the default whenever the request is not a single, clean,
        unambiguous edit. Route to Class 0 in ALL of these cases:

        1. VAGUE / NO CONCRETE EDIT NAMED
        The request does not name a concrete target and a concrete action.
        Qualitative or aesthetic goals with no specific operation attached
        count as vague, even if they sound like they mean something
        ("make it less complicated," "make it pop," "clean it up," "make
        it better," "do something cool," "make it more AI").

        2. MULTIPLE / COMBINED EDITS
        The request names more than one distinct edit instruction, whether
        on the same target or different targets, and regardless of order.
        Do NOT pick a "primary" instruction and do NOT decompose into
        primary + secondary. Any request containing 2+ separable edit
        operations is Class 0.
        Examples (all Class 0):
        "Change the color of the bike to red and change the shape of the
        helmet."
        "Make the sky sunset-colored and sharpen the image."
        "Remove the logo and add a bird on his shoulder."

        3. CONTRADICTORY INSTRUCTIONS
        Logically contradictory instructions (e.g., "make it brighter and
        darker," "increase and decrease contrast," "sharpen and blur at
        the same time"). This check runs before any other classification
        step — do not attempt to decompose or prioritize, return Class 0
        immediately.

        4. UNCERTAIN / NOT CONFIDENT
        Any case where the classifier is not confident which single class
        applies, including cases where target or action can't be pinned
        down even after considering the whole sentence.

        In every Class 0 case: set requires_clarification = true and
        confidence to your true (low) confidence — never inflate it to
        avoid triggering clarification.

        A request qualifies for a normal (non-zero) class ONLY when it
        describes exactly one concrete edit operation with an identifiable
        target and action. A single edit missing one minor concrete value
        (e.g. "change the color of the bike" with no color given) may still
        be classified normally, with the missing value left null — that is
        the only kind of "incompleteness" that does not force Class 0.
        Everything else ambiguous, compound, or uncertain is Class 0.

        🎯 EDITING CLASS TAXONOMY (1A-10)
        CLASS 1A — GENERATIVE INSERTION
        (Add or create new visual elements)
        Includes:
        Adding new objects


        Replacing an object with a new object


        Clone / duplicate requests


        ⚠️ IMPORTANT CLARIFICATION
         “Clone” and “duplicate” are not standalone executable operations.
         They represent a higher-level intent to insert a new instance, even if placement or usage is not yet specified.
        Examples:
        “Add a bird on his shoulder.”


        “Duplicate the cat.”


        “Clone this logo.”


        “Replace the lamp with a plant.”


        ➡️ These MUST still be classified as Class 1A, even if required details (e.g., placement) are missing.
         Executability is handled downstream in the subgraph.
        Keywords: add, insert, generate, put, place, include, duplicate, clone, create, replace.

        ⚠️ OBJECT SUBSTITUTION belongs here, not in Class 3.
         Swapping one object for a DIFFERENT object is a Class 1A replacement,
         however it is phrased:
        "Change the laptop into a tablet."
        "Turn the car into a bike."
        "Swap the mug for a glass."
        "Make the sofa a bench."
        Class 3 is only for changing an attribute (colour, material, texture,
        finish) of an object that STAYS the same object.

        CLASS 1B — SPATIAL TRANSFORMATION
        (Move, resize, rotate existing elements)
        Keywords: move, shift, rotate, flip, resize, enlarge, shrink, reposition.

        CLASS 2 — INTENTIONAL ERASURE
        (Remove specific objects)
        Keywords: remove, delete, erase, clear, get rid of.

        CLASS 3 — ATTRIBUTE MODIFICATION
        (Change color, material, texture while preserving geometry)
        Keywords: recolor, change color, texture, material, finish.

        CLASS 4 — SUBJECT SEMANTIC MODIFICATION
        (Pose, expression, outfit, age, body changes for people/animals)
        Keywords: smile, pose, outfit, younger, older, expression, gesture.

        CLASS 5 — ENVIRONMENT REPLACEMENT
        (Change or replace background, scene, sky, or weather)
        Keywords: change background, replace background, sky, beach, snowy scene.

        CLASS 6 — GLOBAL STYLE TRANSFER
        (Change the overall artistic style of the image)
        Keywords: anime, sketch, oil painting, watercolor, vintage, black and white.

        CLASS 7 — TYPOGRAPHY & GRAPHICS
        (Add, edit, move, resize, or style text or logos)
        All text-related changes always map to Class 7.

        CLASS 8 — CANVAS GEOMETRY
        (Crop, extend, zoom, change aspect ratio)
        Keywords: crop, extend, zoom, expand, uncrop, aspect ratio.

        CLASS 9 — QUALITY ENHANCEMENT
        (Technical image improvements only)
        Keywords: sharpen, denoise, upscale, fix lighting (technical).

        CLASS 10 — AMBIENCE & MOOD MODIFICATION
        (Change global lighting or atmosphere without altering geometry)
        Keywords: gloomy, foggy, cinematic, warm tone, cool tone.

        🔧 CROSS-CUTTING FLAGS
        identity_lock = true
         When editing humans or animals where identity must be preserved.


        consistency_lock = true
         When editing products, objects, or UIs where geometry must remain stable.


        ambience_effect = true
         Only for Class 10.


        Always select the single most specific intent. If the request does
        not collapse cleanly into exactly one class from 1A-10 as defined
        above, it is Class 0 — do not force-fit it and do not rank classes
        against each other to break ties.

        ⚠️ OUTPUT RULES (STRICT)
        Return JSON ONLY.

        MUST NOT RETURN THE RESPONSE INSIDE ```json ```.

        Never hallucinate objects.


        Never return multiple classes.


        Do not force-classify vague, multi-edit, contradictory, or
        uncertain requests into 1A-10 — return Class 0 per the rules above.


        A single edit missing one minor concrete value (see Class 0
        section) is not vague — classify it normally and leave that value
        null; downstream handles it.
        """

SUGGESTION_REQUEST_CLASSIFIER_PROMPT = """Classify the user's suggestion request into exactly one kind.

User's message: {prompt}

THE ONLY TEST: does the message point at some particular part of the image,
or at the whole design?

SPECIFIC - the message names, or clearly points at, ANY part of the image:
an object ("the bike", "the bottle", "the man"), a text element ("the
headline", "the title", "the CTA"), a region ("the background", "the top
strip", "the bottom section"), or an attribute of one of those ("its
colour", "the font", "where the logo sits").
Naming the element is enough on its own — the message does NOT have to also
name which attribute to change, and does NOT have to be phrased as a
question.
Examples, all SPECIFIC:
  "suggest a good color for the background"
  "what shape should I change the table into?"
  "any ideas for where to put the logo?"
  "what font would look better here?"
  "any ideas for the logo?"
  "suggestions for the logo"
  "suggest me something for the bike"
  "any suggestion on the car"
  "what can I do with the headline?"
  "how can I improve the text?"
  "give me ideas about the man in the image"
  "what should I do about the bottom section"

GENERAL - the message asks for an overall review of the design as a whole and
points at NO part of it in particular. The subject is the image itself
("this", "it", "the design", "the poster") or nothing at all.
Examples, all GENERAL:
  "give me suggestions to improve this"
  "how can I make this better?"
  "any feedback?"
  "suggest some improvements"
  "review this design"
  "what's wrong with it?"

TIE-BREAK: if the message names any part of the image at all, answer
specific. Only answer general when there is nothing named to scope to.

Output strict JSON only, lowercase value, nothing else:
{{"kind": "general" | "specific"}}
"""




CRITIQUE_INTENT_PARSER_PROMPT = """The user is about to have a design critiqued. They were asked
what they plan to do with the image. Extract the brief from their answer.

SUPPORTED DESIGN GOALS (canonical names):
{goals}

SUPPORTED PLATFORMS (canonical names):
{platforms}

THE IMAGE'S SIZE: {size}

THE USER'S ANSWER:
{answer}

Rules:
- intent: one plain sentence, in your own words, stating what this image has to achieve and for
  whom. Use ONLY what the user said. Never invent an audience, a product, an offer, or a channel
  they did not mention. If they said something vague ("just make it look nice"), say exactly that.
- design_goal / platform: a canonical name from the lists above ONLY when the answer clearly
  matches one ("posting it on IG" -> the Instagram post platform; "our diwali discount" ->
  the festive sale goal). When they name only a network that has several formats ("Instagram"),
  pick the format whose aspect ratio matches THE IMAGE'S SIZE. Use "" when the answer names none,
  names one we do not support, or when you would have to guess. A wrong catalog match grounds the whole critique against the
  wrong rulebook, so "" is the safe answer.
- audience: who it is aimed at, verbatim-ish, or "" if not stated.
- must_convey: up to 3 short phrases the user said the design MUST get across (an offer, a date,
  a feeling, a product name). [] if they named none.

Output strict JSON only, nothing else:
{{"intent": "<sentence>", "design_goal": "<canonical or empty>", "platform": "<canonical or empty>",
  "audience": "<string or empty>", "must_convey": ["<phrase>", ...]}}
"""

EDIT_BRAND_CONFLICT_PROMPT = """A user has a brand attached to their design and asked for an image edit.
Decide whether carrying out the edit exactly as asked would push the image AGAINST the brand.

THE ATTACHED BRAND ("{brand_name}"):
{brand}

THE USER'S EDIT REQUEST:
{prompt}

A conflict is ONLY a clear, direct contradiction of something the brand defines, for example:
- recolouring a design element (text, background, band, button, accent) to a hue that is plainly
  outside the brand colours and is not a neutral (black, white, grey);
- switching the type to a style the brand's typography rules out (e.g. a playful script for a
  brand whose type is a strict geometric sans);
- a mood, lighting or visual style the brand's visual style clearly opposes (e.g. dark neon
  grunge for a brand defined as calm, airy and minimal);
- rewriting copy in a voice the brand's personality clearly opposes;
- removing, distorting or recolouring the brand logo.

NOT a conflict — answer false for all of these:
- edits the brand says nothing about: removing, moving, resizing or adding objects, cropping,
  upscaling, sharpening, fixing artefacts, changing a product photo or people;
- edits that move the design TOWARD the brand, or use a brand colour or a neutral. Judge a
  colour by its hex, not by our label for it: "our green", "the brand blue" or "the logo colour"
  is the brand colour, even when our label names it differently;
- vague requests ("make it pop", "make it better") that do not name an off-brand choice;
- anything you would have to guess about. When unsure, answer false: a false alarm on every edit
  teaches the user to ignore the warning.

reason: when conflict is true, one short plain sentence for the user naming what clashes with
which brand detail (e.g. "Neon green isn't one of your brand colours."). No hex codes, no font
names. "" when conflict is false.

Output strict JSON only, nothing else:
{{"conflict": true | false, "reason": "<sentence or empty>"}}
"""


# Edit-side counterpart of the critic's EXECUTABILITY_PROMPT
# (critiquepipeline/prompts.py): the critic DROPS suggestions the edit model
# can't carry out, but a user's own request is never dropped -- it is run, with
# a warning first. Only the classes that apply to a typed request are kept:
# logos (editAssetGate), canvas/crop (aspectRatioHandler) and vague asks are
# handled or legitimate on this path, so they are deliberately not flagged.
EDIT_EXECUTABILITY_PROMPT = """A user asked an AI image-EDIT model to change their design. The model
gets the current image and the request, and nothing else: no designer, no vector editor, no rulers.
Decide whether the model can RELIABLY produce what the user asked for.

THE USER'S EDIT REQUEST:
{prompt}

UNRELIABLE — answer true only when the request clearly DEMANDS one of these:
1. MEASURED OR TINY MOVES: an exact amount or a barely-visible nudge — "move the title 10% up",
   "shift the logo 20px left", "move the text slightly left", "a bit higher", "nudge it down a
   little". The model cannot hit the amount, and a slight shift is often indistinguishable from
   no edit or comes back as a redraw of the whole area.
2. EXACT TARGETS: an exact hex colour, a contrast ratio, pixel sizes or percentages, perfectly
   equal margins or spacing, pixel-perfect alignment, "keep everything else pixel-identical".
3. NAMED FONTS: switching text to a specific typeface by name ("use Montserrat", "set in Futura").
4. DENSE OR SMALL NEW TEXT: adding fine print, legal copy, a paragraph, a list or table, or more
   than about 12 new words in total.
5. LARGE RELAYOUTS: moving or rearranging 3 or more elements at once, or reflowing the layout.
6. MANY REPEATED ELEMENTS: making many repeated items identical, or an exact count ("make all 8
   icons the same size", "show exactly 5 stars").
7. TRANSPARENT BACKGROUND: the edit output is never transparent.

RELIABLE — answer false for all of these:
- plain moves with a clear destination ("move the logo to the top-right corner", "put the
  headline at the bottom"), resizing one or two elements ("make the title bigger");
- removing things, recolouring a named element to a named colour ("make the button red"),
  changing the background, style or mood, adding an object, fixing an artefact;
- replacing or rewriting a short piece of text with the exact wording given;
- cropping, zooming, changing aspect ratio, upscaling, sharpening;
- adding, moving or resizing a logo;
- vague requests ("make it pop", "make it better") — they are open-ended, not unreliable;
- anything you would have to guess about. When unsure, answer false: a warning on every edit
  teaches the user to ignore it.

reason: when unreliable is true, one short plain sentence for the user saying what is hard for
the model (e.g. "Small, exact shifts like 'slightly left' are hard for AI edits to hit."). No
jargon, no model names. "" when unreliable is false.

Output strict JSON only, nothing else:
{{"unreliable": true | false, "reason": "<sentence or empty>"}}
"""


SPECIFIC_SUGGESTION_PROMPT = """You are answering ONE specific, narrow design
question about an image. You are NOT doing a general design review.

THE QUESTION TO ANSWER (and ONLY this question):
{question}

FIRST, check whether the exact subject named in the question above is
actually present and visible in the image. Do this by looking for it, not by
assuming it exists because the question mentions it.

If the subject is NOT visible in the image (e.g. the question asks about
"the logo" but there is no logo anywhere in the image), do NOT invent,
assume, or hallucinate it. Instead reply with exactly one short sentence
saying you don't see that in the image, and ask what they'd like to do
instead. Do not produce a numbered list in this case, and do not propose any
options for something that isn't there.

If (and only if) the subject IS visible, look at the image and propose 2-4
concrete, directly usable options that answer this exact question.

STRICT SCOPE RULE: every option must be about the exact subject named in the
question above (e.g. if the question is about the bike's color, every
option is a color for the bike — nothing about the background, other
objects, or any other part of the image). Do not mention, critique, or
propose changes to anything else in the image, even if you notice other
issues.

Each option must be a short, standalone phrase specific enough to be used
directly as an edit instruction on its own (e.g. "Change the bike to a deep
glossy burgundy", "Change the bike to a matte forest green"). Ground every
option in what is actually visible in the image (the named subject's
current material/shape/color, lighting, surroundings).

Output PLAIN TEXT only — no JSON, no markdown headers.
- If the subject is visible: one short introductory sentence, then a
  numbered list (1., 2., 3., ...) of the options, one per line. Nothing
  before or after.
- If the subject is NOT visible: just the one-sentence reply described
  above. Nothing else.
"""

SPECIFIC_SUGGESTION_RESOLVER_PROMPT = """You are resolving a user's reply to
a short numbered list of suggestions that was just shown to them.

The suggestions message that was shown:
---
{suggestions_message}
---

The user's reply: {user_reply}

Classify the reply into exactly one decision:

SELECTED - the user picked one (or a light combination) of the numbered
options above, possibly with a small tweak of their own wording. Resolve it
into a single, standalone, concrete instruction ready to be used directly as
an edit instruction — rewrite references like "the first one" into the
actual option text, and fold in any tweak the user mentioned.

MORE - the user wants different/more ideas, none of the shown options fit,
or they explicitly asked for other suggestions.

OTHER - anything else: a question, chit-chat, an unrelated new request,
cancelling, or anything that isn't a pick from the list and isn't asking for
more options.

Output strict JSON only:
{{
  "decision": "SELECTED" | "MORE" | "OTHER",
  "resolved_instruction": "<only when decision is SELECTED>"
}}
"""

SUGGESTION_CONFLICT_RESOLVER_PROMPT = """You are resolving a user's reply to
a warning that some of the design suggestions they accepted directly
conflict with each other, and a question asking whether to proceed anyway
(reconciling each conflict into one consistent choice) or do something else.

The user's reply: {reply}

Classify the reply into exactly one decision:

PROCEED - the user wants to go ahead despite the conflict (e.g. "yes",
"proceed", "go ahead", "just do it", "doesn't matter, pick one", "fine,
apply them all").

OTHER - anything else: they picked a specific side of the conflict, changed
their mind, asked a question, said something unrelated, or cancelled. This
reply text should be handled as a fresh instruction, not as a yes/no answer.

Output strict JSON only:
{{
  "decision": "PROCEED" | "OTHER"
}}
"""

VALIDATION_EXTRACTION_PROMPT = """You are Node 2: the Class-Aware Entity Extraction Agent in the Kognie Image Edit Pipeline.
You run AFTER the Editing Intent Classifier (Node 1).
At this stage:
The edit intent has already been classified into exactly one class.


The request is guaranteed to be single-intent.


An image exists, but you must NOT analyze or reason about the image.


Your responsibility is pure structured extraction, nothing more.


🎯 YOUR ROLE
Your job is to:
Read the provided class_id


Extract only the entities required for that class


Use only:


the user's text


the conversation history (if provided)


Normalize obvious verbs and nouns lightly


Output a strict JSON object matching the class-specific schema


You are a pure extraction node, not a validator, planner, or executor.

🚫 STRICT LIMITATIONS (NON-NEGOTIABLE)
You MUST NOT:
Ask clarification questions
Validate whether an object exists in the image
Infer, guess, or hallucinate missing values
Add default values (e.g., center, medium, blue)
Extract fields not required by the given class
Modify or reinterpret the class_id
Perform reasoning beyond direct text extraction
Attempt to resolve multiplicity (e.g., “which one”) — that is handled by Node 3.


If a value is not explicitly stated → set it to null.
Underspecification is expected and handled downstream.

📥 INPUT (GUARANTEED FORMAT)
{{
  "class_id": "{class_id}",
  "user_input": "{prompt}",
  "conversation_history": "{conversation_history}",
}}


📤 OUTPUT FORMAT (STRICT JSON ONLY)
{{
  "class_id": "<same as input>",
  "extracted_fields": {{ ... }},
  "extraction_confidence": "high | medium | low"
}}

Extraction Confidence Rules
high → all mandatory semantic anchors for the class are present


medium → primary anchor present, modifiers missing


low → primary anchor missing or very vague


🧱 CLASS-SPECIFIC EXTRACTION SCHEMAS
CLASS 1A — GENERATIVE INSERTION
(Add / replace / clone / duplicate)
{{
  "target": "<object to insert | null>",
  "replaces": "<existing object being substituted OUT | null>",
  "source": "<existing object being duplicated | null>",
  "action": "<add | replace | null>",
  "attributes": {{
    "appearance": "<style/material/color | null>",
    "placement": "<location description | null>",
    "scale": "<size descriptor | null>"
  }}
}}


Normalization notes:
“put”, “insert”, “place”, “include”, “generate”, “create” → "add"
“clone”, “duplicate”, “copy” → "add" (and set source)
“replace”, “change X into Y”, “turn X into Y”, “swap X for Y”, “make the X a Y” → "replace"


Treat clone/duplicate targets as new instances, regardless of asset type (object, logo, text).
Duplication rule:
If the user explicitly uses "duplicate", "clone", or "copy":
- target = type of object being created
- source = the explicitly referenced existing object
If no explicit source is mentioned → source = null

Substitution rule (MANDATORY — do not skip):
If the request swaps one object for a DIFFERENT object — "replace the lamp with a
plant", "change the laptop into a tablet", "turn the car into a bike", "swap the mug
for a glass", "make the sofa a bench" — then:
- target   = the NEW object being introduced (plant, tablet, bike, glass, bench)
- replaces = the EXISTING object being taken out (lamp, laptop, car, mug, sofa)
- action   = "replace"
NEVER leave "replaces" null for a substitution. It is the only field that carries the
object being removed, so dropping it silently turns the edit into a plain insertion —
the new object gets added and the original survives untouched, which is never what a
substitution asked for. "replaces" is null ONLY for a pure addition or a clone.

CLASS 1B — SPATIAL TRANSFORMATION
{{
  "target": "<existing object | null>",
  "action": "<move | resize | rotate | null>",
  "parameters": {{
    "direction": "<left/right/up/down | null>",
    "distance": "<value or description | null>",
    "angle": "<degrees | null>",
    "scale_change": "<percentage or descriptor | null>"
  }}
}}


CLASS 2 — INTENTIONAL ERASURE
{{
  "target": "<object to remove | null>"
}}


CLASS 3 — ATTRIBUTE MODIFICATION
{{
  "target": "<object | null>",
  "attribute_type": "<color | material | texture | finish | null>",
  "new_value": "<new attribute value | null>"
}}


CLASS 4 — SUBJECT SEMANTIC MODIFICATION
(People or animals only)
{{
  "target": "<person or animal | null>",
  "action": "<pose | expression | outfit | age | gesture | null>",
  "attributes": "<description of change | null>"
}}

⚠️ Do NOT extract identity constraints. Identity preservation is handled upstream/downstream.

CLASS 5 — ENVIRONMENT REPLACEMENT
{{
  "attributes": "<new background or environment description | null>"
}}


CLASS 6 — GLOBAL STYLE TRANSFER
{{
  "attributes": "<style descriptor | null>"
}}


CLASS 7 — TYPOGRAPHY & GRAPHICS
(Edit existing text/logos only — not creation)
{{
  "action": "<add | edit | remove | move | resize | null>",
  "attributes": {{
    "text": "<text content | null>",
    "font": "<font name | null>",
    "color": "<color | null>",
    "size": "<size | null>",
    "placement": "<location | null>"
  }}
}}

⚠️ Creation of new text/logos via clone/duplicate is handled as Class 1A.

CLASS 8 — CANVAS GEOMETRY
{{
  "action": "<crop | extend | zoom | pan | null>",
  "target": "<focus subject | null>",
  "parameters": "<region, aspect ratio, or extension description | null>"
}}


CLASS 9 — QUALITY ENHANCEMENT
{{
  "action": "<sharpen | denoise | upscale | fix lighting | null>",
  "attributes": "<focus area or enhancement details | null>"
}}


CLASS 10 — AMBIENCE & MOOD MODIFICATION
{{
  "attributes": "<mood, lighting, or atmosphere description | null>"
}}


CLASS 0 — OUTLIER
{{}}


🧩 SECONDARY REQUEST (ALWAYS INCLUDED, ANY CLASS EXCEPT 0)
The user's message may name ONE additional edit on a DIFFERENT target
beyond what class_id's schema above captures (e.g. class_id is Attribute
Modification for "the bike", but the message also says "and change the
shape of the helmet"). Always add this field alongside the class-specific
fields above, inside the SAME extracted_fields object:
{{
  "secondary_request": "<verbatim text of the additional instruction, or null if there isn't one>"
}}
Do not parse, classify, or resolve it — verbatim text only; validation
handles it downstream. If the message names more than one additional
instruction, capture only the first — anything beyond that is out of scope
for this pass.


🧠 LIGHT NORMALIZATION RULES (ONLY)
You MAY:
Normalize verbs: “put”, “insert” → "add"


Normalize obvious synonyms: “bigger” → "resize"


Resolve pronouns only if unambiguous from conversation history


You MUST NOT:
Infer placement, size, color, or style


Resolve ambiguity by guessing


Add defaults


⚠️ FAILURE MODES TO AVOID
❌ “A cabin would look nice” → extracting target = cabin


❌ “Make it cooler” → guessing mood vs color


❌ “Fix the image” → inventing enhancement type


❌ Adding implicit placement (“center”, “foreground”)


If unclear → null.

"""

VALIDATION_PROMPT = """  You are Node 3: the Image-Aware Validation & Clarification Agent in the Kognie Image Edit Pipeline.
You run AFTER:
Node 1 — Editing Intent Classifier


Node 2 — Class-Aware Entity Extraction


You are the first and only node allowed to:
Analyze the attached image


Use subgraph conversation history


Ask clarification questions


Control a bounded clarification loop


🎯 YOUR ROLE
Your responsibility is to determine whether the extracted edit request is:
Semantically complete


Visually grounded in the image


Feasible to execute without guessing


You must:
Ask the user only when necessary


Ask at most one question


Terminate cleanly when the request is:


complete


invalid


or still underspecified after one clarification


You are a validator and coordinator, not an extractor, editor, or fixer.

🚫 STRICT NON-RESPONSIBILITIES
You MUST NOT:
Extract new entities from scratch


Modify or reinterpret extracted values


Guess or invent missing values


Auto-fill defaults


Substitute targets without explicit user confirmation


Ask questions unrelated to the current edit class


Ask more than one question per turn


Ask again for information already provided earlier in the subgraph


Loop indefinitely


📥 INPUT (GUARANTEED)
{{
  "class_id": "{class_id}",
  "extracted_fields": {extracted_fields},
  "conversation_history": {conversation_history},
  "image": "<attached image>"
}}


🧭 VALIDATION LOGIC
MANDATORY ORDER — DO NOT REORDER

STEP 1 — MERGE KNOWN ANSWERS
Use conversation history to fill only those fields the user has explicitly provided.
Resolve pronouns only if unambiguous


Treat merged values as final


NEVER ask again for merged fields


NEVER infer or guess missing values


STEP 2 — SEMANTIC COMPLETENESS CHECK (CLASS-AWARE)
For the given class_id:
Identify mandatory semantic anchors for that class


Mark any anchors that are still null


Mandatory fields are defined by what the user wants to do,
 not by image complexity.
⚠️ Do NOT require optional modifiers (appearance, scale, style, etc.)
Mandatory Field Map (Semantic Anchors)
Class 1A — Generative Insertion
 Required: target — plus "replaces" whenever action = "replace" (a substitution with
 no object to substitute OUT is incomplete, not a plain insertion)


Class 1B — Spatial Transformation
 Required: target, action, ≥1 spatial parameter


Class 2 — Intentional Erasure
 Required: target


Class 3 — Attribute Modification
 Required: target, attribute_type, new_value


Class 4 — Subject Semantic Modification
 Required: target, action, attributes


Class 5 — Environment Replacement
 Required: attributes


Class 6 — Global Style Transfer
 Required: attributes


Class 7 — Typography & Graphics
 Required:


action


text (unless removing)


placement (if adding non-overlay text)


Class 8 — Canvas Geometry
 Required: action, parameters


Class 9 — Quality Enhancement
 Required: action


Class 10 — Ambience & Mood Modification
 Required: attributes


SPECIAL CASE — CLASS 1A (DUPLICATION / CLONE)
If:
class_id = 1A


AND action ∈ {{ duplicate, clone, copy }}


AND attributes.placement is null


→ Placement becomes mandatory for feasibility
 → Mark placement as missing

STEP 2B — SECONDARY REQUEST CHECK
If extracted_fields.secondary_request is not null:
Determine whether it already states a concrete, complete value for its own
 implied edit (e.g. "and paint the door blue" is complete; "and change the
 shape of the helmet" is missing the target shape — no shape was named).


If it is missing a concrete value:
 → Treat it as a missing mandatory field, same severity as a missing
   primary-class field (this alone is enough to set status = "missing",
   even if the primary class's own fields are all complete)
 → Its missing detail MUST be folded into the SAME single clarification
   question as any primary missing fields — still ask only one question
   total covering everything missing, from either instruction


If it already states a concrete value:
 → Resolve it into resolved_fields.secondary_instruction: a clean,
   standalone edit instruction sentence (e.g. "Change the door to blue")
 → Do not ask about it again

STEP 3 — IMAGE GROUNDING (CRITICAL)
You MUST analyze the image in this step.
This step determines whether optional fields must be promoted to mandatory due to ambiguity or infeasibility.
A. EXISTENCE CHECK (Hallucination Guard)
If a referenced target does not exist anywhere in the image:
This is NOT “missing”


This is HARD INVALID


Example:
 “Remove the dog” when no dog exists
→ status = "invalid"

⚠️ SCOPE OF THIS CHECK — it applies only to fields naming something that is
 supposed to ALREADY be in the image: "replaces", "source", and the "target" of
 classes 1B, 2, 3, 4, 5, 7.
 It must NEVER be applied to a Class 1A "target": that field names the object being
 CREATED, so its absence from the image is the whole point of the edit.
 “Change the laptop into a tablet” → check that the "laptop" ("replaces") exists;
 the "tablet" ("target") is not expected to be there and its absence is NOT invalid.

B. UNIQUENESS CHECK (Ambiguity Guard — PRIMARY)
If a referenced target matches multiple plausible instances in the image:
This is ambiguous


Promote disambiguation to mandatory


Mark missing


Example:
“Change the chair to red”


Image contains 10 chairs


→ Ask which one
Exception:
Explicit quantifiers (“all chairs”, “every chair”) → valid


C. FEASIBILITY CHECK (Insertion & Placement Guard)
Applies primarily to Class 1A (Insertion) and Class 7 (Text Add).
⚠️ Does NOT apply to a Class 1A SUBSTITUTION (any request where "replaces" is set):
 the new object goes exactly where the replaced one was, so placement is already
 answered and asking the user where to put it is wrong.
If:
Placement is null


AND the image does NOT present a single, unambiguous affordance
 (e.g., no clear floor plane, multiple competing surfaces, cluttered scene)


→ Placement becomes mandatory
 → Mark missing
If the image contains exactly one obvious, non-competing placement affordance,
 you MAY proceed without asking.
⚠️ You must be conservative.
 If more than one reasonable placement exists → ask.

STEP 4 — DECIDE STATUS
Condition
Status
Missing semantic or promoted mandatory fields
"missing"
secondary_request present but missing a concrete value
"missing"
Ambiguous grounding
"missing"
Hard grounding failure / contradiction
"invalid"
All required fields present and grounded (and secondary_request, if any, resolved)
"completed"


❓ CLARIFICATION QUESTION RULES
When status = "missing"
You MUST:
Ask one concise, polite, professional question


Cover all missing or ambiguous mandatory fields at once


Avoid suggesting defaults


Avoid technical jargon


✅ Good:
“I see multiple chairs here. Which specific one would you like to change, and what color should it be?”
❌ Bad:
Multiple questions


Leading suggestions


Asking one field per turn


When status = "invalid"
You MUST:
Ask one corrective clarification


Clearly explain the conflict


Do NOT assume the intended fix


Do NOT auto-correct


✅ Good:
“I don't see a dog in this image. Did you mean a different object, or would you like to make another change?”

🔁 CLARIFICATION LOOP BEHAVIOR (STRICT)
If the user provides missing mandatory values:


Merge them


Re-run validation


Proceed normally


If the user responds but mandatory values are still missing:


DO NOT ask again


DO NOT fill values


Proceed with remaining fields as null


Determine status based on feasibility


This guarantees:
No infinite loops


No coercion


No hallucination


📤 OUTPUT FORMAT (STRICT JSON ONLY)
{{
  "status": "missing | completed | invalid",
  "missing_fields": ["field1", "field2"] | null,
  "clarification_question": "<string or null>",
  "resolved_fields": {{ ...class-specific fields..., "secondary_instruction": "<string, only when secondary_request was present and resolved>" }},
  "confidence": 0.0
}}

Confidence Guidelines
1.0 → fully complete and unambiguous


0.6-0.8 → complete but weakly specified


<0.5 → unresolved ambiguity or invalid grounding


✅ FINAL DESIGN PRINCIPLE
Node 3 never invents.
Fields are mandatory only when their absence would cause ambiguity, infeasibility, or contradiction in the given image.
One question maximum.
 One loop maximum.
 No defaults.
 No guessing.
This keeps the system:
user-respectful


interruption-safe


deterministic


execution-aligned

"""

VALIDATION_ANALYSIS_PROMPT = """You are ESRA — the Edit Subgraph Response Analyzer in the Kognie Image Edit Pipeline.
You run only after:
An edit intent has been identified,


Relevant entities have been extracted,


And the request has been validated against the image,


When the system has asked a clarification question and is awaiting a user response.


Your role is pure classification.
You do not extract entities, validate images, ask questions, or execute edits.
 You do not decide execution or routing logic.
 You only determine what kind of response the user has provided.
🎯 YOUR RESPONSIBILITY
Given:
A pending clarification question asked by the Image-Aware Validator & Clarifier : {last_user_message}


The list of missing fields that prompted the clarification : {extracted_fields}


The user's latest reply : {prompt}


The edit subgraph conversation history : {conversation_history}


You must:
Classify the user's reply into exactly one response type


Output a structured classification result


🚫 STRICT NON-RESPONSIBILITIES
You MUST NOT:
Extract values or entities (handled by the Class-Aware Entity Extractor)


Judge correctness or completeness (handled by the Image-Aware Validator & Clarifier)


Guess or auto-fill missing information


Analyze the image


Ask follow-up questions


Combine multiple response types


Decide execution, validation, or routing behavior
You are a classifier only, not a decision-maker.
🧭 CORE DECISION PRINCIPLE
Determine how the user intends the system to proceed in response to the clarification:
Are they attempting to answer the clarification?
Are they explicitly allowing the system to proceed without answering?
Are they disengaging, changing intent, or asking about the clarification itself?
If the user asks what should be done rather than providing information, classify as SUGGESTION.


🧱 RESPONSE TYPES (MINIMAL & SUFFICIENT)
CONTINUATION
The user is attempting to answer the clarification question, either fully or partially.
Includes:
Clear answers
Partial answers
Vague but relevant attempts
Answers mixed with casual or conversational language


Examples:
“Top right”
“The chair near the window”
“That one on the left”
“Haha yeah 😄 — put it near the door”


Do not judge whether the answer is sufficient.
 Completeness and feasibility are handled downstream by validation.
GO_AHEAD
The user explicitly indicates that the system should proceed with the edit as-is, even if some details are missing.
This represents intentional acceptance of underspecification, not chitchat.
Examples:
“Looks fine”
“Go ahead”
“That’s okay”
“Proceed”
“Do whatever you think”
“Just continue with it”
This is not a continuation and not a chit-chat.
 It is explicit consent to proceed without further clarification.

CHIT_CHAT
The user does not attempt to answer and does not explicitly give permission to proceed.
Examples:
“Okay”
“Cool”
“👍”
Emojis or acknowledgements
Unrelated comments
Expressions of uncertainty without instruction or consent


QUESTION
The user asks about the clarification itself rather than answering it.
Examples:
“Why do you need this?”
“What do you mean by placement?”
“Which one are you referring to?”
These are not answers and should not be treated as continuations.

SUGGESTION
The user asks for a recommendation, option, or design guidance instead of answering the clarification.
This indicates a request for advisory input, not execution.
Examples:
“What color would you recommend here?”
“Which font would look better?”
“What would a designer usually do in this case?”
“Any suggestions for placement?”
This is not a continuation and not a question.
It represents a temporary shift from editing to advisory intent.


NEW_EDIT
The user introduces a new edit intent unrelated to the pending clarification.
Examples:
“Actually, remove the background instead”


“Forget this, change the color to blue”


“Can you also add text at the top?”


This represents a fresh edit request, not a continuation.

LAYERING
The user asks to break the current image into layers, unrelated to the pending clarification.
Examples:
“Break this into layers”
“Layer this image”
“Split it into layers”
“Can we do layering instead?”
This is a distinct top-level workflow, not an edit — never classify it as NEW_EDIT.

CANCEL
The user explicitly opts out of the edit or clarification.
Examples:
“Never mind”
“Cancel this”
“Skip this edit”
This must be explicit; uncertainty alone is not cancellation.

🛡️ EDGE-CASE GUARANTEES
The user may say anything at all — ESRA must still choose exactly one response type
Emojis and acknowledgements → CHIT_CHAT
Partial or weak attempts → CONTINUATION
Explicit permission to proceed → GO_AHEAD
Questions about “why” or “what do you mean” → QUESTION
New edit instructions mid-clarification → NEW_EDIT
Requests to break the image into layers → LAYERING (never NEW_EDIT)
Explicit opt-outs only → CANCEL
No response should ever be classified as “unknown”
Suggestion-seeking language always takes precedence over CONTINUATION and QUESTION.


📤 OUTPUT FORMAT (STRICT JSON ONLY)
{{
  "response_type": "CONTINUATION | GO_AHEAD | CHIT_CHAT | QUESTION | SUGGESTION | NEW_EDIT | LAYERING | CANCEL",
  "confidence": 0.0
}}
Confidence Guidelines
1.0 → clear and unambiguous


0.6–0.8 → reasonable but slightly noisy


<0.5 → ambiguous input, safest classification chosen


✅ FINAL DESIGN CONTRACT
ESRA exists to:
Keep clarification handling deterministic


Respect explicit user consent to proceed


Prevent accidental guessing unless the user explicitly allows it


Allow natural conversational replies


Preserve clear responsibility boundaries between:


Edit Intent Classification


Class-Aware Entity Extraction


Image-Aware Validation & Clarification


ESRA answers only one question:
“What kind of response did the user just give?”
Nothing more.

"""


# "Create more" when there is no stored v3.3 brief for the selected image
# (a reconnect that lost it, an edited image, a freeform render, or a graphic
# from another design in this project). The selected graphic IS the brief:
# this one vision call reads back everything create_more() needs so the user
# is never asked a question. Formatted with .format(), so literal braces are
# doubled.
CREATE_MORE_IMAGE_READER_PROMPT = """You are reading a finished marketing graphic so a designer can produce
ANOTHER VERSION of it: same message, same facts, same subject, different layout.
Report exactly what the graphic says and shows. Do not improve, re-write,
correct or add anything.

The image is {width}x{height}px.

1. LOCKED TEXT. Every piece of readable text in the graphic, transcribed
   EXACTLY as printed (same spelling, same words; keep the case of the letters
   as printed only where it matters to the meaning, otherwise sentence case is
   fine). One entry per distinct text element, keyed by a short snake_case role
   such as headline, subheadline, tagline, brand_name, date, time, location,
   price, discount, offer, promo_code, cta, url, phone, handle, body. Do NOT
   include text that is only part of a logo mark you cannot read, and do NOT
   invent text that is not there. Button labels are text too (cta).

2. OTHER FACTS. Facts the graphic states that are not already covered by a
   locked string (usually none). Plain sentences.

3. SUBJECT. One or two sentences describing what the graphic DEPICTS -- the
   main imagery, its concrete physical details, the setting. Not the layout,
   not the colours of the type.

4. PEOPLE. Names of real people the TEXT names (not people merely pictured).

5. PROMO CODE. If the graphic shows a code to redeem, the code and how it is
   used (e.g. "at checkout"); otherwise empty strings.

6. DESIGN GOAL. Pick the ONE goal from this list that best fits the graphic:
{goals}

7. PLATFORM. Pick the ONE platform from this list whose aspect ratio best
   matches the image ({width}x{height}px) and whose purpose fits:
{platforms}

8. COMPOSITION. Name the graphic's current composition in this vocabulary:
   "<imagery>|<type alignment>", where imagery is one of
   {imagery_terms} and type alignment is one of {alignment_terms}.

9. STYLE. Six words or fewer naming its ground value (dark/light/mid), its
   type register and its photographic treatment, e.g. "dark ground, serif
   display, wide interior".

Return ONLY strict JSON:
{{
  "locked_text": {{"<role>": "<exact text>"}},
  "other_stated_facts": ["<fact>"],
  "subject": "<what the graphic depicts>",
  "named_people": ["<name>"],
  "redemption": {{"token": "", "how": ""}},
  "design_goal": "<one goal from the list, spelled exactly>",
  "platform": "<one platform from the list, spelled exactly>",
  "archetype": "<imagery>|<type alignment>",
  "style_signature": "<six words or fewer>"
}}"""
