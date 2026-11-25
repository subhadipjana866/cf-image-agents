from langgraph.graph import StateGraph, START, END
from ..states import SuggestionSubGraph
from langgraph.types import Command, interrupt
from ..utils import parseJson, calculateCost, profile, invoke_with_fallback
from ..prompts import SUGGESTION_SRC_PROMPT
from ..config import SUGGESTION_NEXT_STEP
from ..common_nodes import chitChatHandler, questionHandler, unclearHandler


def _withFocus(stack):
    """Return ``stack`` with exactly one unit focused.

    ``focus`` is what both the classifier prompt and :func:`updateStack` key
    off: the prompt classifies only the in-focus unit, and every status change
    in updateStack is written under ``if suggestion['focus']``. A stack with no
    focus therefore classifies as ambiguous and updates nothing — a silent
    dead end. An existing focus is always respected; otherwise the first
    pending unit takes it.
    """
    if not stack or any(u.get('focus') for u in stack):
        return stack
    target = next((u for u in stack if u.get('status') == 'pending'), stack[0])
    return [{**u, "focus": u is target} for u in stack]


@profile
def classifier(state: SuggestionSubGraph) -> SuggestionSubGraph:
    """Classify the user’s response to a pending suggestion.

    Uses Gemini 2.5 Flash (with GPT-5-mini fallback) to classify the
    user’s reply into one of 12 response types:

    * **R1** — Accept
    * **R2** — Accept with modification
    * **R3** — Reject
    * **R4** — Reject with alternative
    * **R5** — Justification / clarification (question)
    * **R6** — Chit-chat
    * **R7** — New edit request (still an edit, just a different one)
    * **R8** — Unrelated operation (not an edit at all — layering,
      generation, expand, ...)
    * **R9** — Exit suggestion mode
    * **R10** — Ignore / topic shift
    * **R11** — Ambiguous
    * **R12** — Conflicting

    The parsed classification and any extracted data (modifications,
    new suggestions) are stored in ``state['classification_result']``.

    :param state: The suggestion subgraph state with ``prompt``,
        ``s_stack``, and ``conversation_history``.
    :type state: SuggestionSubGraph
    :returns: A ``Command`` routing to ``updateStack`` with the
        classification result.
    :rtype: Command
    """

    cost = state.get('cost', 0.0)

    # ANY accepted unit ends the subgraph: the parent graph executes an
    # accepted suggestion immediately, one at a time, instead of waiting for
    # the whole stack to be decided.
    any_accepted = any(suggestion['status'] == 'accepted' for suggestion in state['s_stack'])
    if any_accepted or len(state['s_stack']) == 0:
        return Command(goto=END, update={**state, "s_stack": state['s_stack'], "cost": cost})

    # SRC only classifies the unit with focus: true, and updateStack only ever
    # edits that unit -- so a stack where nothing is focused can ONLY come back
    # ambiguous (R11/R12), and even a correct verdict would update nothing. A
    # frontend that sets focus keeps its choice; one that just sends text (or
    # forgets the flag) falls back to the first pending unit, which is the one
    # the user is looking at in a one-at-a-time flow.
    s_stack = _withFocus(state['s_stack'])

    prompt = SUGGESTION_SRC_PROMPT.format(
        stack=s_stack,
        history=state['conversation_history']
    )

    response, model_used = invoke_with_fallback([
        {"role": "user", "content": prompt}
    ], temperature=0, thinking_budget=0)
    cost += calculateCost(model_used, response.usage_metadata['total_tokens'])

    result = parseJson(response.content)

    return Command(goto='updateStack', update={"classification_result": result,
                                               "s_stack": s_stack, "cost": cost})

@profile
def updateStack(state: SuggestionSubGraph) -> Command:
    """Update the suggestion stack based on the classifier's result.

    Processes the ``classification_result`` and modifies ``s_stack``:

    * **R1** (Accept) — Marks the focused suggestion as ``accepted`` and
      exits the subgraph at once, so the parent graph executes that one
      suggestion immediately instead of batching it with the others.
    * **R2** (Accept + modify) — Updates the focused suggestion's
      ``nano_prompt`` and ``user_friendly_text`` in place (no duplicate unit)
      and keeps it ``pending`` so the user confirms the rewritten version on
      their next turn instead of it being accepted silently.
    * **R3** (Reject) — Removes the focused suggestion.
    * **R4** (Reject + alternative) — Same treatment as R2: updates
      ``nano_prompt``/``user_friendly_text`` in place and stays ``pending``.
    * **R5/R6** — Routes to ``sugQuestion`` / ``sugChitchat`` without
      modifying the stack.
    * **R7** (New edit) — Clears the stack and routes to ``END``.
    * **R8** (Unrelated operation — not an edit) — Clears the stack and
      routes to ``END``, same as R7, so the top-level Navigator picks the
      right flow (layering, generation, expand, ...) for it.
    * **R9** (Exit) — Clears the stack and routes to ``END``.
    * **R10** (Topic shift) — Routes to ``sugQuestion``.
    * **R11/R12** (Ambiguous / Conflicting) — Routes to ``sugUnclear``.

    :param state: The suggestion subgraph state with
        ``classification_result`` and ``s_stack``.
    :type state: SuggestionSubGraph
    :returns: A ``Command`` routing to the next node with the updated
        suggestion stack.
    :rtype: Command
    """

    cost = state.get('cost', 0.0)

    # Logic to update the suggestion stack
    classification_result = state['classification_result']
    if classification_result['response_type'] == 'R1':
        updated_stack = []
        for suggestion in state['s_stack']:
            if suggestion['focus']:
                updated_stack.append({
                    **suggestion,
                    "status": "accepted",
                    "edits" : 0
                })
            else:
                updated_stack.append(suggestion)

        # Straight out of the subgraph: suggestionHandler executes this one
        # accepted unit right away and then re-presents whatever is left.
        return Command(goto=END, update={"s_stack": updated_stack, "cost": cost})
            
    elif classification_result['response_type'] == 'R2':
        # Update the focused unit in place — never append a second copy — and
        # keep it pending so the rewritten nano_prompt/user_friendly_text get
        # a confirmation on the next turn instead of being locked in silently.
        updated_stack = []
        for suggestion in state['s_stack']:
            if suggestion['focus']:
                updated_stack.append({
                    **suggestion,
                    "status": "pending",
                    "nano_prompt": classification_result['modification'],
                    "user_friendly_text": classification_result.get('modification_summary') or suggestion['user_friendly_text'],
                    "edits": max(suggestion.get('edits', 0) - 1, 0)
                })
            else:
                updated_stack.append(suggestion)

        return Command(goto='askUser', update={"s_stack": updated_stack, "cost": cost})
    
    elif classification_result['response_type'] == 'R3':
        updated_stack = []
        for suggestion in state['s_stack']:
            if suggestion['focus']:
                pass # Remove the rejected suggestion
            else:
                updated_stack.append(suggestion)
        
        return Command(goto='askUser', update={"s_stack": updated_stack, "cost": cost})
    
    elif classification_result['response_type'] == 'R4':
        # Same treatment as R2: update the focused unit in place and keep it
        # pending — a rejection-with-alternative is still an unconfirmed
        # proposal, not something to auto-accept just because the revision
        # budget ran out.
        updated_stack = []
        for suggestion in state['s_stack']:
            if suggestion['focus']:
                updated_stack.append({
                    **suggestion,
                    "status": "pending",
                    "nano_prompt": classification_result['modification'],
                    "user_friendly_text": classification_result.get('modification_summary') or suggestion['user_friendly_text'],
                    "edits": max(suggestion.get('edits', 0) - 1, 0)
                })
            else:
                updated_stack.append(suggestion)

        return Command(goto='askUser', update={"s_stack": updated_stack, "cost": cost})
    
    elif classification_result['response_type'] == 'R5':
        updated_stack = state['s_stack']  # No change
        return Command(goto='sugQuestion', update={"prompt": classification_result['unrelated_query'], "s_stack": updated_stack, "cost": cost})
    
    elif classification_result['response_type'] == 'R6':
        updated_stack = state['s_stack']  # No change
        return Command(goto='sugChitchat', update={"prompt": classification_result['unrelated_query'], "s_stack": updated_stack, "cost": cost})
    
    elif classification_result['response_type'] == 'R7':
        updated_stack = state['s_stack']  # No change

        return Command(goto=END, update={'s_stack':[], "prompt": classification_result['new_suggestion'], "cost": cost})

    elif classification_result['response_type'] == 'R8':
        # Not an edit at all (layering, generation, expand, ...) — same exit
        # shape as R7, so suggestionHandler hands the request to the
        # top-level Navigator, which is the only place that knows how to
        # route a non-edit operation.
        return Command(goto=END, update={'s_stack': [], "prompt": classification_result['new_suggestion'], "cost": cost})

    elif classification_result['response_type'] == 'R9':
        return Command(goto=END, update={**state, "s_stack": [], "prompt": classification_result['unrelated_query'], "cost": cost})
    elif classification_result['response_type'] == 'R10':
        return Command(goto="sugQuestion", update={**state, "s_stack": state['s_stack'], "prompt": classification_result['unrelated_query'], "cost": cost, "current_image_url": state['current_image_url']})
    elif classification_result['response_type'] == 'R11':
        return Command(goto='sugUnclear', update={**state, "cost": cost})
    elif classification_result['response_type'] == 'R12':
        return Command(goto='sugUnclear', update={**state, "cost": cost})

@profile
def askUser(state: SuggestionSubGraph) -> Command:
    """Present the updated suggestion stack and collect the user’s next response.

    Sends an interrupt displaying the current ``s_stack`` along with
    the current image.  The user’s response is then fed back into the
    ``classifier`` node for re-classification.

    This creates the interactive suggestion loop:
    ``classifier`` → ``updateStack`` → ``askUser`` → ``classifier``.

    The loop runs only while nothing is accepted. The moment any suggestion
    is accepted, the subgraph ends and the parent graph executes it.

    :param state: The suggestion subgraph state with ``s_stack`` and
        ``current_image_url``.
    :type state: SuggestionSubGraph
    :returns: A ``Command`` routing to ``classifier`` with the user’s
        latest response.
    :rtype: Command
    """

    cost = state.get('cost', 0.0)
    # Any accepted unit leaves immediately so the parent graph can execute it.
    any_accepted = any(suggestion['status'] == 'accepted' for suggestion in state['s_stack'])
    if any_accepted or not state['s_stack']:
        return Command(goto=END, update={**state, "s_stack": state['s_stack'], "cost": cost})
    # Nothing accepted yet — ask user for next input

    user_input = interrupt({
        "type": "text_input",
        "question": SUGGESTION_NEXT_STEP,
        "suggestions": state['s_stack'],
        "cost": cost,
        "current_image_url": state.get('current_image_url', '')
    })
    user_response = parseJson(user_input)
    cost = 0.0

    # The user can re-select on the canvas between turns, and every exit out of
    # this subgraph hands current_image_url back to the parent graph -- so a
    # selection made here has to be picked up, or the next flow analyses the
    # image they moved away from. Falls back to the state value when the reply
    # omits the key.
    selected_url = user_response.get('current_image_url') or state.get('current_image_url', '')

    any_accepted = any(suggestion['status'] == 'accepted' for suggestion in user_response['suggestions'])
    if any_accepted:
        return Command(goto=END, update={**state, "s_stack": user_response['suggestions'], "cost": cost, "current_image_url": selected_url})

    return Command(goto="classifier", update={"prompt": user_response['response'], "s_stack": user_response['suggestions'], "cost": cost, "current_image_url": selected_url})


suggestionGraph = StateGraph(SuggestionSubGraph)

suggestionGraph.add_node("classifier", classifier)
suggestionGraph.add_node("sugQuestion", questionHandler)
suggestionGraph.add_node("sugChitchat", chitChatHandler)
suggestionGraph.add_node("sugUnclear", unclearHandler)
suggestionGraph.add_node("updateStack", updateStack)
suggestionGraph.add_node("askUser", askUser)

suggestionGraph.add_edge(START, "classifier")

sgb = suggestionGraph.compile()