from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain_google_genai import ChatGoogleGenerativeAI
import pystache
import logging

from ..config import GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, editPrompts, GEN_PRICES
from ..prompts import VALIDATION_EXTRACTION_PROMPT, VALIDATION_PROMPT, VALIDATION_ANALYSIS_PROMPT
from ..utils import parseJson, calculateCost, profile, invoke_with_fallback, TEXT_MODEL_FALLBACK_CHAIN, messageText, invokeMultimodalWithFallback
from ..states import PromptValidatorSubGraph
from ..common_nodes import chitChatHandler, questionHandler, specificSuggestionCreator, specificSuggestionHandler
from langchain_openai import ChatOpenAI


@profile
def knownCategoryEntityExtractor(state: PromptValidatorSubGraph) -> PromptValidatorSubGraph:
    """Extract structured entities from the user prompt for a known edit class.

    Uses Gemini 2.5 Flash (with GPT-5-mini fallback) to parse the user’s
    prompt and conversation history into the entity schema defined for the
    classified edit class (e.g. object name, colour, position).  The
    extractor strictly adheres to the per-class field definitions from
    :data:`config.editPrompts`.

    :param state: The validation subgraph state with ``prompt``,
        ``intent``, ``conversation_history``, and ``current_image_url``.
    :type state: PromptValidatorSubGraph
    :returns: A ``Command`` routing to ``validator`` with
        ``extracted_fields`` populated.
    :rtype: Command
    :raises Exception: If the LLM call or JSON parsing fails.
    """
    cost = state.get('cost', 0.0)

    prompt = state['prompt']
    conversation_history = state.get('conversation_history', [])
    current_image_url = state['current_image_url']
    class_id = state['intent']

    extraction_prompt = VALIDATION_EXTRACTION_PROMPT.format(
        class_id=class_id,
        prompt=prompt,
        conversation_history=conversation_history,
    )
    
    try:
        response, model_used = invoke_with_fallback([
            {"role": "user", "content": extraction_prompt}
        ], temperature=0)
        cost += calculateCost(model_used, response.usage_metadata['total_tokens'])
    except Exception as e:
        raise e
    result = parseJson(response.content)
    # result = json.loads(response.content)
    if result['class_id'] != "0":
        return Command(goto="validator", update={"extracted_fields": result['extracted_fields'], "count": state.get('count', 0), "cost": cost})
    elif result['class_id'] == "0":
        return Command(goto=END, update={"original_prompt": state["prompt"], "cost": cost})


@profile
def knownCategoryPromptValidator(state: PromptValidatorSubGraph) -> PromptValidatorSubGraph:
    """Validate extracted entities against the image and check feasibility.

    Uses Gemini 2.5 Flash (with GPT-5-mini fallback) with image input to:

    1. Verify **semantic completeness** — required fields are present.
    2. Verify **visual grounding** — referenced objects exist in the image.
    3. Verify **feasibility** — the edit can be executed by available models.

    Produces one of three statuses:

    * ``'missing'`` — Fields are incomplete; generates a single
      clarification question.
    * ``'completed'`` — All fields validated; ready for prompt creation.
    * ``'invalid'`` — The edit request is not feasible.

    :param state: The validation subgraph state with ``extracted_fields``,
        ``intent``, ``current_image_url``, and ``conversation_history``.
    :type state: PromptValidatorSubGraph
    :returns: A ``Command`` routing to ``incompleteHandler``,
        ``knownCategoryPromptCreator``, or ``END`` based on status.
    :rtype: Command
    :raises Exception: If the LLM call or JSON parsing fails.
    """
    cost = state.get('cost', 0.0)
    prompt = state['prompt']
    conversation_history = state.get('conversation_history', [])
    current_image_url = state['current_image_url']
    class_id = state['intent']
    extracted_fields = state['extracted_fields']

    validation_prompt = VALIDATION_PROMPT.format(
        class_id=class_id,
        conversation_history=conversation_history,
        extracted_fields=extracted_fields,
    )
    import logging
    from langchain_openai import ChatOpenAI
    multimodal_messages = [
        {"role": "system", "content": validation_prompt},
        {"role": "user", "content": [
            {"type":"text", "text": f"""{{
                "class_id": {class_id},
                "extracted_fields": {extracted_fields},
                "conversation_history": {conversation_history}
            }}"""},
            {"type": "image_url", "image_url": {
                "url": current_image_url,
            }}
        ]}
    ]
    response, model_used = invokeMultimodalWithFallback(
        multimodal_messages[0]["content"], temperature=0, role="classify_extract")
    cost += calculateCost(model_used, response.usage_metadata['total_tokens'])
    result = parseJson(messageText(response))
    if result['status'] == 'completed':
        return Command(goto="knownCategoryPromptCreator", update={"extracted_fields": result['resolved_fields'], "status": 'completed', "cost": cost})
    elif result['status'] == 'missing':
        conversation_history.append({"role": "assistant", "content": result['clarification_question']})
        return Command(goto="incompleteHandler", update={"conversation_history": conversation_history, "prompt": prompt, "current_image_url": current_image_url, "intent": state['intent'], "extracted_fields": result['resolved_fields'], "status": 'missing', "clarification_question": result['clarification_question'], "cost": cost})
        

@profile
def incompleteHandler(state: PromptValidatorSubGraph) -> PromptValidatorSubGraph:
    """Present a clarification question and collect the user’s response.

    Sends an interrupt displaying the question generated by
    :func:`knownCategoryPromptValidator`.  The user’s reply is appended
    to the conversation history and the flow continues to the
    ``analyzer`` node.

    :param state: The validation subgraph state with the latest
        clarification question in ``conversation_history[-1]``.
    :type state: PromptValidatorSubGraph
    :returns: A ``Command`` routing to ``analyzer`` with the updated
        conversation history and user’s response.
    :rtype: Command
    """
    cost = state.get('cost', 0.0)
    user_response = interrupt({
        "type": "text_input",
        "question": state['conversation_history'][-1]['content'],
        "cost": cost,
        "current_image_url": state['current_image_url']
    })


    user_response = parseJson(user_response)
    conversation_history = state.get('conversation_history', []) + [{"role": "user", "content": user_response['response']}]
    current_image_url = user_response['current_image_url']
    prompt = user_response['response']
    
    return Command(goto="analyzer", update={"prompt": prompt, "conversation_history": conversation_history, "current_image_url": current_image_url, "intent": state['intent'], "cost": 0.0})


@profile
def analyzer(state: PromptValidatorSubGraph) -> PromptValidatorSubGraph:
    """Classify the user’s response to a clarification question.

    Uses GPT-5-mini to determine the intent of the user’s reply:

    * **CONTINUATION** — User provided the requested information;
      routes back to ``knownCategoryEntityExtractor`` for re-extraction.
    * **GO_AHEAD** — User wants to proceed as-is; routes to
      ``knownCategoryPromptCreator``.
    * **CHIT_CHAT** — Off-topic; routes to ``valChitchat``.
    * **QUESTION** — User asked a question; routes to ``valQuestion``.
    * **NEW_EDIT** — User introduced a new edit intent; routes to ``END``
      with ``status='new_edit'``.
    * **LAYERING** — User asked to break the image into layers; routes to
      ``END`` with ``status='layering'``.
    * **CANCEL** — User cancelled the edit; routes to ``END`` with
      ``status='cancel'``.

    Enforces a maximum of 3 clarification attempts before automatically
    routing to ``knownCategoryPromptCreator``.

    :param state: The validation subgraph state with ``prompt``,
        ``extracted_fields``, ``count``, and ``conversation_history``.
    :type state: PromptValidatorSubGraph
    :returns: A ``Command`` routing to the appropriate next node.
    :rtype: Command
    :raises Exception: If LLM invocation or JSON parsing fails.
    """
    prompt = state['prompt']
    conversation_history = state.get('conversation_history', [])
    extracted_fields = state['extracted_fields']
    cost = state.get('cost', 0.0)
    analysis_prompt = VALIDATION_ANALYSIS_PROMPT.format(
        prompt=prompt,
        conversation_history=conversation_history,
        extracted_fields=extracted_fields,
        last_user_message=conversation_history[-1]['content'] if conversation_history else '',
    )
    
    response, model_used = invoke_with_fallback([
        {"role": "user", "content": analysis_prompt}
    ], temperature=0)

    cost += calculateCost(model_used, response.usage_metadata['total_tokens'])
    classification = parseJson(response.content.strip()) # I will use conversation_history as well as user prompt to determine the class
    
    # Increment count for clarification attempts
    current_count = state.get('count', 0)
    
    if current_count < 2:
      if classification['response_type'] == 'CHIT_CHAT':
          return Command(goto='sub_chitchat', update={"cost": cost})
      elif classification['response_type'] == 'CONTINUATION':
          return Command(goto='knownCategoryEntityExtractor', update={"cost": cost, "count": current_count + 1})
      elif classification['response_type'] == 'QUESTION':
          return Command(goto="sub_question", update={"prompt":prompt, "cost": cost})
      elif classification['response_type'] == 'NEW_EDIT':
          return Command(goto=END, update={"extracted_fields": {"status": "new_edit", "prompt": prompt, "cost": cost}})
      elif classification['response_type'] == 'LAYERING':
          return Command(goto=END, update={"extracted_fields": {"status": "layering", "prompt": prompt, "cost": cost}})
      elif classification['response_type'] == 'CANCEL':
          return Command(goto=END, update={"extracted_fields": {"status": "cancel", "cost": cost}})
      elif classification['response_type'] == 'SUGGESTION':
          # Hand off to the shared specific-suggestion flow (common_nodes.py):
          # plain-text options scoped to state['clarification_question'] (set
          # at the source by knownCategoryPromptValidator), not a suggestion
          # stack. It forwards control + chat history back into this subgraph
          # (knownCategoryEntityExtractor) once the user picks one.
          return Command(goto='specificSuggestionCreator', update={
              "prompt": prompt,
              "cost": cost
          })
      elif classification['response_type'] == 'GO_AHEAD':
          return Command(goto='knownCategoryPromptCreator', update={"cost": cost})
    else:
        return Command(goto=END, update={"extracted_fields": {"status": "max_attempts_reached", "prompt": state['original_prompt']}})
    
  
@profile
def knownCategoryPromptCreator(state: PromptValidatorSubGraph) -> PromptValidatorSubGraph:
    """Render a structured editing prompt from the extracted entity fields.

    Uses :func:`pystache.render` to interpolate the ``extracted_fields``
    into the Mustache template defined in :data:`config.editPrompts` for
    the current ``intent`` class.  The resulting prompt is stored in
    ``state['prompt']``.

    If the status is ``'missing'`` (max attempts reached), the original
    prompt is returned unchanged.

    If the compound-request extraction step (``STEP 2B`` in
    ``VALIDATION_PROMPT``) resolved a ``secondary_instruction`` (an
    additional edit on a different target, e.g. "change the shape of the
    helmet" alongside a primary "change the bike's color" edit), it's
    appended to the rendered prompt as a second instruction.

    :param state: The validation subgraph state with ``intent``,
        ``extracted_fields``, and ``status``.
    :type state: PromptValidatorSubGraph
    :returns: The updated state with the rendered prompt.
    :rtype: PromptValidatorSubGraph
    """
    if state['status'] == 'missing':
        return state
    template = editPrompts[state['intent']]
    prompt = pystache.render(template, state["extracted_fields"])

    secondary_instruction = state["extracted_fields"].get("secondary_instruction")
    if secondary_instruction:
        prompt = f"{prompt}\n\nAdditionally: {secondary_instruction}"

    state['prompt'] = prompt

    return state


validatorSubGraph = StateGraph(PromptValidatorSubGraph)
validatorSubGraph.add_node("knownCategoryEntityExtractor", knownCategoryEntityExtractor)
validatorSubGraph.add_node("validator", knownCategoryPromptValidator)
validatorSubGraph.add_node("incompleteHandler", incompleteHandler)
validatorSubGraph.add_node("analyzer", analyzer)
validatorSubGraph.add_node("sub_chitchat", chitChatHandler)
validatorSubGraph.add_node("sub_question", questionHandler)
validatorSubGraph.add_node("knownCategoryPromptCreator", knownCategoryPromptCreator)
validatorSubGraph.add_node("specificSuggestionCreator", specificSuggestionCreator)
validatorSubGraph.add_node("specificSuggestionHandler", specificSuggestionHandler)

validatorSubGraph.add_edge(START, "knownCategoryEntityExtractor")

validatorSubGraph.add_edge("knownCategoryPromptCreator", END)

subGraph = validatorSubGraph.compile()


