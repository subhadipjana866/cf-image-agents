from openai import embeddings
from .states import GraphicDesigner
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command, interrupt
from .config import (
    OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, missingDetailsNotification, suggestionResponseNotification,
    suggestionUnclearNotification, commonUnclearNotification, INDEX_PATH,
    RAGBOT_PRICES, CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER,
    LAYERING_EXECUTED, HELPER_AGENT_ERROR, SLOP_RESUME_NOTE,
)
from .prompts import (
    CHIT_CHAT_SYSTEM_PROMPT, QUESTION_HANDLER_CLASSIFIER_PROMPT,
    SPECIFIC_SUGGESTION_PROMPT, SPECIFIC_SUGGESTION_RESOLVER_PROMPT,
)
from .utils import parseJson, calculateCost, deferredInterruptCost, profile, invoke_with_fallback, TEXT_MODEL_FALLBACK_CHAIN, detectCriticism, messageText, invokeMultimodalWithFallback, apiLog
from .model_router import chatModelForRole
from research_pipeline import ResearchPipeline
import random
import pystache
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# RAG Tool for the agent
@tool
def rag_search(query: str) -> str:
    """
    Search the design knowledge base of kognie for relevant information.
    Use this tool to retrieve design-related documentation, best practices, costs of everything,
    and answers to design questions from the FAISS knowledge base.
    
    Args:
        query: The search query or question about graphic design or kognie 
        
    Returns:
        Relevant information from the knowledge base
    """
    try:
        emb = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            emb,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        
        # Format the retrieved documents
        result = "\n\n".join([doc.page_content for doc in docs])
        return result if result else "No relevant information found in the knowledge base."
    except Exception as e:
        return f"Error retrieving information: {str(e)}"


@profile
def helperAgent(state: GraphicDesigner) -> Command:
    """
    A helper agent who handles all kind of questions from the user.
    
    This agent uses a RAG (Retrieval-Augmented Generation) tool to search
    a FAISS knowledge base for relevant design information before responding
    to the user's query. The agent can leverage this tool to provide
    informed answers about graphic design, best practices, and specifications.
    
    :param state: The current graph state with ``prompt``, ``conversation_history``,
        and other metadata.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing back to ``navigator`` with the agent's
        response appended to conversation history.
    :rtype: Command
    """
    print("[helperAgent] answering the user's question via RAG")
    cost = state.get('cost', 0.0)

    # Initialize the RAG tool
    tools = [rag_search]
    
    # A tool-calling agent needs a model OBJECT, not a response, so this cannot
    # go through invoke_with_fallback. It used to walk TEXT_MODEL_FALLBACK_CHAIN
    # by hand and break on the first model it could *construct* -- which always
    # succeeds, since constructing a chat client makes no network call. That
    # loop therefore pinned this agent to the head of the chain permanently and
    # never once fell back. The registry picks the model instead.
    model = chatModelForRole("classify_extract", temperature=0.7)
    
    # Create a prompt template for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions about graphic design. Use the provided tools to search the knowledge base when needed."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent with the RAG tool
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    # Invoke the agent
    try:
        result = agent_executor.invoke({
            "input": state['prompt'],
            "chat_history": state.get('conversation_history', [])
        })
        
        response_content = result.get('output', 'No response generated')

        # Calculate cost (if tokens are available)
        if hasattr(result, 'usage_metadata'):
            cost += calculateCost(model.__class__.__name__, result.usage_metadata.get('total_tokens', 0))

    except Exception as e:
        response_content = HELPER_AGENT_ERROR.format(error=str(e))

    # Non-image spend stays deferred below one cent; once the threshold is
    # reached, the next interrupt includes the accumulated amount.
    user_response = interrupt({
        "type": "text_input",
        "question": response_content,
        "cost": deferredInterruptCost(cost),
        "current_image_url": state.get('current_image_url', '')
    })
    
    user_response = parseJson(user_response)
    
    # Return command to navigate based on subgraph context
    if state['name'] == 'validation_subgraph':
        return Command(goto='analyzer', update={
            "prompt": user_response['response'],
            "conversation_history": state.get('conversation_history', []) + [
                {"role": "assistant", "content": response_content},
                {"role": "user", "content": user_response['response']}
            ],
            "cost": cost,
            "current_image_url": user_response.get('current_image_url', '')
        })
    elif state['name'] == 'suggestion_subgraph':
        return Command(goto='classifier', update={
            "prompt": user_response['response'],
            "s_stack": user_response.get('suggestions', state.get('s_stack', [])),
            "conversation_history": state.get('conversation_history', []) + [
                {"role": "assistant", "content": response_content},
                {"role": "user", "content": user_response['response']}
            ],
            "cost": cost,
            "current_image_url": user_response.get('current_image_url', '')
        })
    else:
        return Command(goto="navigator", update={
            "prompt": user_response['response'],
            "cost": cost,
            "current_image_url": user_response.get('current_image_url', ''),
            "attached_images": user_response.get('attached_images', []),
            "conversation_history": state.get('conversation_history', []) + [
                {"role": "assistant", "content": response_content},
                {"role": "user", "content": user_response['response']}
            ],
            "model": user_response.get("model", state.get("model", "")),
            "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))
        })


@profile
def specificSuggestionCreator(state: GraphicDesigner) -> Command:
    """Generate a short, narrowly-scoped answer to one specific suggestion request.

    Unlike the general fidelity/design suggestion pipeline (which produces a
    suggestion stack of cards and critiques the whole image), this answers
    exactly what was asked — e.g. "suggest a good color for the background",
    "what shape should I change the table into?" — as 2-4 concrete options,
    sent as a normal plain-text chat message (numbered list), not a
    structured suggestion stack.

    Shared across two contexts, distinguished by ``state['name']``:

    * ``'validation_subgraph'`` — invoked from the edit-clarification loop
      (:mod:`graphic_designer.subgraphs.validation`) when the user asks for
      suggestions instead of answering a clarification question.
    * anything else (main graph) — invoked from :func:`suggestionManager`
      when a fresh user turn is classified as a specific (not general)
      suggestion request.

    :param state: The current graph state with ``prompt`` and, in the
        validation context, ``clarification_question``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``specificSuggestionHandler`` with the
        suggestions appended to ``conversation_history`` as a plain-text
        assistant message.
    :rtype: Command
    """
    print("[specificSuggestionCreator] answering a specific suggestion request")
    cost = state.get('cost', 0.0)
    suggestion_request = state.get('suggestion_request') or state.get('prompt', '')
    current_image_url = state.get('current_image_url', '')

    # In the validation context, the pending clarification_question (e.g.
    # "What color would you like to change the bike to?") IS the actual
    # narrow question to answer — it's far more specific than the user's
    # trigger phrase ("suggest me good colors") and must take priority, or
    # the model has nothing to scope itself to and drifts into critiquing
    # the whole image.
    question = state.get('clarification_question') or suggestion_request

    suggestion_prompt = SPECIFIC_SUGGESTION_PROMPT.format(question=question)

    multimodal_messages = [
        {"role": "user", "content": [
            {"type": "text", "text": suggestion_prompt},
            {"type": "image_url", "image_url": {"url": current_image_url}}
        ]}
    ]

    response, model_used = invokeMultimodalWithFallback(
        multimodal_messages[0]["content"], temperature=0, role="classify_extract")
    cost += calculateCost(model_used, response.usage_metadata['total_tokens'])

    message_text = messageText(response).strip()
    conversation_history = state.get('conversation_history', []) + [
        {"role": "assistant", "content": message_text}
    ]

    return Command(goto="specificSuggestionHandler", update={
        "suggestion_request": suggestion_request,
        "conversation_history": conversation_history,
        "cost": cost,
    })


@profile
def specificSuggestionHandler(state: GraphicDesigner) -> Command:
    """Show the specific-suggestion answer and route the user's reply.

    Presents the plain-text suggestion message from :func:`specificSuggestionCreator`
    via a normal interrupt (no suggestion stack). The reply is classified by
    :data:`SPECIFIC_SUGGESTION_RESOLVER_PROMPT` into:

    * **SELECTED** — the user picked (or lightly modified) one of the
      options. In the validation context this feeds back into
      ``knownCategoryEntityExtractor`` as a normal clarification
      continuation (validation completes and proceeds to execution as
      usual); at the top level it's executed directly via the single-edit
      pipeline (``aspectRatioHandler`` → ``gptImageExecutor``).
    * **MORE** — loops back to :func:`specificSuggestionCreator` for a fresh
      batch.
    * **OTHER** — anything else (question, chit-chat, new request, cancel)
      is hand-off unmodified to the context's own top-of-turn classifier
      (``analyzer`` in validation, ``navigator`` at the top level), which
      already handles all of those robustly.

    :param state: The current graph state with ``conversation_history``
        (last entry is the suggestion message just shown).
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``knownCategoryEntityExtractor``,
        ``aspectRatioHandler``, ``specificSuggestionCreator``, ``analyzer``,
        or ``navigator``.
    :rtype: Command
    """
    print("[specificSuggestionHandler] presenting the specific suggestion and routing the reply")
    cost = state.get('cost', 0.0)
    message_text = state['conversation_history'][-1]['content']
    in_validation = state.get('name') == 'validation_subgraph'

    user_response = interrupt({
        "type": "text_input",
        "question": message_text,
        "cost": cost,
        "current_image_url": state.get('current_image_url', '')
    })
    user_response = parseJson(user_response)
    cost = 0.0
    reply = user_response['response']
    current_image_url = user_response.get('current_image_url', state.get('current_image_url', ''))

    resolver_prompt = SPECIFIC_SUGGESTION_RESOLVER_PROMPT.format(
        suggestions_message=message_text,
        user_reply=reply,
    )
    response, model_used = invoke_with_fallback([
        {"role": "user", "content": resolver_prompt}
    ], temperature=0)
    cost += calculateCost(model_used, response.usage_metadata['total_tokens'])
    result = parseJson(response.content)

    conversation_history = state.get('conversation_history', []) + [
        {"role": "user", "content": reply}
    ]

    decision = result.get('decision')

    if decision == 'SELECTED':
        resolved_instruction = result.get('resolved_instruction') or reply
        if in_validation:
            return Command(goto="knownCategoryEntityExtractor", update={
                "prompt": resolved_instruction,
                "conversation_history": conversation_history,
                "count": state.get('count', 0) + 1,
                "current_image_url": current_image_url,
                "cost": cost
            })
        # Route through the same classify -> extract/validate -> class-specific
        # templated prompt chain a directly-typed edit request goes through
        # (editIntentClassifier -> knownCategoryPromptHandler -> validation
        # subgraph's knownCategoryPromptCreator), instead of handing the raw
        # resolved_instruction straight to the image model. That template is
        # what actually carries the anti-drift guardrails (e.g. "Preserve the
        # object's geometry... Do NOT change the background or other object
        # attributes" for Class 3) -- going straight to aspectRatioHandler
        # skipped all of that, so a specific-suggestion edit had none of the
        # scoping a same-worded typed edit would have gotten.
        return Command(goto="editIntentClassifier", update={
            "prompt": resolved_instruction,
            "conversation_history": conversation_history,
            "current_image_url": current_image_url,
            "cost": cost
        })

    if decision == 'MORE':
        return Command(goto="specificSuggestionCreator", update={
            "prompt": reply,
            "conversation_history": conversation_history,
            "current_image_url": current_image_url,
            "cost": cost
        })

    # OTHER - hand off to the context's own top-of-turn classifier, which
    # already handles chit-chat/question/new-request/cancel robustly.
    if in_validation:
        return Command(goto="analyzer", update={
            "prompt": reply,
            "conversation_history": conversation_history,
            "current_image_url": current_image_url,
            "cost": cost
        })

    return Command(goto="navigator", update={
        "prompt": reply,
        "conversation_history": conversation_history,
        "current_image_url": current_image_url,
        "attached_images": user_response.get('attached_images', state.get('attached_images', [])),
        "model": user_response.get("model", state.get("model", "")),
        "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', '')),
        "cost": cost
    })


@profile
def chitChatHandler(state: GraphicDesigner) -> GraphicDesigner:
    """Handle casual chit-chat and guide the user back to the main task.

    Uses GPT-5-mini with a friendly conversational persona to reply to
    non-task-related messages.  The assistant's response is woven into a
    subtle redirect that reminds the user they can generate or edit images.
    An interrupt is issued with the response, and the user's follow-up is
    routed back to ``navigator``.

    This node is reused across the main graph and both subgraphs
    (suggestion and validation).

    :param state: The current graph state with ``prompt`` and
        ``conversation_history``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` (main graph) or
        ``classifier``/``analyzer`` (subgraphs) with the updated
        conversation history.
    :rtype: Command
    """
    print("[chitChatHandler] handling casual chit-chat")
    # Non-image spend stays deferred below one cent; once the threshold is
    # reached, the next interrupt includes the accumulated amount.
    cost = state.get('cost', 0.0)

    response, model_used = invoke_with_fallback([
        {
            "role": "system",
            "content": CHIT_CHAT_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": state['prompt']
        }
    ], temperature=0.7)
    cost += calculateCost(model_used, response.usage_metadata['total_tokens'])

    if state['name'] == 'validation_subgraph':
        response.content += f"\n\n{missingDetailsNotification}"
        user_response = interrupt({
        "type": "text_input",
        "question": response.content.strip(),
        "cost": deferredInterruptCost(cost),
        "current_image_url": state['current_image_url']
    })

        user_response = parseJson(user_response)

        return Command(goto='analyzer',update={"prompt": user_response['response'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": response.content.strip()
        }] + [{
            "role": "user",
            "content": user_response['response']
        }], "cost": cost, "current_image_url": user_response['current_image_url']})

    elif state['name'] == 'layering_subgraph':
        response.content += f"\n\nDo you wish to proceed with layering?"
        user_response = interrupt({
        "type": "text_input",
        "question": response.content.strip(),
        "cost": deferredInterruptCost(cost),
        "current_image_url": state['current_image_url']
    })

        user_response = parseJson(user_response)

        return Command(goto='layeringHandler',update={"prompt": user_response['response'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": response.content.strip()
        }] + [{
            "role": "user",
            "content": user_response['response']
        }], "cost": cost, "current_image_url": user_response['current_image_url']})
    
    elif state['name'] == 'slop_subgraph':
        return _slopResume(state, response.content.strip(), cost)

    elif state['name'] == 'suggestion_subgraph':
        response.content += f"\n\n{pystache.render(suggestionResponseNotification, {'username': state.get('username', 'User')})}"
        user_input = interrupt({
        "type": "text_input",
        "question": response.content.strip(),
        "suggestions": state['s_stack'],
        "cost": deferredInterruptCost(cost),
        "current_image_url": state['current_image_url']
        })

        user_response = parseJson(user_input)

        return Command(goto='classifier',update={"prompt": user_response['response'],"s_stack":user_response['suggestions'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": response.content.strip()
        }] + [{
            "role": "user",
            "content": user_response['response']
        }], "cost": cost})

    else:
        user_response = interrupt({
        "type": "text_input",
        "question": response.content.strip(),
        "cost": deferredInterruptCost(cost),
        "current_image_url": state['current_image_url'],
    })

    
        user_response = parseJson(user_response)

    

    
        conversation_history = state.get('conversation_history', []) + [{
            "role": "system",
            "content": response.content.strip()
        }] + [{
            "role": "user",
            "content": user_response['response']
        }]
        return Command(goto="navigator", update={"prompt": user_response['response'], "cost": cost, "current_image_url": user_response['current_image_url'], "attached_images": user_response['attached_images'], "edit_reference_images": _replyAttachments(user_response), "conversation_history": conversation_history, "model": user_response.get("model", state.get("model", "")), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))})

def askRagbot(question: str, session_context: str = "",
              pricing: bool = False) -> tuple:
    """Answer a question from one of the ragbot corpora.

    ragbot is called in-process rather than over its own HTTP routes, and
    imported lazily and defensively for the same reason ``main.py`` guards its
    router import: the retrieval extras (faiss, sentence-transformers) are
    optional, so a missing one has to degrade to "no ragbot" instead of taking
    the question flow down with it.

    :param question: The user's question about Kognie or its pricing.
    :type question: str
    :param session_context: Live facts about the run in progress, handed to
        ragbot as a synthetic source so a question like "what will this
        generation cost me?" can pair the selected model with its documented
        price.
    :type session_context: str
    :param pricing: Read the pricing collection — the price list and nothing
        else — instead of the product corpus, and answer in a sentence or two.
        A price is a number the user wants now, not a topic to explain, so the
        terse style rides along with the route. The collection names are
        resolved here rather than by the caller so ragbot stays a lazy, guarded
        import.
    :type pricing: bool
    :returns: ``(answer, cost)``. ``answer`` is an empty string when ragbot
        cannot serve the question, which tells the caller to fall back to a
        plain generative reply.
    :rtype: tuple[str, float]
    """
    try:
        from ragbot import config as ragbot_config
        from ragbot.engine import RagBot
    except Exception as e:
        print(f"[questionHandler] ragbot unavailable: {e}")
        return "", 0.0

    name = (ragbot_config.PRICING_COLLECTION if pricing
            else ragbot_config.DEFAULT_COLLECTION)
    try:
        bot = RagBot(name)
        if not bot.store.exists():
            print(f"[questionHandler] ragbot collection '{name}' is empty — "
                  "POST /ragbot/sync (or /ragbot/pricing/sync) to build it")
            return "", 0.0
        answer = bot.ask(question, session_context=session_context or None,
                         brief=pricing)
    except Exception as e:
        print(f"[questionHandler] ragbot query failed: {e}")
        return "", 0.0

    cost = ragbotCost(ragbot_config.ANSWER_MODEL, answer.usage or {})
    text = (answer.answer or "").strip()

    # ragbotCost already applies the real per-token rate (input/output split
    # plus the cache-read/write discount, RAGBOT_PRICES) -- more precise than
    # apiLog's generic providerCost() derivation, so hand it in directly as
    # actual_cost rather than let it be recomputed from a flatter split.
    usage = answer.usage or {}
    provider = "anthropic" if "claude" in ragbot_config.ANSWER_MODEL else "google"
    apiLog(provider, ragbot_config.ANSWER_MODEL, "askRagbot", cost=cost, actual_cost=cost,
           tokens_in=usage.get('input_tokens'), tokens_out=usage.get('output_tokens'))
    return text, cost


def ragbotCost(model: str, usage: dict) -> float:
    """Price one ragbot answer from its Anthropic-shaped usage counters.

    ``usage`` splits input tokens four ways — uncached input, output, cache
    writes and cache reads — and the last two are billed at a discount, so each
    bucket is priced separately against :data:`RAGBOT_PRICES`.

    :param model: The ragbot answer model that produced the response.
    :type model: str
    :param usage: ``{input_tokens, output_tokens, cache_creation_input_tokens,
        cache_read_input_tokens}`` as returned on ``Answer.usage``.
    :type usage: dict
    :returns: Cost in USD, rounded to 6 decimal places. ``0.0`` when the model
        has no entry in the price table.
    :rtype: float
    """
    prices = RAGBOT_PRICES.get(model)
    if not prices:
        # A custom RAGBOT_ANSWER_MODEL we have no rate for is not a reason to
        # lose the answer — bill nothing and say so in the logs.
        print(f"[questionHandler] no price entry for ragbot model {model}; billing 0")
        return 0.0

    input_rate, output_rate = prices
    cost = (
        usage.get('input_tokens', 0) * input_rate
        + usage.get('output_tokens', 0) * output_rate
        + usage.get('cache_creation_input_tokens', 0) * input_rate * CACHE_WRITE_MULTIPLIER
        + usage.get('cache_read_input_tokens', 0) * input_rate * CACHE_READ_MULTIPLIER
    ) / 1_000_000
    return round(cost, 6)


@profile
def questionHandler(state: GraphicDesigner) -> Command:
    """Answer a user question, routed by what the question is about.

    A single classifier call splits the question three ways, answering one of
    them in the same call:

    * **pricing** — anything about money is answered by ragbot over the pricing
      collection, which holds the price list and nothing else.
    * **kognie** — questions about the platform itself are answered by ragbot
      over the product collection, which cannot see the price list.
    * **generative** — everything else is answered inline from the model's own
      knowledge.

    The two corpora are deliberately separate stores rather than one index with
    a pricing document in it. Retrieval over a single index is free to mix them,
    which means a product answer can quote internal rates and a pricing answer
    can drift into marketing copy; splitting them makes each answer's reachable
    material an explicit routing decision. Prices are then revised by replacing
    a file in ``ragbot/documents_pricing/`` and syncing, never by editing code.

    The run context (current process, selected model, spend so far) goes to
    ragbot as a synthetic per-query source, which is what lets one answer draw
    on both halves: "what will this generation cost me?" needs the selected
    model from the session and that model's price from the price list.

    The answer is presented via an interrupt, and the user's follow-up is routed
    back to ``navigator`` (main graph) or ``classifier`` / ``analyzer``
    (subgraphs).

    :param state: The current graph state with ``prompt`` and
        ``conversation_history``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to the appropriate next node with
        the answer appended to conversation history.
    :rtype: Command
    """
    print("[questionHandler] answering the user's question")
    current_model = state.get('model', '') or 'not selected yet'
    accrued_cost = state.get('cost', 0.0)
    process = state.get('name', '') or 'main graph'

    prompt = QUESTION_HANDLER_CLASSIFIER_PROMPT.format(
        current_model=current_model,
        accrued_cost=accrued_cost,
        user_query=state['prompt'],
        chat_history=state.get('conversation_history', []),
        query_about=process
    )

    classify_response, model_used = invoke_with_fallback([{
        "role": "user",
        "content": prompt
    }], temperature=0)
    # Non-image spend stays deferred below one cent; once the threshold is
    # reached, the next interrupt includes the accumulated amount.
    cost = accrued_cost + calculateCost(model_used, classify_response.usage_metadata['total_tokens'])
    classification = parseJson(classify_response.content)
    question_type = classification.get('type', 'generative')
    print(f"[questionHandler] question type: {question_type}")

    if question_type in ('pricing', 'kognie'):
        session_context = (
            f"- Process the user is in right now: {process}\n"
            f"- Image model currently selected: {current_model}\n"
            f"- Spent so far in this process (USD): {round(accrued_cost, 4)}"
        )
        response, ragbot_cost = askRagbot(state['prompt'], session_context,
                                          pricing=question_type == 'pricing')
        cost += ragbot_cost
    else:
        # generative answers come back inline from the classifier.
        response = (classification.get('response') or '').strip()

    if not response:
        # Either ragbot could not serve the question or the classifier returned a
        # bare type — answer generatively rather than showing the user nothing.
        print("[questionHandler] falling back to a generative answer")
        gen_response, gen_model = invoke_with_fallback([{
            "role": "user",
            "content": state['prompt']
        }], temperature=0)
        cost += calculateCost(gen_model, gen_response.usage_metadata['total_tokens'])
        response = gen_response.content


    if state['name'] == 'validation_subgraph':
        response += "\n\nPlease provide the missing details to continue editing your image."
        user_response = interrupt({
            "type": "text_input",
            "question": response,
            "cost": deferredInterruptCost(cost),
            "current_image_url": state['current_image_url']
        })
        user_response = parseJson(user_response)
        return Command(goto='analyzer',update={"prompt": user_response['response'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": response
        }] + [{
            "role": "user",
            "content": user_response['response']
        }], "cost": cost, "current_image_url": user_response['current_image_url']})

    elif state['name'] == 'layering_subgraph':
        response += "\n\nDo you wish to proceed with layering?"
        user_response = interrupt({
        "type": "text_input",
        "question": response,
        "cost": deferredInterruptCost(cost),
        "current_image_url": state['current_image_url']
    })
        user_response = parseJson(user_response)
        return Command(goto='layeringHandler',update={"prompt": user_response['response'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": response
        }] + [{
            "role": "user",
            "content": user_response['response']
        }], "cost": cost, "current_image_url": user_response['current_image_url']})
    elif state['name'] == 'slop_subgraph':
        return _slopResume(state, response, cost)
    elif state['name'] == 'suggestion_subgraph':
        response += "\n\nPlease provide your feedback on the suggestions to continue."
        user_input = interrupt({
        "type": "text_input",
        "question": response,
        "suggestions": state['s_stack'],
        "cost": deferredInterruptCost(cost),
        "current_image_url": state['current_image_url']
        })

        user_response = parseJson(user_input)

        return Command(goto='classifier',update={"prompt": user_response['response'],"s_stack":user_response['suggestions'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": response
        }] + [{
            "role": "user",
            "content": user_response['response']
        }], "cost": cost})
    else:
        user_response = interrupt({
            "type": "text_input",
            "question": response,
            "cost": deferredInterruptCost(cost),
            "current_image_url": state['current_image_url']
        })


        user_response = parseJson(user_response)

        conversation_history = state.get('conversation_history', []) + [{
            "role": "system",
            "content": response
        }] + [{
            "role": "user",
            "content": user_response['response']
        }]
        return Command(goto="navigator", update={"prompt": user_response['response'], "cost": cost, "current_image_url": user_response['current_image_url'], "attached_images": user_response['attached_images'], "edit_reference_images": _replyAttachments(user_response), "conversation_history": conversation_history, "model": user_response.get("model", state.get("model", "")), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))})


@profile
def unclearHandler(state: GraphicDesigner) -> GraphicDesigner:
    """Handle unclear or ambiguous user input by requesting clarification.

    Sends an interrupt with a polite notification message asking the user
    to rephrase or clarify their request.  The selected notification
    message is chosen randomly from a configured set
    (``commonUnclearNotification``).  The user's clarified response is
    then routed back to ``navigator``.

    This node is used in the main graph and the suggestion subgraph.

    :param state: The current graph state with ``prompt``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` (main graph) or
        ``classifier`` (subgraphs) with the user's clarified input.
    :rtype: Command
    """
    print("[unclearHandler] requesting clarification for unclear input")
    cost = state.get('cost', 0.0)
    if state['name'] == 'suggestion_subgraph':
        user_response = interrupt({
            "type": "text_input",
            "question": suggestionUnclearNotification,
            "suggestions": state['s_stack'],
            "cost": deferredInterruptCost(cost),
            "current_image_url": state.get('current_image_url', '')
        })

        user_response_parsed = parseJson(user_response)

        return Command(goto='classifier',update={"prompt": user_response_parsed['response'],"s_stack":user_response_parsed['suggestions'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": suggestionUnclearNotification
        }] + [{
            "role": "user",
            "content": user_response_parsed['response']
        }], "cost": cost, "current_image_url": user_response_parsed.get('current_image_url', '')})

    elif state['name'] == 'layering_subgraph':
        response = commonUnclearNotification + "\n\nDo you wish to proceed with layering?"
        user_response = interrupt({
            "type": "text_input",
            "question": response,
            "cost": deferredInterruptCost(cost),
            "current_image_url": state['current_image_url']
        })
        user_response = parseJson(user_response)
        return Command(goto='layeringHandler', update={"prompt": user_response['response'], "conversation_history": state.get('conversation_history', []) + [{
            "role": "system",
            "content": response
        }] + [{
            "role": "user",
            "content": user_response['response']
        }], "cost": cost, "current_image_url": user_response['current_image_url']})

    elif state['name'] == 'slop_subgraph':
        return _slopResume(state, commonUnclearNotification, cost)
        
    else:
        user_response = interrupt({
            "type": "text_input",
            "question": commonUnclearNotification,
            "cost": deferredInterruptCost(cost),
            "current_image_url": state.get('current_image_url', '')
        })


    user_response = parseJson(user_response)

    return Command(goto="navigator", update={"prompt": user_response['response'], "cost": cost, "current_image_url": user_response.get('current_image_url', ''), "attached_images": user_response['attached_images'], "edit_reference_images": _replyAttachments(user_response), "name": "main_graph", "model": user_response.get("model", state.get("model", "")), "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')), "style": user_response.get('style', state.get('style', ''))})



def _slopResume(state: GraphicDesigner, response: str, cost: float) -> Command:
    """Answer inside the slop subgraph and hand the reply back to it.

    Shared by the question, chit-chat and unclear handlers when they run as
    nodes of :mod:`graphic_designer.subgraphs.slop`. The answer is followed
    by the question the user was answering, and the reply goes back to that
    subgraph's ``slopClassifier`` -- with the same image, attachment and
    picker keys the main-graph branch hands to ``navigator``, since the reply
    may well end up there.
    """
    question = f"{response}\n\n{SLOP_RESUME_NOTE}"
    user_response = parseJson(interrupt({
        "type": "text_input",
        "question": question,
        "cost": deferredInterruptCost(cost),
        "current_image_url": state.get('current_image_url', '')
    }))
    return Command(goto='slopClassifier', update={
        "prompt": user_response.get('response', ''),
        "conversation_history": state.get('conversation_history', []) + [
            {"role": "system", "content": question},
            {"role": "user", "content": user_response.get('response', '')}
        ],
        "cost": cost,
        "current_image_url": user_response.get('current_image_url') or state.get('current_image_url', ''),
        "attached_images": user_response.get('attached_images', []),
        "edit_reference_images": _replyAttachments(user_response),
        "model": user_response.get("model", state.get("model", "")),
        "aspect_ratio": user_response.get('aspect_ratio', state.get('aspect_ratio', '')),
        "style": user_response.get('style', state.get('style', '')),
    })


@profile
def _replyAttachments(user_response: dict) -> list:
    """Images that arrived with THIS reply.

    Mirrors ``nodes._reply_attachments``; defined locally because nodes.py
    imports this module, not the other way round. Every handler that ends a
    turn writes it -- empty included -- so ``edit_reference_images`` only ever
    holds one turn's attachments and a reference can never leak into a later,
    unrelated edit.
    """
    if not isinstance(user_response, dict):
        return []
    return [src for src in (user_response.get('attached_images') or [])
            if isinstance(src, str) and src.strip()]


def nextStep(state: GraphicDesigner) -> Command:
    """Present post-suggestion results and collect the user's next action.

    Sends an interrupt displaying the current image (after suggestion
    execution) along with a prompt asking "What do you want to do next?".
    It collects the user's response and routes back to ``navigator`` with
    updated state including the user's new prompt and conversation history.

    This node is typically used after a suggestion subgraph has completed
    its execution and the system is ready for the next user command.

    :param state: The current graph state with ``current_image_url``,
        and ``cost``.
    :type state: GraphicDesigner
    :returns: A ``Command`` routing to ``navigator`` with the user's
        follow-up input and updated state.
    :rtype: Command
    """
    print("[nextStep] presenting the layering result and asking what's next")
    cost = state.get('cost', 0.0)
    # Logic to proceed to the next step
    user_response = interrupt({
        "type": "layering_executed",
        "question": LAYERING_EXECUTED,
        "prompt": state.get('prompt', ''),
        "cost": cost,
        "current_image_url": state.get('current_image_url', ''),
        "segmentation_result": state.get('segmentation_result', None),
        "bboxes": state.get('bboxes', []),
        "extracted_texts": state.get('extracted_texts', [])
    })

    user_response = parseJson(user_response)

    # Per-process navigator (layering flow).
    is_criticism, detect_cost = detectCriticism(state, user_response['response'], 'layering')
    if is_criticism:
        return Command(goto="criticismHandler", update={**state, "criticism_origin": "layering",
        "criticism_text": user_response['response'], "cost": detect_cost,
        "current_image_url": user_response.get('current_image_url', state.get('current_image_url', '')),
        "conversation_history": state.get('conversation_history', []) + [{"role": "user", "content": user_response['response']}]})

    return Command(goto="navigator", update={**state, 'prompt': user_response['response'], "cost": 0.0, "current_image_url": user_response.get('current_image_url', ''), "attached_images": user_response['attached_images'], "edit_reference_images": _replyAttachments(user_response), "conversation_history": state.get('conversation_history', []) + [{
        "role": "user",
        "content": user_response['response'],
    }], "suggestion_type": "user_driven", "model": user_response.get("model", state.get("model", "")), "aspect_ratio": user_response.get('aspect_ratio', ''), "style": user_response.get('style', state.get('style', ''))})