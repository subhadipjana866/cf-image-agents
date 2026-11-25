"""Model access for the orchestrator steps.

Every step makes one call: a rendered prompt in, one JSON object out. The
`step` name identifies the caller ("interpret", "select", "readiness",
"editor_plan"), so a different model can be used per step and tests can
script replies per step.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from typing import Any, Callable, Protocol


class LLM(Protocol):
    def complete_json(self, step: str, prompt: str) -> dict: ...


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply, tolerating code fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("model reply contains no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("model reply is not a JSON object")
    return obj


class ChatModelLLM:
    """Wraps LangChain chat models. Requires the `llm` extra.

    `models` maps a step name to a model string understood by
    `langchain.chat_models.init_chat_model`, with "default" as the fallback,
    for example {"default": "openai:gpt-5.4-mini", "select": "openai:gpt-5.4"}.
    """

    def __init__(self, models: dict[str, str], *, max_retries: int = 1, **model_kwargs: Any):
        from langchain.chat_models import init_chat_model  # imported lazily

        self._init = init_chat_model
        self._names = models
        self._kwargs = model_kwargs
        self._cache: dict[str, Any] = {}
        self._max_retries = max_retries

    def _model(self, step: str):
        name = self._names.get(step) or self._names["default"]
        if name not in self._cache:
            self._cache[name] = self._init(name, **self._kwargs)
        return self._cache[name]

    def complete_json(self, step: str, prompt: str) -> dict:
        last: Exception | None = None
        for _ in range(self._max_retries + 1):
            reply = self._model(step).invoke(prompt)
            content = reply.content
            if isinstance(content, list):  # content blocks
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            try:
                return extract_json(content)
            except ValueError as err:  # malformed JSON: ask again
                last = err
        raise ValueError(f"step '{step}': model did not return valid JSON ({last})")


Reply = dict | Callable[[str], dict]


class ScriptedLLM:
    """Offline stand-in for tests and demos.

    Queue replies per step with `add`. A reply is a dict, or a callable that
    receives the rendered prompt and returns a dict. Prompts are recorded in
    `calls` for assertions.
    """

    def __init__(self) -> None:
        self._queues: dict[str, deque[Reply]] = defaultdict(deque)
        self.calls: list[tuple[str, str]] = []

    def add(self, step: str, *replies: Reply) -> "ScriptedLLM":
        self._queues[step].extend(replies)
        return self

    def complete_json(self, step: str, prompt: str) -> dict:
        self.calls.append((step, prompt))
        if not self._queues[step]:
            raise AssertionError(f"ScriptedLLM: no reply queued for step '{step}'")
        reply = self._queues[step].popleft()
        return reply(prompt) if callable(reply) else reply

    def remaining(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._queues.items() if v}
