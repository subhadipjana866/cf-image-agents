"""Model access for the navigator: the `image_agents.llm.LLM` protocol, plus usage.

`image_agents.llm.ChatModelLLM` drops the token counts, which we want when
comparing models. This wrapper is the same idea (one rendered prompt in, one
JSON object out, a model per step) and records tokens and latency per call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from image_agents.llm import extract_json


@dataclass
class Call:
    step: str
    model: str
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class UsageLog:
    calls: list[Call] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)

    def totals(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": len(self.calls),
                "input_tokens": sum(c.input_tokens for c in self.calls),
                "output_tokens": sum(c.output_tokens for c in self.calls),
                "seconds": round(sum(c.seconds for c in self.calls), 1),
            }


class TrackedChatLLM:
    """LangChain chat models keyed by step, with "default" as the fallback.

    `models` example: {"default": "openai:gpt-5.4-mini", "select": "openai:gpt-5.4"}.
    """

    def __init__(self, models: dict[str, str], *, max_retries: int = 1, **model_kwargs: Any):
        from langchain.chat_models import init_chat_model  # imported lazily

        self._init = init_chat_model
        self._names = models
        self._kwargs = model_kwargs
        self._cache: dict[str, Any] = {}
        self._max_retries = max_retries
        self.usage = UsageLog()

    def _model(self, step: str):
        name = self._names.get(step) or self._names["default"]
        if name not in self._cache:
            self._cache[name] = self._init(name, **self._kwargs)
        return name, self._cache[name]

    def complete_json(self, step: str, prompt: str) -> dict:
        name, model = self._model(step)
        last: Exception | None = None
        for _ in range(self._max_retries + 1):
            start = time.monotonic()
            reply = model.invoke(prompt)
            meta = getattr(reply, "usage_metadata", None) or {}
            self.usage.add(Call(step, name, time.monotonic() - start,
                                meta.get("input_tokens", 0), meta.get("output_tokens", 0)))
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
