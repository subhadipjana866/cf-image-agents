"""One model call that looks at images and returns JSON. Used by the collector and the
review agents' tools."""
from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from image_agents.llm import extract_json

from .images import to_data_uri

VISION_MODEL = "openai:gpt-5.4"
_models: dict[str, Any] = {}


def image_part(url: str, detail: str = "auto") -> dict:
    src = url if url.startswith(("http://", "https://", "data:")) else to_data_uri(url)
    return {"type": "image_url", "image_url": {"url": src, "detail": detail}}


def vision_json(prompt: str, images: Optional[list[str]] = None, *,
                model: str = VISION_MODEL, retries: int = 1) -> tuple[dict, float, float]:
    """(parsed JSON, cost in dollars, seconds). Images are URLs, paths or data URIs."""
    from langchain.chat_models import init_chat_model
    from ..utils import calculateCost

    if model not in _models:
        _models[model] = init_chat_model(model)
    content: list[dict] = [{"type": "text", "text": prompt}]
    content += [image_part(u) for u in images or []]
    started, last = time.monotonic(), None
    for _ in range(retries + 1):
        reply = _models[model].invoke([HumanMessage(content=content)])
        usage = getattr(reply, "usage_metadata", None) or {}
        cost = calculateCost(model.split(":", 1)[-1], int(usage.get("total_tokens") or 0))
        text = reply.content if isinstance(reply.content, str) else "".join(
            b.get("text", "") for b in reply.content if isinstance(b, dict))
        try:
            return extract_json(text), cost, time.monotonic() - started
        except ValueError as err:
            last = err
    raise ValueError(f"vision call returned no JSON ({last})")
