"""OpenAI moderation for every message before it reaches the chat engine.

What the user sent is checked with ``omni-moderation-latest``: their text (the message,
form answers, edited suggestion text) and every image they attached or uploaded with it.
Flagging uses the same per-category cut-offs as the REST endpoints
(``settings.MODERATION_THRESHOLD``), not OpenAI's own ``flagged`` bit.

The moderation endpoint is free. Each image goes in its own request (the endpoint takes
text plus one image), and all requests run in parallel. If the endpoint cannot be
reached the message is let through and the failure is logged, so an outage does not take
the chat down; the image services still apply their own safety checks when rendering.
"""
from __future__ import annotations

import base64
import imghdr
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

MODEL = "omni-moderation-latest"
IMAGE_MAX_BYTES = 4 * 1024 * 1024     # as main.MODERATION_IMAGE_MAX_BYTES
DOWNLOAD_TIMEOUT_S = 15

# How each category is named to the user.
_PLAIN = {
    "sexual": "sexual content", "sexual/minors": "sexual content involving minors",
    "harassment": "harassment", "harassment/threatening": "threats",
    "hate": "hateful content", "hate/threatening": "hateful threats",
    "illicit": "illegal activity", "illicit/violent": "violent illegal activity",
    "self-harm": "self-harm", "self-harm/intent": "self-harm", "self-harm/instructions": "self-harm",
    "violence": "violence", "violence/graphic": "graphic violence",
}

BLOCKED_MESSAGE = ("I can't help with that because it goes against our content guidelines "
                   "({reasons}). Please rephrase your request{image_note}.")


@dataclass
class Verdict:
    flagged: bool = False
    categories: list[str] = field(default_factory=list)   # over the threshold, worst first
    in_image: bool = False                                 # an attached image was flagged
    checked: int = 0                                       # moderation requests made
    errors: list[str] = field(default_factory=list)

    def message(self) -> str:
        reasons = ", ".join(dict.fromkeys(_PLAIN.get(c, c) for c in self.categories)) or "unsafe content"
        note = " or attach a different image" if self.in_image else ""
        return BLOCKED_MESSAGE.format(reasons=reasons, image_note=note)


def _thresholds() -> dict:
    import settings
    return getattr(settings, "MODERATION_THRESHOLD", {}) or {}


def _over(result, thresholds: dict) -> list[tuple[str, float]]:
    scores = result.category_scores
    scores = scores.model_dump(by_alias=True) if hasattr(scores, "model_dump") else vars(scores)
    hits = [(c, float(s or 0.0)) for c, s in scores.items() if float(s or 0.0) > thresholds.get(c, 1.0)]
    return sorted(hits, key=lambda x: -x[1])


def _image_input(src: str) -> str:
    """A data URI the moderation endpoint accepts (downloaded and shrunk if needed)."""
    if src.startswith("data:"):
        data = base64.b64decode(src.partition(",")[2])
    else:
        import requests
        resp = requests.get(src, timeout=DOWNLOAD_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.content
    kind = imghdr.what(None, data) or "png"
    if len(data) > IMAGE_MAX_BYTES or kind not in ("png", "jpeg", "webp", "gif"):
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        while True:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            if buf.tell() <= IMAGE_MAX_BYTES or img.width < 256:
                break
            img = img.resize((img.width // 2, img.height // 2))
        data, kind = buf.getvalue(), "jpeg"
    return f"data:image/{kind};base64,{base64.b64encode(data).decode()}"


def check(text: str, images: Optional[list[str]] = None, *, client=None) -> Verdict:
    """Moderate the user's text and attached images. Never raises."""
    text = (text or "").strip()
    images = [i for i in dict.fromkeys(images or []) if isinstance(i, str) and i]
    if not text and not images:
        return Verdict()
    if client is None:
        from openai import OpenAI
        client = OpenAI(timeout=20, max_retries=1)
    thresholds = _thresholds()

    def run(kind: str, payload: str):
        if kind == "text":
            inputs = [{"type": "text", "text": payload}]
        else:
            inputs = [{"type": "image_url", "image_url": {"url": _image_input(payload)}}]
        res = client.moderations.create(model=MODEL, input=inputs).results[0]
        return kind, _over(res, thresholds)

    jobs = ([("text", text)] if text else []) + [("image", i) for i in images]
    verdict = Verdict(checked=len(jobs))
    hits: list[tuple[str, float]] = []
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        for future in [pool.submit(run, k, p) for k, p in jobs]:
            try:
                kind, over = future.result()
            except Exception as err:  # noqa: BLE001 - an outage must not block the chat
                verdict.errors.append(f"{type(err).__name__}: {err}")
                continue
            if over:
                hits += over
                verdict.in_image = verdict.in_image or kind == "image"
    if hits:
        verdict.flagged = True
        verdict.categories = [c for c, _ in sorted(hits, key=lambda x: -x[1])]
        verdict.categories = list(dict.fromkeys(verdict.categories))
    for e in verdict.errors:
        print(f"[moderation] check failed, letting the message through: {e}")
    return verdict
