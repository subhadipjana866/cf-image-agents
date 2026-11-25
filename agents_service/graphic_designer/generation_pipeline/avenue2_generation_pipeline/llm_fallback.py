"""Provider fallback for the reasoning/COT text calls.

Every COT stage (scope check, intent clarifier, intent router, multicall
fusion) calls Gemini directly. This helper wraps that call with a provider
chain so a Gemini outage degrades gracefully instead of failing the whole
generation:

    Gemini (per-stage model)  ->  OpenAI (gpt-5.6-terra)  ->  Anthropic (claude-haiku-4-5)

The returned object mirrors the google-genai response surface the call sites
already consume (``.text`` and ``.usage_metadata.prompt_token_count`` /
``.candidates_token_count``), plus ``.model_used`` for logging.

Fallback models are overridable via env:
    COT_OPENAI_FALLBACK_MODEL     (default gpt-5.6-terra)
    COT_ANTHROPIC_FALLBACK_MODEL  (default claude-haiku-4-5-20251001)
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

OPENAI_FALLBACK_MODEL = os.environ.get("COT_OPENAI_FALLBACK_MODEL", "gpt-5.6-terra")
ANTHROPIC_FALLBACK_MODEL = os.environ.get("COT_ANTHROPIC_FALLBACK_MODEL", "claude-haiku-4-5-20251001")


@dataclass
class _Usage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0


@dataclass
class FallbackResponse:
    text: str
    usage_metadata: _Usage = field(default_factory=_Usage)
    model_used: str = ""
    # The GEN_PRICES key for whatever answered. Separate from model_used (the
    # wire id) so a caller can bill correctly without knowing which rung of the
    # chain served it -- routing a call to another provider must not change
    # what the customer pays.
    billing_id: str = ""


def _apiLog(provider, model, status="ok", tokens=None, detail="",
            tokens_in=None, tokens_out=None, role=None):
    """Report into the shared api_calls log. Imported lazily and guarded: this
    module is loaded via sys.path (not as a package), so the parent package may
    not be importable in standalone/test runs."""
    try:
        from graphic_designer.utils import apiLog
        apiLog(provider, model, "cot_multicall", tokens=tokens,
               status=status, detail=detail,
               tokens_in=tokens_in, tokens_out=tokens_out, role=role)
    except Exception:
        pass


def _strip_md_fences(text: str) -> str:
    """OpenAI/Anthropic sometimes wrap JSON in ```json fences; Gemini's
    response_mime_type prevents that natively — normalize the others."""
    t = (text or "").strip()
    if t.startswith("```"):
        import re
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
        if m:
            return m.group(1)
    return t


def _image_png_bytes(path) -> bytes:
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Provider rungs ─────────────────────────────────────────────────────────
# One function per provider, each a literal lift of the block that used to sit
# inline in generate_with_fallback -- same request shaping, same logging, same
# empty-response rule. Extracting them is what lets the chain become data
# (a registry role) instead of three hardcoded try/except blocks.


def _call_google(model, prompt_text, temperature, response_mime_type,
                 system_instruction, max_output_tokens, images, role=None):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg = {"temperature": temperature}
    if response_mime_type:
        cfg["response_mime_type"] = response_mime_type
    if system_instruction:
        cfg["system_instruction"] = system_instruction
    if max_output_tokens:
        cfg["max_output_tokens"] = max_output_tokens
    # Gemini 3 thinks by default, and thinking tokens are drawn from the SAME
    # max_output_tokens budget as the answer. At the pipeline's 2048-token
    # stage budget a default-thinking call can spend the whole allowance
    # reasoning and return empty text (finish_reason=MAX_TOKENS), so every
    # stage here asks for the low thinking level. Older models that don't
    # know the field are handled by the retry below.
    thinking_cfg = types.ThinkingConfig(thinking_level="low")

    parts = [types.Part.from_bytes(data=_image_png_bytes(p), mime_type="image/png") for p in images]
    parts.append(types.Part(text=prompt_text))
    _apiLog("google", model, status="start", role=role,
            detail=f"{len(images)} image(s)" if images else "text only")
    try:
        r = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=parts, role="user")],
            config=types.GenerateContentConfig(thinking_config=thinking_cfg, **cfg),
        )
    except Exception as thinking_err:
        if "thinking" not in str(thinking_err).lower():
            raise
        r = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=parts, role="user")],
            config=types.GenerateContentConfig(**cfg),
        )
    u = r.usage_metadata
    _apiLog("google", model, role=role,
            tokens_in=getattr(u, "prompt_token_count", 0) or 0,
            tokens_out=getattr(u, "candidates_token_count", 0) or 0)
    text = r.text or ""
    # An empty body is a failure, not a result: fall through to the next
    # provider rather than handing "" to a stage that will build a prompt
    # out of it.
    if not text.strip():
        raise RuntimeError(
            f"empty response (finish_reason="
            f"{getattr((r.candidates or [None])[0], 'finish_reason', '?')})")
    if response_mime_type == "application/json":
        text = _strip_md_fences(text)
    return FallbackResponse(
        text=text,
        usage_metadata=_Usage(getattr(u, "prompt_token_count", 0) or 0,
                              getattr(u, "candidates_token_count", 0) or 0),
        model_used=model,
    )


def _call_openai(model, prompt_text, temperature, response_mime_type,
                 system_instruction, max_output_tokens, images, role=None,
                 api_base="https://api.openai.com/v1", api_key=None):
    """OpenAI chat-completions, and any OpenAI-compatible endpoint: the only
    per-provider difference is the base URL and key, so one rung serves them."""
    import requests

    content = []
    for p in images:
        b64 = base64.b64encode(_image_png_bytes(p)).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt_text})

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": str(system_instruction)})
    messages.append({"role": "user", "content": content})

    body = {"model": model, "messages": messages}
    # json_object mode requires the word 'json' somewhere in the messages
    if response_mime_type == "application/json" and "json" in prompt_text.lower():
        body["response_format"] = {"type": "json_object"}

    _apiLog("openai", model, status="start", role=role,
            detail=f"{len(images)} image(s)" if images else "text only")
    resp = requests.post(
        f"{api_base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key or OPENAI_API_KEY}"},
        json=body, timeout=120,
    )
    resp.raise_for_status()
    d = resp.json()
    usage = d.get("usage", {})
    out = d["choices"][0]["message"]["content"] or ""
    _apiLog("openai", model, role=role,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0))
    if response_mime_type == "application/json":
        out = _strip_md_fences(out)
    if not (out or "").strip():
        raise RuntimeError("empty response")
    return FallbackResponse(
        text=out,
        usage_metadata=_Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        model_used=model,
    )


def _call_anthropic(model, prompt_text, temperature, response_mime_type,
                    system_instruction, max_output_tokens, images, role=None):
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    content = []
    for p in images:
        b64 = base64.b64encode(_image_png_bytes(p)).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
    content.append({"type": "text", "text": prompt_text})

    kwargs = {
        "model": model,
        "max_tokens": max_output_tokens or 4096,
        "messages": [{"role": "user", "content": content}],
        "temperature": min(max(temperature, 0.0), 1.0),
    }
    if system_instruction:
        kwargs["system"] = str(system_instruction)
    _apiLog("anthropic", model, status="start", role=role,
            detail=f"{len(images)} image(s)" if images else "text only")
    r = client.messages.create(**kwargs)
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    _apiLog("anthropic", model, role=role,
            tokens_in=r.usage.input_tokens, tokens_out=r.usage.output_tokens)
    if response_mime_type == "application/json":
        text = _strip_md_fences(text)
    if not (text or "").strip():
        raise RuntimeError("empty response")
    return FallbackResponse(
        text=text,
        usage_metadata=_Usage(r.usage.input_tokens, r.usage.output_tokens),
        model_used=model,
    )


_PROVIDER_CALLS = {
    "google": _call_google,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


def _legacy_chain(model):
    """The hardcoded Gemini -> OpenAI -> Anthropic order this module shipped
    with. Used whenever no role is given, so every existing caller and every
    standalone script keeps its exact behavior."""
    return [
        {"provider": "google", "api_id": model, "billing_id": model},
        {"provider": "openai", "api_id": OPENAI_FALLBACK_MODEL, "billing_id": OPENAI_FALLBACK_MODEL},
        {"provider": "anthropic", "api_id": ANTHROPIC_FALLBACK_MODEL,
         "billing_id": ANTHROPIC_FALLBACK_MODEL},
    ]


def _role_chain(role, pin=None):
    """The registry's chain for `role`, or None when the router is unreachable.

    Guarded the same way _apiLog is: this module is loaded via sys.path in the
    standalone pipeline runs, where the parent package may not import at all.
    A missing router must degrade to the legacy chain, never fail the call.
    """
    try:
        from graphic_designer.model_router import resolveRole
        return resolveRole(role, pin=pin)
    except Exception:
        return None


def generate_with_fallback(
    model: str,
    prompt_text: str,
    temperature: float = 0.0,
    response_mime_type: str | None = None,
    system_instruction: str | None = None,
    max_output_tokens: int | None = None,
    image_paths: list | None = None,
    role: str | None = None,
    pin: str | None = None,
) -> FallbackResponse:
    """Run one text-generation call with provider fallback.

    The chain comes from one of two places:

    * ``role`` given — :mod:`graphic_designer.model_router` resolves the
      registry policy for that role (which provider goes first, and what
      follows it). This is how the pipeline should ask for work.
    * otherwise — the module's original hardcoded order, Gemini(``model``) ->
      OpenAI -> Anthropic, so every existing caller and every standalone script
      behaves exactly as before.

    :param model: Gemini model id for the legacy chain. Ignored when ``role``
        is given, because the registry names the model then.
    :param prompt_text: The full user prompt.
    :param image_paths: Optional reference images (e.g. wireframes) attached
        to the request on every provider.
    :param role: Registry role to route by (``cot_reasoning``, ...).
    :param pin: An ``api_id`` to force to the head of the chain — how a
        multi-call stage keeps every one of its calls on the provider the run
        started with.
    :raises RuntimeError: Only when ALL providers fail.
    """
    errors = []
    images = [p for p in (image_paths or []) if p and Path(p).exists()]

    chain = (_role_chain(role, pin) if role else None) or _legacy_chain(model)

    for entry in chain:
        provider = entry.get("provider")
        api_id = entry.get("api_id") or entry.get("name")
        call = _PROVIDER_CALLS.get(provider)
        if call is None or not api_id:
            continue
        try:
            resp = call(api_id, prompt_text, temperature, response_mime_type,
                        system_instruction, max_output_tokens, images,
                        role=entry.get("role") or role)
            resp.billing_id = entry.get("billing_id") or api_id
            return resp
        except Exception as e:
            errors.append(f"{provider}({api_id}): {e}")
            _apiLog(provider, api_id, status="failed", detail=str(e),
                    role=entry.get("role") or role)
            print(f"[llm_fallback] {provider}({api_id}) failed: {str(e)[:200]}")

    raise RuntimeError("[llm_fallback] ALL providers failed: " + " | ".join(errors))
