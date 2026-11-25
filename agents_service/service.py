"""HTTP front for the Kognie agents, run in a Cloudflare Container.

The Worker's ImageChatAgent (a Durable Object) owns the conversation, the task
list and the images. For each task it calls ``POST /invoke`` with the agent id,
the task, and the session's images; this service runs the copied LangChain agent
(graphic_designer/agents) and returns one of three outcomes, as chat._invoke does:

* completed: the agent's reply plus any new image versions (bytes, base64)
* input_required: the agent called ask_user; the question and options
* failed: an error, or an image job where every tool failed

The container is stateless between calls except for the LangGraph checkpoint of a
run that is waiting on the user. A container can sleep and lose it, so a resume
whose checkpoint is gone is restarted with the question and answer in the message.
"""
from __future__ import annotations

import container_shims  # noqa: I001 - must run before any graphic_designer import

container_shims.install_store()

import base64
import os
import re
import traceback
from typing import Any, Literal, Optional

from fastapi import FastAPI
from langchain.agents.middleware import wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from graphic_designer.agents import build, edit_tools, generation_tools, support_agents
from graphic_designer.agents.common import CHECKPOINTER, make_agent
from graphic_designer.agents.images import load_bytes, to_data_uri
from graphic_designer.agents.session import Brief, DesignSession

MODEL = os.getenv("AGENT_MODEL", build.DEFAULT_MODEL)
AGENT_IDS = ("image-generator", "image-editor", "chitchat-handler")

# ------------------------------------------------------------------ agents
# The same builders as graphic_designer/agents/build.py and support_agents.py, with
# the tool sets this app uses: freeform generation, variations and regenerate; the
# fifteen generative edit tools; chitchat as it is.
GENERATION_TOOLS = [generation_tools.generate_freeform_design,
                    generation_tools.create_more_variations,
                    generation_tools.regenerate_design]
EDIT_TOOLS = edit_tools.EDIT_TOOLS[:15]     # the generative block of EDIT_TOOLS

SCOPE = {
    "image-generator": (
        "IN THIS APP: there is no design catalog. For every new design use "
        "generate_freeform_design; create_more_variations and regenerate_design work as "
        "described. list_catalog and generate_catalog_design do not exist here."),
    "image-editor": (
        "IN THIS APP: only the generative edit tools exist. There are no exact tools "
        "(crop, resize, rotate, tone, border, logo, platform resize, upscale, restore, "
        "export, image info). Do mechanical changes with custom_edit or "
        "apply_multiple_edits, and say the result is AI-made, not pixel-exact."),
    "chitchat-handler": (
        "IN THIS APP: three agents exist: image generation (new designs, variations, "
        "regenerate), image editing (generative edits to the current image), and you. "
        "Describe only those when asked what you can do."),
}


def _scope(agent_id: str):
    @wrap_model_call
    def scope_note(request, handler):
        current = request.system_message.content if request.system_message else ""
        note = SCOPE[agent_id]
        return handler(request.override(system_message=SystemMessage(content=f"{current}\n\n{note}")))
    return scope_note


@wrap_model_call
def see_current_image(request, handler):
    """build.see_current_image, with images from the container store sent inline:
    their URLs exist only inside this container, so OpenAI cannot fetch them."""
    session = request.runtime.context
    v = session.current if session else None
    if v is not None and container_shims.stored(v.url):
        data, mime = container_shims.stored(v.url)
        look = HumanMessage(content=[
            {"type": "text", "text": f"[Current image: version {v.number}, shown by the app, "
                                     f"not sent by the user.]"},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(data).decode()}", "detail": "auto"}},
        ])
        return handler(request.override(messages=[*request.messages, look]))
    return build.see_current_image.wrap_model_call(request, handler)


def _build(agent_id: str):
    model = init_chat_model(MODEL)
    if agent_id == "image-generator":
        return make_agent(model, GENERATION_TOOLS,
                          middleware=[build.generation_prompt, build.one_tool_at_a_time,
                                      _scope(agent_id)],
                          context_schema=DesignSession, name=agent_id)
    if agent_id == "image-editor":
        return make_agent(model, EDIT_TOOLS,
                          middleware=[build.edit_prompt, see_current_image,
                                      build.one_tool_at_a_time, _scope(agent_id)],
                          context_schema=DesignSession, name=agent_id)
    return make_agent(model, [support_agents.explain_capabilities],
                      system_prompt=support_agents.CHITCHAT_PROMPT,
                      middleware=[_scope(agent_id)],
                      context_schema=DesignSession, name=agent_id)


_AGENTS: dict[str, Any] = {}


def agent(agent_id: str):
    if agent_id not in _AGENTS:
        _AGENTS[agent_id] = _build(agent_id)
    return _AGENTS[agent_id]


# ------------------------------------------------------------------ contract
class VersionIn(BaseModel):
    number: int
    image: str                       # data URI or http(s) URL
    made_by: str = "upload"
    tool: str = "upload"
    instruction: str = ""
    parent: Optional[int] = None
    model: Optional[str] = None


class BriefIn(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = None
    style: Optional[str] = None
    num_images: int = 1
    photo: Optional[str] = None      # data URI of an attached product photo
    photo_mode: Optional[str] = None
    logo: Optional[str] = None       # data URI of a logo
    field_answers: dict[str, str] = Field(default_factory=dict)


class SessionIn(BaseModel):
    user_id: str = "cf-user"
    session_id: str
    brief: Optional[BriefIn] = None
    versions: list[VersionIn] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class InvokeIn(BaseModel):
    agent_id: Literal["image-generator", "image-editor", "chitchat-handler"]
    thread_id: str                   # "<chat>:<task>:<attempt>", as chat._thread
    message: str = ""                # the task goal for a new run
    resume: Optional[str] = None     # the user's answer to the agent's question
    question: Optional[str] = None   # that question, used if the checkpoint is gone
    history: list[Turn] = Field(default_factory=list)
    session: SessionIn


class NewVersion(BaseModel):
    number: int
    parent: Optional[int]
    tool: str
    instruction: str
    model: Optional[str]
    mime: str
    image_b64: str


class Ask(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)
    multiple: bool = False
    allow_free_text: bool = True


class InvokeOut(BaseModel):
    status: Literal["completed", "input_required", "failed"]
    reply: str = ""
    ask: Optional[Ask] = None
    new_versions: list[NewVersion] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    cost: float = 0.0
    error: Optional[str] = None


# ------------------------------------------------------------------ helpers
def _session(s: SessionIn) -> DesignSession:
    session = DesignSession(user_id=s.user_id, session_id=s.session_id,
                            attachments=list(s.attachments))
    if s.brief:
        b = s.brief
        session.brief = Brief(prompt=b.prompt, aspect_ratio=b.aspect_ratio, style=b.style,
                              num_images=b.num_images, photo_url=b.photo,
                              photo_mode=b.photo_mode if b.photo else None,
                              logo_url=b.logo, logo_mode="provided" if b.logo else "none",
                              field_answers=dict(b.field_answers))
    for v in sorted(s.versions, key=lambda v: v.number):
        session.add_version(v.image, made_by=v.made_by, tool=v.tool,
                            instruction=v.instruction, parent=v.parent, model=v.model)
    return session


def _image_bytes(url: str) -> tuple[bytes, str]:
    hit = container_shims.stored(url)
    if hit:
        return hit
    data = load_bytes(url)
    mime = to_data_uri(data).split(";", 1)[0][5:]
    return data, mime


_LINK = re.compile(r"(https://%s\S+|file://\S+|data:image/\S+)" % re.escape(container_shims.URL_PREFIX[8:]))


_REF_LINE = re.compile(r"^\s*[-*]?\s*(final version|version|image|images|link|links)\s*\d*\s*:?\s*"
                       r"(version\s*\d+)?\s*(\(image\))?\s*$", re.IGNORECASE)


def _user_text(text: str) -> str:
    """The reply without internal storage links or this run's version numbers (they
    count from the image the orchestrator sent, not the chat's own numbering); the
    UI shows the images itself."""
    text = re.sub(r"\[([^\]]*)\]\((?:%s)\)" % _LINK.pattern, r"\1", text)
    text = _LINK.sub("(image)", text)
    lines = [ln for ln in text.splitlines()
             if not _REF_LINE.match(ln) and not re.fullmatch(r"\s*[-*]?\s*\(image\)\s*", ln)]
    text = "\n".join(ln.replace(" (image)", "").replace("(image)", "") for ln in lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _has_checkpoint(a, config: dict) -> bool:
    try:
        return bool(a.get_state(config).next)
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ API
app = FastAPI(title="Kognie agents")


@app.get("/")
@app.get("/ping")
def ping() -> dict:
    """Readiness probe used by the Cloudflare Container class."""
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": MODEL, "agents": list(AGENT_IDS),
            "openai": bool(os.getenv("OPENAI_API_KEY")), "gemini": bool(os.getenv("GEMINI_API_KEY"))}


@app.post("/invoke", response_model=InvokeOut)
def invoke(req: InvokeIn) -> InvokeOut:
    session = _session(req.session)
    before = len(session.versions)
    a = agent(req.agent_id)
    usage = UsageMetadataCallbackHandler()
    config = {"recursion_limit": 60, "callbacks": [usage],
              "configurable": {"thread_id": req.thread_id}}

    if req.resume is not None and _has_checkpoint(a, config):
        inputs: Any = Command(resume=req.resume)
    else:
        message = req.message
        if req.resume is not None:          # checkpoint lost: restart with the answer
            message = (f"{message}\n\nYou asked the user: {req.question or '(a question)'}\n"
                       f"They replied: {req.resume}")
        history = [(t.role, t.content) for t in req.history]
        inputs = {"messages": [*history, ("user", message)]}

    try:
        out = a.invoke(inputs, context=session, config=config)
    except Exception as err:  # noqa: BLE001 - reported to the orchestrator
        traceback.print_exc()
        return InvokeOut(status="failed", error=f"{type(err).__name__}: {err}",
                         trace=session.trace, cost=session.cost)

    stopped = out.get("__interrupt__") if isinstance(out, dict) else None
    if stopped:
        ask = getattr(stopped[0], "value", None) or {}
        return InvokeOut(status="input_required",
                         ask=Ask(question=str(ask.get("question") or "Could you tell me a bit more?"),
                                 options=[str(o) for o in ask.get("options") or []],
                                 multiple=bool(ask.get("multiple")),
                                 allow_free_text=bool(ask.get("allow_free_text", True))),
                         trace=session.trace, cost=session.cost)

    CHECKPOINTER.delete_thread(req.thread_id)
    messages = out.get("messages") or []
    reply = _user_text(str(messages[-1].content)) if messages else ""
    new_versions = []
    for v in session.versions[before:]:
        data, mime = _image_bytes(v.url)
        container_shims.forget(v.url)
        new_versions.append(NewVersion(number=v.number, parent=v.parent, tool=v.tool,
                                       instruction=v.instruction, model=v.model, mime=mime,
                                       image_b64=base64.b64encode(data).decode()))
    tool_failed = any(not e.get("ok") for e in session.trace)
    image_agent = req.agent_id in ("image-generator", "image-editor")
    status = "failed" if image_agent and not new_versions and tool_failed else "completed"
    return InvokeOut(status=status, reply=reply, new_versions=new_versions,
                     trace=session.trace, cost=session.cost)
