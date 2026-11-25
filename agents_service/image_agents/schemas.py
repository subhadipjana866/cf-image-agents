"""Data contracts shared by the orchestrator steps and the agents."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Performer = Literal["system", "user"]


class AssetRef(BaseModel):
    asset_id: str
    version: int


# ---------------------------------------------------------------- interpretation
class RequestedChange(BaseModel):
    change: str
    target: str
    performed_by: Performer = "system"


class Conflict(BaseModel):
    between: list[str] = Field(default_factory=list)
    kind: Literal["tension", "contradiction"]
    explanation: str = ""
    resolutions: list[str] = Field(default_factory=list)


class TaskIntent(BaseModel):
    """One task as stated by the interpretation step."""

    id: str
    goal: str
    desired_result: str = "other"
    performed_by: Performer = "system"
    target_assets: list[AssetRef] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    supports_task_id: Optional[str] = None


class PendingTaskInput(BaseModel):
    task_id: str
    input: str


class Clarification(BaseModel):
    request: str
    options: list[str] = Field(default_factory=list)
    why: str = ""


class Interpretation(BaseModel):
    analysis: str = ""
    input_type: Literal[
        "new_task", "input_for_pending_task", "supporting_task", "amendment", "cancellation"
    ]
    pending_task_input: Optional[PendingTaskInput] = None
    affected_task_id: Optional[str] = None
    requested_changes: list[RequestedChange] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    tasks: list[TaskIntent] = Field(default_factory=list)
    clarification: Optional[Clarification] = None


# ---------------------------------------------------------------------- selection
class Alternate(BaseModel):
    agent_id: str
    why_not_first: str = ""


class ExternalHandoff(BaseModel):
    consent_required: bool = False
    share: list[str] = Field(default_factory=list)


class Selection(BaseModel):
    task_id: Optional[str] = None
    analysis: str = ""
    decision: Literal["assign", "missing_inputs", "no_capable_agent"]
    agent_id: Optional[str] = None
    location: Optional[Literal["internal", "external"]] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    alternates: list[Alternate] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    capability_gap: Optional[str] = None
    external_handoff: Optional[ExternalHandoff] = None


# ---------------------------------------------------------------------- readiness
class MissingInput(BaseModel):
    name: str
    request: str
    options: list[str] = Field(default_factory=list)


class Readiness(BaseModel):
    task_id: Optional[str] = None
    analysis: str = ""
    resolved: dict[str, Any] = Field(default_factory=dict)
    defaults_used: list[str] = Field(default_factory=list)
    missing: list[MissingInput] = Field(default_factory=list)


# --------------------------------------------------------------------------- tasks
TaskStatus = Literal[
    "ready",            # runnable (new, or resumable)
    "in_progress",
    "pending",          # suspended, waiting for input from the user
    "waiting_on_task",  # suspended until another task's result arrives
    "parked",           # was pending or blocked; the user moved on
    "completed",
    "failed",
    "cancelled",
]
FINAL_STATUSES = {"completed", "failed", "cancelled"}


class Task(BaseModel):
    """A task as tracked by the orchestrator. Stored in graph state as a dict."""

    id: str
    plan_id: str
    parent_id: Optional[str] = None
    origin: str = "user"                    # "user" or the requesting agent's id
    depth: int = 0

    goal: str
    desired_result: str = "other"
    performed_by: Performer = "system"
    target_assets: list[AssetRef] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    supports_task_id: Optional[str] = None

    status: TaskStatus = "ready"
    agent_id: Optional[str] = None
    excluded_agents: list[str] = Field(default_factory=list)
    selection: Optional[dict] = None
    consent: bool = False

    inputs: dict[str, Any] = Field(default_factory=dict)   # gathered required inputs
    inputs_ready: bool = False
    readiness_cache: Optional[list] = None   # missing inputs found in a batched check
    batch_ok: bool = True                # False after a batched call failed for this task
    record: list[dict] = Field(default_factory=list)       # task record: facts and findings
    request: Optional[dict] = None       # what it asked the user: text, options, from
    waiting_on: Optional[str] = None     # a task id, or "ui:<task id>"
    resume: Optional[dict] = None        # payload for the next agent step
    agent_state: dict[str, Any] = Field(default_factory=dict)
    supporting_results: list[dict] = Field(default_factory=list)

    result: Optional[dict] = None
    failure: Optional[str] = None
    attempts: int = 0
    steps: int = 0


# ------------------------------------------------------------------ agent contract
class TaskEnvelope(BaseModel):
    """What every agent receives."""

    task_id: str
    parent_task_id: Optional[str] = None
    plan_id: str
    session_id: str
    origin: str
    goal: str
    desired_result: str
    performed_by: Performer
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    target_assets: list[AssetRef] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    context: list[dict] = Field(default_factory=list)      # the task record
    limits: dict[str, int] = Field(default_factory=dict)
    resume: Optional[dict] = None       # {"user_input": ...} or {"assistance": ...}
    agent_state: dict[str, Any] = Field(default_factory=dict)


class Outcome(BaseModel):
    """What every agent step returns: exactly one of four kinds."""

    kind: Literal["completed", "input_required", "assistance_required", "failed"]

    # completed
    result: dict[str, Any] = Field(default_factory=dict)
    assets: list[AssetRef] = Field(default_factory=list)
    summary: str = ""
    client_action: Optional[dict] = None
    generative: bool = False            # a generative method produced the assets

    # input_required
    request: str = ""
    options: list[str] = Field(default_factory=list)

    # assistance_required
    need: str = ""
    need_inputs: dict[str, Any] = Field(default_factory=dict)
    acceptance: str = ""

    # failed
    reason: str = ""
    retryable: bool = False

    # carried to the agent's next step on this task
    agent_state: dict[str, Any] = Field(default_factory=dict)
    findings: list[dict] = Field(default_factory=list)     # written to the task record
