"""Run context shared by the generation and edit agents and their tools.

A tool reads and writes the session through ``runtime.context``. Every image a
tool produces becomes a new version with its parent, so a chain of edits can be
followed, undone, or reported step by step, and nothing is applied silently.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Version:
    number: int
    url: str
    made_by: str                    # agent id, or "upload"
    tool: str
    instruction: str
    parent: Optional[int] = None
    model: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)   # text_layers, archetype, notices...


@dataclass
class Brief:
    """What the input-collector hands the generator: everything already settled.

    Until the collector exists, tests fill this in directly. Nothing in it is
    asked for again; a gap is filled by the pipeline's own defaults.
    """
    prompt: str
    design_goal: Optional[str] = None          # catalog goal, None = freeform
    platform: Optional[str] = None             # catalog platform name
    aspect_ratio: Optional[str] = None         # e.g. "4:5"; used by freeform
    field_answers: dict[str, str] = field(default_factory=dict)
    photo_url: Optional[str] = None
    photo_mode: Optional[str] = None           # "as_is" | "reference"
    logo_url: Optional[str] = None
    logo_mode: str = "none"                    # "provided" | "omitted" | "none"
    style: Optional[str] = None                # preset id from styles.json, or free text
    cta_requested: bool = False
    num_images: int = 1


@dataclass
class DesignSession:
    user_id: str = "local"
    session_id: str = "agents-test"
    brief: Optional[Brief] = None
    brand_dna: Optional[dict] = None
    brand_name: Optional[str] = None
    versions: list[Version] = field(default_factory=list)
    cost: float = 0.0
    trace: list[dict] = field(default_factory=list)
    # Set when an agent asked the user to pick an image; the next frontend reply
    # carrying a selected image continues `pending_request`.
    awaiting_image: bool = False
    pending_request: Optional[str] = None
    # Interfaces ManualOps asked the frontend to open (the user acts in them).
    client_actions: list[dict] = field(default_factory=list)
    # Suggestions from the critic / suggestion agents, so "apply the 2nd one" can be resolved.
    suggestions: list[dict] = field(default_factory=list)
    # Images the user attached this turn (a product photo, a reference); not the canvas.
    attachments: list[str] = field(default_factory=list)
    # Generator bookkeeping kept between turns (v33 result, create-more lineage).
    v33_prev: Optional[dict] = None
    lineage: list[dict] = field(default_factory=list)
    lineage_urls: list[str] = field(default_factory=list)
    # Layering: boxes the user drew on the image this turn (normalised, image pixels), and
    # every layering result ({image_url, version, segmentation_result, bboxes, extracted_texts}).
    manual_bboxes: list[dict] = field(default_factory=list)
    layers: list[dict] = field(default_factory=list)
    # Whether the frontend has an image selected on the canvas this turn (the websocket sets
    # it per message). A new design ignores the session's image when nothing is selected.
    canvas_selected: bool = True
    # The project's latest graphic, used only by requests that work on an image when the
    # session has none yet (as the old last-graphic fallback did for image routes).
    fallback_image: Optional[str] = None

    # ------------------------------------------------------------ versions
    @property
    def current(self) -> Optional[Version]:
        return self.versions[-1] if self.versions else None

    def add_version(self, url: str, *, made_by: str, tool: str, instruction: str,
                    parent: Optional[int] = None, model: Optional[str] = None,
                    meta: Optional[dict] = None) -> Version:
        if parent is None and self.current:
            parent = self.current.number
        v = Version(len(self.versions) + 1, url, made_by, tool, instruction, parent, model,
                    meta or {})
        self.versions.append(v)
        return v

    def get(self, number: Optional[int]) -> Version:
        if not self.versions:
            raise LookupError("there is no image in this session yet")
        if number is None:
            return self.versions[-1]
        if not 1 <= number <= len(self.versions):
            raise LookupError(f"version {number} does not exist; versions are 1-{len(self.versions)}")
        return self.versions[number - 1]

    def start_with_image(self, url: str, description: str = "uploaded image") -> Version:
        return self.add_version(url, made_by="upload", tool="upload", instruction=description,
                                parent=None)

    def select_image(self, url: str) -> Version:
        """The user picked an image in the frontend: it becomes the current image."""
        if self.current and self.current.url == url:
            return self.current
        return self.add_version(url, made_by="user", tool="selected",
                                instruction="image selected in the frontend", parent=None)

    # --------------------------------------------------------------- trace
    def record(self, tool: str, *, ok: bool, detail: str, cost: float = 0.0,
               started: Optional[float] = None, **extra: Any) -> None:
        self.cost += cost
        self.trace.append({"tool": tool, "ok": ok, "detail": detail, "cost": round(cost, 6),
                           "seconds": round(time.monotonic() - started, 2) if started else None,
                           **extra})
