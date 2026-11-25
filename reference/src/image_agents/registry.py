"""Agent registry: one YAML card per agent.

The selection prompt sees `Card.prompt_view()`, which leaves out the
`operational` section (endpoints, timeouts, credentials). Hard rules such as
exclusions, external-agent policy and user-performed tasks are applied here in
code, before any prompt is built.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from .config import Policy

CARDS_DIR = Path(__file__).parent / "cards"


class RequiredInput(BaseModel):
    name: str
    meaning: str = ""
    required: bool = True
    applies_when: Optional[str] = None
    default: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    on_missing: Optional[str] = None


class Skill(BaseModel):
    id: str
    summary: str = ""
    interface: Optional[str] = None           # ManualOps: the interface this skill opens
    prefill: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class Behaviour(BaseModel):
    may_request_user_input: bool = False
    may_request_assistance: bool = False
    opens_interface: bool = False             # True only for agents like ManualOps
    readiness_model: bool = True              # False: required inputs are checked in code only
    batchable: bool = False                   # True: chained tasks may be planned in one call
    method: str = "deterministic"
    quality_gate: str = "none"
    cost_class: str = "low"
    typical_latency_seconds: Any = None
    typical_assistance_needs: list[str] = Field(default_factory=list)


class Card(BaseModel):
    id: str
    name: str
    version: str = "1.0.0"
    location: str = "internal"                # "internal" | "external"
    provider: Optional[str] = None
    description: str
    skills: list[Skill] = Field(default_factory=list)
    not_for: list[str] = Field(default_factory=list)
    required_inputs: list[RequiredInput] = Field(default_factory=list)
    produces: dict[str, Any] = Field(default_factory=dict)
    behaviour: Behaviour = Field(default_factory=Behaviour)
    data_handling: Optional[dict] = None      # external agents
    claims_verified: Optional[bool] = None    # external agents
    observed: dict[str, Any] = Field(default_factory=dict)
    operational: dict[str, Any] = Field(default_factory=dict)

    def prompt_view(self) -> dict:
        """Fields that bear on selection. Never includes `operational`."""
        view = self.model_dump(exclude={"operational"}, exclude_none=True)
        view["behaviour"] = {
            k: v for k, v in view["behaviour"].items() if k not in ("readiness_model", "batchable")
        }
        return view

    def needs_asset(self) -> bool:
        return any(
            r.name == "target_asset" and r.required and not r.applies_when
            for r in self.required_inputs
        )


class Registry:
    def __init__(self, cards: list[Card]):
        self._cards = {c.id: c for c in cards}

    @classmethod
    def load(cls, directory: str | Path = CARDS_DIR) -> "Registry":
        cards = []
        for path in sorted(Path(directory).glob("*.yaml")):
            cards.append(Card.model_validate(yaml.safe_load(path.read_text())))
        return cls(cards)

    def add(self, card: Card) -> None:
        self._cards[card.id] = card

    def get(self, agent_id: str) -> Card:
        return self._cards[agent_id]

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._cards

    def all(self) -> list[Card]:
        return list(self._cards.values())

    def candidates(
        self,
        *,
        excluded: list[str],
        user_performed: bool,
        policy: Policy,
    ) -> list[Card]:
        """Cards that may be offered to the selection prompt for one task."""
        out = []
        for card in self._cards.values():
            if card.id in excluded:
                continue
            if card.location == "external" and not policy.allow_external_agents:
                continue
            # A user-performed task may only go to an agent that opens an
            # interface; a system-performed one may never go to such an agent.
            if card.behaviour.opens_interface != user_performed:
                continue
            out.append(card)
        return out
