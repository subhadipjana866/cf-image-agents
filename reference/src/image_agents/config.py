"""Limits and operator policy. Values are starting points to tune."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Limits:
    max_stack_depth: int = 3          # assistance chains
    max_steps_per_task: int = 12      # agent steps for one task
    max_attempts_per_task: int = 3    # retries after a retryable failure
    max_steps_per_turn: int = 60      # scheduler passes in one user turn
    max_batch_size: int = 4           # chained tasks handed to one agent in a single call
    graph_recursion_limit: int = 400  # LangGraph supersteps per invocation


@dataclass
class Policy:
    # External agents
    allow_external_agents: bool = False
    external_consent_required: bool = True
    # Result kinds that only the user may carry out, whatever the phrasing.
    user_only_results: set[str] = field(default_factory=lambda: {"account_action"})
    # A failed link in a chain: continue on the last good version, or stop.
    continue_on_failed_link: bool = True
    # "generative": gate the final version of a plan when any step was
    # generative. "off": never.
    quality_gate: str = "generative"
    evaluator_agent_id: str = "image-evaluator"

    def for_prompt(self) -> dict:
        return {
            "external_agents_allowed": self.allow_external_agents,
            "consent_required_before_sharing": self.external_consent_required,
            "user_only_results": sorted(self.user_only_results),
        }


@dataclass
class Config:
    limits: Limits = field(default_factory=Limits)
    policy: Policy = field(default_factory=Policy)
    history_turns_in_prompt: int = 10
