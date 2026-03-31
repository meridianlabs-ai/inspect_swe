"""Top-level reliability campaign specification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .concurrency import OrchestratorConcurrency

PhaseName = Literal[
    "baseline",
    "fault",
    "prompt",
    "structural",
    "safety",
    "abstention",
]


class ReliabilitySpec(BaseModel):
    """Configuration for a reliability campaign.

    Notes:
    - Inspect `.eval` logs are always canonical for execution truth.
    - Sidecar records are derived analysis projections.
    """

    benchmark: str
    agents: list[str] = Field(default_factory=list)
    phases: list[PhaseName] = Field(
        default_factory=lambda: [
            "baseline",
            "fault",
            "prompt",
            "structural",
            "safety",
            "abstention",
        ]
    )
    seed: int = 0
    sidecar_dir: str = "reliability_sidecars"
    canonical_log_format: Literal["eval"] = "eval"
    strict_identity_tags: bool = True
    fail_on_missing_hooks: bool = True
    concurrency: OrchestratorConcurrency = Field(
        default_factory=OrchestratorConcurrency
    )

    @field_validator("benchmark")
    @classmethod
    def _validate_benchmark(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("benchmark cannot be empty")
        return value

    @field_validator("agents")
    @classmethod
    def _validate_agents(cls, value: list[str]) -> list[str]:
        cleaned = [agent.strip() for agent in value if agent.strip()]
        if not cleaned:
            raise ValueError("agents must contain at least one non-empty agent name")
        return cleaned

    @model_validator(mode="after")
    def _validate_unique_phases(self) -> "ReliabilitySpec":
        if len(self.phases) != len(set(self.phases)):
            raise ValueError("phases must be unique")
        return self

    def has_phase(self, phase: PhaseName) -> bool:
        return phase in self.phases
