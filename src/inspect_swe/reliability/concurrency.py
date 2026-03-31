"""Concurrency policies for reliability orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ConcurrencyPolicyError(ValueError):
    """Raised when an invalid reliability concurrency policy is requested."""


class OrchestratorConcurrency(BaseModel):
    """Concurrency controls for campaign orchestration and eval execution."""

    orchestrator_mode: Literal["single_process", "multi_process"] = "multi_process"
    orchestrator_workers: int = Field(default=1, ge=1)
    max_tasks: int | None = Field(default=None, ge=1)
    max_samples: int | None = Field(default=None, ge=1)
    max_subprocesses: int | None = Field(default=None, ge=1)
    max_sandboxes: int | None = Field(default=None, ge=1)
    max_connections: int | None = Field(default=None, ge=1)
    reproducible_profile: bool = True

    @model_validator(mode="after")
    def _validate_mode_constraints(self) -> "OrchestratorConcurrency":
        if self.orchestrator_mode == "single_process" and self.orchestrator_workers != 1:
            raise ConcurrencyPolicyError(
                "single_process mode requires orchestrator_workers=1 because "
                "Inspect permits one active eval_async per process."
            )
        return self


def validate_orchestrator_policy(policy: OrchestratorConcurrency) -> None:
    """Validate policy in a standalone helper for explicit preflight checks."""
    # Trigger pydantic/model-level validation rules.
    OrchestratorConcurrency.model_validate(policy.model_dump())
