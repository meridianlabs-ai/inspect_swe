"""Identity schema for reliability records."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .spec import PhaseName


class ReliabilityRunIdentity(BaseModel):
    """Identity envelope with explicit reliability semantics.

    Distinguishes:
    - repeat_id: independent K-run repetition index.
    - sample_retry_id: Inspect retry-on-error index.
    - agent_attempt_id: in-agent continuation/retry loop index.
    """

    eval_set_id: str
    run_id: str
    phase: PhaseName
    agent: str
    task: str
    sample_id: int | str
    sample_uuid: str
    repeat_id: int = Field(default=0, ge=0)
    sample_retry_id: int = Field(default=0, ge=0)
    agent_attempt_id: int = Field(default=0, ge=0)

    @field_validator("eval_set_id", "run_id", "agent", "task", "sample_uuid")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    def key(self) -> str:
        """Deterministic key for grouping and dedupe."""
        return (
            f"{self.eval_set_id}:{self.run_id}:{self.phase}:{self.agent}:{self.task}:"
            f"{self.sample_uuid}:r{self.repeat_id}:sr{self.sample_retry_id}:aa{self.agent_attempt_id}"
        )
