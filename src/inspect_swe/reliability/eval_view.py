"""Eval-native reliability view helpers."""

from __future__ import annotations

from typing import Any

from inspect_ai.log import EvalLog
from pydantic import BaseModel, Field


_REQUIRED_IDENTITY_METADATA_KEYS = (
    "reliability_phase",
    "reliability_repeat_id",
    "reliability_agent_attempt_id",
    "reliability_agent",
)


class BaselineSampleView(BaseModel):
    """Normalized baseline telemetry view derived from one eval sample."""

    eval_set_id: str
    run_id: str
    task: str
    sample_id: int | str
    sample_uuid: str
    phase: str
    agent: str
    repeat_id: int = Field(default=0, ge=0)
    sample_retry_id: int = Field(default=0, ge=0)
    agent_attempt_id: int = Field(default=0, ge=0)
    scores: dict[str, Any] = Field(default_factory=dict)
    event_count: int = Field(default=0, ge=0)
    event_types: list[str] = Field(default_factory=list)
    tool_call_count: int = Field(default=0, ge=0)
    total_time: float | None = None
    working_time: float | None = None
    confidence_value: float | None = None
    confidence_source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def identity_key(self) -> str:
        return (
            f"{self.eval_set_id}:{self.run_id}:{self.phase}:{self.agent}:{self.task}:"
            f"{self.sample_uuid}:r{self.repeat_id}:sr{self.sample_retry_id}:aa{self.agent_attempt_id}"
        )


def extract_baseline_sample_views(
    eval_log: EvalLog,
    *,
    expected_phase: str,
    expected_agent: str,
    expected_repeat_id: int,
    strict_identity_tags: bool,
) -> tuple[list[BaselineSampleView], int]:
    """Extract normalized baseline telemetry views from an eval log."""
    samples = eval_log.samples or []
    run_metadata = dict(getattr(eval_log.eval, "metadata", {}) or {})
    task_name = str(getattr(eval_log.eval, "task", "unknown_task"))
    run_id = str(getattr(eval_log.eval, "run_id", "unknown_run"))
    eval_set_id = str(getattr(eval_log.eval, "eval_set_id", "no_eval_set"))
    warnings = 0
    views: list[BaselineSampleView] = []

    for sample in samples:
        sample_metadata = dict(getattr(sample, "metadata", {}) or {})
        merged_metadata = dict(run_metadata)
        merged_metadata.update(sample_metadata)

        if strict_identity_tags:
            missing = [
                key for key in _REQUIRED_IDENTITY_METADATA_KEYS if key not in merged_metadata
            ]
            if missing:
                warnings += 1

        phase = str(merged_metadata.get("reliability_phase", expected_phase))
        repeat_id = _read_int(merged_metadata.get("reliability_repeat_id"), expected_repeat_id)
        agent_attempt_id = _read_int(merged_metadata.get("reliability_agent_attempt_id"), 0)
        agent = _read_agent(merged_metadata, expected_agent)

        if strict_identity_tags:
            if phase != expected_phase or repeat_id != expected_repeat_id or agent != expected_agent:
                warnings += 1

        scores = _normalize_scores(getattr(sample, "scores", {}) or {})
        events = list(getattr(sample, "events", []) or [])
        event_types = [type(event).__name__ for event in events]
        confidence_value, confidence_source = _extract_confidence(
            sample=sample,
            metadata=merged_metadata,
            scores=scores,
        )

        views.append(
            BaselineSampleView(
                eval_set_id=eval_set_id,
                run_id=run_id,
                task=task_name,
                sample_id=getattr(sample, "id"),
                sample_uuid=_sample_uuid_text(getattr(sample, "uuid", None)),
                phase=phase,
                agent=agent,
                repeat_id=repeat_id,
                agent_attempt_id=agent_attempt_id,
                scores=scores,
                event_count=len(events),
                event_types=event_types,
                tool_call_count=sum(1 for event_type in event_types if event_type == "ToolEvent"),
                total_time=getattr(sample, "total_time", None),
                working_time=getattr(sample, "working_time", None),
                confidence_value=confidence_value,
                confidence_source=confidence_source,
                metadata=merged_metadata,
            )
        )

    return views, warnings


def _read_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(0, default)
    return max(0, parsed)


def _read_agent(metadata: dict[str, Any], default: str) -> str:
    agent = metadata.get("reliability_agent")
    if isinstance(agent, str) and agent.strip():
        return agent.strip()
    return default


def _normalize_scores(scores: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, score in scores.items():
        normalized[name] = getattr(score, "value", score)
    return normalized


def _sample_uuid_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_confidence(
    *,
    sample: Any,
    metadata: dict[str, Any],
    scores: dict[str, Any],
) -> tuple[float | None, str | None]:
    del sample
    for key in ("reliability_confidence", "confidence", "self_confidence"):
        value = _as_percentage(metadata.get(key))
        if value is not None:
            return value, f"metadata:{key}"

    for key in ("reliability_confidence", "confidence", "self_confidence"):
        value = _as_percentage(scores.get(key))
        if value is not None:
            return value, f"score:{key}"

    return None, None


def _as_percentage(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    if parsed <= 1:
        parsed *= 100.0
    if parsed > 100:
        return None
    return round(parsed, 4)
