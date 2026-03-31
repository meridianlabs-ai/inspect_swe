"""Inspect hooks for reliability sidecar telemetry."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any, cast

from inspect_ai.hooks import (
    Hooks,
    SampleAttemptStart,
    SampleEnd,
    TaskEnd,
    TaskStart,
    hooks,
)
from inspect_ai.hooks._hooks import get_all_hooks
from pydantic import BaseModel

from .artifacts import ReliabilityRecord, SidecarWriter
from .identity import ReliabilityRunIdentity
from .spec import PhaseName


class ReliabilityHookConfig(BaseModel):
    """Runtime config for reliability hook telemetry."""

    enabled: bool = False
    sidecar_path: str | None = None
    strict_identity_tags: bool = True
    repeat_metadata_key: str = "reliability_repeat_id"
    agent_attempt_metadata_key: str = "reliability_agent_attempt_id"
    phase_metadata_key: str = "reliability_phase"


_CONFIG: ContextVar[ReliabilityHookConfig | None] = ContextVar(
    "inspect_swe_reliability_hook_config", default=None
)


def configure_reliability_hooks(config: ReliabilityHookConfig) -> None:
    _CONFIG.set(config)


def disable_reliability_hooks() -> None:
    _CONFIG.set(None)


def _active_config() -> ReliabilityHookConfig | None:
    config = _CONFIG.get()
    if config is not None:
        return config

    # Environment fallback for CLI-driven runs.
    enabled = os.getenv("INSPECT_SWE_RELIABILITY_ENABLED", "").lower() in {
        "1",
        "true",
        "yes",
    }
    sidecar_path = os.getenv("INSPECT_SWE_RELIABILITY_SIDECAR")
    if enabled and sidecar_path:
        return ReliabilityHookConfig(enabled=True, sidecar_path=sidecar_path)
    return None


def assert_reliability_hooks_active(require_enabled: bool = True) -> None:
    """Fail-fast check for hook registration + activation."""
    reliability_hooks = [hook for hook in get_all_hooks() if isinstance(hook, ReliabilityHooks)]
    if not reliability_hooks:
        raise RuntimeError(
            "ReliabilityHooks is not registered. Ensure inspect_swe reliability "
            "hooks are imported via package entrypoint before running reliability jobs."
        )
    if require_enabled and not any(hook.enabled() for hook in reliability_hooks):
        raise RuntimeError(
            "ReliabilityHooks is registered but disabled. Configure hook runtime "
            "state before starting reliability execution."
        )


@hooks(
    name="inspect_swe/reliability",
    description="Collect reliability sidecar telemetry from Inspect sample lifecycle.",
)
class ReliabilityHooks(Hooks):
    """Hook subscriber that emits normalized sidecar records."""

    def __init__(self) -> None:
        self._writer: SidecarWriter | None = None
        self._writer_path: str | None = None
        self._sample_attempt: dict[str, int] = {}
        self._task_name_by_eval_id: dict[str, str] = {}
        self._agent_by_eval_id: dict[str, str] = {}
        self._task_metadata_by_eval_id: dict[str, dict[str, Any]] = {}

    def enabled(self) -> bool:
        config = _active_config()
        return bool(config and config.enabled and config.sidecar_path)

    async def on_task_start(self, data: TaskStart) -> None:
        self._task_name_by_eval_id[data.eval_id] = data.spec.task
        task_args = data.spec.task_args_passed or {}
        agent = cast(str | None, task_args.get("agent"))
        self._agent_by_eval_id[data.eval_id] = agent or data.spec.model
        self._task_metadata_by_eval_id[data.eval_id] = dict(data.spec.metadata or {})

    async def on_task_end(self, data: TaskEnd) -> None:
        self._task_name_by_eval_id.pop(data.eval_id, None)
        self._agent_by_eval_id.pop(data.eval_id, None)
        self._task_metadata_by_eval_id.pop(data.eval_id, None)

    async def on_sample_attempt_start(self, data: SampleAttemptStart) -> None:
        self._sample_attempt[data.sample_id] = data.attempt

    async def on_sample_end(self, data: SampleEnd) -> None:
        config = _active_config()
        if config is None or not config.enabled or not config.sidecar_path:
            return

        writer = self._resolve_writer(config.sidecar_path)
        metadata = dict(self._task_metadata_by_eval_id.get(data.eval_id, {}))
        metadata.update(data.sample.metadata or {})
        phase = self._phase_from_metadata(metadata, config.phase_metadata_key)
        attempt = self._sample_attempt.get(data.sample_id, 1)

        repeat_id = self._read_int_metadata(metadata, config.repeat_metadata_key, 0)
        agent_attempt_id = self._read_int_metadata(
            metadata, config.agent_attempt_metadata_key, 0
        )
        sample_retry_id = max(0, attempt - 1)
        identity_agent = self._identity_agent(metadata, data.eval_id)

        if config.strict_identity_tags:
            missing = [
                key
                for key in (
                    config.repeat_metadata_key,
                    config.agent_attempt_metadata_key,
                    config.phase_metadata_key,
                )
                if key not in metadata
            ]
            if missing:
                metadata = dict(metadata)
                metadata["reliability_identity_warning"] = {
                    "missing_keys": missing,
                    "message": (
                        "Missing explicit reliability identity tags; default indices applied."
                    ),
                }

        identity = ReliabilityRunIdentity(
            eval_set_id=data.eval_set_id or "no_eval_set",
            run_id=data.run_id,
            phase=phase,
            agent=identity_agent,
            task=self._task_name_by_eval_id.get(data.eval_id, "unknown_task"),
            sample_id=data.sample.id,
            sample_uuid=data.sample_id,
            repeat_id=repeat_id,
            sample_retry_id=sample_retry_id,
            agent_attempt_id=agent_attempt_id,
        )

        record = ReliabilityRecord(
            identity=identity,
            outcome=self._outcome_payload(data.sample.scores or {}),
            behavior=self._behavior_payload(data.sample.events),
            resources=self._resource_payload(data.sample.total_time, data.sample.working_time),
            confidence=self._confidence_payload(metadata),
            perturbation=self._select_prefixed(metadata, "reliability_perturbation_"),
            safety=self._select_prefixed(metadata, "reliability_safety_"),
            abstention=self._select_prefixed(metadata, "reliability_abstention_"),
            metadata=metadata,
        )
        writer.write(record)

    def _resolve_writer(self, path: str) -> SidecarWriter:
        if self._writer is None or self._writer_path != path:
            self._writer = SidecarWriter(path)
            self._writer_path = path
        return self._writer

    def _phase_from_metadata(self, metadata: dict[str, Any], key: str) -> PhaseName:
        phase = metadata.get(key, "baseline")
        if phase in {
            "baseline",
            "fault",
            "prompt",
            "structural",
            "safety",
            "abstention",
        }:
            return cast(PhaseName, phase)
        return "baseline"

    def _read_int_metadata(
        self, metadata: dict[str, Any], key: str, default: int
    ) -> int:
        value = metadata.get(key, default)
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, ivalue)

    def _identity_agent(self, metadata: dict[str, Any], eval_id: str) -> str:
        metadata_agent = metadata.get("reliability_agent")
        if isinstance(metadata_agent, str) and metadata_agent.strip():
            return metadata_agent.strip()
        return self._agent_by_eval_id.get(eval_id, "unknown_agent")

    def _outcome_payload(self, scores: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        first_numeric: float | None = None
        for name, score in scores.items():
            value = getattr(score, "value", score)
            values[name] = value
            if first_numeric is None and isinstance(value, (int, float)):
                first_numeric = float(value)
        values["pass"] = first_numeric == 1.0 if first_numeric is not None else None
        return values

    def _behavior_payload(self, events: list[Any]) -> dict[str, Any]:
        event_types = [type(event).__name__ for event in events]
        return {
            "event_count": len(events),
            "event_types": event_types,
            "tool_call_count": sum(1 for t in event_types if t == "ToolEvent"),
        }

    def _resource_payload(
        self, total_time: float | None, working_time: float | None
    ) -> dict[str, Any]:
        return {
            "total_time": total_time,
            "working_time": working_time,
        }

    def _confidence_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        for key in ("reliability_confidence", "confidence", "self_confidence"):
            if key in metadata:
                return {"value": metadata[key], "source_key": key}
        return {}

    def _select_prefixed(
        self, metadata: dict[str, Any], prefix: str
    ) -> dict[str, Any]:
        return {k: v for k, v in metadata.items() if k.startswith(prefix)}
