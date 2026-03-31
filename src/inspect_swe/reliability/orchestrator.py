"""Reliability campaign orchestration primitives.

This module intentionally focuses on preflight and sharding policy first:
- validates canonical storage contract (`.eval` logs),
- validates concurrency policy constraints,
- validates hook readiness and telemetry requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .concurrency import validate_orchestrator_policy
from .hooks import assert_reliability_hooks_active
from .spec import PhaseName, ReliabilitySpec


@dataclass(frozen=True)
class PhaseShard:
    """One campaign execution shard."""

    benchmark: str
    phase: PhaseName
    agent: str
    shard_index: int
    output_dir: str


def preflight_reliability_spec(spec: ReliabilitySpec) -> None:
    """Fail-fast validation for reliability runs."""
    if spec.canonical_log_format != "eval":
        raise ValueError(
            "Only Inspect `.eval` canonical logs are supported for reliability execution."
        )
    validate_orchestrator_policy(spec.concurrency)
    if spec.fail_on_missing_hooks:
        assert_reliability_hooks_active(require_enabled=True)


def build_phase_shards(spec: ReliabilitySpec, output_root: str | Path) -> list[PhaseShard]:
    """Create deterministic phase shards from a reliability spec."""
    root = Path(output_root)
    shards: list[PhaseShard] = []
    shard_index = 0
    for phase in spec.phases:
        for agent in spec.agents:
            output_dir = root / spec.benchmark / phase / agent
            shards.append(
                PhaseShard(
                    benchmark=spec.benchmark,
                    phase=phase,
                    agent=agent,
                    shard_index=shard_index,
                    output_dir=str(output_dir),
                )
            )
            shard_index += 1
    return shards
