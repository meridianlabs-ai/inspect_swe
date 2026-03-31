from __future__ import annotations

from pathlib import Path

import pytest
from inspect_swe.reliability import (
    ReliabilityHookConfig,
    ReliabilitySpec,
    build_phase_shards,
    configure_reliability_hooks,
    disable_reliability_hooks,
    preflight_reliability_spec,
)


def test_build_phase_shards_cartesian_product(tmp_path: Path) -> None:
    spec = ReliabilitySpec(
        benchmark="swe_bench_verified",
        agents=["codex_cli", "claude_code"],
        phases=["baseline", "fault"],
    )
    shards = build_phase_shards(spec, tmp_path)
    assert len(shards) == 4
    assert shards[0].shard_index == 0
    assert shards[-1].shard_index == 3
    assert shards[0].phase == "baseline"
    assert shards[0].agent == "codex_cli"
    assert shards[-1].phase == "fault"
    assert shards[-1].agent == "claude_code"


def test_preflight_requires_enabled_hooks_when_configured(tmp_path: Path) -> None:
    spec = ReliabilitySpec(
        benchmark="swe_bench_verified",
        agents=["codex_cli"],
        phases=["baseline"],
        fail_on_missing_hooks=True,
    )

    disable_reliability_hooks()
    with pytest.raises(RuntimeError):
        preflight_reliability_spec(spec)

    configure_reliability_hooks(
        ReliabilityHookConfig(
            enabled=True,
            sidecar_path=str(tmp_path / "records.jsonl"),
        )
    )
    preflight_reliability_spec(spec)
    disable_reliability_hooks()
