from __future__ import annotations

from pathlib import Path

import pytest
from inspect_swe.reliability import (
    OrchestratorConcurrency,
    ReliabilityRecord,
    ReliabilityRunIdentity,
    ReliabilitySpec,
    SidecarWriter,
    load_sidecar_records,
)
from inspect_swe.reliability.artifacts import assert_canonical_eval_log_path
from pydantic import ValidationError


def test_reliability_spec_requires_eval_canonical_format() -> None:
    with pytest.raises(ValidationError):
        ReliabilitySpec(
            benchmark="swe_bench_verified",
            agents=["codex_cli"],
            canonical_log_format="json",  # type: ignore[arg-type]
        )


def test_reliability_spec_rejects_duplicate_phases() -> None:
    with pytest.raises(ValidationError):
        ReliabilitySpec(
            benchmark="swe_bench_verified",
            agents=["codex_cli"],
            phases=["baseline", "baseline"],
        )


def test_orchestrator_concurrency_rejects_inprocess_parallel_eval_async() -> None:
    with pytest.raises(ValidationError):
        OrchestratorConcurrency(
            orchestrator_mode="single_process",
            orchestrator_workers=2,
        )


def test_identity_key_keeps_retry_and_attempt_dimensions() -> None:
    identity = ReliabilityRunIdentity(
        eval_set_id="eval-set",
        run_id="run",
        phase="baseline",
        agent="codex_cli",
        task="swe_bench_task",
        sample_id=123,
        sample_uuid="sample-uuid",
        repeat_id=2,
        sample_retry_id=1,
        agent_attempt_id=3,
    )
    key = identity.key()
    assert ":r2:" in key
    assert ":sr1:" in key
    assert key.endswith(":aa3")


def test_sidecar_writer_roundtrip(tmp_path: Path) -> None:
    sidecar = tmp_path / "reliability.jsonl"
    writer = SidecarWriter(sidecar)
    writer.write(
        ReliabilityRecord(
            identity=ReliabilityRunIdentity(
                eval_set_id="e",
                run_id="r",
                phase="baseline",
                agent="codex_cli",
                task="task",
                sample_id=1,
                sample_uuid="uuid",
            ),
            outcome={"pass": True},
        )
    )
    records = load_sidecar_records(sidecar)
    assert len(records) == 1
    assert records[0].outcome["pass"] is True


def test_eval_log_path_must_be_eval() -> None:
    assert_canonical_eval_log_path("results/run.eval")
    with pytest.raises(ValueError):
        assert_canonical_eval_log_path("results/run.json")
