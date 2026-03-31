from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from inspect_ai.log import EvalLog
from inspect_swe.reliability import (
    ReliabilityRecord,
    ReliabilityRunIdentity,
    assess_sidecar_coverage,
)


def _record(sample_uuid: str, repeat_id: int = 0) -> ReliabilityRecord:
    return ReliabilityRecord(
        identity=ReliabilityRunIdentity(
            eval_set_id="set",
            run_id="run",
            phase="baseline",
            agent="codex_cli",
            task="task",
            sample_id=sample_uuid,
            sample_uuid=sample_uuid,
            repeat_id=repeat_id,
        ),
    )


def test_assess_sidecar_coverage_detects_missing_samples() -> None:
    eval_log = cast(
        EvalLog,
        SimpleNamespace(
            samples=[
                SimpleNamespace(uuid="uuid-a"),
                SimpleNamespace(uuid="uuid-b"),
            ]
        ),
    )

    report = assess_sidecar_coverage(eval_log, [_record("uuid-a")])
    assert report.expected_samples == 2
    assert report.observed_records == 1
    assert report.missing_sample_uuids == ["uuid-b"]
    assert report.complete is False


def test_assess_sidecar_coverage_detects_duplicate_identity_keys() -> None:
    eval_log = cast(
        EvalLog,
        SimpleNamespace(samples=[SimpleNamespace(uuid="uuid-a")]),
    )
    record = _record("uuid-a")
    report = assess_sidecar_coverage(eval_log, [record, record])
    assert len(report.duplicate_identity_keys) == 1
