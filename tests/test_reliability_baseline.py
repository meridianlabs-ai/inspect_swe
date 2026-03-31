from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from inspect_ai.log import EvalLog
from inspect_swe.reliability import (
    BaselineExecutionError,
    BaselinePhaseConfig,
    ReliabilitySpec,
    run_baseline_phase,
)
from inspect_swe.reliability.telemetry import TelemetryCoverageReport


class _EvalCallRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[EvalLog]:
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        log = cast(
            EvalLog,
            SimpleNamespace(
                location=f"/tmp/logs/run-{idx}.eval",
                eval=SimpleNamespace(run_id=f"run-{idx}"),
            ),
        )
        return [log]


def test_run_baseline_phase_executes_repeats_and_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _EvalCallRecorder()
    monkeypatch.setattr("inspect_swe.reliability.baseline.eval", recorder)
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline.preflight_reliability_spec", lambda spec: None
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline._assess_repeat_coverage",
        lambda **kwargs: TelemetryCoverageReport(
            expected_samples=1,
            observed_records=1,
            missing_sample_uuids=[],
            duplicate_identity_keys=[],
            identity_warning_count=0,
        ),
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline._default_solver_for_agent",
        lambda agent: None,
    )

    spec = ReliabilitySpec(
        benchmark="swe_bench_verified",
        agents=["codex_cli", "claude_code"],
        phases=["baseline"],
        fail_on_missing_hooks=False,
    )
    result = run_baseline_phase(
        spec=spec,
        tasks="examples/multiple_attempts",
        config=BaselinePhaseConfig(
            repeats=2,
            campaign_id="campaign_a",
            log_root=str(tmp_path),
        ),
    )

    assert len(recorder.calls) == 4
    assert len(result.results) == 4
    first_call = recorder.calls[0]
    assert first_call["log_format"] == "eval"
    assert "task_args" not in first_call
    assert first_call["metadata"]["reliability_phase"] == "baseline"
    assert first_call["metadata"]["reliability_campaign_id"] == "campaign_a"
    assert first_call["metadata"]["reliability_repeat_id"] == 0
    assert first_call["metadata"]["reliability_agent_attempt_id"] == 0
    assert "campaign_a" in first_call["log_dir"]
    assert result.campaign_id == "campaign_a"
    assert "campaign_a" in result.sidecar_path


def test_run_baseline_phase_fails_on_incomplete_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _EvalCallRecorder()
    monkeypatch.setattr("inspect_swe.reliability.baseline.eval", recorder)
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline.preflight_reliability_spec", lambda spec: None
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline._assess_repeat_coverage",
        lambda **kwargs: TelemetryCoverageReport(
            expected_samples=1,
            observed_records=0,
            missing_sample_uuids=["uuid-a"],
            duplicate_identity_keys=[],
            identity_warning_count=0,
        ),
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline._default_solver_for_agent",
        lambda agent: None,
    )

    spec = ReliabilitySpec(
        benchmark="swe_bench_verified",
        agents=["codex_cli"],
        phases=["baseline"],
        fail_on_missing_hooks=False,
    )
    with pytest.raises(BaselineExecutionError):
        run_baseline_phase(
            spec=spec,
            tasks="examples/multiple_attempts",
            config=BaselinePhaseConfig(
                repeats=1,
                log_root=str(tmp_path),
                fail_on_incomplete_telemetry=True,
            ),
        )


def test_run_baseline_phase_uses_generated_campaign_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _EvalCallRecorder()
    monkeypatch.setattr("inspect_swe.reliability.baseline.eval", recorder)
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline.preflight_reliability_spec", lambda spec: None
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline._assess_repeat_coverage",
        lambda **kwargs: TelemetryCoverageReport(
            expected_samples=1,
            observed_records=1,
            missing_sample_uuids=[],
            duplicate_identity_keys=[],
            identity_warning_count=0,
        ),
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline._default_solver_for_agent",
        lambda agent: None,
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.baseline._default_campaign_id",
        lambda: "auto_campaign",
    )

    spec = ReliabilitySpec(
        benchmark="gaia_level1",
        agents=["codex_cli"],
        phases=["baseline"],
        fail_on_missing_hooks=False,
    )
    result = run_baseline_phase(
        spec=spec,
        tasks="inspect_evals/gaia_level1",
        config=BaselinePhaseConfig(repeats=1, log_root=str(tmp_path)),
    )

    assert result.campaign_id == "auto_campaign"
    assert "auto_campaign" in result.sidecar_path
    assert "auto_campaign" in recorder.calls[0]["log_dir"]
