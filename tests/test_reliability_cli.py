from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import inspect_swe.reliability.cli as cli
from inspect_swe.reliability import BaselinePhaseResult, BaselineRepeatResult


def test_cli_baseline_parses_args_and_invokes_runner(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_baseline_phase(*, spec: Any, tasks: Any, config: Any) -> BaselinePhaseResult:
        captured["spec"] = spec
        captured["tasks"] = tasks
        captured["config"] = config
        return BaselinePhaseResult(
            benchmark=spec.benchmark,
            repeats=config.repeats,
            campaign_id="campaign-test",
            sidecar_path=str(tmp_path / "records.jsonl"),
            results=[
                BaselineRepeatResult(
                    agent="codex_cli",
                    repeat_id=0,
                    run_ids=["run-1"],
                    log_paths=["logs/run-1.eval"],
                    coverage_complete=True,
                    missing_sample_uuids=[],
                    duplicate_identity_keys=[],
                    identity_warning_count=0,
                )
            ],
        )

    monkeypatch.setattr(cli, "run_baseline_phase", _fake_run_baseline_phase)

    code = cli.main(
        [
            "baseline",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--agent",
            "codex_cli",
            "--agent",
            "gemini_cli",
            "--repeats",
            "2",
            "--seed",
            "42",
            "--model",
            "openai/gpt-5.4-2026-03-05",
            "--task-arg",
            "attempts=2",
            "--task-arg",
            "mode=smoke",
            "--metadata",
            "team=reliability",
            "--metadata",
            "dry_run=true",
            "--limit",
            "10-12",
            "--sample-id",
            "1,abc",
            "--log-root",
            "logs/custom",
            "--campaign-id",
            "run-alpha",
            "--sidecar-dir",
            "custom_sidecars",
            "--orchestrator-mode",
            "single_process",
            "--orchestrator-workers",
            "1",
            "--max-samples",
            "1",
            "--max-subprocesses",
            "2",
            "--max-sandboxes",
            "1",
            "--max-connections",
            "5",
            "--no-fail-on-missing-hooks",
            "--no-strict-identity-tags",
            "--no-fail-on-incomplete-telemetry",
        ]
    )

    assert code == 0
    assert captured["tasks"] == "inspect_evals/gaia_level1"
    assert captured["spec"].agents == ["codex_cli", "gemini_cli"]
    assert captured["spec"].fail_on_missing_hooks is False
    assert captured["spec"].strict_identity_tags is False
    assert captured["spec"].concurrency.orchestrator_mode == "single_process"
    assert captured["spec"].concurrency.orchestrator_workers == 1
    assert captured["spec"].concurrency.max_samples == 1
    assert captured["spec"].concurrency.max_subprocesses == 2
    assert captured["spec"].concurrency.max_sandboxes == 1
    assert captured["spec"].concurrency.max_connections == 5
    assert captured["config"].repeats == 2
    assert captured["config"].campaign_id == "run-alpha"
    assert captured["config"].limit == (10, 12)
    assert captured["config"].sample_id == [1, "abc"]
    assert captured["config"].task_args == {"attempts": 2, "mode": "smoke"}
    assert captured["config"].metadata == {"team": "reliability", "dry_run": True}
    assert captured["config"].model == "openai/gpt-5.4-2026-03-05"

    out = capsys.readouterr().out
    assert "Baseline reliability run complete" in out
    assert "coverage=complete" in out


def test_cli_baseline_json_output(monkeypatch: Any, capsys: Any, tmp_path: Path) -> None:
    def _fake_run_baseline_phase(*, spec: Any, tasks: Any, config: Any) -> BaselinePhaseResult:
        return BaselinePhaseResult(
            benchmark=spec.benchmark,
            repeats=config.repeats,
            campaign_id="campaign-test",
            sidecar_path=str(tmp_path / "records.jsonl"),
            results=[],
        )

    monkeypatch.setattr(cli, "run_baseline_phase", _fake_run_baseline_phase)

    code = cli.main(
        [
            "baseline",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "gaia_level1"
    assert payload["repeats"] == 5
    assert payload["results"] == []


def test_cli_baseline_rejects_bad_key_value_pair(capsys: Any) -> None:
    code = cli.main(
        [
            "baseline",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--metadata",
            "missing_separator",
        ]
    )

    assert code == 2
    assert "invalid arguments: expected KEY=VALUE pair" in capsys.readouterr().err
