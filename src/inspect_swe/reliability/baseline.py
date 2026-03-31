"""Baseline reliability phase execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inspect_ai import eval
from inspect_ai.log import EvalLog
from pydantic import BaseModel, Field, field_validator

from .artifacts import assert_canonical_eval_log_path, load_sidecar_records
from .hooks import ReliabilityHookConfig, configure_reliability_hooks
from .orchestrator import preflight_reliability_spec
from .spec import ReliabilitySpec
from .telemetry import TelemetryCoverageReport, assess_sidecar_coverage


class BaselineExecutionError(RuntimeError):
    """Raised when baseline execution violates reliability constraints."""


class BaselinePhaseConfig(BaseModel):
    """Options for baseline phase execution."""

    repeats: int = Field(default=5, ge=1)
    campaign_id: str | None = None
    log_root: str = "logs/reliability"
    sidecar_path: str | None = None
    model: str | None = None
    solver: Any | None = None
    task_args: dict[str, Any] = Field(default_factory=dict)
    inject_agent_task_arg: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    sandbox: str | None = None
    limit: int | tuple[int, int] | None = None
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None
    verify_telemetry: bool = True
    fail_on_incomplete_telemetry: bool = True
    configure_hooks: bool = True

    @field_validator("campaign_id")
    @classmethod
    def _validate_campaign_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("campaign_id cannot be empty")
        return cleaned


class BaselineRepeatResult(BaseModel):
    """One baseline repeat execution summary."""

    agent: str
    repeat_id: int
    run_ids: list[str]
    log_paths: list[str]
    coverage_complete: bool
    missing_sample_uuids: list[str]
    duplicate_identity_keys: list[str]
    identity_warning_count: int


class BaselinePhaseResult(BaseModel):
    """Complete baseline phase execution summary."""

    benchmark: str
    repeats: int
    campaign_id: str
    sidecar_path: str
    results: list[BaselineRepeatResult]


def run_baseline_phase(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: BaselinePhaseConfig | None = None,
) -> BaselinePhaseResult:
    """Run K independent baseline repeats for all agents in the spec."""
    config = config or BaselinePhaseConfig()
    if "baseline" not in spec.phases:
        raise BaselineExecutionError(
            "ReliabilitySpec does not include baseline phase; refusing baseline run."
        )

    campaign_id = config.campaign_id or _default_campaign_id()
    sidecar_path = _resolve_sidecar_path(spec, config, campaign_id)
    if config.configure_hooks:
        configure_reliability_hooks(
            ReliabilityHookConfig(
                enabled=True,
                sidecar_path=sidecar_path,
                strict_identity_tags=spec.strict_identity_tags,
            )
        )

    preflight_reliability_spec(spec)

    repeat_results: list[BaselineRepeatResult] = []
    for agent in spec.agents:
        for repeat_id in range(config.repeats):
            logs = _run_single_repeat(
                spec=spec,
                tasks=tasks,
                config=config,
                campaign_id=campaign_id,
                agent=agent,
                repeat_id=repeat_id,
            )

            coverage = _assess_repeat_coverage(
                logs=logs,
                sidecar_path=sidecar_path,
                agent=agent,
                repeat_id=repeat_id,
                phase="baseline",
            )

            if (
                config.verify_telemetry
                and config.fail_on_incomplete_telemetry
                and not coverage.complete
            ):
                raise BaselineExecutionError(
                    "Incomplete reliability telemetry detected for baseline repeat "
                    f"(agent={agent}, repeat_id={repeat_id}): "
                    f"missing_sample_uuids={coverage.missing_sample_uuids}, "
                    f"duplicate_identity_keys={coverage.duplicate_identity_keys}"
                )

            repeat_results.append(
                BaselineRepeatResult(
                    agent=agent,
                    repeat_id=repeat_id,
                    run_ids=[log.eval.run_id for log in logs],
                    log_paths=[log.location for log in logs if log.location],
                    coverage_complete=coverage.complete,
                    missing_sample_uuids=coverage.missing_sample_uuids,
                    duplicate_identity_keys=coverage.duplicate_identity_keys,
                    identity_warning_count=coverage.identity_warning_count,
                )
            )

    return BaselinePhaseResult(
        benchmark=spec.benchmark,
        repeats=config.repeats,
        campaign_id=campaign_id,
        sidecar_path=sidecar_path,
        results=repeat_results,
    )


def _resolve_sidecar_path(
    spec: ReliabilitySpec, config: BaselinePhaseConfig, campaign_id: str
) -> str:
    if config.sidecar_path:
        return config.sidecar_path
    return str(
        Path(config.log_root)
        / spec.sidecar_dir
        / spec.benchmark
        / "baseline"
        / campaign_id
        / "baseline_records.jsonl"
    )


def _run_single_repeat(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: BaselinePhaseConfig,
    campaign_id: str,
    agent: str,
    repeat_id: int,
) -> list[EvalLog]:
    repeat_log_dir = (
        Path(config.log_root)
        / spec.benchmark
        / "baseline"
        / campaign_id
        / agent
        / f"rep_{repeat_id:03d}"
    )

    run_task_args = dict(config.task_args)
    if config.inject_agent_task_arg:
        run_task_args["agent"] = agent

    run_metadata = dict(config.metadata)
    run_metadata.update(
        {
            "reliability_phase": "baseline",
            "reliability_campaign_id": campaign_id,
            "reliability_repeat_id": repeat_id,
            "reliability_agent_attempt_id": 0,
            "reliability_agent": agent,
            "reliability_benchmark": spec.benchmark,
            "reliability_seed": spec.seed,
        }
    )

    eval_kwargs: dict[str, Any] = {
        "tasks": tasks,
        "metadata": run_metadata,
        "log_dir": str(repeat_log_dir),
        "log_format": "eval",
        "score": True,
        "sample_shuffle": False,
        "max_tasks": spec.concurrency.max_tasks,
        "max_samples": spec.concurrency.max_samples,
        "max_subprocesses": spec.concurrency.max_subprocesses,
        "max_sandboxes": spec.concurrency.max_sandboxes,
        "sandbox": config.sandbox,
        "limit": config.limit,
        "sample_id": config.sample_id,
    }
    if run_task_args:
        eval_kwargs["task_args"] = run_task_args
    if config.model is not None:
        eval_kwargs["model"] = config.model
    solver = config.solver or _default_solver_for_agent(agent)
    if solver is not None:
        eval_kwargs["solver"] = solver
    if spec.concurrency.max_connections is not None:
        eval_kwargs["max_connections"] = spec.concurrency.max_connections

    # Strip None values for cleaner call signatures.
    eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
    logs = eval(**eval_kwargs)

    for log in logs:
        if not log.location:
            continue
        assert_canonical_eval_log_path(log.location)
    return logs


def _default_solver_for_agent(agent: str) -> Any | None:
    if agent == "codex_cli":
        from inspect_swe._codex_cli.codex_cli import codex_cli

        return codex_cli()
    if agent == "claude_code":
        from inspect_swe._claude_code.claude_code import claude_code

        return claude_code()
    if agent == "gemini_cli":
        from inspect_swe._gemini_cli.gemini_cli import gemini_cli

        return gemini_cli()
    if agent == "mini_swe_agent":
        from inspect_swe._mini_swe_agent.mini_swe_agent import mini_swe_agent

        return mini_swe_agent()
    return None


def _default_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def _assess_repeat_coverage(
    *,
    logs: list[EvalLog],
    sidecar_path: str,
    agent: str,
    repeat_id: int,
    phase: str,
) -> TelemetryCoverageReport:
    records = load_sidecar_records(sidecar_path)
    filtered = [
        record
        for record in records
        if (
            record.identity.agent == agent
            or record.metadata.get("reliability_agent") == agent
        )
        and record.identity.repeat_id == repeat_id
        and record.identity.phase == phase
    ]

    reports: list[TelemetryCoverageReport] = []
    for log in logs:
        run_filtered = [
            record for record in filtered if record.identity.run_id == log.eval.run_id
        ]
        reports.append(assess_sidecar_coverage(log, run_filtered))

    return _combine_coverage_reports(reports)


def _combine_coverage_reports(
    reports: list[TelemetryCoverageReport],
) -> TelemetryCoverageReport:
    missing: list[str] = []
    duplicates: list[str] = []
    expected = 0
    observed = 0
    warnings = 0
    for report in reports:
        expected += report.expected_samples
        observed += report.observed_records
        missing.extend(report.missing_sample_uuids)
        duplicates.extend(report.duplicate_identity_keys)
        warnings += report.identity_warning_count
    return TelemetryCoverageReport(
        expected_samples=expected,
        observed_records=observed,
        missing_sample_uuids=sorted(set(missing)),
        duplicate_identity_keys=sorted(set(duplicates)),
        identity_warning_count=warnings,
    )
