"""Baseline reliability phase execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inspect_ai.agent import as_solver, is_agent
from inspect_ai import eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic import BaseModel, Field, field_validator

from .concurrency import validate_orchestrator_policy
from .spec import ReliabilitySpec
from .telemetry import TelemetryCoverageReport, assess_eval_coverage


class BaselineExecutionError(RuntimeError):
    """Raised when baseline execution violates reliability constraints."""


def assert_canonical_eval_log_path(path: str | Path) -> None:
    """Validate that a log path points to Inspect native `.eval` logs."""
    if Path(path).suffix != ".eval":
        raise ValueError(
            f"expected Inspect .eval log path, received: {path!s}. "
            "Reliability execution truth must come from `.eval` logs."
        )


def preflight_reliability_spec(spec: ReliabilitySpec) -> None:
    """Fail-fast validation for reliability runs."""
    if spec.canonical_log_format != "eval":
        raise ValueError(
            "Only Inspect `.eval` canonical logs are supported for reliability execution."
        )
    validate_orchestrator_policy(spec.concurrency)


class BaselinePhaseConfig(BaseModel):
    """Options for baseline phase execution."""

    repeats: int = Field(default=5, ge=1)
    campaign_id: str | None = None
    log_root: str = "logs/reliability"
    model: str | None = None
    solver: Any | None = None
    task_args: dict[str, Any] = Field(default_factory=dict)
    inject_agent_task_arg: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    sandbox: str | None = None
    limit: int | tuple[int, int] | None = None
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None
    compute_confidence: bool = True
    verify_telemetry: bool = True
    fail_on_incomplete_telemetry: bool = True

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
                agent=agent,
                repeat_id=repeat_id,
                phase="baseline",
                strict_identity_tags=spec.strict_identity_tags,
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
        results=repeat_results,
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
    default_solver = _default_solver_for_agent(agent)
    solver = config.solver or default_solver
    if config.compute_confidence:
        solver = _wrap_solver_with_confidence(
            solver if solver is not None else default_solver
        )
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

        return codex_cli(version="0.118.0")
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


def _wrap_solver_with_confidence(base_solver: Any | None) -> Solver | Any | None:
    if base_solver is None:
        return None
    wrapped = as_solver(base_solver) if is_agent(base_solver) else base_solver

    @solver(name="baseline_confidence_solver")
    def confidence_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state = await wrapped(state, generate)
            confidence = await _compute_confidence_with_same_model(state)
            if confidence is not None:
                metadata = dict(getattr(state, "metadata", {}) or {})
                metadata["reliability_confidence"] = confidence
                metadata["reliability_confidence_source"] = "same_model_followup"
                state.metadata = metadata
            return state

        return solve

    return confidence_solver()


async def _compute_confidence_with_same_model(state: TaskState) -> float | None:
    messages = list(getattr(state, "messages", []) or [])
    if not messages:
        return None

    prompt = ChatMessageUser(
        content=(
            "Review the full conversation above, including any tool use and intermediate steps.\n"
            "Estimate confidence that the final submitted answer is correct.\n"
            "Return only one number between 0 and 100.\n"
            "No explanation, no units, no extra text.\n\n"
            "Confidence (0-100):"
        )
    )
    confidence_messages = messages + [prompt]

    _log_confidence_prompt_messages(confidence_messages)

    model = get_model()
    confidence_output = await model.generate(confidence_messages)
    value = _parse_confidence_value(getattr(confidence_output, "completion", None))
    return value


def _parse_confidence_value(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    token = text.split()[0].strip().rstrip("%")
    try:
        parsed = float(token)
    except ValueError:
        return None
    if parsed < 0 or parsed > 100:
        return None
    return round(parsed, 4)


def _log_confidence_prompt_messages(messages: list[Any]) -> None:
    payload = [_message_for_log(message) for message in messages]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    log_path = Path.cwd() / f"src_inspect_swe_{digest}.log"
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _message_for_log(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    return {
        "role": role if isinstance(role, str) else str(role),
        "content": _content_for_log(content),
    }


def _content_for_log(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[Any] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append({"text": text})
            elif isinstance(item, dict):
                parts.append(item)
            else:
                parts.append(str(item))
        return parts
    return content if isinstance(content, (dict, int, float, bool)) else str(content)


def _default_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def _assess_repeat_coverage(
    *,
    logs: list[EvalLog],
    agent: str,
    repeat_id: int,
    phase: str,
    strict_identity_tags: bool,
) -> TelemetryCoverageReport:
    reports: list[TelemetryCoverageReport] = []
    for log in logs:
        reports.append(
            assess_eval_coverage(
                log,
                expected_phase=phase,
                expected_agent=agent,
                expected_repeat_id=repeat_id,
                strict_identity_tags=strict_identity_tags,
            )
        )

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
