"""Command-line interface for Inspect SWE reliability phases."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Sequence, cast

from .baseline import (
    BaselineExecutionError,
    BaselinePhaseConfig,
    BaselinePhaseResult,
    run_baseline_phase,
)
from .concurrency import OrchestratorConcurrency
from .spec import ReliabilitySpec


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for reliability commands."""
    parser = argparse.ArgumentParser(
        prog="inspect-swe-reliability",
        description="Run reliability phases for inspect_swe.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline",
        help="Run baseline reliability phase.",
        description=(
            "Run K independent baseline repeats and emit sidecar telemetry "
            "alongside canonical Inspect `.eval` logs."
        ),
    )
    baseline.add_argument("--benchmark", required=True, help="Benchmark label.")
    baseline.add_argument(
        "--tasks",
        required=True,
        help="Inspect task path (e.g. inspect_evals/gaia_level1).",
    )
    baseline.add_argument(
        "--agent",
        dest="agents",
        action="append",
        required=True,
        help="Agent identifier (repeat for multi-agent runs).",
    )
    baseline.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent baseline repeats per agent (default: 5).",
    )
    baseline.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Campaign seed recorded in reliability metadata.",
    )
    baseline.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to inspect eval.",
    )
    baseline.add_argument(
        "--task-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task arg pair (repeatable).",
    )
    baseline.add_argument(
        "--inject-agent-task-arg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject task arg `agent=<agent>` for each run.",
    )
    baseline.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata pair (repeatable).",
    )
    baseline.add_argument(
        "--sandbox",
        default=None,
        help="Sandbox setting passed to inspect eval.",
    )
    baseline.add_argument(
        "--limit",
        default=None,
        help="Sample limit (e.g. 3 or 10-20).",
    )
    baseline.add_argument(
        "--sample-id",
        default=None,
        help="Sample id(s): one value or comma-separated list.",
    )
    baseline.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "Campaign identifier used to segregate log/sidecar paths. "
            "Defaults to an auto-generated timestamped id."
        ),
    )
    baseline.add_argument(
        "--log-root",
        default="logs/reliability",
        help="Root directory for eval logs and default sidecars.",
    )
    baseline.add_argument(
        "--sidecar-dir",
        default="reliability_sidecars",
        help="Relative sidecar directory under log root.",
    )
    baseline.add_argument(
        "--sidecar-path",
        default=None,
        help="Explicit sidecar JSONL path (overrides sidecar-dir).",
    )
    baseline.add_argument(
        "--strict-identity-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reliability identity metadata tags.",
    )
    baseline.add_argument(
        "--fail-on-missing-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preflight when reliability hooks are not active.",
    )
    baseline.add_argument(
        "--verify-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate sidecar coverage against eval logs.",
    )
    baseline.add_argument(
        "--fail-on-incomplete-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when telemetry coverage is incomplete.",
    )
    baseline.add_argument(
        "--configure-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure reliability hooks automatically.",
    )
    baseline.add_argument(
        "--orchestrator-mode",
        choices=("single_process", "multi_process"),
        default="multi_process",
        help="Orchestrator policy mode.",
    )
    baseline.add_argument(
        "--orchestrator-workers",
        type=int,
        default=1,
        help="Orchestrator worker count.",
    )
    baseline.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Inspect max_tasks setting.",
    )
    baseline.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Inspect max_samples setting.",
    )
    baseline.add_argument(
        "--max-subprocesses",
        type=int,
        default=None,
        help="Inspect max_subprocesses setting.",
    )
    baseline.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Inspect max_sandboxes setting.",
    )
    baseline.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Inspect max_connections setting.",
    )
    baseline.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    baseline.set_defaults(handler=_handle_baseline_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run reliability CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast("CommandHandler", args.handler)
    try:
        return handler(args)
    except BaselineExecutionError as ex:
        print(f"baseline failed: {ex}", file=sys.stderr)
        return 2
    except ValueError as ex:
        print(f"invalid arguments: {ex}", file=sys.stderr)
        return 2


def _handle_baseline_command(args: argparse.Namespace) -> int:
    spec = ReliabilitySpec(
        benchmark=args.benchmark,
        agents=_unique_agents(args.agents),
        phases=["baseline"],
        seed=args.seed,
        sidecar_dir=args.sidecar_dir,
        strict_identity_tags=args.strict_identity_tags,
        fail_on_missing_hooks=args.fail_on_missing_hooks,
        concurrency=OrchestratorConcurrency(
            orchestrator_mode=args.orchestrator_mode,
            orchestrator_workers=args.orchestrator_workers,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            max_subprocesses=args.max_subprocesses,
            max_sandboxes=args.max_sandboxes,
            max_connections=args.max_connections,
        ),
    )

    config = BaselinePhaseConfig(
        repeats=args.repeats,
        campaign_id=args.campaign_id,
        log_root=args.log_root,
        sidecar_path=args.sidecar_path,
        model=args.model,
        task_args=_parse_key_value_pairs(args.task_arg),
        inject_agent_task_arg=args.inject_agent_task_arg,
        metadata=_parse_key_value_pairs(args.metadata),
        sandbox=args.sandbox,
        limit=_parse_limit_arg(args.limit),
        sample_id=_parse_sample_id_arg(args.sample_id),
        verify_telemetry=args.verify_telemetry,
        fail_on_incomplete_telemetry=args.fail_on_incomplete_telemetry,
        configure_hooks=args.configure_hooks,
    )

    result = run_baseline_phase(
        spec=spec,
        tasks=args.tasks,
        config=config,
    )
    _print_baseline_result(result, json_output=args.json)
    return 0


def _print_baseline_result(result: BaselinePhaseResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print(
        "Baseline reliability run complete: "
        f"benchmark={result.benchmark} repeats={result.repeats} "
        f"campaign_id={result.campaign_id}"
    )
    print(f"Sidecar: {result.sidecar_path}")
    for row in result.results:
        status = "complete" if row.coverage_complete else "incomplete"
        print(
            f"- agent={row.agent} repeat={row.repeat_id} "
            f"coverage={status} logs={len(row.log_paths)} "
            f"missing={len(row.missing_sample_uuids)} "
            f"duplicates={len(row.duplicate_identity_keys)}"
        )


def _parse_key_value_pairs(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected KEY=VALUE pair, received: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key cannot be empty in pair: {raw}")
        parsed[key] = _coerce_scalar(value.strip())
    return parsed


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _parse_limit_arg(value: str | None) -> int | tuple[int, int] | None:
    if value is None:
        return None
    token = value.strip()
    if not token:
        raise ValueError("limit cannot be empty")
    if "-" in token:
        start_raw, end_raw = token.split("-", 1)
        start = int(start_raw.strip())
        end = int(end_raw.strip())
        if start < 1 or end < start:
            raise ValueError("limit range must be in form start-end with 1 <= start <= end")
        return (start, end)
    parsed = int(token)
    if parsed < 1:
        raise ValueError("limit must be >= 1")
    return parsed


def _parse_sample_id_arg(
    value: str | None,
) -> str | int | list[str] | list[int] | list[str | int] | None:
    if value is None:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("sample-id cannot be empty")

    parsed: list[str | int] = []
    for token in tokens:
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            parsed.append(int(token))
        else:
            parsed.append(token)

    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _unique_agents(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        name = raw.strip()
        if not name or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    if not ordered:
        raise ValueError("at least one non-empty --agent value is required")
    return ordered


CommandHandler = Callable[[argparse.Namespace], int]


if __name__ == "__main__":
    raise SystemExit(main())
