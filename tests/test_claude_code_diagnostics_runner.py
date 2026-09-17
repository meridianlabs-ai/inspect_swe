import importlib
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import anyio
import pytest
from inspect_ai import Task, eval_async
from inspect_ai.agent import AgentState, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import ExecCompleted, ExecStderr, ExecStdout
from inspect_swe import claude_code
from pydantic import TypeAdapter


def _stdout(record: object) -> ExecStdout:
    return ExecStdout(data=TypeAdapter(object).dump_json(record).decode() + "\n")


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invocations: list[list[object] | BaseException],
    retry_uncaught_errors: int | None,
) -> tuple[EvalLog, int, list[BaseException]]:
    module = importlib.import_module("inspect_swe._claude_code.claude_code")
    launches: list[object] = []
    failures: list[BaseException] = []

    async def process(events: list[object]) -> AsyncIterator[object]:
        for event in events:
            if isinstance(event, BaseException):
                raise event
            yield event

    async def launch(**kwargs: Any) -> Any:
        events = invocations[len(launches)]
        launches.append(kwargs)
        if isinstance(events, BaseException):
            raise events
        return process(events)

    @asynccontextmanager
    async def bridge(state: AgentState, **kwargs: Any) -> AsyncIterator[Any]:
        yield SimpleNamespace(state=state, port=3001, mcp_server_configs=[])

    monkeypatch.setattr(module, "sandbox_agent_bridge", bridge)
    monkeypatch.setattr(
        module, "sandbox_env", lambda _: SimpleNamespace(exec_remote=launch)
    )
    monkeypatch.setattr(
        module, "ensure_agent_binary_installed", AsyncMock(return_value="/claude")
    )
    monkeypatch.setattr(
        module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")
    )
    monkeypatch.setattr(module, "_seed_claude_config", AsyncMock())

    @solver
    def record_failures() -> Solver:
        agent_solver = as_solver(
            claude_code(
                debug=False,
                retry_uncaught_errors=retry_uncaught_errors,
            )
        )

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            try:
                return await agent_solver(state, generate)
            except BaseException as ex:
                failures.append(ex)
                raise

        return solve

    async def run() -> list[EvalLog]:
        return await eval_async(
            Task(dataset=[Sample(input="test")], solver=record_failures()),
            model="mockllm/model",
            log_dir=str(tmp_path),
        )

    logs = anyio.run(run)
    return logs[0], len(launches), failures


@pytest.mark.parametrize("exit_code", [1, 2, 137])
def test_failure_preserves_sanitized_results_without_debug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exit_code: int
) -> None:
    log, launches, failures = _run(
        monkeypatch,
        tmp_path,
        [
            [
                _stdout(
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "errors": [{"type": "api_error", "message": "SECRET"}],
                        "result": "SECRET",
                    }
                ),
                ExecStderr(data="[claude-code:unrecognized_model]"),
                ExecCompleted(exit_code=exit_code),
            ]
        ],
        3,
    )
    assert log.status == "error"
    assert log.samples is not None
    sample = log.samples[0]
    assert sample.error is not None
    assert (
        f"agent {exit_code}: [claude-code:unrecognized_model]" in sample.error.message
    )
    assert "error_during_execution" in sample.error.message
    assert "api_error" in sample.error.message
    assert "SECRET" not in sample.error.message
    assert (
        "error_during_execution" in TypeAdapter(object).dump_json(sample.store).decode()
    )
    assert "SECRET" not in TypeAdapter(object).dump_json(sample.store).decode()
    assert launches == 1


def test_retry_resets_diagnostics_and_does_not_change_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log, launches, failures = _run(
        monkeypatch,
        tmp_path,
        [
            [
                _stdout({"type": "error", "error": {"type": "api_error"}}),
                ExecCompleted(exit_code=1),
            ],
            [ExecCompleted(exit_code=1)],
        ],
        1,
    )
    assert launches == 2
    assert log.error is not None
    assert '"result_available":false' in log.error.message
    assert '"error_records":0' in log.error.message
    assert "api_error" not in log.error.message


@pytest.mark.parametrize("record", ["SECRET", None, ["SECRET"]])
def test_non_object_jsonl_is_counted_without_masking_exit_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record: object,
) -> None:
    log, launches, failures = _run(
        monkeypatch,
        tmp_path,
        [
            [
                _stdout(record),
                ExecStdout(data="{invalid SECRET\n"),
                ExecCompleted(exit_code=2),
            ]
        ],
        0,
    )
    assert log.error is not None
    assert "agent 2:" in log.error.message
    assert '"malformed_records":2' in log.error.message
    assert "SECRET" not in log.error.message
    assert launches == 1


def test_success_preserves_summary_without_introducing_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log, launches, failures = _run(
        monkeypatch,
        tmp_path,
        [
            [
                _stdout({"type": "result", "subtype": "success", "is_error": False}),
                ExecCompleted(exit_code=0),
            ]
        ],
        3,
    )
    assert log.status == "success"
    assert log.error is None
    assert log.samples is not None
    assert any(value == "success" for value in log.samples[0].store.values())
    assert launches == 1


def test_stream_exception_preserves_diagnostics_and_original_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log, launches, failures = _run(
        monkeypatch,
        tmp_path,
        [
            [
                _stdout({"type": "error", "error": {"type": "api_error"}}),
                RuntimeError("transport interrupted"),
            ]
        ],
        3,
    )
    assert log.error is not None
    assert "transport interrupted" in log.error.message
    assert log.samples is not None
    assert "api_error" in TypeAdapter(object).dump_json(log.samples[0].store).decode()
    assert launches == 1


def test_cancellation_is_not_retried_or_converted_to_cli_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asyncio

    log, launches, failures = _run(
        monkeypatch,
        tmp_path,
        [
            [
                _stdout({"type": "error", "error": {"type": "api_error"}}),
                asyncio.CancelledError(),
            ]
        ],
        3,
    )
    assert len(failures) == 1
    assert isinstance(failures[0], asyncio.CancelledError)
    assert log.error is None
    assert launches == 1


def test_launch_failure_does_not_reuse_previous_invocation_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log, launches, failures = _run(
        monkeypatch,
        tmp_path,
        [
            [
                _stdout({"type": "error", "error": {"type": "api_error"}}),
                ExecCompleted(exit_code=1),
            ],
            RuntimeError("launch unavailable"),
        ],
        1,
    )
    assert launches == 2
    assert len(failures) == 1
    assert log.error is not None
    assert "launch unavailable" in log.error.message
    assert log.samples is not None
    summary = TypeAdapter(object).dump_json(log.samples[0].store).decode()
    assert "api_error" not in summary
    assert any(
        key.endswith("error_records") and value == 0
        for key, value in log.samples[0].store.items()
    )
