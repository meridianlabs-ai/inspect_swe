from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from importlib import import_module
from subprocess import Popen
from sys import executable
from typing import TypeVar, cast, final
from unittest.mock import patch

import anyio
import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageUser
from inspect_ai.util import SandboxEnvironment
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions
from inspect_ai.util._sandbox import exec_remote as core_exec_remote
from inspect_ai.util._subprocess import ExecResult
from inspect_swe._antigravity.antigravity import antigravity
from inspect_swe._codex_cli.codex_cli import codex_cli
from pydantic import BaseModel

T = TypeVar("T")


@final
class _FakeCheckpointer:
    attempt: str = ""

    async def __aenter__(self) -> "_FakeCheckpointer":
        return self

    async def __aexit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        return None

    def track(self, _key: str, _value: object, default: T) -> T:
        return default


@final
class _FakeBridge:
    state: AgentState
    port: int
    mcp_server_configs: list[object]

    def __init__(self, state: AgentState) -> None:
        self.state = state
        self.port = 3001
        self.mcp_server_configs = []


@final
class _FakeStore:
    values: dict[str, int]

    def __init__(self) -> None:
        self.values = {}

    def get(self, key: str, default: int) -> int:
        return self.values.get(key, default)

    def set(self, key: str, value: int) -> None:
        self.values[key] = value


ModelT = TypeVar("ModelT", bound=BaseModel)


@final
class _CoreKillSandbox:
    _tools_user: str | None
    process: Popen[bytes] | None
    exit_code: int | None
    kill_rpc_count: int

    def __init__(self) -> None:
        self._tools_user = None
        self.process = None
        self.exit_code = None
        self.kill_rpc_count = 0

    @contextmanager
    def no_events(self) -> Generator[None, None, None]:
        yield

    async def write_file(self, _path: str, _content: str) -> None:
        return None

    async def exec(self, cmd: list[str], **_: object) -> ExecResult[str]:
        assert cmd
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    async def exec_remote(
        self,
        cmd: list[str],
        *,
        options: ExecRemoteAwaitableOptions,
        stream: bool,
    ) -> ExecResult[str]:
        assert stream is False
        # A deadline the test owns, far above the sub-second timeouts these cases
        # exercise, so it only trips when the production bounding is missing. Without
        # it a regression to a bare await hangs the suite instead of failing it, and
        # leaves the real subprocess alive past `cleanup()`.
        with anyio.fail_after(30):
            return await core_exec_remote.exec_remote_awaitable(
                cast(SandboxEnvironment, self),
                cmd,
                sandbox_default_poll_interval=0.001,
                options=options,
            )

    async def exec_model_request(
        self,
        *,
        method: str,
        params: dict[str, object],
        result_type: type[ModelT],
        **_kwargs: object,
    ) -> ModelT:
        if method == "exec_remote_start":
            assert params["command"]
            self.process = Popen([executable, "-c", "import time; time.sleep(60)"])
            return result_type.model_validate({"pid": self.process.pid})
        if method == "exec_remote_poll":
            return result_type.model_validate(
                {"state": "running", "seq": 0, "stdout": "", "stderr": ""}
            )
        if method == "exec_remote_kill":
            assert self.process is not None
            self.kill_rpc_count += 1
            self.process.kill()
            self.exit_code = self.process.wait(timeout=1)
            return result_type.model_validate({"seq": 0, "stdout": "", "stderr": ""})
        raise AssertionError(f"Unexpected RPC method: {method}")

    def cleanup(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=1)


@final
class _FastSandbox:
    timeout: float | None

    def __init__(self) -> None:
        self.timeout = None

    async def write_file(self, _path: str, _content: str) -> None:
        return None

    async def exec(self, cmd: list[str], **_: object) -> ExecResult[str]:
        assert cmd
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    async def exec_remote(
        self,
        cmd: list[str],
        *,
        options: ExecRemoteAwaitableOptions,
        stream: bool,
    ) -> ExecResult[str]:
        assert cmd
        assert stream is False
        self.timeout = options.timeout
        return ExecResult(
            success=True,
            returncode=0,
            stdout="stdout-marker",
            stderr="stderr-marker",
        )


def _agent_state() -> AgentState:
    return AgentState(messages=[ChatMessageUser(content="Solve the task.")])


@asynccontextmanager
async def _sandbox_agent_bridge(
    state: AgentState,
    **_kwargs: object,
) -> AsyncGenerator[_FakeBridge, None]:
    yield _FakeBridge(state)


async def _ensure_binary(*_args: object, **_kwargs: object) -> str:
    return "codex"


async def _resolve_cwd(*_args: object, **_kwargs: object) -> str:
    return "/workdir"


async def _resolve_model(*_args: object, **_kwargs: object) -> str:
    return "gpt-5"


def _run_codex(
    sandbox: _CoreKillSandbox | _FastSandbox,
    *,
    exec_timeout: float | None = None,
    debug: bool = False,
) -> AgentState:
    from inspect_swe._codex_cli import codex_cli as codex_module

    with (
        patch.object(codex_module, "checkpointer", return_value=_FakeCheckpointer()),
        patch.object(codex_module, "sandbox_agent_bridge", _sandbox_agent_bridge),
        patch.object(codex_module, "sandbox_env", return_value=sandbox),
        patch.object(codex_module, "ensure_agent_binary_installed", _ensure_binary),
        patch.object(codex_module, "resolve_agent_cwd", _resolve_cwd),
        patch.object(codex_module, "resolve_codex_model", _resolve_model),
        patch.object(codex_module, "store", return_value=_FakeStore()),
    ):
        if exec_timeout is None:
            agent = codex_cli(debug=debug)
        else:
            agent = codex_cli(exec_timeout=exec_timeout, debug=debug)
        return anyio.run(agent, _agent_state())


def test_codex_exec_timeout_kills_real_core_process() -> None:
    sandbox = _CoreKillSandbox()

    try:
        with (
            patch.object(
                core_exec_remote, "exec_model_request", sandbox.exec_model_request
            ),
            pytest.raises(
                RuntimeError,
                match="Codex CLI execution timed out after 0.01 seconds",
            ),
        ):
            _ = _run_codex(sandbox, exec_timeout=0.01)

        assert sandbox.kill_rpc_count == 1
        assert sandbox.exit_code == -9
    finally:
        sandbox.cleanup()


def test_codex_debug_trace_includes_cli_output() -> None:
    sandbox = _FastSandbox()
    traced: list[str] = []

    with patch("inspect_swe._codex_cli.codex_cli.trace", traced.append):
        _ = _run_codex(sandbox, debug=True)

    assert traced == ["Codex CLI Debug Output:\nstdout-marker\nstderr-marker"]


def test_codex_exec_timeout_leaves_fast_cli_invocation_unchanged() -> None:
    sandbox = _FastSandbox()

    result = _run_codex(sandbox)

    assert result.messages[-1].text == "Solve the task."
    assert sandbox.timeout == 1800.0


async def _ensure_antigravity_sdk(*_args: object, **_kwargs: object) -> str:
    return "python"


def _run_antigravity(
    sandbox: _CoreKillSandbox | _FastSandbox,
    *,
    exec_timeout: float | None = None,
) -> AgentState:
    antigravity_module = import_module("inspect_swe._antigravity.antigravity")

    with (
        patch.object(antigravity_module, "sandbox_agent_bridge", _sandbox_agent_bridge),
        patch.object(antigravity_module, "sandbox_env", return_value=sandbox),
        patch.object(
            antigravity_module, "ensure_antigravity_sdk", _ensure_antigravity_sdk
        ),
        patch.object(antigravity_module, "resolve_agent_cwd", _resolve_cwd),
        patch.object(antigravity_module, "store", return_value=_FakeStore()),
    ):
        agent = (
            antigravity()
            if exec_timeout is None
            else antigravity(exec_timeout=exec_timeout)
        )
        return anyio.run(agent, _agent_state())


def test_antigravity_exec_timeout_kills_real_core_process() -> None:
    sandbox = _CoreKillSandbox()

    try:
        with (
            patch.object(
                core_exec_remote, "exec_model_request", sandbox.exec_model_request
            ),
            pytest.raises(
                RuntimeError,
                match="Antigravity CLI execution timed out after 0.01 seconds",
            ),
        ):
            _ = _run_antigravity(sandbox, exec_timeout=0.01)

        assert sandbox.kill_rpc_count == 1
        assert sandbox.exit_code == -9
    finally:
        sandbox.cleanup()
