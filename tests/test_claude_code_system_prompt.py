from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import AgentState, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import INCORRECT, Score
from inspect_ai.util import ExecCompleted, ExecRemoteProcess, SandboxEnvironment
from inspect_swe import claude_code
from inspect_swe._claude_code import claude_code as claude_code_module
from inspect_swe._claude_code.claude_code import _system_prompt_args


def test_system_prompt_appends_to_default() -> None:
    assert _system_prompt_args(["Task prompt", "Agent prompt"], None) == [
        "--append-system-prompt",
        "Task prompt\n\nAgent prompt",
    ]


def test_system_prompt_can_replace_default() -> None:
    assert _system_prompt_args([], "Replacement prompt") == [
        "--system-prompt",
        "Replacement prompt",
    ]


def test_task_prompt_is_appended_to_replacement() -> None:
    assert _system_prompt_args(["Task prompt"], "Replacement prompt") == [
        "--system-prompt",
        "Replacement prompt",
        "--append-system-prompt",
        "Task prompt",
    ]


def test_empty_system_prompts_add_no_cli_flags() -> None:
    assert _system_prompt_args([], None) == []


def test_append_and_replace_system_prompts_are_mutually_exclusive() -> None:
    with pytest.raises(
        ValueError,
        match="system_prompt and replace_system_prompt cannot both be specified",
    ):
        claude_code(
            system_prompt="Additional prompt",
            replace_system_prompt="Replacement prompt",
        )


class _ExitedProcess(ExecRemoteProcess):
    """A Claude Code launch that exits 0 without output."""

    def __init__(self) -> None:
        self._exited = False

    def __aiter__(self) -> "_ExitedProcess":
        return self

    async def __anext__(self) -> ExecCompleted:
        if self._exited:
            raise StopAsyncIteration
        self._exited = True
        return ExecCompleted(0)


def test_every_launch_appends_the_callers_system_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resumed launch re-sends the caller's system prompt, once.

    Claude Code does not keep --append-system-prompt in its session, so a
    --resume without it runs on Claude Code's default prompt alone. The bridge
    replaces the state's system messages with Claude Code's own prompt after a
    launch; re-sending those would duplicate it.
    """
    commands: list[list[str]] = []

    @asynccontextmanager
    async def bridge(state: AgentState, **kwargs: Any) -> AsyncIterator[Any]:
        async def exec_remote(cmd: list[str], **kwargs: Any) -> ExecRemoteProcess:
            commands.append(cmd)
            state.messages = [
                ChatMessageSystem(content="Claude Code's own prompt\n\nTask prompt"),
                *[m for m in state.messages if not isinstance(m, ChatMessageSystem)],
            ]
            return _ExitedProcess()

        sandbox.exec_remote = exec_remote
        yield SimpleNamespace(state=state, port=3001, mcp_server_configs=[])

    @asynccontextmanager
    async def checkpoint() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            attempt="start", track=lambda name, getter, initial: initial
        )

    sandbox = Mock(spec=SandboxEnvironment)
    module = claude_code_module
    monkeypatch.setattr(module, "sandbox_agent_bridge", bridge)
    monkeypatch.setattr(module, "checkpointer", checkpoint)
    monkeypatch.setattr(module, "sandbox_env", lambda _: sandbox)
    monkeypatch.setattr(
        module, "ensure_agent_binary_installed", AsyncMock(return_value="claude")
    )
    monkeypatch.setattr(
        module, "resolve_agent_cwd", AsyncMock(return_value=str(tmp_path))
    )
    monkeypatch.setattr(module, "_seed_claude_config", AsyncMock())
    monkeypatch.setattr(
        module, "score", AsyncMock(return_value=[Score(value=INCORRECT)])
    )

    agent = claude_code(attempts=2)
    log = eval(
        Task(
            dataset=[
                Sample(
                    input=[
                        ChatMessageSystem(content="Task prompt"),
                        ChatMessageUser(content="Solve it."),
                    ]
                )
            ],
            # two turns of one agent: each turn's attempts relaunch with
            # --resume, and the second turn starts from the bridged state
            solver=[as_solver(agent), as_solver(agent)],
        ),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path),
    )[0]

    assert log.status == "success", log.error
    assert len(commands) == 4
    assert "--resume" in commands[1] and "--resume" in commands[3]
    for command in commands:
        flags = [i for i, arg in enumerate(command) if arg == "--append-system-prompt"]
        assert [command[i + 1] for i in flags] == ["Task prompt"]
