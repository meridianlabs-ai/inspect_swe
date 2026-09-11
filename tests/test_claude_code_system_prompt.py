import os
import subprocess
from pathlib import Path

import anyio
import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from inspect_swe import claude_code
from inspect_swe._claude_code import claude_code as claude_code_module
from inspect_swe._claude_code.claude_code import (
    _centaur_claude_cmd,
    _system_prompt_args,
    _system_texts,
    run_claude_code_centaur,
)
from inspect_swe._util.centaur import CentaurOptions


def test_system_prompt_appends_to_default() -> None:
    assert _system_prompt_args(
        ["Task prompt", "Agent prompt"], None, is_resume=False
    ) == [
        "--append-system-prompt",
        "Task prompt\n\nAgent prompt",
    ]


def test_system_prompt_can_replace_default() -> None:
    assert _system_prompt_args([], "Replacement prompt", is_resume=False) == [
        "--system-prompt",
        "Replacement prompt",
    ]


def test_task_prompt_is_appended_to_replacement() -> None:
    assert _system_prompt_args(
        ["Task prompt"], "Replacement prompt", is_resume=False
    ) == [
        "--system-prompt",
        "Replacement prompt",
        "--append-system-prompt",
        "Task prompt",
    ]


def test_empty_system_prompts_add_no_cli_flags() -> None:
    assert _system_prompt_args([], None, is_resume=False) == []


def test_resume_reapplies_replacement_without_appended_messages() -> None:
    assert _system_prompt_args(
        ["Round-tripped prompt"], "Replacement prompt", is_resume=True
    ) == [
        "--system-prompt",
        "Replacement prompt",
    ]


def test_append_and_replace_system_prompts_are_mutually_exclusive() -> None:
    with pytest.raises(
        ValueError,
        match="system_prompt and replace_system_prompt cannot both be specified",
    ):
        claude_code(
            system_prompt="Additional prompt",
            replace_system_prompt="Replacement prompt",
        )


def test_system_texts_takes_task_messages_then_caller_prompt() -> None:
    messages: list[ChatMessage] = [
        ChatMessageSystem(content="Task prompt"),
        ChatMessageUser(content="Do the thing"),
    ]

    assert _system_texts(messages, "Agent prompt") == ["Task prompt", "Agent prompt"]


def test_system_texts_without_caller_prompt_keeps_only_task_messages() -> None:
    assert _system_texts([ChatMessageSystem(content="Task prompt")], None) == [
        "Task prompt"
    ]


def test_centaur_alias_carries_the_task_and_caller_system_prompts() -> None:
    """Regression: centaur dropped the caller's system prompt entirely.

    The alias was built from ``[claude_binary] + cmd`` while
    ``_system_prompt_args`` was computed only in the unattended branch, so
    ``system_prompt``/``replace_system_prompt`` never reached the operator's
    ``claude`` and a human session silently ran with Claude Code's stock prompt.
    """
    cmd = _centaur_claude_cmd(
        "/usr/bin/claude",
        ["--model", "sonnet"],
        ["Task prompt"],
        None,
    )

    assert cmd == [
        "/usr/bin/claude",
        "--model",
        "sonnet",
        "--append-system-prompt",
        "Task prompt",
    ]


def test_centaur_alias_replaces_the_stock_prompt_when_asked() -> None:
    cmd = _centaur_claude_cmd(
        "/usr/bin/claude",
        ["--model", "sonnet"],
        [],
        "Replacement prompt",
    )

    assert cmd == [
        "/usr/bin/claude",
        "--model",
        "sonnet",
        "--system-prompt",
        "Replacement prompt",
    ]


def test_centaur_alias_adds_no_prompt_flags_when_there_is_no_prompt() -> None:
    assert _centaur_claude_cmd("/usr/bin/claude", ["--model", "sonnet"], [], None) == [
        "/usr/bin/claude",
        "--model",
        "sonnet",
    ]


def test_centaur_claude_resume_omits_appended_messages_but_reapplies_replacement_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`claude --resume` must not append prompts already in the session."""
    captured: dict[str, str] = {}

    async def fake_run_centaur(
        options: CentaurOptions, instructions: str, bashrc: str, state: AgentState
    ) -> None:
        captured["bashrc"] = bashrc

    monkeypatch.setattr(claude_code_module, "run_centaur", fake_run_centaur)

    capture_file = tmp_path / "claude-args"
    claude_binary = tmp_path / "claude"
    claude_binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CLAUDE_CAPTURE"\n')
    claude_binary.chmod(0o755)

    anyio.run(
        lambda: run_claude_code_centaur(
            options=CentaurOptions(),
            claude_cmd=_centaur_claude_cmd(
                str(claude_binary),
                ["--model", "sonnet"],
                ["Task prompt"],
                "Replacement prompt",
            ),
            resume_claude_cmd=_centaur_claude_cmd(
                str(claude_binary),
                ["--model", "sonnet"],
                ["Task prompt"],
                "Replacement prompt",
                is_resume=True,
            ),
            agent_env={},
            state=AgentState(messages=[]),
        )
    )

    bashrc = tmp_path / "bashrc"
    bashrc.write_text(captured["bashrc"])
    shell = tmp_path / "run-claude"
    shell.write_text(
        f'shopt -s expand_aliases\nsource "{bashrc}"\nclaude --resume session-123\n'
    )
    subprocess.run(
        ["bash", str(shell)],
        check=True,
        env={
            "CLAUDE_CAPTURE": str(capture_file),
            "HOME": str(tmp_path / "home"),
            "PATH": os.environ["PATH"],
        },
    )

    assert capture_file.read_text().splitlines() == [
        "--model",
        "sonnet",
        "--system-prompt",
        "Replacement prompt",
        "--resume",
        "session-123",
    ]
