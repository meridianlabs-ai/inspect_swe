from inspect_swe._claude_code.claude_code import (
    claude_code_command,
    claude_code_effort_args,
)


def test_effort_adds_cli_flag_when_configured() -> None:
    assert claude_code_effort_args("max") == ["--effort", "max"]


def test_effort_adds_no_cli_flag_when_unconfigured() -> None:
    assert claude_code_effort_args(None) == []


def test_effort_is_present_in_claude_code_command_after_model() -> None:
    assert claude_code_command(
        ["--permission-mode", "auto"], "claude-opus-5", "max"
    ) == [
        "--permission-mode",
        "auto",
        "--model",
        "claude-opus-5",
        "--effort",
        "max",
    ]
