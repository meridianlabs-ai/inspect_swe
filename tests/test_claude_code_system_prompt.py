import pytest
from inspect_swe import claude_code
from inspect_swe._claude_code.claude_code import _system_prompt_args


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
