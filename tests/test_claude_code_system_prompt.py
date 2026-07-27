from inspect_swe._claude_code.claude_code import _system_prompt_args


def test_system_prompt_replaces_default() -> None:
    assert _system_prompt_args(["Task prompt", "Agent prompt"], None) == [
        "--system-prompt",
        "Task prompt\n\nAgent prompt",
    ]


def test_system_prompt_can_be_appended() -> None:
    assert _system_prompt_args([], "Additional prompt") == [
        "--append-system-prompt",
        "Additional prompt",
    ]


def test_system_prompt_can_be_replaced_and_appended() -> None:
    assert _system_prompt_args(["Replacement prompt"], "Additional prompt") == [
        "--system-prompt",
        "Replacement prompt",
        "--append-system-prompt",
        "Additional prompt",
    ]


def test_empty_system_prompts_add_no_cli_flags() -> None:
    assert _system_prompt_args([], None) == []
