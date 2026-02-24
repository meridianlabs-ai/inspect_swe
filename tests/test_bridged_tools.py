from typing import Literal

from inspect_ai.model import ChatMessageAssistant

from tests.conftest import (
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_google,
    skip_if_no_openai,
)


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_bridged_tools() -> None:
    check_bridged_tools(
        "claude_code", "anthropic/claude-sonnet-4-0", "mcp__secrets__secret_lookup"
    )


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_bridged_tools() -> None:
    check_bridged_tools("codex_cli", "openai/gpt-5", "mcp__secrets__secret_lookup")


@skip_if_no_google
@skip_if_no_docker
def test_gemini_cli_bridged_tools() -> None:
    # Gemini CLI uses plain tool names without mcp__/server__ prefix
    check_bridged_tools("gemini_cli", "google/gemini-2.5-pro", "secret_lookup")


def check_bridged_tools(
    agent: Literal["claude_code", "codex_cli", "gemini_cli"],
    model: str,
    expected_tool_name: str,
) -> None:
    log = run_example("bridged_tools", agent, model)[0]
    assert log.status == "success"
    assert log.samples

    # Verify the bridged tool was called
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]

    # Check that secret_lookup was called
    secret_lookup_call = next(
        (tc for tc in tool_calls if tc.function == expected_tool_name),
        None,
    )
    assert secret_lookup_call is not None, (
        f"Expected {expected_tool_name} tool call, "
        f"found: {[tc.function for tc in tool_calls]}"
    )
