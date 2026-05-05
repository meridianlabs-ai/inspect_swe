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
def test_claude_code_mcp() -> None:
    check_mcp("claude_code", "anthropic/claude-sonnet-4-0", "mcp__memory__create_entities")


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_mcp() -> None:
    check_mcp("codex_cli", "openai/gpt-5", "mcp__memory__create_entities")


@skip_if_no_google
@skip_if_no_docker
def test_gemini_cli_mcp() -> None:
    check_mcp("gemini_cli", "google/gemini-3.1-pro-preview", None)


@skip_if_no_anthropic
@skip_if_no_docker
def test_opencode_mcp() -> None:
    check_mcp("opencode", "anthropic/claude-sonnet-4-0", "memory_create_entities")


def check_mcp(
    agent: Literal["claude_code", "codex_cli", "gemini_cli", "opencode"],
    model: str,
    expected_tool: str | None,
) -> None:
    log = run_example("mcp", agent, model)[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    tool_names = {tc.function for tc in tool_calls}

    # Gemini doesn't reliably use MCP tools, so callers may pass None to skip
    if expected_tool is not None:
        assert expected_tool in tool_names, (
            f"Expected '{expected_tool}' in tool calls, got: {sorted(tool_names)}"
        )
