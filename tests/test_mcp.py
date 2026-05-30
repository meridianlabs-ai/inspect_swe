from typing import Literal

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

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
    check_mcp(
        "claude_code", "anthropic/claude-sonnet-4-0", "mcp__memory__create_entities"
    )


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_mcp() -> None:
    check_mcp("codex_cli", "openai/gpt-5", "create_entities")


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
    messages = log.samples[0].messages
    assistant_messages = [m for m in messages if isinstance(m, ChatMessageAssistant)]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    tool_names = {tc.function for tc in tool_calls}

    # Gemini doesn't reliably use MCP tools, so callers may pass None to skip
    if expected_tool is None:
        return

    # the model must have requested the MCP tool
    assert expected_tool in tool_names, (
        f"Expected '{expected_tool}' in tool calls, got: {sorted(tool_names)}"
    )

    # ...and the tool must have actually executed and returned a result. The
    # MCP server (not inspect_swe) produces the result, so we can't assert an
    # exact value; we confirm a ChatMessageTool came back for the tool with
    # non-empty content. Note: the `error` field is not populated for these
    # bridged CLI agents, so it can't be used to detect failure here.
    tool_results = [
        m
        for m in messages
        if isinstance(m, ChatMessageTool) and m.function == expected_tool
    ]
    assert tool_results, (
        f"No tool result message found for '{expected_tool}' "
        "(tool was requested but never executed?)"
    )
    assert any(m.text.strip() for m in tool_results), (
        f"Tool '{expected_tool}' returned empty content for all results"
    )
