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
    check_mcp("claude_code", "anthropic/claude-sonnet-4-0", "mcp__")


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_mcp() -> None:
    check_mcp("codex_cli", "openai/gpt-5", "mcp__")


@skip_if_no_google
@skip_if_no_docker
def test_gemini_cli_mcp() -> None:
    check_mcp("gemini_cli", "google/gemini-2.5-pro")


def check_mcp(
    agent: Literal["claude_code", "codex_cli", "gemini_cli"],
    model: str,
    prefix: str = "",
) -> None:
    log = run_example("mcp", agent, model)[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    tool_names = {tc.function for tc in tool_calls}

    # Gemini doesn't reliably use MCP tools, so we only assert success above
    if agent != "gemini_cli":
        assert f"{prefix}memory__create_entities" in tool_names
