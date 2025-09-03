from typing import Literal

from inspect_ai.model import ChatMessageAssistant

from tests.conftest import run_example, skip_if_no_anthropic, skip_if_no_docker


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_mcp() -> None:
    check_mcp("claude_code", "anthropic/claude-sonnet-4-0")


def check_mcp(agent: Literal["claude_code", "codex_cli"], model: str) -> None:
    log = run_example("mcp", agent, model)[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    assert next(
        (tc for tc in tool_calls if tc.function == "mcp__memory__create_entities"), None
    )
