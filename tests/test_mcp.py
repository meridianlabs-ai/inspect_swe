from typing import Literal

import pytest
from inspect_ai.model import ChatMessageAssistant

from tests.conftest import (
    get_available_sandboxes,
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_claude_code_mcp(sandbox: str) -> None:
    check_mcp("claude_code", "anthropic/claude-sonnet-4-0", "mcp__", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_mcp(sandbox: str) -> None:
    check_mcp("codex_cli", "openai/gpt-5", sandbox=sandbox)


def check_mcp(
    agent: Literal["claude_code", "codex_cli"],
    model: str,
    prefix: str = "",
    sandbox: str | None = None,
) -> None:
    log = run_example("mcp", agent, model, sandbox=sandbox)[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    assert next(
        (tc for tc in tool_calls if tc.function == f"{prefix}memory__create_entities"),
        None,
    )
