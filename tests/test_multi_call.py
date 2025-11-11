from typing import Literal

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

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
def test_claude_code_multi_call(sandbox: str) -> None:
    check_multi_call("claude_code", "anthropic/claude-sonnet-4-0", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_multi_call(sandbox: str) -> None:
    check_multi_call("codex_cli", "openai/gpt-5", sandbox)


def check_multi_call(
    agent: Literal["claude_code", "codex_cli"], model: str, sandbox: str
) -> None:
    log = run_example("multi_call", agent, model, sandbox=sandbox)[0]
    assert log.samples
    sample = log.samples[0]

    user_messages = [m for m in sample.messages if isinstance(m, ChatMessageUser)]
    assistant_messages = [
        m for m in sample.messages if isinstance(m, ChatMessageAssistant)
    ]

    assert len(user_messages) == 4
    assert len(assistant_messages) == 4
