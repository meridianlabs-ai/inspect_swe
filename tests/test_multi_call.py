from typing import Literal

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

from tests.conftest import (
    get_available_sandboxes,
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_google,
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


@skip_if_no_google
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_gemini_cli_multi_call(sandbox: str) -> None:
    check_multi_call("gemini_cli", "google/gemini-2.5-pro", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_mini_swe_agent_multi_call_openai(sandbox: str) -> None:
    check_multi_call("mini_swe_agent", "openai/gpt-5-mini", sandbox)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_mini_swe_agent_multi_call_anthropic(sandbox: str) -> None:
    check_multi_call("mini_swe_agent", "anthropic/claude-haiku-4-5", sandbox)


def check_multi_call(
    agent: Literal["claude_code", "codex_cli", "gemini_cli", "mini_swe_agent"],
    model: str,
    sandbox: str,
) -> None:
    log = run_example("multi_call", agent, model, sandbox=sandbox)[0]
    assert log.samples
    sample = log.samples[0]

    user_messages = [m for m in sample.messages if isinstance(m, ChatMessageUser)]
    assistant_messages = [
        m for m in sample.messages if isinstance(m, ChatMessageAssistant)
    ]

    # Codex CLI includes extra scaffold messages in conversation history
    # Gemini CLI may use built-in tools (e.g. google_web_search), causing variable counts
    match agent:
        case "claude_code":
            assert len(user_messages) == 4
            assert len(assistant_messages) == 4
        case "codex_cli":
            assert len(user_messages) == 12
            assert len(assistant_messages) == 4
        case "gemini_cli":
            assert len(user_messages) >= 4
            assert len(assistant_messages) >= 4
        case "mini_swe_agent":
            # model may have more messages due to bash tool use(user/assistant share tool call and tool response)
            assert len(user_messages) >= 4
            assert len(assistant_messages) >= 4
