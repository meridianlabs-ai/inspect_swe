from typing import Literal

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.tool import Tool, tool
from inspect_swe import codex_cli

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
        "claude_code", "anthropic/claude-sonnet-4-5", "mcp__secrets__secret_lookup"
    )


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_bridged_tools() -> None:
    check_bridged_tools("codex_cli", "openai/gpt-5", "secret_lookup")


@tool
def secret_lookup() -> Tool:
    async def execute(key: str) -> str:
        """Look up a secret value by key.

        Args:
            key: The key to look up.
        """
        return "ALPHA-SECRET-12345"

    return execute


@pytest.mark.slow
@skip_if_no_docker
def test_codex_cli_code_mode_requires_proposal_opt_out() -> None:
    """Codex code mode with a server that requires a proposal fails before launch.

    `model_config` selects a `code_mode_only` catalog entry, so the eval model
    can be a mock: the check runs before Codex sends any request.
    """
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", "unused")],
    )
    task = Task(
        dataset=[Sample(input="Look up the secret for the key 'alpha'.")],
        solver=codex_cli(
            model_config="gpt-5.6-sol",
            bridged_tools=[BridgedToolsSpec(name="secrets", tools=[secret_lookup()])],
        ),
        sandbox="docker",
    )
    log = eval(task, model=model, limit=1, time_limit=300)[0]

    assert log.status == "error"
    assert log.error is not None
    # the message is the exception's repr, so match fragments without quotes
    assert "in code mode" in log.error.message
    assert "secrets" in log.error.message
    assert "require_proposal=False" in log.error.message
    assert log.samples
    assert not any(isinstance(m, ChatMessageAssistant) for m in log.samples[0].messages)


@skip_if_no_google
@skip_if_no_docker
def test_gemini_cli_bridged_tools() -> None:
    check_bridged_tools(
        "gemini_cli", "google/gemini-3.1-pro-preview", "mcp_secrets_secret_lookup"
    )


@skip_if_no_anthropic
@skip_if_no_docker
def test_kimi_code_bridged_tools() -> None:
    check_bridged_tools(
        "kimi_code", "anthropic/claude-sonnet-4-5", "mcp__secrets__secret_lookup"
    )


@skip_if_no_anthropic
@skip_if_no_docker
def test_opencode_bridged_tools() -> None:
    check_bridged_tools(
        "opencode", "anthropic/claude-sonnet-4-5", "secrets_secret_lookup"
    )


def check_bridged_tools(
    agent: Literal["claude_code", "codex_cli", "gemini_cli", "kimi_code", "opencode"],
    model: str,
    expected_tool_name: str,
) -> None:
    log = run_example("bridged_tools", agent, model)[0]
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
