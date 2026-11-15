from typing import Literal

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import Tool, ToolChoice, ToolInfo, tool
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


# Codex release whose catalog the code-mode tests rely on (gpt-5.6-sol is
# code_mode_only, gpt-5.5 has no tool mode)
_CODEX_VERSION = "0.160.1"


class _CaptureToolNames:
    """A bridge ``GenerateFilter`` that records the first request's tool names."""

    def __init__(self) -> None:
        self.tool_names: list[str] | None = None

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        if self.tool_names is None:
            self.tool_names = [t.name for t in tools]
        return ModelOutput.from_content(str(model), "done")


def _run_codex_code_mode(
    model_config: str,
    require_proposal: bool,
) -> tuple[EvalLog, _CaptureToolNames]:
    """Run codex_cli on a mock model with one bridged server, in Docker.

    `model_config` picks the Codex catalog entry, so the eval model can be a
    mock; the filter answers every request and records the tools Codex
    offered in the first one.
    """
    capture = _CaptureToolNames()
    task = Task(
        dataset=[Sample(input="Look up the secret for the key 'alpha'.")],
        solver=codex_cli(
            model_config=model_config,
            version=_CODEX_VERSION,
            bridged_tools=[
                BridgedToolsSpec(
                    name="secrets",
                    tools=[secret_lookup()],
                    require_proposal=require_proposal,
                )
            ],
            filter=capture,
        ),
        sandbox="docker",
    )
    log = eval(task, model="mockllm/model", limit=1, time_limit=300)[0]
    return log, capture


def _assert_fails_before_launch(log: EvalLog) -> None:
    assert log.status == "error"
    assert log.error is not None
    # the message is the exception's repr, so match fragments without quotes
    assert "in code mode" in log.error.message
    assert "secrets" in log.error.message
    assert "require_proposal=False" in log.error.message
    assert log.samples
    assert not any(isinstance(m, ChatMessageAssistant) for m in log.samples[0].messages)


@pytest.mark.slow
@skip_if_no_docker
def test_codex_cli_code_mode_requires_proposal_opt_out() -> None:
    """Codex code mode with a server that requires a proposal fails before launch."""
    log, capture = _run_codex_code_mode("gpt-5.6-sol", require_proposal=True)
    _assert_fails_before_launch(log)
    assert capture.tool_names is None

    # with the opt-out Codex runs, and offers the tool only inside exec
    log, capture = _run_codex_code_mode("gpt-5.6-sol", require_proposal=False)
    assert log.status == "success"
    assert capture.tool_names is not None
    assert "exec" in capture.tool_names
    assert not any("secret_lookup" in name for name in capture.tool_names)


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
