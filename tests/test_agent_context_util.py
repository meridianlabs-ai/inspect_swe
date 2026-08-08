"""Tests for the shared agent-context classify wrapper."""

import logging
from typing import Any

import pytest
from inspect_ai.agent import (
    AgentBridgeContext,
    current_agent_bridge_context,
)
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.model._model import GenerateInput
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe._util.agentcontext import classify_filter, static_root_classifier


def _args() -> tuple[Model, list[ChatMessage], list[ToolInfo], None, GenerateConfig]:
    return (
        get_model("mockllm/model"),
        [ChatMessageUser(content="hi")],
        [],
        None,
        GenerateConfig(),
    )


async def test_wrapper_sets_context_before_user_filter() -> None:
    seen: dict[str, Any] = {}

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        seen["ctx"] = current_agent_bridge_context()
        return None

    def classify(
        model: Model, messages: list[ChatMessage], tools: list[ToolInfo]
    ) -> AgentBridgeContext:
        return AgentBridgeContext("subagent")

    wrapped = classify_filter(user_filter, classify)
    with bridged_request_scope("slug"):
        result = await wrapped(*_args())
    assert result is None
    assert seen["ctx"] == AgentBridgeContext("subagent")


async def test_wrapper_without_user_filter() -> None:
    wrapped = classify_filter(None, static_root_classifier)
    with bridged_request_scope("slug"):
        await wrapped(*_args())
        assert current_agent_bridge_context() == AgentBridgeContext("root")


async def test_wrapper_supports_legacy_str_filter() -> None:
    seen: dict[str, Any] = {}

    async def legacy_filter(
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        seen["model"] = model
        seen["ctx"] = current_agent_bridge_context()
        return None

    wrapped = classify_filter(legacy_filter, static_root_classifier)
    with bridged_request_scope("slug"):
        await wrapped(*_args())
    # legacy dispatch passes model.name (bare model name, no provider prefix) —
    # matches inspect_ai's bridge dispatch (agent/_bridge/util.py) and kimi's
    # pre-existing combined_filter behavior.
    assert seen["model"] == get_model("mockllm/model").name
    assert seen["ctx"] == AgentBridgeContext("root")


async def test_wrapper_passes_through_user_filter_result() -> None:
    async def rewriting_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        return GenerateInput(
            input=messages, tools=tools, tool_choice=tool_choice, config=config
        )

    wrapped = classify_filter(rewriting_filter, static_root_classifier)
    with bridged_request_scope("slug"):
        result = await wrapped(*_args())
    assert isinstance(result, GenerateInput)


async def test_classify_exceptions_do_not_break_generation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken_classify(
        model: Model, messages: list[ChatMessage], tools: list[ToolInfo]
    ) -> AgentBridgeContext:
        raise RuntimeError("boom")

    wrapped = classify_filter(None, broken_classify)
    with caplog.at_level(logging.WARNING, logger="inspect_swe._util.agentcontext"):
        with bridged_request_scope("slug"):
            result = await wrapped(*_args())  # must not raise
            assert current_agent_bridge_context() == AgentBridgeContext("unknown")
        assert result is None

        # a broken classifier fails on every request; only the first
        # occurrence of a given error should be logged.
        with bridged_request_scope("slug"):
            await wrapped(*_args())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
