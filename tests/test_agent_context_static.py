"""Agent-context wiring for the non-delegating agents (root, plus kimi compaction→utility)."""

from typing import Any

from inspect_ai.agent import AgentBridgeContext, current_agent_bridge_context
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe._antigravity.antigravity import build_antigravity_filter
from inspect_swe._kimi_code.kimi_code import (
    _COMPACTION_INSTRUCTION_MARKER,
    build_kimi_filter,
)
from inspect_swe._mini_swe_agent.mini_swe_agent import build_mini_swe_filter
from inspect_swe._util.agentcontext import ModelFilter


async def _invoke(
    wrapped: ModelFilter, messages: list[ChatMessage] | None = None
) -> AgentBridgeContext | None:
    with bridged_request_scope("slug"):
        await wrapped(
            get_model("mockllm/model"),
            messages if messages is not None else [ChatMessageUser(content="hi")],
            [],
            None,
            GenerateConfig(),
        )
        return current_agent_bridge_context()


async def test_kimi_filter_stamps_root() -> None:
    assert await _invoke(build_kimi_filter(None)) == AgentBridgeContext("root")


async def test_mini_swe_filter_stamps_root() -> None:
    assert await _invoke(build_mini_swe_filter(None)) == AgentBridgeContext("root")


async def test_antigravity_filter_stamps_root() -> None:
    assert await _invoke(build_antigravity_filter(None)) == AgentBridgeContext("root")


async def test_kimi_user_filter_sees_root() -> None:
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

    await _invoke(build_kimi_filter(user_filter))
    assert seen["ctx"] == AgentBridgeContext("root")


async def test_kimi_filter_classifies_compaction_request_as_utility() -> None:
    # Synthetic stand-in for kimi's own auto-compaction summarizer request: a
    # plain ChatMessageUser appended as the last message, carrying the
    # compaction instruction's stable opening line (see
    # _COMPACTION_INSTRUCTION_MARKER's provenance comment in kimi_code.py).
    compaction_request: list[ChatMessage] = [
        ChatMessageUser(content="earlier turn"),
        ChatMessageUser(
            content=f"{_COMPACTION_INSTRUCTION_MARKER} to yourself so you can continue."
        ),
    ]
    result = await _invoke(build_kimi_filter(None), compaction_request)
    assert result == AgentBridgeContext("utility")


async def test_kimi_filter_strips_repeat_reminder_and_returns_generate_input() -> None:
    reminder = (
        "<system-reminder>The same tool call has been repeated 3 "
        "times.</system-reminder>"
    )
    messages: list[ChatMessage] = [
        ChatMessageUser(content="hi"),
        ChatMessageTool(content=f"tool output{reminder}", tool_call_id="call_1"),
    ]
    with bridged_request_scope("slug"):
        result = await build_kimi_filter(None)(
            get_model("mockllm/model"), messages, [], None, GenerateConfig()
        )
    assert isinstance(result, GenerateInput)
    assert not any("system-reminder" in m.text for m in result.input)
    assert any("tool output" in m.text for m in result.input)
