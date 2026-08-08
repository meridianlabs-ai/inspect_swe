"""Static-root agent context wiring for non-delegating agents."""

from typing import Any

from inspect_ai.agent import AgentBridgeContext, current_agent_bridge_context
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe._antigravity.antigravity import build_antigravity_filter
from inspect_swe._kimi_code.kimi_code import build_kimi_filter
from inspect_swe._mini_swe_agent.mini_swe_agent import build_mini_swe_filter
from inspect_swe._util.agentcontext import ModelFilter


async def _invoke(wrapped: ModelFilter) -> AgentBridgeContext | None:
    with bridged_request_scope("slug"):
        await wrapped(
            get_model("mockllm/model"),
            [ChatMessageUser(content="hi")],
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
