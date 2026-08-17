"""Regression tests for antigravity model-tool-declaration confinement.

localharness advertises engine builtin tools (list_resources, read_resource,
manage_task, schedule) in every generateContent request, and no SDK config
(CapabilitiesConfig.enabled_tools / McpServerConfig.disabled_tools / policy)
removes them from the declaration. The antigravity agent therefore strips the
model's tool declaration to an allow-list via a default GenerateFilter. These
tests lock that in: the model must only ever be offered ``call_mcp_tool``.
"""

from __future__ import annotations

import asyncio
import logging

import anyio
import pytest
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo, ToolParams
from inspect_swe._antigravity.antigravity import (
    _ALLOWED_TOOL_NAMES,
    _confine_declared_tools,
    _ModelGenerateFilter,
)

_MODEL = get_model("mockllm/model")


def _tools(names: list[str]) -> list[ToolInfo]:
    return [ToolInfo(name=n, description="x", parameters=ToolParams()) for n in names]


async def _apply_filter(
    filter: _ModelGenerateFilter,
    tools: list[ToolInfo],
) -> ModelOutput | GenerateInput | None:
    return await filter(_MODEL, [], tools, None, GenerateConfig())


def test_confine_strips_engine_tools_to_allowlist() -> None:
    confine = _confine_declared_tools(None)
    tools = _tools(
        [
            "call_mcp_tool",
            "view_file",
            "list_resources",
            "read_resource",
            "manage_task",
            "schedule",
        ]
    )
    result = asyncio.run(_apply_filter(confine, tools))
    assert isinstance(result, GenerateInput)
    assert [tool.name for tool in result.tools] == ["call_mcp_tool", "view_file"]
    assert set(_ALLOWED_TOOL_NAMES) == {"call_mcp_tool", "view_file"}


def test_confine_is_noop_when_already_confined() -> None:
    confine = _confine_declared_tools(None)
    result = asyncio.run(_apply_filter(confine, _tools(["call_mcp_tool", "view_file"])))
    assert result is None


def test_confine_preserves_user_filter_short_circuit() -> None:
    substitute = ModelOutput.from_content("inspect", "short-circuited")

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput:
        return substitute

    confine = _confine_declared_tools(user_filter)
    result = asyncio.run(_apply_filter(confine, _tools(["call_mcp_tool", "schedule"])))
    assert result is substitute


def test_confine_warns_once_on_unrecognized_declared_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SDK drift (a tool outside allowlist + known engine tools) is loud."""
    confined_filter = _confine_declared_tools(None)

    async def _run() -> None:
        tools = _tools(["call_mcp_tool", "brand_new_engine_tool"])
        with caplog.at_level(logging.WARNING):
            first = await confined_filter(_MODEL, [], tools, None, GenerateConfig())
            second = await confined_filter(_MODEL, [], tools, None, GenerateConfig())
        assert isinstance(first, GenerateInput)
        assert [tool.name for tool in first.tools] == ["call_mcp_tool"]
        assert isinstance(second, GenerateInput)
        drift_warnings = [
            record
            for record in caplog.records
            if "brand_new_engine_tool" in record.message
        ]
        assert len(drift_warnings) == 1, "drift warning should fire once per name"

    anyio.run(_run)
