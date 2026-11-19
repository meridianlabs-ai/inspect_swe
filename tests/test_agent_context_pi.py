from functools import partial
from typing import Any

import anyio
import pytest
from inspect_ai.agent import AgentBridgeContext, current_agent_bridge_context
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo, ToolParams
from inspect_swe._pi.pi import (
    _SUMMARY_SYSTEM_PREFIX,
    PiEventSink,
    pi_classifier,
    pi_request_filter,
)
from inspect_swe._util.download import download_text_file


def model_for_test() -> Model:
    return get_model(
        model="openai/example", api_key="dummy", base_url="http://localhost:1/v1"
    )


def summary_messages() -> list[ChatMessage]:
    return [
        ChatMessageSystem(content=_SUMMARY_SYSTEM_PREFIX),
        ChatMessageUser(content="<conversation>\nuser: hello\n</conversation>"),
    ]


def test_pi_classifier() -> None:
    model = model_for_test()
    summary = summary_messages()
    assert pi_classifier(model=model, messages=summary, tools=[]).kind == "utility"
    negatives: list[list[ChatMessage]] = [
        [],
        [summary[1]],
        [ChatMessageSystem(content="regular prompt"), summary[1]],
        [summary[0], ChatMessageUser(content="prefix <conversation>\ntext")],
        [
            summary[0],
            ChatMessageTool(content=summary[1].text, tool_call_id="1", function="read"),
        ],
        summary + [ChatMessageUser(content="extra")],
    ]
    for messages in negatives:
        assert pi_classifier(model=model, messages=messages, tools=[]).kind == "root"
    tool = ToolInfo(name="read", description="read", parameters=ToolParams())
    assert pi_classifier(model=model, messages=summary, tools=[tool]).kind == "root"


@pytest.mark.anyio
async def test_user_filter_sees_pi_context() -> None:
    seen: list[AgentBridgeContext | None] = []

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> None:
        seen.append(current_agent_bridge_context())

    wrapped: Any = pi_request_filter(user_filter=user_filter, sink=PiEventSink())
    with bridged_request_scope("slug"):
        result = await wrapped(
            model=model_for_test(),
            messages=summary_messages(),
            tools=[],
            tool_choice=None,
            config=GenerateConfig(),
        )
    assert result is None
    assert seen == [AgentBridgeContext("utility")]


@pytest.mark.anyio
async def test_no_user_filter_sets_context() -> None:
    wrapped: Any = pi_request_filter(user_filter=None, sink=PiEventSink())
    with bridged_request_scope("slug"):
        assert (
            await wrapped(
                model=model_for_test(),
                messages=summary_messages(),
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is None
        )
        assert current_agent_bridge_context() == AgentBridgeContext("utility")


@pytest.mark.anyio
async def test_legacy_filter_receives_model_name() -> None:
    seen: list[tuple[str, AgentBridgeContext | None]] = []

    async def legacy_filter(
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> None:
        seen.append((model, current_agent_bridge_context()))

    with pytest.warns(DeprecationWarning):
        wrapped: Any = pi_request_filter(user_filter=legacy_filter, sink=PiEventSink())
    model = model_for_test()
    with bridged_request_scope("slug"):
        await wrapped(
            model=model,
            messages=summary_messages(),
            tools=[],
            tool_choice=None,
            config=GenerateConfig(),
        )
    assert seen == [(model.name, AgentBridgeContext("utility"))]


@pytest.mark.anyio
async def test_filter_results_and_stop_reason() -> None:
    sink = PiEventSink()
    output = ModelOutput.from_content(
        model="example", content="blocked", stop_reason="content_filter"
    )
    rewritten = GenerateInput(
        input=[ChatMessageUser(content="rewritten")],
        tools=[],
        tool_choice=None,
        config=GenerateConfig(),
    )
    results: list[ModelOutput | GenerateInput | None] = [
        output,
        output,
        rewritten,
        None,
    ]

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        return results.pop(0)

    wrapped: Any = pi_request_filter(user_filter=user_filter, sink=sink)
    root: list[ChatMessage] = [ChatMessageUser(content="task")]
    with bridged_request_scope("slug"):
        assert (
            await wrapped(
                model=model_for_test(),
                messages=root,
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is output
        )
        assert sink.last_stop_reason == "content_filter"
        assert (
            await wrapped(
                model=model_for_test(),
                messages=summary_messages(),
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is output
        )
        assert sink.last_stop_reason == "content_filter"
        assert (
            await wrapped(
                model=model_for_test(),
                messages=root,
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is rewritten
        )
        assert sink.last_stop_reason is None
        assert (
            await wrapped(
                model=model_for_test(),
                messages=root,
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is None
        )


@pytest.mark.slow
def test_live_pi_summary_markers() -> None:
    base = "https://raw.githubusercontent.com/earendil-works/pi/main/packages/coding-agent/src/core/compaction/"

    async def fetch_source(*, filename: str) -> str:
        return await download_text_file(url=base + filename)

    utils, compaction, branch = [
        anyio.run(func=partial(fetch_source, filename=filename))
        for filename in ("utils.ts", "compaction.ts", "branch-summarization.ts")
    ]
    assert _SUMMARY_SYSTEM_PREFIX in utils
    for source in (compaction, branch):
        assert "systemPrompt: SUMMARIZATION_SYSTEM_PROMPT" in source
        assert "`<conversation>\\n${conversationText}\\n</conversation>" in source
