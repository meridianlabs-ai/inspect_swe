import json
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.event import ModelEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_swe._pi.config import pi_models_json, pi_session_header
from inspect_swe._pi.pi import PiEventSink, pi_request_filter
from inspect_swe._util.model_event_sink import filter_generation


def test_session_header_and_model_identity() -> None:
    header = json.loads(pi_session_header(cwd="/workspace"))
    assert header["type"] == "session"
    assert header["version"] == 3
    assert header["cwd"] == "/workspace"
    UUID(header["id"])
    assert datetime.fromisoformat(header["timestamp"]).utcoffset() == timedelta(0)
    models = json.loads(
        pi_models_json(
            port=3201,
            model="openai/example",
            context_size=1000,
            max_tokens=500,
            reasoning=False,
        )
    )
    provider = models["providers"]["inspect"]
    assert provider["baseUrl"] == "http://localhost:3201/v1"
    assert (
        provider["models"][0]["id"] == provider["models"][0]["name"] == "openai/example"
    )


@pytest.mark.anyio
async def test_updated_project_instructions_are_not_overwritten() -> None:
    model = get_model(
        model="openai/example", api_key="dummy", base_url="http://localhost:1/v1"
    )
    utility: list[ChatMessage] = [
        ChatMessageSystem(content="You are a context summarization assistant. utility"),
        ChatMessageUser(content="<conversation>\ntext"),
    ]
    root: list[ChatMessage] = [
        ChatMessageSystem(content="first"),
        ChatMessageUser(content="task"),
    ]
    changed: list[ChatMessage] = [
        ChatMessageSystem(content="changed"),
        ChatMessageUser(content="new compacted history"),
    ]
    first: Any = pi_request_filter(user_filter=None, sink=PiEventSink())
    with bridged_request_scope("slug"):
        assert (
            await first(
                model=model,
                messages=utility,
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is None
        )
        assert (
            await first(
                model=model,
                messages=root,
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is None
        )
        resumed: Any = pi_request_filter(user_filter=None, sink=PiEventSink())
        result = await resumed(
            model=model,
            messages=changed,
            tools=[],
            tool_choice=None,
            config=GenerateConfig(),
        )
        assert result is None
        assert changed[0].text == "changed"
        assert (
            await resumed(
                model=model,
                messages=utility,
                tools=[],
                tool_choice=None,
                config=GenerateConfig(),
            )
            is None
        )


def test_refusal_tracks_only_latest_root_event() -> None:
    sink = PiEventSink()
    root: list[ChatMessage] = [
        ChatMessageSystem(content="system"),
        ChatMessageUser(content="task"),
    ]
    utility: list[ChatMessage] = [
        ChatMessageSystem(content="You are a context summarization assistant."),
        ChatMessageUser(content="<conversation>\ntext"),
    ]

    def event(messages: list[ChatMessage], reason: str) -> ModelEvent:
        output = ModelOutput.from_content(
            model="test",
            content="",
            stop_reason="content_filter" if reason == "refusal" else "stop",
        )
        return ModelEvent(
            model="test",
            input=messages,
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
            output=output,
        )

    refusal = event(root, "refusal")
    sink.on_pending(refusal)
    sink.on_complete(refusal)
    assert sink.last_stop_reason == "content_filter"
    utility_event = event(utility, "stop")
    sink.on_pending(utility_event)
    sink.on_complete(utility_event)
    assert sink.last_stop_reason == "content_filter"
    later = event(root, "stop")
    sink.on_pending(later)
    assert sink.last_stop_reason is None
    sink.on_complete(later)
    assert sink.last_stop_reason == "stop"
    refusal = event(root, "refusal")
    sink.on_pending(refusal)
    sink.on_complete(refusal)
    failed = event(utility, "stop")
    failed.error = "later failure"
    sink.on_pending(failed)
    sink.on_complete(failed)
    assert sink.last_stop_reason is None
    intermediate = event(messages=root, reason="refusal")
    sink.on_pending(event=intermediate)
    with filter_generation(sink=sink):
        sink.on_complete(event=intermediate)
    assert sink.last_stop_reason is None
