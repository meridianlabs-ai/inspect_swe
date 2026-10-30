"""Claude Code sub-agent attribution by the bridge's `ModelEventSink` (`LiveConsumer`).

A parent's Task/Agent tool call opens a sub-agent span in `on_complete`. Each
later request is attributed by substring-matching its first user message
against the pending sub-agent prompts: in `on_start` on an inspect_ai with
`ModelEventSink.on_start` (once per generate, which inspect_ai then runs under
the span), else in `on_pending`.
"""

import contextvars
from typing import Any

import anyio
import pytest
from inspect_ai.event import SpanBeginEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._model import use_model_event_sink
from inspect_ai.tool import ToolCall
from inspect_ai.util._span import current_span_id, span
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.live_consumer import LiveConsumer

PROMPT = "Find every TODO comment in the repository and list them."


def _model_event(
    input: list[ChatMessage], tool_calls: list[ToolCall] | None = None
) -> ModelEvent:
    message = ChatMessageAssistant(content="ok", tool_calls=tool_calls)
    return ModelEvent(
        model="anthropic/claude-opus-5-5",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(message),
    )


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)

    def _event_updated(self, event: Any) -> None:
        pass

    def span_begins(self) -> list[SpanBeginEvent]:
        return [e for e in self.events if isinstance(e, SpanBeginEvent)]


def _spawned(monkeypatch: Any) -> tuple[LiveConsumer, LiveConsumer, _TranscriptStub]:
    """Two consumers that have each seen a parent spawn one Task sub-agent."""
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    consumers = LiveConsumer(), LiveConsumer()
    for consumer in consumers:
        consumer.on_complete(
            _model_event(
                [ChatMessageUser(content="look for TODOs")],
                tool_calls=[
                    ToolCall(
                        id="toolu_1",
                        function="Task",
                        arguments={"prompt": PROMPT, "subagent_type": "explore"},
                    )
                ],
            )
        )
    return consumers[0], consumers[1], stub


REQUESTS: dict[str, list[ChatMessage]] = {
    "sub-agent": [
        ChatMessageSystem(content="You are a sub-agent."),
        ChatMessageUser(content=f"{PROMPT}\n\nBe thorough."),
    ],
    "main": [ChatMessageUser(content="look for TODOs")],
}


def test_on_start_attributes_as_on_pending_did(monkeypatch: Any) -> None:
    starting, legacy, stub = _spawned(monkeypatch)
    # outside any span the main thread's span is None
    expected = {"sub-agent": "agent-toolu_1", "main": None}
    for name, input in REQUESTS.items():
        event = _model_event(input)
        contextvars.copy_context().run(legacy.on_pending, event)
        assert event.span_id == expected[name], name
        assert (
            contextvars.copy_context().run(starting.on_start, input) == (expected[name])
        ), name
    assert stub.span_begins()[0].id == "agent-toolu_1"


def test_on_pending_leaves_the_span_on_start_chose(monkeypatch: Any) -> None:
    consumer, _, stub = _spawned(monkeypatch)
    input = REQUESTS["sub-agent"]
    attributed: list[list[ChatMessage]] = []
    attribute = consumer._attribute

    def counted(input: list[ChatMessage]) -> str | None:
        attributed.append(input)
        return attribute(input)

    monkeypatch.setattr(consumer, "_attribute", counted)

    def generate() -> list[ModelEvent]:
        span_id = consumer.on_start(input)
        # a parallel parent spawns a second sub-agent with the same prompt, so
        # attributing again would no longer find a single match
        consumer.on_complete(
            _model_event(
                [ChatMessageUser(content="look again")],
                tool_calls=[
                    ToolCall(
                        id="toolu_2", function="Task", arguments={"prompt": PROMPT}
                    )
                ],
            )
        )
        # inspect_ai creates each attempt's event under the span on_start returned
        attempts = [_model_event(input) for _ in range(2)]
        for event in attempts:
            event.span_id = span_id
            consumer.on_pending(event)
            consumer.on_complete(event)
        return attempts

    attempts = contextvars.copy_context().run(generate)
    assert [e.span_id for e in attempts] == ["agent-toolu_1"] * 2
    # attributed once, by on_start, not again by either attempt's on_pending
    assert len(attributed) == 1
    assert all(event in stub.events for event in attempts)


def test_without_on_start_on_pending_attributes_as_before(monkeypatch: Any) -> None:
    consumer, _, _ = _spawned(monkeypatch)
    event = _model_event(REQUESTS["sub-agent"])
    event.span_id = "the bridge's span"
    contextvars.copy_context().run(consumer.on_pending, event)
    assert event.span_id == "agent-toolu_1"


def test_a_concurrent_request_s_on_start_does_not_mark_another_request(
    monkeypatch: Any,
) -> None:
    consumer, _, _ = _spawned(monkeypatch)
    started, other = contextvars.copy_context(), contextvars.copy_context()
    started.run(consumer.on_start, REQUESTS["main"])
    elsewhere = _model_event(REQUESTS["sub-agent"])
    other.run(consumer.on_pending, elsewhere)
    assert elsewhere.span_id == "agent-toolu_1"


# mockllm inherits inspect_ai's asyncio-backed connection limiter
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_generates_through_the_model_are_attributed_once_each(
    monkeypatch: Any,
) -> None:
    # inspect_ai's own generate(): through on_start when it has one, else on_pending
    init_transcript(Transcript())
    consumer = LiveConsumer()
    attributed: list[list[ChatMessage]] = []
    attribute = consumer._attribute

    def counted(input: list[ChatMessage]) -> str | None:
        attributed.append(input)
        return attribute(input)

    monkeypatch.setattr(consumer, "_attribute", counted)
    model = get_model("mockllm/model")

    async def generate(input: list[ChatMessage]) -> None:
        with use_model_event_sink(consumer):
            await model.generate(input)

    async with span("outer", type="agent"):
        outer = current_span_id()
        consumer.on_complete(
            _model_event(
                [ChatMessageUser(content="look for TODOs")],
                tool_calls=[
                    ToolCall(
                        id="toolu_1", function="Task", arguments={"prompt": PROMPT}
                    )
                ],
            )
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(generate, REQUESTS["sub-agent"])
            tg.start_soon(generate, REQUESTS["main"])
        await generate(REQUESTS["sub-agent"])

    placed = [
        (e.input[-1].text, e.span_id)
        for e in transcript().events
        if isinstance(e, ModelEvent)
    ]
    sub, main = REQUESTS["sub-agent"][-1].text, REQUESTS["main"][-1].text
    assert sorted(placed) == sorted(
        [(sub, "agent-toolu_1"), (main, outer), (sub, "agent-toolu_1")]
    )
    assert len(attributed) == 3
