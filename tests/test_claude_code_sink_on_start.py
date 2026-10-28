"""Claude Code sub-agent attribution in `ModelEventSink.on_start` and `on_pending`."""

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
from tenacity import wait_fixed

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


def _attempts(
    consumer: LiveConsumer, input: list[ChatMessage], outcomes: list[bool]
) -> str | None:
    """A generate whose attempts fail (False) or succeed (True); its span."""
    span_id = consumer.on_start(input)
    for ok in outcomes:
        event = _model_event(input)
        event.span_id = span_id
        if not ok:
            event.error = "overloaded"
        consumer.on_pending(event)
        consumer.on_complete(event)
    return span_id


def test_retrying_generate_keeps_its_agent(monkeypatch: Any) -> None:
    # two agents with the same prompt: A's first attempt fails and B's first
    # generate starts before A's retry; B must not take A's span
    consumer, _, _ = _spawned(monkeypatch)
    consumer.on_complete(
        _model_event(
            [ChatMessageUser(content="look again")],
            tool_calls=[
                ToolCall(id="toolu_2", function="Task", arguments={"prompt": PROMPT})
            ],
        )
    )
    input = REQUESTS["sub-agent"]
    a, b = contextvars.copy_context(), contextvars.copy_context()
    a_span = a.run(_attempts, consumer, input, [False])
    b_span = b.run(_attempts, consumer, input, [True])
    a.run(_attempts, consumer, input, [])  # (no new generate: A retries)
    assert (a_span, b_span) == ("agent-toolu_1", "agent-toolu_2")


def test_refusal_retry_keeps_its_agent(
    monkeypatch: Any,
) -> None:
    # the bridge retries a refusal with a new generate, in the same task
    consumer, _, _ = _spawned(monkeypatch)
    consumer.on_complete(
        _model_event(
            [ChatMessageUser(content="look again")],
            tool_calls=[
                ToolCall(id="toolu_2", function="Task", arguments={"prompt": PROMPT})
            ],
        )
    )
    input = REQUESTS["sub-agent"]

    def refused_then_retried() -> tuple[str | None, str | None]:
        return _attempts(consumer, input, [False]), _attempts(consumer, input, [True])

    a = contextvars.copy_context().run(refused_then_retried)
    b = contextvars.copy_context().run(_attempts, consumer, input, [True])
    assert a == ("agent-toolu_1", "agent-toolu_1")
    assert b == "agent-toolu_2"


# mockllm inherits inspect_ai's asyncio-backed connection limiter
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_model_retry_keeps_its_agent() -> None:
    # inspect_ai's own generate and retries: A's first attempt fails, B's
    # first generate runs before A's retry; each keeps its own span
    init_transcript(Transcript())
    a_failed, b_done = anyio.Event(), anyio.Event()

    class _Consumer(LiveConsumer):
        def on_complete(self, event: ModelEvent) -> None:
            super().on_complete(event)
            if event.error:
                a_failed.set()

    consumer = _Consumer()
    attempts = 0

    def a_output(*args: Any, **kwargs: Any) -> ModelOutput:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient provider failure")
        return ModelOutput.from_message(
            ChatMessageAssistant(
                content="A",
                tool_calls=[ToolCall(id="a_tool", function="Bash", arguments={})],
            )
        )

    b_output = ModelOutput.from_message(
        ChatMessageAssistant(
            content="B",
            tool_calls=[ToolCall(id="b_tool", function="Bash", arguments={})],
        )
    )
    a = get_model(
        "mockllm/a", custom_outputs=a_output, config=GenerateConfig(max_retries=1)
    )
    b = get_model("mockllm/b", custom_outputs=[b_output])
    a.api.should_retry = lambda ex: True  # type: ignore[method-assign]
    a.api.retry_wait = lambda: wait_fixed(0)  # type: ignore[method-assign]

    async def wait_for_b(ex: BaseException) -> None:
        await b_done.wait()

    a.before_retry = wait_for_b  # type: ignore[method-assign]

    async def run_a() -> None:
        with use_model_event_sink(consumer):
            await a.generate(REQUESTS["sub-agent"])

    async def run_b() -> None:
        await a_failed.wait()
        with use_model_event_sink(consumer):
            await b.generate(REQUESTS["sub-agent"])
        b_done.set()

    async with span("outer", type="agent"):
        consumer.on_complete(
            _model_event(
                [ChatMessageUser(content="look for TODOs")],
                tool_calls=[
                    ToolCall(id=t, function="Task", arguments={"prompt": PROMPT})
                    for t in ("toolu_1", "toolu_2")
                ],
            )
        )
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_a)
                tg.start_soon(run_b)

    placed = [
        (e.model, e.span_id, e.error is None)
        for e in transcript().events
        if isinstance(e, ModelEvent)
    ]
    assert sorted(placed) == sorted(
        [
            ("mockllm/a", "agent-toolu_1", False),
            ("mockllm/b", "agent-toolu_2", True),
            ("mockllm/a", "agent-toolu_1", True),
        ]
    )
