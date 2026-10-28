"""Claude Code sub-agent attribution by the agent id request headers."""

import contextvars
from inspect import signature
from types import SimpleNamespace
from typing import Any

import pytest
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import use_model_event_sink
from inspect_ai.tool import ToolCall
from inspect_ai.util._span import current_span_id, span
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.live_consumer import (
    AGENT_ID_HEADER,
    PARENT_AGENT_ID_HEADER,
    LiveConsumer,
    bridge_header_args,
)

from tests.test_claude_code_subagent_spans import (
    PROMPT,
    PROMPT_2,
    _async_ack,
    _bash_call,
    _call,
    _consumer,
    _launch,
    _model_event,
    _notification,
    _started,
    _TranscriptStub,
)

_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "headers", default=None
)


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch) -> tuple[LiveConsumer, _TranscriptStub]:
    monkeypatch.setattr(
        consumer_module,
        "_bridge_request",
        lambda: SimpleNamespace(headers=_headers.get() or {}),
    )
    return _consumer(monkeypatch)


def _from(
    consumer: LiveConsumer,
    agent_id: str | None,
    event: ModelEvent,
    parent_id: str | None = None,
) -> str | None:
    """A call made by `agent_id` (None: the main agent); its span."""
    headers = {AGENT_ID_HEADER: agent_id} if agent_id else {}
    if parent_id:
        headers[PARENT_AGENT_ID_HEADER] = parent_id

    def call() -> str | None:
        _headers.set(headers)
        return _call(consumer, event).span_id

    return contextvars.copy_context().run(call)


def _first(tool_call_id: str, prompt: str = PROMPT) -> ModelEvent:
    return _model_event(
        [ChatMessageUser(content=prompt)], _bash_call(tool_call_id).tool_calls
    )


def _next(tool_call_id: str, prompt: str = PROMPT) -> ModelEvent:
    return _model_event(
        [
            ChatMessageUser(content=prompt),
            _bash_call(tool_call_id),
            ChatMessageTool(content="ok", tool_call_id=tool_call_id),
        ]
    )


def test_same_prompt_first_calls_before_acks(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    a, b = _first("a1"), _first("b1")
    assert _from(consumer, "aA", a) == "agent-t1"
    assert _from(consumer, "aB", b) == "agent-t2"
    assert _from(consumer, "aB", _next("x")) == "agent-t2"
    assert _from(consumer, "aA", _next("y")) == "agent-t1"


def test_out_of_order_first_calls_move_on_ack(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    # B's first call arrives first and is bound to t1 for now
    b = _first("b1")
    assert _from(consumer, "aB", b) == "agent-t1"
    a = _first("a1")
    assert _from(consumer, "aA", a) == "agent-t2"
    # the launch results say which is which: the calls move
    consumer.process_jsonl_line(_async_ack("t1", PROMPT, "aA"))
    consumer.process_jsonl_line(_async_ack("t2", PROMPT, "aB"))
    assert (a.span_id, b.span_id) == ("agent-t1", "agent-t2")
    assert _from(consumer, "aA", _next("a1")) == "agent-t1"
    assert _from(consumer, "aB", _next("b1")) == "agent-t2"
    # B's notification closes B's span
    consumer.process_jsonl_line(_notification(None, "aB"))
    ends = [e.id for e in stub.events if isinstance(e, SpanEndEvent)]
    assert ends == ["agent-t2"]


def test_task_started_confirms_before_first_call(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    consumer.process_jsonl_line(_started("aB", "t2", "r1"))
    assert _from(consumer, "aB", _first("b1")) == "agent-t2"
    assert _from(consumer, "aA", _first("a1")) == "agent-t1"


def test_retry_and_sibling_never_share_a_span(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    failed = _first("a0")
    failed.output, failed.error = ModelOutput(), "overloaded"
    assert _from(consumer, "aA", failed) == "agent-t1"
    assert _from(consumer, "aB", _first("b1")) == "agent-t2"
    assert _from(consumer, "aA", _first("a1")) == "agent-t1"


def test_main_quoting_grandchild_stays_outer(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    delegate = "Delegate the TODO survey to a child agent and report back."
    launch_a = ToolCall(id="tA", function="Agent", arguments={"prompt": delegate})
    main: list[ChatMessage] = [ChatMessageUser(content=PROMPT)]
    assert _from(consumer, None, _model_event(main, [launch_a])) == "outer"
    launch_b = ToolCall(id="tB", function="Agent", arguments={"prompt": PROMPT})
    child = _model_event([ChatMessageUser(content=delegate)], [launch_b])
    assert _from(consumer, "aA", child) == "agent-tA"
    assert _from(consumer, "aB", _first("g1"), parent_id="aA") == "agent-tB"
    # main's history is the grandchild's prompt, verbatim
    assert _from(consumer, None, _model_event(main)) == "outer"
    begin = [
        e for e in stub.events if isinstance(e, SpanBeginEvent) and e.id == "agent-tB"
    ]
    assert begin[0].parent_id == "agent-tA"


def test_nested_agent_bound_among_parents_launches(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    # the main agent and a child launch agents with the same prompt
    _launch(consumer, {"tA": PROMPT_2, "tM": PROMPT})
    launch = ToolCall(id="tB", function="Agent", arguments={"prompt": PROMPT})
    child = _model_event([ChatMessageUser(content=PROMPT_2)], [launch])
    assert _from(consumer, "aA", child) == "agent-tA"
    assert _from(consumer, "aB", _first("g1"), parent_id="aA") == "agent-tB"
    assert _from(consumer, "aM", _first("m1")) == "agent-tM"


def test_resumed_run_by_id(setup: tuple[LiveConsumer, _TranscriptStub]) -> None:
    consumer, stub = setup
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack("toolu_1", PROMPT, "aA"))
    history: list[ChatMessage] = [ChatMessageUser(content=PROMPT)]
    assert _from(consumer, "aA", _model_event(history)) == "agent-toolu_1"
    consumer.process_jsonl_line(_notification("toolu_1", "aA"))
    history += [ChatMessageAssistant(content="done"), ChatMessageUser(content="more")]
    assert _from(consumer, "aA", _model_event(history)) == "agent-toolu_1-run2"


def test_side_request_stays_in_the_current_run(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    assert _from(consumer, "aA", _first("a1")) == "agent-toolu_1"
    side: list[ChatMessage] = [
        ChatMessageUser(content="Summarise this page."),
        ChatMessageAssistant(content="It says..."),
        ChatMessageUser(content="And this one."),
    ]
    assert _from(consumer, "aA", _model_event(side)) == "agent-toolu_1"


def test_unmatched_first_call_stays_with_launcher(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT_2})
    unknown = _model_event([ChatMessageUser(content="something else entirely")])
    assert _from(consumer, "aX", unknown) == "outer"
    assert _from(consumer, "aB", _first("b1", PROMPT_2)) == "agent-t2"


def test_without_headers_history_decides(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    assert _from(consumer, None, _first("a1")) == "agent-toolu_1"


def test_without_the_api_history_decides(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    monkeypatch.setattr(consumer_module, "_bridge_request", lambda: None)
    _launch(consumer)
    assert _call(consumer, _first("a1")).span_id == "agent-toolu_1"


def test_bridge_header_args(monkeypatch: pytest.MonkeyPatch) -> None:
    def old_bridge(model: str | None = None) -> None: ...

    def new_bridge(expose_client_headers: Any = None) -> None: ...

    monkeypatch.setattr(consumer_module, "sandbox_agent_bridge", old_bridge)
    assert bridge_header_args() == {}
    monkeypatch.setattr(consumer_module, "sandbox_agent_bridge", new_bridge)
    args = bridge_header_args()
    assert args == {"expose_client_headers": [AGENT_ID_HEADER, PARENT_AGENT_ID_HEADER]}
    assert set(args) <= set(signature(new_bridge).parameters)


# mockllm inherits inspect_ai's asyncio-backed connection limiter
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.anyio
async def test_headers_through_inspect_ai() -> None:
    context = pytest.importorskip("inspect_ai.agent._bridge.context")
    if not hasattr(context, "client_request_headers_scope"):
        pytest.skip("inspect_ai does not expose bridged request headers")
    init_transcript(Transcript())
    consumer = LiveConsumer()
    model = get_model("mockllm/model")
    names = frozenset({AGENT_ID_HEADER, PARENT_AGENT_ID_HEADER})

    async def request(agent_id: str | None, input: list[ChatMessage]) -> None:
        headers = {AGENT_ID_HEADER: agent_id} if agent_id else {}
        with (
            context.client_request_headers_scope(headers, names),
            context.bridged_request_scope("claude"),
            use_model_event_sink(consumer),
        ):
            await model.generate(input)

    async with span("outer", type="agent"):
        outer = current_span_id()
        consumer.on_complete(
            _model_event(
                [ChatMessageUser(content="main task")],
                [
                    ToolCall(id=t, function="Agent", arguments={"prompt": PROMPT})
                    for t in ("t1", "t2")
                ],
            )
        )
        first: list[ChatMessage] = [ChatMessageUser(content=PROMPT)]
        await request("aB", first)
        await request("aA", first)
        await request(None, first)
        consumer.process_jsonl_line(_async_ack("t1", PROMPT, "aA"))
        consumer.process_jsonl_line(_async_ack("t2", PROMPT, "aB"))
        await request("aB", first + [ChatMessageAssistant(content="hm")])

    placed = [e.span_id for e in transcript().events if isinstance(e, ModelEvent)]
    assert placed == ["agent-t2", "agent-t1", outer, "agent-t2"]
