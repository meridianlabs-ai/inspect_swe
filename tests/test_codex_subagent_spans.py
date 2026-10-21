"""Codex Multi-Agent V2 sub-agent span reconstruction (bridge-only).

Multi-Agent V2 (codex >= 0.146, forced for gpt-5.6-terra/sol) changed every
signal the CodexConsumer used to build the agent-span tree:

- `spawn_agent` calls carry `task_name` (not `agent_type`) and an *encrypted*
  `message`, so spans were named the generic "agent" and prompt-substring
  attribution matched nothing (all model events landed on the outer span).
- `spawn_agent` results carry `{"task_name": "/root/<name>"}` (not
  `agent_id`/`nickname`).
- completion arrives as a FINAL_ANSWER `agent_message` from the child (the
  `wait_agent` result no longer carries per-thread status).

The agent bridge (inspect_ai >= 0.3.253) converts each `agent_message` input
item to a ChatMessageUser and preserves the raw item (author/recipient) on
ContentText.internal — every agent_message in a request is inbound to the
requester, so `recipient` identifies the calling agent exactly.

Fixture shapes below are copied from a real gpt-5.6-sol eval log
(codex 0.147.0, 2026-08-07).
"""

import contextvars
from typing import Any

import anyio
import pytest
from inspect_ai._util.content import ContentText
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import use_model_event_sink
from inspect_ai.tool import ToolCall
from inspect_ai.util._span import current_span_id, span
from inspect_swe._codex_cli._events import consumer as consumer_module
from inspect_swe._codex_cli._events.consumer import CodexConsumer
from inspect_swe._codex_cli._events.detection import (
    COMPACTION_MARKER,
    agent_message_recipients,
    final_answer_authors,
    find_spawned_agents,
    spawn_result,
)

# ---------------------------------------------------------------------------
# fixtures (shapes from a real codex 0.147.0 / gpt-5.6-sol run)
# ---------------------------------------------------------------------------


def _v2_spawn_call(call_id: str, task_name: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={
            "task_name": task_name,
            "fork_turns": "all",
            "message": "gAAAAABqdjXI1bVTsv-encrypted-payload",
        },
    )


def _v1_spawn_call(call_id: str, prompt: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={"agent_type": "explore", "message": prompt},
    )


def _agent_message_user(
    author: str, recipient: str, message_type: str = "MESSAGE", payload: str = ""
) -> ChatMessageUser:
    """A ChatMessageUser as the bridge produces it from an agent_message item."""
    envelope = (
        f"Message Type: {message_type}\n"
        f"Task name: {recipient}\n"
        f"Sender: {author}\n"
        f"Payload:\n{payload}"
    )
    raw_item: dict[str, Any] = {
        "type": "agent_message",
        "author": author,
        "recipient": recipient,
        "content": [{"type": "input_text", "text": envelope}],
    }
    return ChatMessageUser(
        content=[
            ContentText(
                text=f"Agent message from {author}:\n{envelope}",
                internal={"agent_message": raw_item},
            )
        ]
    )


def _spawn_result_tool_message(call_id: str, text: str) -> ChatMessageTool:
    return ChatMessageTool(content=text, tool_call_id=call_id, function="spawn_agent")


def _model_event(
    input: list[ChatMessage], tool_calls: list[ToolCall] | None = None
) -> ModelEvent:
    message = ChatMessageAssistant(content="ok", tool_calls=tool_calls)
    return ModelEvent(
        model="openai/gpt-5.6-sol",
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

    def span_ends(self) -> list[SpanEndEvent]:
        return [e for e in self.events if isinstance(e, SpanEndEvent)]


def _consumer(monkeypatch: Any) -> tuple[CodexConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    return CodexConsumer(), stub


# ---------------------------------------------------------------------------
# 1. detection: V2 spawn calls and results
# ---------------------------------------------------------------------------


def test_find_spawned_agents_uses_v2_task_name() -> None:
    spawned = find_spawned_agents([_v2_spawn_call("call_1", "write_fizzbuzz")])
    assert len(spawned) == 1
    assert spawned[0].name == "write_fizzbuzz"


def test_find_spawned_agents_keeps_v1_agent_type() -> None:
    spawned = find_spawned_agents(
        [_v1_spawn_call("call_1", "write a fizzbuzz program to /tmp/fizzbuzz.py")]
    )
    assert len(spawned) == 1
    assert spawned[0].name == "explore"


def test_spawn_result_parses_v2_task_name() -> None:
    result = spawn_result(
        _spawn_result_tool_message("call_1", '{"task_name":"/root/write_fizzbuzz"}')
    )
    assert result is not None
    assert result.agent_id == "/root/write_fizzbuzz"
    assert result.nickname is None


def test_spawn_result_still_parses_v1_agent_id() -> None:
    result = spawn_result(
        _spawn_result_tool_message(
            "call_1", '{"agent_id":"thread_abc","nickname":"Explorer"}'
        )
    )
    assert result is not None
    assert result.agent_id == "thread_abc"
    assert result.nickname == "Explorer"


# ---------------------------------------------------------------------------
# 2. detection: agent_message identity and completion signals
# ---------------------------------------------------------------------------


def test_agent_message_recipients_identifies_requester() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_fizzbuzz"),
        _agent_message_user(
            "/root/write_fizzbuzz/implement_file", "/root/write_fizzbuzz"
        ),
    ]
    assert agent_message_recipients(input) == {"/root/write_fizzbuzz"}


def test_agent_message_recipients_empty_without_agent_messages() -> None:
    assert agent_message_recipients([ChatMessageUser(content="plain task")]) == set()


def test_final_answer_authors_detects_completion() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_primes"),
        _agent_message_user(
            "/root/write_primes", "/root", "FINAL_ANSWER", "primes written"
        ),
    ]
    assert final_answer_authors(input) == {"/root/write_primes"}


def test_final_answer_authors_ignores_plain_messages() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root/write_primes", "/root", "MESSAGE")
    ]
    assert final_answer_authors(input) == set()


# ---------------------------------------------------------------------------
# 3. consumer: spans are named after V2 task_name
# ---------------------------------------------------------------------------


def test_consumer_names_v2_spans_after_task_name(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="spawn two agents")],
        tool_calls=[
            _v2_spawn_call("call_fb", "write_fizzbuzz"),
            _v2_spawn_call("call_pr", "write_primes"),
        ],
    )
    consumer.on_complete(parent)

    names = [e.name for e in stub.span_begins()]
    assert names == ["write_fizzbuzz", "write_primes"]


# ---------------------------------------------------------------------------
# 4. consumer: attribution by agent_message recipient
# ---------------------------------------------------------------------------


def test_consumer_attributes_subagent_call_by_recipient(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    # parent spawns two agents (V2: encrypted prompts, so no substring match)
    parent = _model_event(
        [ChatMessageUser(content="spawn two agents")],
        tool_calls=[
            _v2_spawn_call("call_fb", "write_fizzbuzz"),
            _v2_spawn_call("call_pr", "write_primes"),
        ],
    )
    consumer.on_complete(parent)
    fb_span, pr_span = (e.id for e in stub.span_begins())

    # each sub-agent's first call carries its inbound spawn agent_message
    fb_call = _model_event([_agent_message_user("/root", "/root/write_fizzbuzz")])
    consumer.on_pending(fb_call)
    assert fb_call.span_id == fb_span

    pr_call = _model_event([_agent_message_user("/root", "/root/write_primes")])
    consumer.on_pending(pr_call)
    assert pr_call.span_id == pr_span

    # the parent's own calls (inbound messages addressed to /root) stay outer
    parent_call = _model_event(
        [
            ChatMessageUser(content="spawn two agents"),
            _agent_message_user("/root/write_primes", "/root"),
        ]
    )
    consumer.on_pending(parent_call)
    assert parent_call.span_id not in (fb_span, pr_span)


def test_consumer_nests_grandchild_span_under_child(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_fb", "write_fizzbuzz")],
    )
    consumer.on_complete(parent)
    fb_span = stub.span_begins()[0].id

    # child call spawns its own sub-agent
    child_call = _model_event(
        [_agent_message_user("/root", "/root/write_fizzbuzz")],
        tool_calls=[_v2_spawn_call("call_impl", "implement_file")],
    )
    consumer.on_pending(child_call)
    consumer.on_complete(child_call)

    impl_begin = stub.span_begins()[1]
    assert impl_begin.name == "implement_file"
    assert impl_begin.parent_id == fb_span


# ---------------------------------------------------------------------------
# 5. consumer: FINAL_ANSWER closes the child's span
# ---------------------------------------------------------------------------


def test_consumer_closes_span_on_final_answer(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_pr", "write_primes")],
    )
    consumer.on_complete(parent)
    pr_span = stub.span_begins()[0].id

    # child identifies itself (binds /root/write_primes to the span) ...
    child_call = _model_event([_agent_message_user("/root", "/root/write_primes")])
    consumer.on_pending(child_call)
    assert child_call.span_id == pr_span

    # ... then the parent sees the child's FINAL_ANSWER -> span closes
    parent_call = _model_event(
        [
            ChatMessageUser(content="go"),
            _agent_message_user(
                "/root/write_primes", "/root", "FINAL_ANSWER", "all primes written"
            ),
        ]
    )
    consumer.on_pending(parent_call)

    assert [e.id for e in stub.span_ends()] == [pr_span]


# ---------------------------------------------------------------------------
# 6. consumer: V1 prompt-substring attribution still works
# ---------------------------------------------------------------------------


def test_consumer_v1_prompt_attribution_unchanged(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    prompt = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"
    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v1_spawn_call("call_1", prompt)],
    )
    consumer.on_complete(parent)
    span = stub.span_begins()[0].id

    # V1: codex re-sends the spawn prompt as the sub-agent's user message
    child_call = _model_event([ChatMessageUser(content=prompt)])
    consumer.on_pending(child_call)
    assert child_call.span_id == span


# ---------------------------------------------------------------------------
# 7. consumer: on_start (inspect_ai with ModelEventSink.on_start)
# ---------------------------------------------------------------------------


def _spawned_consumers(
    monkeypatch: Any,
) -> tuple[CodexConsumer, CodexConsumer, _TranscriptStub]:
    """Two consumers that have each seen a parent spawn write_fizzbuzz (V2) and a V1 agent."""
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    consumers = CodexConsumer(), CodexConsumer()
    for consumer in consumers:
        consumer.on_complete(
            _model_event(
                [ChatMessageUser(content="go")],
                tool_calls=[
                    _v2_spawn_call("call_fb", "write_fizzbuzz"),
                    _v1_spawn_call("call_v1", V1_PROMPT),
                ],
            )
        )
    return consumers[0], consumers[1], stub


V1_PROMPT = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"

REQUESTS: dict[str, list[ChatMessage]] = {
    "v2-recipient": [_agent_message_user("/root", "/root/write_fizzbuzz")],
    "v1-prompt": [ChatMessageUser(content=V1_PROMPT)],
    "outer": [ChatMessageUser(content="go")],
}


def test_on_start_attributes_as_on_pending_did(monkeypatch: Any) -> None:
    starting, legacy, stub = _spawned_consumers(monkeypatch)
    fb_span, v1_span = (e.id for e in stub.span_begins()[:2])
    # outside any span the outer span is None
    expected = {"v2-recipient": fb_span, "v1-prompt": v1_span, "outer": None}
    for name, input in REQUESTS.items():
        event = _model_event(input)
        contextvars.copy_context().run(legacy.on_pending, event)
        assert event.span_id == expected[name], name
        assert (
            contextvars.copy_context().run(starting.on_start, input) == (expected[name])
        ), name


def test_on_pending_leaves_the_span_on_start_chose(monkeypatch: Any) -> None:
    consumer, _, stub = _spawned_consumers(monkeypatch)
    input = REQUESTS["v2-recipient"]

    def generate() -> ModelEvent:
        span_id = consumer.on_start(input)
        # meanwhile the parent sees the child's FINAL_ANSWER, closing its span,
        # so attributing again would give the outer span
        consumer.on_start(
            [
                ChatMessageUser(content="go"),
                _agent_message_user(
                    "/root/write_fizzbuzz", "/root", "FINAL_ANSWER", "done"
                ),
            ]
        )
        # inspect_ai creates the event under the span on_start returned
        event = _model_event(input)
        event.span_id = span_id
        consumer.on_pending(event)
        return event

    event = contextvars.copy_context().run(generate)
    assert event.span_id == stub.span_begins()[0].id
    assert event in stub.events


def test_without_on_start_on_pending_attributes_as_before(monkeypatch: Any) -> None:
    consumer, _, stub = _spawned_consumers(monkeypatch)
    event = _model_event(REQUESTS["v2-recipient"])
    event.span_id = "the bridge's span"
    contextvars.copy_context().run(consumer.on_pending, event)
    assert event.span_id == stub.span_begins()[0].id


def test_on_start_binds_and_closes_once_per_generate_across_retries(
    monkeypatch: Any,
) -> None:
    consumer, _, stub = _spawned_consumers(monkeypatch)
    fb_span = stub.span_begins()[0].id
    contextvars.copy_context().run(
        consumer.on_start, REQUESTS["v2-recipient"]
    )  # binds /root/write_fizzbuzz
    calls: list[str] = []
    for method in ("_harvest_bindings", "_attribute", "_close_thread"):
        original = getattr(consumer, method)

        def counted(*args: Any, _name: str = method, _original: Any = original) -> Any:
            calls.append(_name)
            return _original(*args)

        monkeypatch.setattr(consumer, method, counted)

    # the parent's request carries the child's FINAL_ANSWER; its provider
    # attempt fails once and is retried
    input: list[ChatMessage] = [
        ChatMessageUser(content="go"),
        _agent_message_user("/root/write_fizzbuzz", "/root", "FINAL_ANSWER", "done"),
    ]

    def generate() -> None:
        span_id = consumer.on_start(input)
        for _ in range(2):
            event = _model_event(input)
            event.span_id = span_id
            consumer.on_pending(event)
            consumer.on_complete(event)

    contextvars.copy_context().run(generate)
    assert calls == ["_harvest_bindings", "_close_thread", "_attribute"]
    assert [e.id for e in stub.span_ends()] == [fb_span]


@pytest.mark.parametrize("on_start", [True, False], ids=["on-start", "on-pending"])
def test_the_compaction_marker_is_emitted_on_the_call_s_span(
    monkeypatch: Any, on_start: bool
) -> None:
    consumer, _, stub = _spawned_consumers(monkeypatch)
    input: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_fizzbuzz"),
        ChatMessageUser(content=f"{COMPACTION_MARKER}\nSummarize."),
    ]

    def generate() -> ModelEvent:
        event = _model_event(input)
        if on_start:
            event.span_id = consumer.on_start(input)
        consumer.on_pending(event)
        return event

    event = contextvars.copy_context().run(generate)
    [marker] = [e for e in stub.events if isinstance(e, CompactionEvent)]
    assert marker.span_id == event.span_id == stub.span_begins()[0].id


def test_a_concurrent_request_s_on_start_does_not_mark_another_request(
    monkeypatch: Any,
) -> None:
    # each request is generated in its own task (its own context): one that
    # called on_start leaves another's placement to its on_pending
    consumer, _, stub = _spawned_consumers(monkeypatch)
    fb_span = stub.span_begins()[0].id
    started, other = contextvars.copy_context(), contextvars.copy_context()
    span_id = started.run(consumer.on_start, REQUESTS["v2-recipient"])
    elsewhere = _model_event(REQUESTS["v2-recipient"])
    other.run(consumer.on_pending, elsewhere)
    assert elsewhere.span_id == span_id == fb_span

    placed = _model_event(REQUESTS["v2-recipient"])
    placed.span_id = "left alone"
    started.run(consumer.on_pending, placed)
    assert placed.span_id == "left alone"


@pytest.mark.anyio
async def test_generates_through_the_model_start_once_each(monkeypatch: Any) -> None:
    # inspect_ai's own generate(): through on_start when it has one, else on_pending
    init_transcript(Transcript())
    consumer = CodexConsumer()
    started: list[list[ChatMessage]] = []
    start = consumer._start

    def counted(input: list[ChatMessage]) -> str | None:
        started.append(input)
        return start(input)

    monkeypatch.setattr(consumer, "_start", counted)
    model = get_model("mockllm/model")

    async def generate(input: list[ChatMessage]) -> None:
        with use_model_event_sink(consumer):
            await model.generate(input)

    async with span("outer", type="agent"):
        outer = current_span_id()
        consumer.on_complete(
            _model_event(
                [ChatMessageUser(content="go")],
                tool_calls=[_v1_spawn_call("call_v1", V1_PROMPT)],
            )
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(generate, REQUESTS["v1-prompt"])
            tg.start_soon(generate, REQUESTS["outer"])
        await generate(REQUESTS["v1-prompt"])

    placed = [
        (e.input[-1].text, e.span_id)
        for e in transcript().events
        if isinstance(e, ModelEvent)
    ]
    assert sorted(placed) == sorted(
        [(V1_PROMPT, "agent-call_v1"), ("go", outer), (V1_PROMPT, "agent-call_v1")]
    )
    assert len(started) == 3
