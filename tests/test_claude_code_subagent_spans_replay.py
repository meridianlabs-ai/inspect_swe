"""Replay recorded Claude Code 2.1.303 runs through the sub-agent span consumers.

Each fixture records, in order, what a consumer saw in a real run (claude-haiku-4-5,
`tests/fixtures/claude_code_traces`): bridge calls (`pending` with the request and
the span it was attributed to, `complete` with the reply), stream-json `user` and
`system` lines, `reset`, and the agent spans the consumer opened (`begin`, with its
parent) and closed (`end`). System prompts are dropped and `<system-reminder>`
bodies elided, except for task notifications. The recorded spans were checked by
hand: every sub-agent call in its own span (a resumed run in its own `-runN` span),
every main-agent call in the outer span, each span closed when its run ended.
Replaying catches changes that would treat these real shapes differently, and
Claude Code changes once fixtures are re-recorded.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from pydantic import TypeAdapter

TRACES = Path(__file__).parent / "fixtures" / "claude_code_traces"
_MESSAGES = TypeAdapter(list[ChatMessage])


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)

    def _event_updated(self, event: Any) -> None:
        pass


def _outer(span_id: str | None) -> str:
    return span_id if span_id is not None and span_id.startswith("agent-") else "outer"


def _replay(
    consumer: LiveConsumer, stub: _TranscriptStub, trace: Path
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """(recorded, replayed) sequences of calls and span events.

    A call is ("call", its span); span events are ("begin", id, parent) and
    ("end", id). Outer spans are "outer".
    """
    events: dict[int, ModelEvent] = {}
    recorded: list[tuple[str, ...]] = []
    replayed: list[tuple[str, ...]] = []

    def span_events() -> None:
        for e in stub.events:
            if isinstance(e, SpanBeginEvent):
                replayed.append(("begin", e.id, _outer(e.parent_id)))
            elif isinstance(e, SpanEndEvent):
                replayed.append(("end", e.id))
        stub.events.clear()

    for line in trace.read_text().splitlines():
        entry = json.loads(line)
        kind = entry["kind"]
        if kind == "begin":
            recorded.append(("begin", entry["id"], _outer(entry["parent"])))
        elif kind == "end":
            recorded.append(("end", entry["id"]))
        elif kind == "pending":
            # (recorded after the spans it opened)
            event = ModelEvent(
                model="anthropic/claude-haiku-4-5",
                input=_MESSAGES.validate_python(entry["input"]),
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
                output=ModelOutput(),
            )
            consumer.on_pending(event)
            events[entry["call"]] = event
            span_events()
            recorded.append(("call", _outer(entry["span_id"])))
            replayed.append(("call", _outer(event.span_id)))
        elif kind == "complete":
            event = events.pop(entry["call"])
            if entry["output"] is not None:
                event.output = ModelOutput.from_message(
                    ChatMessageAssistant.model_validate(entry["output"]),
                    stop_reason=entry["stop_reason"],
                )
            consumer.on_complete(event)
            span_events()
        elif kind == "jsonl":
            consumer.process_jsonl_line(entry["raw"])
            span_events()
        elif kind == "reset":
            consumer.reset()
            replayed.append(("reset",))
            recorded.append(("reset",))
            span_events()
    return recorded, replayed


@pytest.mark.parametrize(
    "trace", sorted(TRACES.glob("cli-*.jsonl")), ids=lambda p: p.stem
)
def test_recorded_run_replays_the_same(
    monkeypatch: pytest.MonkeyPatch, trace: Path
) -> None:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    consumer = LiveConsumer()
    recorded, replayed = _replay(consumer, stub, trace)
    assert any(e[0] == "call" and e[1] != "outer" for e in recorded)
    # every call in the same span, and the same spans opened (under the same
    # parents) and closed at the same points
    assert replayed == recorded
    # spans close when their run ends, not when the attempt does
    final_reset = len(recorded) - 1 - recorded[::-1].index(("reset",))
    assert not [e for e in recorded[final_reset:] if e[0] == "end"]


def test_background_traces_keep_their_notifications() -> None:
    for trace in TRACES.glob("cli-*background*.jsonl"):
        assert "<task-notification>" in trace.read_text(), trace.name
