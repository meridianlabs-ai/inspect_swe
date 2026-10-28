"""A compacting sub-agent's pre-summary work survives transcript reconstruction."""

from pathlib import Path
from typing import Any

import pytest
from inspect_ai.event import CompactionEvent, Event
from inspect_ai.event._model import ModelEvent
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.bridge_only import BridgeOnlyLiveConsumer
from inspect_swe._claude_code._events.live_consumer import (
    LiveConsumer,
)
from inspect_swe._claude_code._events.shapes import is_compaction_request

from tests.test_claude_code_subagent_spans_replay import TRACES, _replay

COMPACTION_TRACES = sorted(TRACES.glob("*subagent_compaction.jsonl"))


class _KeepingStub:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.kept: list[Event] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)
        self.kept.append(event)

    def _event_updated(self, event: Any) -> None:
        pass


def _agent_events(
    monkeypatch: pytest.MonkeyPatch, trace: Path
) -> dict[str, list[Event]]:
    stub = _KeepingStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    consumer = (
        BridgeOnlyLiveConsumer() if trace.stem.startswith("acp-") else LiveConsumer()
    )
    _replay(consumer, stub, trace)  # type: ignore[arg-type]
    by_span: dict[str, list[Event]] = {}
    for e in stub.kept:
        if isinstance(e, ModelEvent | CompactionEvent) and str(e.span_id).startswith(
            "agent-"
        ):
            by_span.setdefault(str(e.span_id), []).append(e)
    return by_span


def _tool_call_ids(events: list[Any]) -> set[str]:
    return {
        tc.id
        for e in events
        if isinstance(e, ModelEvent) and e.output and e.output.choices
        for tc in e.output.message.tool_calls or []
    }


@pytest.mark.parametrize("trace", COMPACTION_TRACES, ids=lambda p: p.stem)
def test_compaction_marked_after_summary(
    monkeypatch: pytest.MonkeyPatch, trace: Path
) -> None:
    by_span = _agent_events(monkeypatch, trace)
    compacting = [
        events
        for events in by_span.values()
        if any(
            isinstance(e, ModelEvent) and is_compaction_request(e.input) for e in events
        )
    ]
    assert compacting
    for events in compacting:
        for i, e in enumerate(events):
            if isinstance(e, ModelEvent) and is_compaction_request(e.input):
                assert isinstance(events[i + 1], CompactionEvent)


@pytest.mark.parametrize("trace", COMPACTION_TRACES, ids=lambda p: p.stem)
def test_scout_keeps_pre_summary_tool_calls(
    monkeypatch: pytest.MonkeyPatch, trace: Path
) -> None:
    scout = pytest.importorskip("inspect_scout")
    for events in _agent_events(monkeypatch, trace).values():
        rebuilt = scout.span_messages(events)
        kept = {
            tc.id
            for m in rebuilt
            if isinstance(m, ChatMessageAssistant)
            for tc in m.tool_calls or []
        }
        assert _tool_call_ids(events) <= kept
