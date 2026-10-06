"""Unit tests for codex rollout envelope/model parsing."""

from datetime import timedelta
from typing import Any

from inspect_swe._codex_cli._events.rollout import _RolloutProcessor
from inspect_swe._codex_cli._events.rollout_extraction import parse_timestamp
from inspect_swe._codex_cli._events.rollout_models import (
    ResponseCompaction,
    ResponseFunctionCall,
    ResponseMessage,
    SessionMetaEvent,
    parse_rollout_event,
    parse_rollout_events,
)

THREAD_ID = "0199aaaa-0000-7000-8000-000000000001"


def _line(
    type_: str, payload: dict[str, Any], ts: str = "2026-08-01T10:00:00Z"
) -> dict[str, Any]:
    return {"timestamp": ts, "type": type_, "payload": payload}


def test_parse_session_meta_backfills_ids() -> None:
    # old files have only id; parse should backfill session_id
    event = parse_rollout_event(_line("session_meta", {"id": THREAD_ID, "cwd": "/x"}))
    assert isinstance(event, SessionMetaEvent)
    assert event.thread_id == THREAD_ID
    assert event.session_id == THREAD_ID
    assert event.timestamp == "2026-08-01T10:00:00Z"

    event = parse_rollout_event(
        {"type": "session_meta", "payload": {"session_id": THREAD_ID}}
    )
    assert isinstance(event, SessionMetaEvent)
    assert event.thread_id == THREAD_ID


def test_subagent_source_classification() -> None:
    review = SessionMetaEvent(id=THREAD_ID, source={"subagent": "review"})
    assert review.subagent_source() == "review"

    spawn = SessionMetaEvent(
        id=THREAD_ID, source={"subagent": {"thread_spawn": {"depth": 1}}}
    )
    subagent = spawn.subagent_source()
    assert isinstance(subagent, dict) and "thread_spawn" in subagent

    cli = SessionMetaEvent(id=THREAD_ID, source="cli")
    assert cli.subagent_source() is None


def test_parse_response_items() -> None:
    events = parse_rollout_events(
        [
            _line(
                "response_item",
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            ),
            _line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": '{"command": ["ls"]}',
                    "call_id": "c1",
                },
            ),
        ]
    )
    assert len(events) == 2
    assert isinstance(events[0], ResponseMessage)
    assert events[0].role == "user"
    assert isinstance(events[1], ResponseFunctionCall)
    assert events[1].call_id == "c1"


def test_unknown_types_dropped() -> None:
    events = parse_rollout_events(
        [
            _line("response_item", {"type": "some_future_item", "data": 1}),
            _line("event_msg", {"type": "some_future_event"}),
            _line("world_state", {"full": True, "state": {}}),
            {"not": "an envelope"},
            _line("response_item", {"type": "message", "role": "user", "content": []}),
        ]
    )
    # only the message survives
    assert len(events) == 1
    assert isinstance(events[0], ResponseMessage)


def test_compaction_alias_parsed() -> None:
    for item_type in ("compaction", "compaction_summary", "context_compaction"):
        event = parse_rollout_event(
            _line("response_item", {"type": item_type, "encrypted_content": "xyz"})
        )
        assert isinstance(event, ResponseCompaction)


def test_timestamp_parsing_and_monotonicity() -> None:
    assert parse_timestamp("2026-08-01T10:00:00.500Z") is not None
    assert parse_timestamp("not a timestamp") is None
    assert parse_timestamp(None) is None

    proc = _RolloutProcessor()
    e1 = ResponseMessage(timestamp="2026-08-01T10:00:01Z", role="user")
    e2 = ResponseMessage(timestamp="2026-08-01T10:00:00Z", role="user")  # earlier!
    t1 = proc.update_timestamp(e1)
    t2 = proc.update_timestamp(e2)
    assert t2 == t1 + timedelta(milliseconds=1)
