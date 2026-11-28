"""Conversion of Claude Code session events to Inspect events."""

from typing import Any

import pytest
from inspect_ai.event import Event, ModelEvent
from inspect_swe._claude_code._events import claude_code_events, process_parsed_events
from inspect_swe._claude_code._events.extraction import sum_scout_tokens
from inspect_swe._claude_code._events.models import (
    consolidate_assistant_events,
    parse_events,
)

USAGE = {
    "input_tokens": 2,
    "output_tokens": 226,
    "cache_read_input_tokens": 43004,
    "cache_creation_input_tokens": 0,
}


def _assistant(
    uuid: str, parent: str, tool_id: str, usage: bool = True
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "model": "claude-opus-4-5",
        "id": "msg_1",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "Read",
                "input": {"file_path": f"{tool_id}.txt"},
            }
        ],
        "stop_reason": "tool_use",
    }
    if usage:
        message["usage"] = USAGE
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-11-11T18:51:40.000Z",
        "sessionId": "s1",
        "message": message,
    }


def _tool_result(uuid: str, parent: str, tool_id: str) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-11-11T18:51:41.000Z",
        "sessionId": "s1",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}
            ],
        },
    }


# One API response (msg_1) with two tool calls: Claude Code runs the first tool
# before the second tool_use has streamed, so the tool result splits the
# response's fragments, and each fragment carries the whole response's usage.
def _split_response(first_usage: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "type": "user",
            "uuid": "u0",
            "parentUuid": None,
            "timestamp": "2026-11-11T18:51:39.000Z",
            "sessionId": "s1",
            "message": {"role": "user", "content": "Read a.txt and b.txt"},
        },
        _assistant("a1", "u0", "toolu_A", usage=first_usage),
        _tool_result("u1", "a1", "toolu_A"),
        _assistant("a2", "u1", "toolu_B"),
        _tool_result("u2", "a2", "toolu_B"),
    ]


async def _convert_batch(raw: list[dict[str, Any]]) -> list[Event]:
    events = consolidate_assistant_events(parse_events(raw))
    return [e async for e in process_parsed_events(events)]


async def _convert_stream(raw: list[dict[str, Any]]) -> list[Event]:
    return [e async for e in claude_code_events(raw)]


@pytest.mark.anyio
@pytest.mark.parametrize("convert", [_convert_batch, _convert_stream])
async def test_split_response_usage_counted_once(convert: Any) -> None:
    events = await convert(_split_response())

    model_events = [e for e in events if isinstance(e, ModelEvent)]
    # One model event per assistant line is kept ...
    assert len(model_events) == 2
    first, second = model_events
    assert first.output.usage is not None
    assert first.output.usage.total_tokens == sum(USAGE.values())
    # ... but the response's usage is only attached once.
    assert second.output.usage is None
    assert sum_scout_tokens(events) == sum(USAGE.values())


@pytest.mark.anyio
@pytest.mark.parametrize("convert", [_convert_batch, _convert_stream])
async def test_split_response_usage_kept_when_first_fragment_has_none(
    convert: Any,
) -> None:
    events = await convert(_split_response(first_usage=False))

    model_events = [e for e in events if isinstance(e, ModelEvent)]
    assert len(model_events) == 2
    first, second = model_events
    assert first.output.usage is None
    assert second.output.usage is not None
    assert sum_scout_tokens(events) == sum(USAGE.values())
