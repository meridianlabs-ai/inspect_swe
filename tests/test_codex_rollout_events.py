"""Regression tests for Codex rollout-file event conversion."""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from inspect_ai.event import Event, ModelEvent, SpanBeginEvent
from inspect_swe._codex_cli._events.rollout import process_rollout_events
from inspect_swe._codex_cli._events.rollout_extraction import (
    is_context_message,
    output_to_result,
)
from inspect_swe._codex_cli._events.rollout_models import (
    SubAgentActivityEvent,
    parse_rollout_events,
)


def _line(line_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": "2026-08-10T20:05:06.000Z",
        "type": line_type,
        "payload": payload,
    }


def _message(role: str, text: str) -> dict[str, Any]:
    return _line(
        "response_item",
        {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    )


async def _convert(raw_lines: Sequence[dict[str, Any]]) -> list[Event]:
    events = parse_rollout_events(list(raw_lines))
    return [event async for event in process_rollout_events(events)]


@pytest.mark.parametrize(
    "text",
    [
        "<recommended_plugins>\n<plugin>example</plugin>\n</recommended_plugins>",
        "# AGENTS.md instructions for /workspace\n\n<INSTRUCTIONS>...</INSTRUCTIONS>",
    ],
)
def test_modern_injected_messages_are_context(text: str) -> None:
    assert is_context_message(text)


@pytest.mark.parametrize(
    "text",
    [
        # A genuine user message may start with the AGENTS.md heading; only
        # the full injected shape (with the closing marker) is context.
        "# AGENTS.md instructions are being ignored, why?",
        "# AGENTS.md instructions\n\nplease review my draft below",
    ],
)
def test_agents_md_prefix_alone_is_genuine_user_speech(text: str) -> None:
    assert not is_context_message(text)


def test_output_to_result_unwraps_legacy_form() -> None:
    legacy = '{"output": "hello\\n", "metadata": {"exit_code": 1, "duration_seconds": 0.2}}'
    assert output_to_result(legacy) == ("hello\n", 1)
    assert output_to_result('{"output": "ok"}') == ("ok", None)


@pytest.mark.parametrize(
    "text",
    [
        # Genuine tool output that happens to be JSON with an "output" key
        # (e.g. `cat config.json`) must pass through verbatim.
        '{"output": "x", "metadata": {"exit_code": 1}, "extra": true}',
        '{"output": {"nested": 1}, "metadata": {"exit_code": 1}}',
        '{"output": "x", "metadata": "not-a-dict"}',
    ],
)
def test_output_to_result_passes_through_non_legacy_json(text: str) -> None:
    assert output_to_result(text) == (text, None)


def test_context_messages_do_not_shift_rollback_boundary() -> None:
    scout_events = asyncio.run(
        _convert(
            [
                _message("user", "first user turn"),
                _message("assistant", "first response"),
                _message("user", "<recommended_plugins>plugins</recommended_plugins>"),
                _message("user", "second user turn"),
                _message("assistant", "second response"),
                _message(
                    "user",
                    "# AGENTS.md instructions for /workspace\n\n"
                    "<INSTRUCTIONS>be helpful</INSTRUCTIONS>",
                ),
                _line("event_msg", {"type": "thread_rolled_back", "num_turns": 1}),
                _message("user", "replacement user turn"),
                _message("assistant", "replacement response"),
            ]
        )
    )

    final_model_event = next(
        event for event in reversed(scout_events) if isinstance(event, ModelEvent)
    )
    input_text = "\n".join(message.text for message in final_model_event.input)
    assert "first user turn" in input_text
    assert "second user turn" not in input_text
    assert "second response" not in input_text
    assert "replacement user turn" in input_text


def test_genuine_agents_md_prefixed_turn_is_a_rollback_boundary() -> None:
    """A real user message starting with the AGENTS.md heading (no closing
    marker) must count as a genuine turn, so num_turns=1 rolls back only it."""
    scout_events = asyncio.run(
        _convert(
            [
                _message("user", "first user turn"),
                _message("assistant", "first response"),
                _message("user", "# AGENTS.md instructions are being ignored, why?"),
                _message("assistant", "second response"),
                _line("event_msg", {"type": "thread_rolled_back", "num_turns": 1}),
                _message("user", "replacement user turn"),
                _message("assistant", "replacement response"),
            ]
        )
    )

    final_model_event = next(
        event for event in reversed(scout_events) if isinstance(event, ModelEvent)
    )
    input_text = "\n".join(message.text for message in final_model_event.input)
    assert "first user turn" in input_text
    assert "first response" in input_text
    assert "AGENTS.md instructions are being ignored" not in input_text
    assert "second response" not in input_text
    assert "replacement user turn" in input_text


def test_subagent_activity_links_modern_spawn_result_to_child_thread() -> None:
    loader_calls: list[tuple[str, int]] = []

    async def fake_loader(thread_id: str, max_depth: int) -> list[Event]:
        loader_calls.append((thread_id, max_depth))
        return []

    parsed = parse_rollout_events(
        [
            _message("user", "delegate this"),
            _line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "arguments": '{"task_name":"importer_qa_fixture"}',
                    "call_id": "call_spawn",
                },
            ),
            _line(
                "event_msg",
                {
                    "type": "sub_agent_activity",
                    "event_id": "call_spawn",
                    "agent_thread_id": "019fed47-6294-7070-b852-b370a8e708cc",
                    "agent_path": "/root/importer_qa_fixture",
                    "kind": "started",
                },
            ),
            _line(
                "response_item",
                {
                    "type": "function_call_output",
                    "call_id": "call_spawn",
                    "output": '{"task_name":"/root/importer_qa_fixture"}',
                },
            ),
        ]
    )
    assert isinstance(parsed[2], SubAgentActivityEvent)

    async def convert() -> list[Event]:
        return [
            event
            async for event in process_rollout_events(parsed, child_loader=fake_loader)
        ]

    scout_events = asyncio.run(convert())

    assert loader_calls == [("019fed47-6294-7070-b852-b370a8e708cc", 4)]
    agent_span = next(
        event
        for event in scout_events
        if isinstance(event, SpanBeginEvent) and event.type == "agent"
    )
    assert agent_span.name == "importer_qa_fixture"
    assert agent_span.metadata == {
        "agent_type": None,
        "task_name": "importer_qa_fixture",
        "thread_id": "019fed47-6294-7070-b852-b370a8e708cc",
        "agent_path": "/root/importer_qa_fixture",
    }
