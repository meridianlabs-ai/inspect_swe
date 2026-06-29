from typing import Any

import pytest
from inspect_swe._claude_code.claude_code import (
    _claude_code_stop_reason,
    _is_claude_code_refusal_exit,
    _update_terminal_stop_reason,
)


def _resolve_terminal_stop_reason(events: list[dict[str, Any]]) -> str | None:
    terminal_stop_reason: str | None = None
    saw_result_event = False
    for raw in events:
        terminal_stop_reason, saw_result_event = _update_terminal_stop_reason(
            raw, terminal_stop_reason, saw_result_event
        )
    return terminal_stop_reason


def _assistant(stop_reason: str | None) -> dict[str, Any]:
    return {"type": "assistant", "message": {"stop_reason": stop_reason}}


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        # refusal assistant message followed by a result event that omits
        # stop_reason (the real Claude Code shape) must not be clobbered
        (
            [_assistant("refusal"), {"type": "result", "subtype": "success"}],
            "refusal",
        ),
        # partial stream: refusal assistant message, no result event
        ([_assistant("refusal")], "refusal"),
        # result event with an explicit stop_reason is authoritative
        (
            [_assistant("end_turn"), {"type": "result", "stop_reason": "refusal"}],
            "refusal",
        ),
        # assistant events after a result event do not override it
        (
            [{"type": "result", "stop_reason": "stop"}, _assistant("refusal")],
            "stop",
        ),
        # ordinary completion stays non-refusal
        (
            [_assistant("end_turn"), {"type": "result", "subtype": "success"}],
            "end_turn",
        ),
    ],
)
def test_resolve_terminal_stop_reason(
    events: list[dict[str, Any]], expected: str | None
) -> None:
    assert _resolve_terminal_stop_reason(events) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {
                "type": "assistant",
                "message": {"role": "assistant", "stop_reason": "refusal"},
            },
            "refusal",
        ),
        ({"type": "result", "stop_reason": "refusal"}, "refusal"),
        ({"type": "assistant", "message": "bad"}, None),
        ({"type": "system", "stop_reason": "refusal"}, None),
    ],
)
def test_claude_code_stop_reason(raw: dict[str, Any], expected: str | None) -> None:
    assert _claude_code_stop_reason(raw) == expected


@pytest.mark.parametrize(
    ("exit_code", "stderr_data", "stop_reason", "expected"),
    [
        (1, "", "refusal", True),
        (1, "", "stop", False),
        (1, "", None, False),
        (1, "boom", "refusal", False),
        (2, "", "refusal", False),
    ],
)
def test_claude_code_refusal_exit(
    exit_code: int,
    stderr_data: str,
    stop_reason: str | None,
    expected: bool,
) -> None:
    assert (
        _is_claude_code_refusal_exit(
            exit_code=exit_code,
            stderr_data=stderr_data,
            stop_reason=stop_reason,
        )
        is expected
    )
