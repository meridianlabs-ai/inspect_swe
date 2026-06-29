from typing import Any

import pytest
from inspect_swe._claude_code.claude_code import (
    _claude_code_stop_reason,
    _is_claude_code_refusal_exit,
)


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
