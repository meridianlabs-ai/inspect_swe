"""Tests for `claude_code_event_stream`'s exit-status contract.

`claude_code()` initialises its exit code to 0 and only an `ExitEvent`
overwrites it, so the stream's promise of exactly one `ExitEvent` is what
stands between a killed Claude Code process and a scored sample. inspect_ai's
`ExecRemoteProcess` stops iterating without an `ExecCompleted` when the
process was killed (`kill()` discards output), so the stream has to treat a
missing `ExecCompleted` as an error rather than fall through silently.
"""

from typing import cast

import pytest
from inspect_ai.util import (
    ExecCompleted,
    ExecOutput,
    ExecRemoteProcess,
    ExecStderr,
    ExecStdout,
)
from inspect_swe._claude_code._events.stream import (
    ClaudeCodeStreamEvent,
    ExitEvent,
    JsonlEvent,
    StderrEvent,
    claude_code_event_stream,
)


class _FakeProcess:
    """`ExecRemoteProcess` stand-in that replays a fixed list of events.

    Like the real one, it raises `StopAsyncIteration` once the list is drained;
    whether an `ExecCompleted` was among the events is up to the test.
    """

    def __init__(self, events: list[ExecOutput]) -> None:
        self._events = list(events)

    def __aiter__(self) -> "_FakeProcess":
        return self

    async def __anext__(self) -> ExecOutput:
        if self._events:
            return self._events.pop(0)
        raise StopAsyncIteration


async def _drain(events: list[ExecOutput]) -> list[ClaudeCodeStreamEvent]:
    proc = cast(ExecRemoteProcess, _FakeProcess(events))
    return [event async for event in claude_code_event_stream(proc)]


_INIT_LINE = '{"type": "system", "subtype": "init"}\n'


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize(
    "exit_code",
    [0, 1],
)
async def test_completed_process_yields_one_final_exit_event(exit_code: int) -> None:
    events = await _drain(
        [
            ExecStdout(data=_INIT_LINE),
            ExecStderr(data="warning\n"),
            # partial trailing line, flushed when ExecCompleted arrives
            ExecStdout(data='{"type": "result"}'),
            ExecCompleted(exit_code=exit_code),
        ]
    )

    assert events == [
        JsonlEvent(raw={"type": "system", "subtype": "init"}, line=_INIT_LINE.strip()),
        StderrEvent(data="warning\n"),
        JsonlEvent(raw={"type": "result"}, line='{"type": "result"}'),
        ExitEvent(code=exit_code),
    ]


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize(
    "events",
    [
        pytest.param([], id="no-output"),
        pytest.param([ExecStdout(data=_INIT_LINE)], id="stdout-only"),
        pytest.param(
            [ExecStdout(data=_INIT_LINE), ExecStderr(data="killed\n")],
            id="stdout-and-stderr",
        ),
    ],
)
async def test_process_without_exit_status_raises(events: list[ExecOutput]) -> None:
    with pytest.raises(RuntimeError, match="without reporting an exit status"):
        await _drain(events)


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_events_before_the_missing_exit_status_are_still_delivered() -> None:
    """The consumer still sees everything the process emitted before dying."""
    proc = cast(
        ExecRemoteProcess,
        _FakeProcess([ExecStdout(data=_INIT_LINE), ExecStderr(data="killed\n")]),
    )
    seen: list[ClaudeCodeStreamEvent] = []
    with pytest.raises(RuntimeError):
        async for event in claude_code_event_stream(proc):
            seen.append(event)

    assert seen == [
        JsonlEvent(raw={"type": "system", "subtype": "init"}, line=_INIT_LINE.strip()),
        StderrEvent(data="killed\n"),
    ]
    assert not any(isinstance(event, ExitEvent) for event in seen)
