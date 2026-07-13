"""Async generator for consuming Claude Code's JSONL output stream.

`claude_code_event_stream` wraps the raw `ExecRemoteProcess` event stream
(which mixes `ExecStdout` / `ExecStderr` / `ExecCompleted`) and yields a
tagged sequence with stdout already parsed into JSONL dicts. The caller
gets a single `async for` loop and doesn't have to manage line-buffering
or JSON parsing inline.

Yields exactly one `ExitEvent` before terminating; trailing partial
stdout lines are flushed before that.
"""

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from inspect_ai.util import (
    ExecCompleted,
    ExecRemoteProcess,
    ExecStderr,
    ExecStdout,
)


@dataclass
class JsonlEvent:
    """A single parsed JSONL line from Claude Code's stdout."""

    raw: dict[str, Any]
    line: str


@dataclass
class JsonlParseError:
    """A stdout line that could not be parsed as JSON."""

    line: str


@dataclass
class StderrEvent:
    """A chunk of stderr output."""

    data: str


@dataclass
class ExitEvent:
    """Process exit; always the final event yielded."""

    code: int


ClaudeCodeStreamEvent = JsonlEvent | JsonlParseError | StderrEvent | ExitEvent


async def claude_code_event_stream(
    proc: ExecRemoteProcess,
) -> AsyncIterator[ClaudeCodeStreamEvent]:
    """Yield parsed JSONL events + stderr + exit from a Claude Code subprocess.

    Owns line-buffering across `ExecStdout` chunks and JSON parsing.
    Malformed lines surface as `JsonlParseError`; the caller decides how to
    handle them (typically: append to a debug log). Trailing partial lines
    are flushed when `ExecCompleted` arrives, before the final `ExitEvent`.
    """
    buffer = ""
    async for event in proc:
        if isinstance(event, ExecStdout):
            buffer += event.data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    yield JsonlEvent(raw=json.loads(line), line=line)
                except json.JSONDecodeError:
                    yield JsonlParseError(line=line)
        elif isinstance(event, ExecStderr):
            yield StderrEvent(data=event.data)
        elif isinstance(event, ExecCompleted):
            tail = buffer.strip()
            if tail:
                try:
                    yield JsonlEvent(raw=json.loads(tail), line=tail)
                except json.JSONDecodeError:
                    yield JsonlParseError(line=tail)
                buffer = ""
            yield ExitEvent(code=event.exit_code)
