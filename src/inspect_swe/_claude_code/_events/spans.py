"""Post-process bridge events to annotate agent spans from JSONL output.

When Claude Code runs with `--output-format stream-json --verbose`, the JSONL
output contains `parent_tool_use_id` on subagent events, revealing the agent
hierarchy. This module post-processes the JSONL after completion to:

1. Emit SpanBeginEvent/SpanEndEvent for each agent span
2. Set span_id on existing bridge events to place them inside the correct span
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from inspect_ai.event import (
    Event,
    ModelEvent,
    SpanBeginEvent,
    SpanEndEvent,
)
from inspect_ai.log import transcript
from inspect_ai.util._span import current_span_id

from .events import to_span_begin_event, to_span_end_event
from .toolview import tool_view


@dataclass
class _AgentSpanInfo:
    """Metadata for an agent span discovered in JSONL."""

    span_id: str
    parent_span_id: str | None
    name: str
    description: str
    # Bridge timestamps from matched ModelEvents (accurate)
    min_timestamp: datetime | None = None
    max_timestamp: datetime | None = None


def annotate_agent_spans(jsonl_output: str) -> None:
    """Parse JSONL output and annotate transcript events with agent spans.

    Walks the JSONL to discover agent hierarchy via parent_tool_use_id,
    then matches bridge ModelEvents by message.id to assign span_id and
    emit SpanBeginEvent/SpanEndEvent.

    Args:
        jsonl_output: Raw JSONL string from Claude Code stdout.
    """
    span_id = current_span_id()
    if span_id is None:
        return
    new_events = _annotate_events(jsonl_output, transcript().events, span_id)
    for event in new_events:
        transcript()._event(event)


def _annotate_events(
    jsonl_output: str,
    events: Sequence[Event],
    agent_span_id: str,
) -> list[Event]:
    """Core annotation logic, decoupled from transcript/span globals.

    Mutates events in-place (reassigning span_id on matched events) and
    returns new SpanBeginEvent/SpanEndEvent instances to be appended.

    Args:
        jsonl_output: Raw JSONL string from Claude Code stdout.
        events: Existing events to annotate (mutated in place).
        agent_span_id: The span_id of the enclosing agent span.

    Returns:
        List of new SpanBeginEvent/SpanEndEvent to append.
    """
    # Apply tool views to ModelEvent ToolCalls that don't have them
    for event in events:
        if not isinstance(event, ModelEvent):
            continue
        if not event.output:
            continue
        for choice in event.output.choices:
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    if tc.view is None:
                        tc.view = tool_view(tc.function, tc.arguments)

    # Step 1: Parse JSONL and build agent hierarchy
    msg_id_to_span, agent_spans = _parse_agent_hierarchy(jsonl_output)

    if not agent_spans:
        return []

    # Step 2: Match ModelEvents and build anchor timeline
    anchors: list[tuple[datetime, str]] = []

    for event in events:
        if not isinstance(event, ModelEvent):
            continue
        if event.span_id != agent_span_id:
            continue
        if event.pending is not None:
            continue

        msg_id = _get_message_id(event)
        if msg_id is None:
            continue

        span_id = msg_id_to_span.get(msg_id)
        if span_id is None:
            continue

        # Reassign this ModelEvent to the agent span
        event.span_id = span_id
        anchors.append((event.timestamp, span_id))

        # Track span boundaries using bridge timestamps
        span_info = agent_spans.get(span_id)
        if span_info is not None:
            if (
                span_info.min_timestamp is None
                or event.timestamp < span_info.min_timestamp
            ):
                span_info.min_timestamp = event.timestamp
            if (
                span_info.max_timestamp is None
                or event.timestamp > span_info.max_timestamp
            ):
                span_info.max_timestamp = event.timestamp

    if not anchors:
        return []

    # Step 3: Assign remaining events via span time ranges
    # For each agent span with matched events, assign unmatched events that fall
    # within that span's [min_timestamp, max_timestamp] range. This avoids
    # sweeping up main-agent events that happen between subagent calls.
    for event in events:
        if isinstance(event, (SpanBeginEvent, SpanEndEvent)):
            continue
        if event.span_id != agent_span_id:
            continue

        for span_info in agent_spans.values():
            if span_info.min_timestamp is None or span_info.max_timestamp is None:
                continue
            if span_info.min_timestamp <= event.timestamp <= span_info.max_timestamp:
                event.span_id = span_info.span_id
                break

    # Step 4: Build SpanBeginEvent/SpanEndEvent
    return _build_span_events(agent_spans, agent_span_id)


def _parse_agent_hierarchy(
    jsonl_output: str,
) -> tuple[dict[str, str], dict[str, _AgentSpanInfo]]:
    """Parse JSONL and extract agent hierarchy.

    Returns:
        Tuple of (msg_id_to_span, agent_spans) where:
        - msg_id_to_span maps message.id -> span_id for subagent messages
        - agent_spans maps span_id -> AgentSpanInfo
    """
    msg_id_to_span: dict[str, str] = {}
    agent_spans: dict[str, _AgentSpanInfo] = {}
    last_parent_tool_use_id: str | None = None

    for line in jsonl_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = raw.get("type")
        if event_type != "assistant":
            continue

        message = raw.get("message", {})
        content = message.get("content", [])
        parent_tool_use_id = raw.get("parent_tool_use_id")

        # Check for Task/Agent tool_use blocks in this assistant event
        has_agent_tool = False
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") not in ("Task", "Agent"):
                continue

            has_agent_tool = True
            tool_use_id = block.get("id", "")
            tool_input = block.get("input", {})
            span_id = f"agent-{tool_use_id}"

            # Determine parent span
            parent_span: str | None = None
            if parent_tool_use_id:
                parent_span = f"agent-{parent_tool_use_id}"

            name = tool_input.get("name") or tool_input.get("subagent_type", "agent")
            description = tool_input.get("description", "")

            agent_spans[span_id] = _AgentSpanInfo(
                span_id=span_id,
                parent_span_id=parent_span,
                name=name,
                description=description,
            )
        # Map message.id to span for subagent messages
        msg_id = message.get("id")
        if msg_id and parent_tool_use_id:
            span_id = f"agent-{parent_tool_use_id}"
            msg_id_to_span[msg_id] = span_id

        # Track last seen parent_tool_use_id to handle orphan final messages.
        # When a subagent's final message lacks parent_tool_use_id, infer it
        # from the preceding subagent context.
        if parent_tool_use_id:
            last_parent_tool_use_id = parent_tool_use_id
        elif last_parent_tool_use_id and not has_agent_tool:
            # This assistant event follows subagent events but has no
            # parent_tool_use_id — likely the subagent's final message
            if msg_id:
                span_id = f"agent-{last_parent_tool_use_id}"
                msg_id_to_span[msg_id] = span_id
            last_parent_tool_use_id = None
        else:
            last_parent_tool_use_id = None

    return msg_id_to_span, agent_spans


def _get_message_id(event: ModelEvent) -> str | None:
    """Extract message.id from a ModelEvent."""
    if event.output and event.output.choices:
        return event.output.choices[0].message.id
    return None


def _build_span_events(
    agent_spans: dict[str, _AgentSpanInfo],
    agent_span_id: str,
) -> list[Event]:
    """Build SpanBeginEvent/SpanEndEvent for discovered agent spans.

    Returns a list of events rather than emitting them directly.
    """
    result: list[Event] = []

    # Sort spans by start timestamp (spans without timestamps go last)
    _FAR_FUTURE = datetime(9999, 1, 1, tzinfo=timezone.utc)
    sorted_spans = sorted(
        agent_spans.values(),
        key=lambda s: s.min_timestamp or _FAR_FUTURE,
    )

    for span_info in sorted_spans:
        if span_info.min_timestamp is None or span_info.max_timestamp is None:
            # No matched ModelEvents — skip this span
            continue

        parent_id = span_info.parent_span_id
        # If parent span exists but has no timestamps (wasn't matched),
        # fall back to our span
        if parent_id and parent_id in agent_spans:
            parent_info = agent_spans[parent_id]
            if parent_info.min_timestamp is None:
                parent_id = agent_span_id
        elif parent_id is None:
            parent_id = agent_span_id

        begin_ts = span_info.min_timestamp - timedelta(milliseconds=1)
        end_ts = span_info.max_timestamp + timedelta(milliseconds=1)

        begin_event = to_span_begin_event(
            span_id=span_info.span_id,
            name=span_info.name,
            span_type="agent",
            timestamp=begin_ts,
            parent_id=parent_id,
            metadata={"description": span_info.description}
            if span_info.description
            else None,
        )
        end_event = to_span_end_event(
            span_id=span_info.span_id,
            timestamp=end_ts,
        )

        result.append(begin_event)
        result.append(end_event)

    return result
