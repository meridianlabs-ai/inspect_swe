"""Conversion of Codex CLI rollout files to Scout/Inspect events.

Mirrors ``_claude_code/_events/events.py``: a ``_RolloutProcessor`` walks the
parsed rollout lines in file order, reconstructing what the model saw
(``accumulated_messages``) and emitting Inspect events:

- assistant-side response items (reasoning / assistant message / tool calls)
  are buffered and flushed into a single ``ModelEvent`` per model response
- ``function_call`` / ``function_call_output`` pairs (matched by ``call_id``)
  become tool spans: SpanBegin(type="tool") → ToolEvent → SpanEnd
- ``spawn_agent`` calls become agent spans with the child thread's events
  nested inside (loaded via a ``ChildThreadLoader`` callback — file-based in
  inspect_scout)
- ``compacted`` items become ``CompactionEvent``s; accumulated messages are
  reset to the item's ``replacement_history`` (the exact post-compaction
  context)
- ``thread_rolled_back`` (undo) truncates accumulated messages so subsequent
  ``ModelEvent.input`` reflects what the model actually saw; already-emitted
  events stay in the timeline with an InfoEvent marking the boundary
- ``token_count`` events supply each ModelEvent's usage; the usage is looked
  ahead at flush time so events are complete when yielded (consumers may
  serialize them as they stream)

Unlike Claude Code there is no uuid/parentUuid tree (rollouts are
append-ordered) and no shared-id consolidation (one model response is a run
of consecutive assistant-side items).
"""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging import getLogger
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol

from inspect_ai.event import (
    CompactionEvent,
    Event,
    InfoEvent,
    ModelEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.model import Content, ContentText, ModelOutput, ModelUsage
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ChatCompletionChoice
from inspect_ai.tool._tool import ToolResult
from inspect_ai.tool._tool_call import ToolCall, ToolCallError

from .detection import SPAWN_AGENT
from .rollout_extraction import (
    content_items_to_content,
    history_to_messages,
    is_context_message,
    output_to_result,
    parse_arguments,
    parse_timestamp,
    reasoning_to_content,
    total_tokens_from_token_info,
    usage_from_token_info,
)
from .rollout_models import (
    CompactedEvent,
    ResponseCompaction,
    ResponseCustomToolCall,
    ResponseCustomToolCallOutput,
    ResponseFunctionCall,
    ResponseFunctionCallOutput,
    ResponseLocalShellCall,
    ResponseMessage,
    ResponseReasoning,
    ResponseWebSearchCall,
    ReviewModeEvent,
    RolloutEvent,
    SessionMetaEvent,
    SubAgentActivityEvent,
    ThreadRolledBackEvent,
    TokenCountEvent,
    TurnAbortedEvent,
    TurnCompleteEvent,
    TurnContextEvent,
)
from .toolview import tool_view

logger = getLogger(__name__)

CODEX_EVENT_SOURCE = "codex_cli"

# Sentinel timestamp for events with unparseable timestamps.
# Using epoch avoids extending timelines to the present day.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class ChildThreadLoader(Protocol):
    """Callback for loading a spawned child thread's events.

    Implementations locate the child rollout file by thread id
    (inspect_scout loads from ``$CODEX_HOME/sessions``) and return its
    converted events for nesting under the parent's agent span.
    """

    async def __call__(self, thread_id: str, max_depth: int) -> list[Event]: ...


@dataclass
class _PendingCall:
    """A tool call awaiting its output item."""

    call_id: str
    function: str
    arguments: dict[str, Any]
    timestamp: datetime
    is_spawn: bool = False
    spawn_agent_type: str | None = None
    spawn_task_name: str | None = None
    spawn_thread_id: str | None = None
    spawn_agent_path: str | None = None


def _str_arg(arguments: dict[str, Any], key: str) -> str | None:
    """A truthy tool-call argument as a string, else None."""
    value = arguments.get(key)
    return str(value) if value else None


def _spawn_result_thread_id(result: str) -> tuple[str | None, str | None]:
    """Extract (thread_id, nickname) from a spawn_agent tool result.

    V1 results carry ``{"agent_id": <thread uuid>, "nickname": ...}``; V2
    results carry ``{"task_name": "/root/<name>", "nickname": ...}`` with no
    thread id at all — the thread id then comes from the ``sub_agent_activity``
    event instead (the task path is not a rollout thread id, so unlike
    ``detection.spawn_result`` it is not used as one here).
    """
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if isinstance(data, dict):
        agent_id = data.get("agent_id")
        nickname = data.get("nickname")
        return (
            agent_id if isinstance(agent_id, str) and agent_id else None,
            nickname if isinstance(nickname, str) and nickname else None,
        )
    return None, None


class _RolloutProcessor:
    """Stateful conversion of parsed rollout events to Inspect events."""

    def __init__(
        self,
        max_depth: int = 5,
        child_loader: ChildThreadLoader | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.child_loader = child_loader

        self.accumulated_messages: list[ChatMessage] = []
        # Index into accumulated_messages where each genuine user turn began
        # (used to replay thread_rolled_back truncation).
        self.user_turn_starts: list[int] = []
        self.pending_calls: dict[str, _PendingCall] = {}
        # Buffered assistant-side items awaiting flush into one ModelEvent
        self.assistant_buffer: list[RolloutEvent] = []
        self.buffer_timestamp: datetime | None = None
        self.current_model: str | None = None
        self.last_total_tokens: int | None = None
        self.last_timestamp: datetime = _EPOCH

    def update_timestamp(self, event: RolloutEvent) -> datetime:
        """Parse and track the latest timestamp, ensuring monotonic ordering."""
        timestamp = parse_timestamp(event.timestamp) or self.last_timestamp
        if timestamp <= self.last_timestamp:
            timestamp = self.last_timestamp + timedelta(milliseconds=1)
        self.last_timestamp = timestamp
        return timestamp

    # ── assistant-side buffering ─────────────────────────────────────────

    def is_assistant_side(self, event: RolloutEvent) -> bool:
        """Whether this item is part of a model response (buffered)."""
        if isinstance(event, ResponseMessage):
            return event.role == "assistant"
        return isinstance(
            event,
            (
                ResponseReasoning,
                ResponseFunctionCall,
                ResponseLocalShellCall,
                ResponseCustomToolCall,
                ResponseWebSearchCall,
            ),
        )

    def buffer_assistant(self, event: RolloutEvent, timestamp: datetime) -> None:
        """Buffer an assistant-side item for the next ModelEvent flush."""
        if not self.assistant_buffer:
            self.buffer_timestamp = timestamp
        self.assistant_buffer.append(event)

    def flush_model(self, usage: ModelUsage | None = None) -> list[Event]:
        """Convert buffered assistant-side items into a ModelEvent.

        ``usage`` comes from the caller's lookahead to the response's
        ``token_count`` event, so the ModelEvent is complete when yielded.

        Also emits self-contained spans for hosted ``web_search_call`` items
        (which have no separate output item) and registers pending tool
        calls for later matching by ``call_id``.
        """
        if not self.assistant_buffer:
            return []

        timestamp = self.buffer_timestamp or self.last_timestamp
        content: list[Content] = []
        tool_calls: list[ToolCall] = []
        web_search_events: list[Event] = []
        message_id: str | None = None

        for item in self.assistant_buffer:
            if isinstance(item, ResponseReasoning):
                reasoning = reasoning_to_content(item)
                if reasoning is not None:
                    content.append(reasoning)
            elif isinstance(item, ResponseMessage):
                if message_id is None and item.id:
                    message_id = item.id
                converted = content_items_to_content(item.content)
                if isinstance(converted, str):
                    if converted:
                        content.append(ContentText(text=converted))
                else:
                    content.extend(converted)
            elif isinstance(item, ResponseFunctionCall):
                arguments = parse_arguments(item.arguments)
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        function=item.name,
                        arguments=arguments,
                        view=tool_view(item.name, arguments),
                    )
                )
                is_spawn = item.name == SPAWN_AGENT
                self.pending_calls[item.call_id] = _PendingCall(
                    call_id=item.call_id,
                    function=item.name,
                    arguments=arguments,
                    timestamp=timestamp,
                    is_spawn=is_spawn,
                    spawn_agent_type=(
                        _str_arg(arguments, "agent_type") if is_spawn else None
                    ),
                    spawn_task_name=(
                        _str_arg(arguments, "task_name") if is_spawn else None
                    ),
                )
            elif isinstance(item, ResponseLocalShellCall):
                call_id = item.call_id or item.id or ""
                arguments = dict(item.action)
                arguments.pop("type", None)
                tool_calls.append(
                    ToolCall(id=call_id, function="local_shell", arguments=arguments)
                )
                if call_id:
                    self.pending_calls[call_id] = _PendingCall(
                        call_id=call_id,
                        function="local_shell",
                        arguments=arguments,
                        timestamp=timestamp,
                    )
            elif isinstance(item, ResponseCustomToolCall):
                arguments = {"input": item.input}
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        function=item.name,
                        arguments=arguments,
                        view=tool_view(item.name, arguments),
                        type="custom",
                    )
                )
                self.pending_calls[item.call_id] = _PendingCall(
                    call_id=item.call_id,
                    function=item.name,
                    arguments=arguments,
                    timestamp=timestamp,
                )
            elif isinstance(item, ResponseWebSearchCall):
                # Hosted tool: no output item, so emit a self-contained span.
                action = item.action or {}
                arguments = {k: v for k, v in action.items() if k != "type"}
                call_id = item.id or f"web_search_{len(web_search_events)}"
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        function="web_search",
                        arguments=arguments,
                        view=tool_view("web_search", arguments),
                    )
                )
                web_search_events.extend(
                    _tool_span_events(
                        call_id=call_id,
                        function="web_search",
                        arguments=arguments,
                        result=item.status or "",
                        error=None,
                        timestamp=timestamp,
                        completed=timestamp,
                    )
                )

        self.assistant_buffer = []
        self.buffer_timestamp = None

        output_content: str | list[Content]
        if len(content) == 1 and isinstance(content[0], ContentText):
            output_content = content[0].text
        else:
            output_content = content if content else ""

        output_message = ChatMessageAssistant(
            id=message_id,
            content=output_content,
            tool_calls=tool_calls if tool_calls else None,
        )

        stop_reason: Literal["stop", "tool_calls"] = (
            "tool_calls" if tool_calls else "stop"
        )
        model = self.current_model or "unknown"
        model_event = ModelEvent(
            model=model,
            input=list(self.accumulated_messages),
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
            output=ModelOutput(
                model=model,
                usage=usage,
                choices=[
                    ChatCompletionChoice(
                        message=output_message, stop_reason=stop_reason
                    )
                ],
            ),
            timestamp=timestamp,
        )
        self.accumulated_messages.append(output_message)

        return [model_event, *web_search_events]

    # ── boundary items ───────────────────────────────────────────────────

    def process_user_message(self, event: ResponseMessage) -> None:
        """Accumulate a user/developer message (no event emitted).

        All model-visible messages are accumulated for ModelEvent.input
        fidelity — including injected context blocks (user_instructions,
        environment_context, ...), which are model context even though they
        are not user speech. Genuine user turns are tracked for rollback.
        """
        converted = content_items_to_content(event.content)
        if not converted:
            return
        if event.role in ("developer", "system"):
            self.accumulated_messages.append(ChatMessageSystem(content=converted))
        else:
            text = converted if isinstance(converted, str) else ""
            genuine = not (text and is_context_message(text))
            if genuine:
                self.user_turn_starts.append(len(self.accumulated_messages))
            self.accumulated_messages.append(ChatMessageUser(content=converted))

    async def process_call_output(
        self,
        call_id: str,
        output: Any,
        timestamp: datetime,
    ) -> list[Event]:
        """Match a tool output to its pending call and emit the tool span."""
        pending = self.pending_calls.pop(call_id, None)
        result, exit_code = output_to_result(output)
        error: ToolCallError | None = None
        if exit_code is not None and exit_code != 0:
            error = ToolCallError(
                type="unknown",
                message=result if isinstance(result, str) else str(result),
            )

        if pending is None:
            # Output with no matching call (e.g. prefix truncated by
            # compaction replay or a malformed file) — record for fidelity.
            logger.debug(f"Tool output with no matching call: {call_id}")
            self.accumulated_messages.append(
                ChatMessageTool(content=result, tool_call_id=call_id)
            )
            return []

        events: list[Event] = []
        if pending.is_spawn:
            events = await self._spawn_agent_span_events(
                pending, result, error, timestamp
            )
        else:
            events = _tool_span_events(
                call_id=pending.call_id,
                function=pending.function,
                arguments=pending.arguments,
                result=result,
                error=error,
                timestamp=pending.timestamp,
                completed=timestamp,
            )

        self.accumulated_messages.append(
            ChatMessageTool(
                content=result,
                tool_call_id=pending.call_id,
                function=pending.function,
            )
        )
        return events

    def process_sub_agent_activity(self, event: SubAgentActivityEvent) -> None:
        """Bind a modern sub-agent activity event to its pending spawn call."""
        pending = self.pending_calls.get(event.event_id)
        if pending is None or not pending.is_spawn:
            logger.debug(
                f"Sub-agent activity with no matching spawn call: {event.event_id}"
            )
            return
        pending.spawn_thread_id = event.agent_thread_id
        # agent_path is optional on the wire; never erase a previously bound
        # path with a later event that omits it.
        if event.agent_path:
            pending.spawn_agent_path = event.agent_path

    async def _spawn_agent_span_events(
        self,
        pending: _PendingCall,
        result: str | list[Content],
        error: ToolCallError | None,
        timestamp: datetime,
    ) -> list[Event]:
        """Create the agent span for a spawn_agent call (child events nested)."""
        agent_span_id = f"agent-{pending.call_id}"
        result_text = result if isinstance(result, str) else ""
        result_thread_id, nickname = _spawn_result_thread_id(result_text)
        thread_id = pending.spawn_thread_id or result_thread_id
        path_name = (
            PurePosixPath(pending.spawn_agent_path).name
            if pending.spawn_agent_path
            else None
        )
        agent_name = (
            nickname
            or pending.spawn_task_name
            or pending.spawn_agent_type
            or path_name
            or "agent"
        )

        # Same shape as the live bridge (consumer.py): keys present only when
        # known, so consumers need not distinguish absent from None.
        metadata = {
            key: value
            for key, value in {
                "agent_type": pending.spawn_agent_type,
                "task_name": pending.spawn_task_name,
                "thread_id": thread_id,
                "agent_path": pending.spawn_agent_path,
            }.items()
            if value is not None
        }

        events: list[Event] = [
            _span_begin(
                span_id=agent_span_id,
                name=agent_name,
                span_type="agent",
                timestamp=pending.timestamp,
                metadata=metadata or None,
            )
        ]

        tool_event = _to_tool_event(
            call_id=pending.call_id,
            function=pending.function,
            arguments=pending.arguments,
            result=result,
            error=error,
            timestamp=pending.timestamp,
            completed=timestamp,
        )
        tool_event.span_id = agent_span_id
        tool_event.agent_span_id = agent_span_id
        events.append(tool_event)

        if thread_id and self.child_loader and self.max_depth > 0:
            child_events = await self.child_loader(thread_id, self.max_depth - 1)
            # Re-parent top-level items so event_tree() nests them under
            # the agent span
            for evt in child_events:
                if isinstance(evt, SpanBeginEvent):
                    if evt.parent_id is None:
                        evt.parent_id = agent_span_id
                elif not isinstance(evt, SpanEndEvent):
                    if evt.span_id is None:
                        evt.span_id = agent_span_id
            events.extend(child_events)

        events.append(SpanEndEvent(id=agent_span_id, timestamp=timestamp))
        return events

    def process_compacted(
        self, event: CompactedEvent, timestamp: datetime
    ) -> list[Event]:
        """Handle a local compaction checkpoint."""
        compaction = CompactionEvent(
            source=CODEX_EVENT_SOURCE,
            tokens_before=self.last_total_tokens,
            metadata={"trigger": "local", "message": event.message},
            timestamp=timestamp,
        )
        # replacement_history is the exact post-compaction context
        if event.replacement_history is not None:
            self.accumulated_messages = history_to_messages(event.replacement_history)
        elif event.message:
            self.accumulated_messages = [ChatMessageAssistant(content=event.message)]
        else:
            self.accumulated_messages = []
        self.user_turn_starts = []
        return [compaction]

    def process_remote_compaction(
        self, event: ResponseCompaction, timestamp: datetime
    ) -> list[Event]:
        """Handle a remote/server-side compaction artifact (encrypted)."""
        compaction = CompactionEvent(
            source=CODEX_EVENT_SOURCE,
            tokens_before=self.last_total_tokens,
            metadata={"trigger": "remote", "encrypted": True},
            timestamp=timestamp,
        )
        self.accumulated_messages = []
        self.user_turn_starts = []
        return [compaction]

    def process_token_count(self, event: TokenCountEvent) -> None:
        """Track the cumulative total (per-response usage is attached by the
        flush-time lookahead in ``process_rollout_events``)."""
        if not event.info:
            return
        total = total_tokens_from_token_info(event.info)
        if total is not None:
            self.last_total_tokens = total

    def process_rolled_back(
        self, event: ThreadRolledBackEvent, timestamp: datetime
    ) -> list[Event]:
        """Replay an undo: truncate accumulated messages by num_turns user turns.

        Already-emitted events stay in the timeline (they really happened);
        this only affects what subsequent ModelEvent.input reflects.
        """
        num_turns = event.num_turns
        if num_turns > 0 and self.user_turn_starts:
            num_turns = min(num_turns, len(self.user_turn_starts))
            truncate_at = self.user_turn_starts[-num_turns]
            del self.accumulated_messages[truncate_at:]
            del self.user_turn_starts[-num_turns:]
        return [
            InfoEvent(
                source=CODEX_EVENT_SOURCE,
                data={"type": "thread_rolled_back", "num_turns": event.num_turns},
                timestamp=timestamp,
            )
        ]

    async def process_turn_aborted(
        self, event: TurnAbortedEvent, timestamp: datetime
    ) -> list[Event]:
        """Handle an interrupt: flush dangling calls and mark the boundary."""
        events = await self.flush_pending(
            error=ToolCallError(
                type="unknown", message=f"aborted: {event.reason or 'interrupted'}"
            ),
            completed=timestamp,
        )
        events.append(
            InfoEvent(
                source=CODEX_EVENT_SOURCE,
                data={"type": "turn_aborted", "reason": event.reason},
                timestamp=timestamp,
            )
        )
        return events

    async def flush_pending(
        self,
        error: ToolCallError | None = None,
        completed: datetime | None = None,
    ) -> list[Event]:
        """Emit spans for tool calls that never received an output.

        Dangling spawn_agent calls (aborted turn, truncated file) still get
        their agent span — the child thread id may already be bound from a
        ``sub_agent_activity`` event, so its events remain reachable.
        """
        events: list[Event] = []
        for pending in self.pending_calls.values():
            if pending.is_spawn:
                events.extend(
                    await self._spawn_agent_span_events(
                        pending,
                        result="",
                        error=error,
                        timestamp=completed or pending.timestamp,
                    )
                )
            else:
                events.extend(
                    _tool_span_events(
                        call_id=pending.call_id,
                        function=pending.function,
                        arguments=pending.arguments,
                        result="",
                        error=error,
                        timestamp=pending.timestamp,
                        completed=completed or pending.timestamp,
                    )
                )
        self.pending_calls.clear()
        return events


# ── event constructors ───────────────────────────────────────────────────


def _span_begin(
    span_id: str,
    name: str,
    span_type: str,
    timestamp: datetime,
    metadata: dict[str, Any] | None = None,
) -> SpanBeginEvent:
    from inspect_ai.util._span import current_span_id

    return SpanBeginEvent(
        id=span_id,
        name=name,
        type=span_type,
        parent_id=current_span_id(),
        timestamp=timestamp,
        working_start=0.0,
        metadata=metadata,
    )


def _to_tool_event(
    call_id: str,
    function: str,
    arguments: dict[str, Any],
    result: str | list[Content],
    error: ToolCallError | None,
    timestamp: datetime,
    completed: datetime | None,
) -> ToolEvent:
    tool_result: ToolResult = result  # type: ignore[assignment]
    return ToolEvent(
        id=call_id,
        type="function",
        function=function,
        arguments=arguments,
        result=tool_result,
        timestamp=timestamp,
        completed=completed,
        error=error,
        view=tool_view(function, arguments),
    )


def _tool_span_events(
    call_id: str,
    function: str,
    arguments: dict[str, Any],
    result: str | list[Content],
    error: ToolCallError | None,
    timestamp: datetime,
    completed: datetime | None,
) -> list[Event]:
    """Regular tool span: SpanBegin(type='tool') → ToolEvent → SpanEnd."""
    tool_span_id = f"tool-{call_id}"
    tool_event = _to_tool_event(
        call_id=call_id,
        function=function,
        arguments=arguments,
        result=result,
        error=error,
        timestamp=timestamp,
        completed=completed,
    )
    tool_event.span_id = tool_span_id
    return [
        _span_begin(
            span_id=tool_span_id,
            name=function,
            span_type="tool",
            timestamp=timestamp,
        ),
        tool_event,
        SpanEndEvent(id=tool_span_id, timestamp=completed or timestamp),
    ]


# ── public entry point ───────────────────────────────────────────────────


async def process_rollout_events(
    events: Sequence[RolloutEvent],
    max_depth: int = 5,
    child_loader: ChildThreadLoader | None = None,
) -> AsyncIterator[Event]:
    """Convert parsed Codex rollout events to Inspect events.

    Args:
        events: Rollout events in file order (from ``parse_rollout_events``).
        max_depth: Maximum depth for loading spawned child threads
            (0 = no loading).
        child_loader: Callback for loading a child thread's events by
            thread id (file-based in inspect_scout).

    Yields:
        Inspect Event objects (ModelEvent, ToolEvent, SpanBeginEvent,
        CompactionEvent, InfoEvent, ...).
    """
    proc = _RolloutProcessor(max_depth=max_depth, child_loader=child_loader)

    def lookahead_usage(start: int) -> ModelUsage | None:
        """Usage for the response being flushed: the first usage-bearing
        token_count at/after ``start``, before the next model response."""
        for ahead in events[start:]:
            if proc.is_assistant_side(ahead):
                return None
            if isinstance(ahead, TokenCountEvent) and ahead.info:
                usage = usage_from_token_info(ahead.info)
                if usage is not None:
                    return usage
        return None

    for index, event in enumerate(events):
        timestamp = proc.update_timestamp(event)

        if proc.is_assistant_side(event):
            proc.buffer_assistant(event, timestamp)
            continue

        # Boundary item: flush any buffered model response first (with its
        # usage looked ahead, so the ModelEvent is complete when yielded)
        if proc.assistant_buffer:
            for evt in proc.flush_model(usage=lookahead_usage(index)):
                yield evt

        if isinstance(event, ResponseMessage):
            proc.process_user_message(event)
        elif isinstance(event, ResponseFunctionCallOutput):
            for evt in await proc.process_call_output(
                event.call_id, event.output, timestamp
            ):
                yield evt
        elif isinstance(event, ResponseCustomToolCallOutput):
            for evt in await proc.process_call_output(
                event.call_id, event.output, timestamp
            ):
                yield evt
        elif isinstance(event, TokenCountEvent):
            proc.process_token_count(event)
        elif isinstance(event, SubAgentActivityEvent):
            proc.process_sub_agent_activity(event)
        elif isinstance(event, CompactedEvent):
            for evt in proc.process_compacted(event, timestamp):
                yield evt
        elif isinstance(event, ResponseCompaction):
            for evt in proc.process_remote_compaction(event, timestamp):
                yield evt
        elif isinstance(event, TurnContextEvent):
            if event.model:
                proc.current_model = event.model
        elif isinstance(event, ThreadRolledBackEvent):
            for evt in proc.process_rolled_back(event, timestamp):
                yield evt
        elif isinstance(event, TurnAbortedEvent):
            for evt in await proc.process_turn_aborted(event, timestamp):
                yield evt
        elif isinstance(event, TurnCompleteEvent):
            if event.error:
                yield InfoEvent(
                    source=CODEX_EVENT_SOURCE,
                    data={"type": "turn_error", "error": event.error},
                    timestamp=timestamp,
                )
        elif isinstance(event, ReviewModeEvent):
            yield InfoEvent(
                source=CODEX_EVENT_SOURCE,
                data={
                    "type": (
                        "entered_review_mode" if event.entered else "exited_review_mode"
                    ),
                },
                timestamp=timestamp,
            )
        elif isinstance(event, SessionMetaEvent):
            # Mid-file session_meta lines occur in fork-copied files; the
            # file's identity is the first meta (handled by the caller).
            pass

    # Flush any trailing model response and dangling tool calls
    for evt in proc.flush_model():
        yield evt
    for evt in await proc.flush_pending():
        yield evt
