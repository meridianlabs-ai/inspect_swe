"""Event conversion for Codex CLI rollout sessions."""

import json
import sqlite3
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar

from inspect_ai.event import Event, ModelEvent, SpanBeginEvent, SpanEndEvent, ToolEvent
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ChatCompletionChoice, ModelUsage
from inspect_ai.tool import ToolCall
from inspect_ai.tool._tool import ToolResult

from .models import (
    CollabAgentSpawnEndPayload,
    EventMsgRecord,
    FunctionCallOutputPayload,
    FunctionCallPayload,
    MessagePayload,
    ReasoningPayload,
    Record,
    SessionMetaRecord,
    TokenCountPayload,
    TokenUsagePayload,
    TurnContextRecord,
    parse_event,
    parse_events,
)
from .toolview import tool_view
from .util import parse_timestamp

T = TypeVar("T")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AgentEventLoader(Protocol):
    """Callback for loading child-agent events from nested rollout files."""

    async def __call__(
        self,
        session_file: Path | None,
        thread_id: str | None,
        max_depth: int,
    ) -> list[Event]: ...


@dataclass
class _PendingTool:
    call: FunctionCallPayload
    timestamp: datetime


@dataclass(frozen=True)
class _SpawnedAgent:
    thread_id: str | None
    nickname: str | None
    role: str | None
    prompt: str | None
    model: str | None
    reasoning_effort: str | None


def _to_tool_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_json_text(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _message_text(message: MessagePayload) -> str:
    return "\n".join(block.text for block in message.content if block.text)


def _reasoning_blocks(reasoning: ReasoningPayload) -> list[ContentReasoning]:
    blocks: list[ContentReasoning] = []
    for summary in reasoning.summary:
        if isinstance(summary, str):
            text = summary
        elif isinstance(summary, dict):
            text = str(summary.get("text") or summary.get("summary") or "")
        else:
            text = str(summary.text or summary.summary or "")
        if text:
            blocks.append(ContentReasoning(reasoning=text, summary=text))
    return blocks


def _message_to_input(
    message: MessagePayload,
) -> ChatMessageSystem | ChatMessageUser | None:
    text = _message_text(message)
    if not text:
        return None
    if text.startswith("<subagent_notification>"):
        return None
    if message.role == "developer":
        return ChatMessageSystem(content=text, metadata={"role": "developer"})
    if message.role == "user":
        return ChatMessageUser(content=text)
    return None


def _tool_output_to_input(
    output: FunctionCallOutputPayload,
    tool_functions: dict[str, str],
) -> ChatMessageTool:
    rendered = output.output if isinstance(output.output, str) else json.dumps(output.output)
    return ChatMessageTool(
        content=rendered,
        tool_call_id=output.call_id,
        function=tool_functions.get(output.call_id),
    )


def _usage_from_payload(payload: TokenUsagePayload | None) -> ModelUsage | None:
    if payload is None:
        return None
    return ModelUsage(
        input_tokens=payload.input_tokens,
        output_tokens=payload.output_tokens,
        total_tokens=payload.total_tokens,
        input_tokens_cache_read=payload.cached_input_tokens or None,
        reasoning_tokens=payload.reasoning_output_tokens or None,
    )


def to_span_begin_event(
    span_id: str,
    name: str,
    span_type: str,
    timestamp: datetime,
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanBeginEvent:
    from inspect_ai.util._span import current_span_id

    return SpanBeginEvent(
        id=span_id,
        name=name,
        type=span_type,
        parent_id=parent_id if parent_id is not None else current_span_id(),
        timestamp=timestamp,
        working_start=0.0,
        metadata=metadata,
    )


def to_span_end_event(span_id: str, timestamp: datetime) -> SpanEndEvent:
    return SpanEndEvent(id=span_id, timestamp=timestamp)


def _assistant_event(
    items: Sequence[MessagePayload | FunctionCallPayload | ReasoningPayload],
    input_messages: list[Any],
    model_name: str,
    timestamp: datetime,
    usage: ModelUsage | None,
) -> ModelEvent | None:
    content: list[ContentText | ContentReasoning] = []
    tool_calls: list[ToolCall] = []
    phases: list[str] = []

    for item in items:
        if isinstance(item, MessagePayload):
            text = _message_text(item)
            if text:
                content.append(ContentText(text=text))
            if item.phase:
                phases.append(item.phase)
        elif isinstance(item, FunctionCallPayload):
            arguments = _to_tool_arguments(item.arguments)
            tool_calls.append(
                ToolCall(
                    id=item.call_id,
                    function=item.name,
                    arguments=arguments,
                    view=tool_view(item.name, arguments),
                )
            )
        elif isinstance(item, ReasoningPayload):
            content.extend(_reasoning_blocks(item))

    if not content and not tool_calls:
        return None

    if len(content) == 1 and isinstance(content[0], ContentText):
        output_content: str | list[Any] = content[0].text
    else:
        output_content = content if content else ""

    metadata = {"phases": phases} if phases else None
    output_message = ChatMessageAssistant(
        content=output_content,
        tool_calls=tool_calls or None,
        model=model_name,
        metadata=metadata,
    )

    output = ModelOutput(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=output_message,
                stop_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=usage,
    )

    return ModelEvent(
        model=model_name,
        input=list(input_messages),
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=output,
        timestamp=timestamp,
        metadata=metadata,
    )


async def _to_async_iter(items: Iterable[T]) -> AsyncIterator[T]:
    for item in items:
        yield item


def _codex_home(session_file: Path) -> Path | None:
    for parent in session_file.parents:
        if parent.name == "sessions":
            return parent.parent
    return None


def _find_child_rollout(session_file: Path, thread_id: str) -> Path | None:
    codex_home = _codex_home(session_file)
    if codex_home is None:
        return None

    state_db = codex_home / "state_5.sqlite"
    if state_db.exists():
        with sqlite3.connect(state_db) as conn:
            row = conn.execute(
                "select rollout_path from threads where id = ?",
                (thread_id,),
            ).fetchone()
        if row and row[0]:
            candidate = Path(str(row[0]))
            if candidate.exists():
                return candidate

    sessions_dir = codex_home / "sessions"
    pattern = f"rollout-*{thread_id}.jsonl"
    matches = sorted(sessions_dir.rglob(pattern))
    return matches[0] if matches else None


def _read_rollout_file(session_file: Path) -> list[dict[str, Any]]:
    raw_events: list[dict[str, Any]] = []
    for line in session_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            raw_events.append(raw)
    return raw_events


async def _load_agent_events(
    session_file: Path | None,
    thread_id: str | None,
    max_depth: int,
    agent_loader: AgentEventLoader | None = None,
) -> list[Event]:
    if max_depth <= 0 or session_file is None or thread_id is None:
        return []

    child_file = _find_child_rollout(session_file, thread_id)
    if child_file is None:
        return []

    if agent_loader is not None:
        return await agent_loader(child_file, thread_id, max_depth - 1)

    parsed = parse_events(_read_rollout_file(child_file))
    result: list[Event] = []
    async for event in process_parsed_events(
        parsed,
        max_depth=max_depth - 1,
        session_file=child_file,
        agent_loader=agent_loader,
    ):
        result.append(event)
    return result


async def codex_cli_events(
    raw_events: Iterable[dict[str, Any]] | AsyncIterable[dict[str, Any]],
    max_depth: int = 5,
    session_file: Path | None = None,
    agent_loader: AgentEventLoader | None = None,
) -> AsyncIterator[Event]:
    """Convert raw Codex rollout JSONL records to Inspect events."""
    if isinstance(raw_events, AsyncIterable):
        event_stream = raw_events
    else:
        event_stream = _to_async_iter(raw_events)

    parsed: list[Record] = []
    async for raw in event_stream:
        event = parse_event(raw)
        if event is not None:
            parsed.append(event)

    async for event in process_parsed_events(
        parsed,
        max_depth=max_depth,
        session_file=session_file,
        agent_loader=agent_loader,
    ):
        yield event


async def process_parsed_events(
    events: Sequence[Record],
    max_depth: int = 5,
    session_file: Path | None = None,
    agent_loader: AgentEventLoader | None = None,
) -> AsyncIterator[Event]:
    """Convert parsed Codex rollout records to Inspect events."""
    accumulated_messages: list[Any] = []
    pending_tools: dict[str, _PendingTool] = {}
    spawned_agents: dict[str, _SpawnedAgent] = {}
    pending_usage: ModelUsage | None = None
    assistant_batch: list[MessagePayload | FunctionCallPayload | ReasoningPayload] = []
    assistant_timestamp: datetime | None = None
    last_timestamp = _EPOCH
    model_name = "unknown"
    model_provider: str | None = None

    def next_timestamp(ts_str: str | None) -> datetime:
        nonlocal last_timestamp
        timestamp = parse_timestamp(ts_str) or last_timestamp
        if timestamp <= last_timestamp:
            timestamp = last_timestamp + timedelta(milliseconds=1)
        last_timestamp = timestamp
        return timestamp

    async def flush_assistant() -> AsyncIterator[Event]:
        nonlocal assistant_batch
        nonlocal assistant_timestamp
        nonlocal pending_usage
        batch = assistant_batch
        batch_timestamp = assistant_timestamp
        if not batch or batch_timestamp is None:
            assistant_batch = []
            assistant_timestamp = None
            return

        model_event = _assistant_event(
            batch,
            accumulated_messages,
            model_name if model_provider is None else f"{model_provider}/{model_name}",
            batch_timestamp,
            pending_usage,
        )
        assistant_batch = []
        assistant_timestamp = None
        pending_usage = None

        if model_event is None:
            return

        yield model_event

        if model_event.output and model_event.output.message:
            accumulated_messages.append(model_event.output.message)

        for raw_item in batch:
            if isinstance(raw_item, FunctionCallPayload):
                pending_tools[raw_item.call_id] = _PendingTool(
                    call=raw_item,
                    timestamp=batch_timestamp,
                )

    for event in events:
        timestamp = next_timestamp(event.timestamp)

        if isinstance(event, SessionMetaRecord):
            if event.payload.model_provider:
                model_provider = event.payload.model_provider
            continue

        if isinstance(event, TurnContextRecord):
            if event.payload.model:
                model_name = event.payload.model
            continue

        if isinstance(event, EventMsgRecord):
            payload = event.payload
            if isinstance(payload, TokenCountPayload):
                usage_payload = None
                if payload.info is not None:
                    usage_payload = (
                        payload.info.last_token_usage or payload.info.total_token_usage
                    )
                pending_usage = _usage_from_payload(usage_payload)
            elif isinstance(payload, CollabAgentSpawnEndPayload):
                spawned_agents[payload.call_id] = _SpawnedAgent(
                    thread_id=payload.new_thread_id,
                    nickname=payload.new_agent_nickname,
                    role=payload.new_agent_role,
                    prompt=payload.prompt,
                    model=payload.model,
                    reasoning_effort=payload.reasoning_effort,
                )
            continue

        payload = event.payload

        if isinstance(payload, MessagePayload) and payload.role in ("user", "developer"):
            async for assistant_event in flush_assistant():
                yield assistant_event
            input_message = _message_to_input(payload)
            if input_message is not None:
                accumulated_messages.append(input_message)
            continue

        if isinstance(payload, FunctionCallOutputPayload):
            async for assistant_event in flush_assistant():
                yield assistant_event

            pending = pending_tools.pop(payload.call_id, None)
            tool_functions = {
                tool_id: pending_tool.call.name
                for tool_id, pending_tool in pending_tools.items()
            }

            if pending is not None:
                tool_functions[payload.call_id] = pending.call.name
                arguments = _to_tool_arguments(pending.call.arguments)
                result: ToolResult
                if isinstance(payload.output, str):
                    result = payload.output
                else:
                    result = json.dumps(payload.output)

                spawned = spawned_agents.get(payload.call_id)
                if pending.call.name == "spawn_agent" and spawned is None:
                    parsed_output = _parse_json_text(payload.output)
                    if parsed_output:
                        spawned = _SpawnedAgent(
                            thread_id=str(parsed_output.get("agent_id"))
                            if parsed_output.get("agent_id")
                            else None,
                            nickname=str(parsed_output.get("nickname"))
                            if parsed_output.get("nickname")
                            else None,
                            role=arguments.get("agent_type")
                            if isinstance(arguments.get("agent_type"), str)
                            else None,
                            prompt=arguments.get("message")
                            if isinstance(arguments.get("message"), str)
                            else None,
                            model=arguments.get("model")
                            if isinstance(arguments.get("model"), str)
                            else None,
                            reasoning_effort=arguments.get("reasoning_effort")
                            if isinstance(arguments.get("reasoning_effort"), str)
                            else None,
                        )

                if pending.call.name == "spawn_agent" and spawned is not None:
                    span_id = f"agent-{payload.call_id}"
                    span_name = spawned.nickname or spawned.role or "agent"
                    metadata = {
                        "thread_id": spawned.thread_id,
                        "prompt": spawned.prompt,
                        "role": spawned.role,
                        "model": spawned.model,
                        "reasoning_effort": spawned.reasoning_effort,
                    }
                    yield to_span_begin_event(
                        span_id=span_id,
                        name=span_name,
                        span_type="agent",
                        timestamp=pending.timestamp,
                        metadata={k: v for k, v in metadata.items() if v is not None},
                    )
                    tool_event = ToolEvent(
                        id=payload.call_id,
                        type="function",
                        function=pending.call.name,
                        arguments=arguments,
                        result=result,
                        timestamp=pending.timestamp,
                        completed=timestamp,
                        span_id=span_id,
                        agent_span_id=span_id,
                        view=tool_view(pending.call.name, arguments),
                    )
                    yield tool_event

                    child_events = await _load_agent_events(
                        session_file=session_file,
                        thread_id=spawned.thread_id,
                        max_depth=max_depth,
                        agent_loader=agent_loader,
                    )
                    for child_event in child_events:
                        if isinstance(child_event, SpanBeginEvent):
                            if child_event.parent_id is None:
                                child_event.parent_id = span_id
                        elif not isinstance(child_event, SpanEndEvent):
                            if child_event.span_id is None:
                                child_event.span_id = span_id
                        yield child_event

                    yield to_span_end_event(span_id, timestamp)
                else:
                    span_id = f"tool-{payload.call_id}"
                    yield to_span_begin_event(
                        span_id=span_id,
                        name=pending.call.name,
                        span_type="tool",
                        timestamp=pending.timestamp,
                    )
                    tool_event = ToolEvent(
                        id=payload.call_id,
                        type="function",
                        function=pending.call.name,
                        arguments=arguments,
                        result=result,
                        timestamp=pending.timestamp,
                        completed=timestamp,
                        span_id=span_id,
                        view=tool_view(pending.call.name, arguments),
                    )
                    yield tool_event
                    yield to_span_end_event(span_id, timestamp)

            accumulated_messages.append(_tool_output_to_input(payload, tool_functions))
            continue

        if isinstance(payload, (MessagePayload, FunctionCallPayload, ReasoningPayload)):
            if not assistant_batch:
                assistant_timestamp = timestamp
            if isinstance(payload, MessagePayload) and payload.role != "assistant":
                continue
            assistant_batch.append(payload)

    async for assistant_event in flush_assistant():
        yield assistant_event

    for pending in pending_tools.values():
        span_id = f"tool-{pending.call.call_id}"
        yield to_span_begin_event(
            span_id=span_id,
            name=pending.call.name,
            span_type="tool",
            timestamp=pending.timestamp,
        )
        arguments = _to_tool_arguments(pending.call.arguments)
        yield ToolEvent(
            id=pending.call.call_id,
            type="function",
            function=pending.call.name,
            arguments=arguments,
            result="",
            timestamp=pending.timestamp,
            completed=pending.timestamp,
            span_id=span_id,
            view=tool_view(pending.call.name, arguments),
        )
        yield to_span_end_event(span_id, pending.timestamp)
