"""Pydantic models for Codex rollout JSONL records."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextContent(BaseModel):
    """Text content block inside Codex message items."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["input_text", "output_text"]
    text: str = ""


class MessagePayload(BaseModel):
    """Codex message response item."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["message"]
    role: str
    content: list[TextContent] = Field(default_factory=list)
    phase: str | None = None


class FunctionCallPayload(BaseModel):
    """Codex tool/function call response item."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function_call"]
    name: str
    arguments: str | dict[str, Any] | None = None
    call_id: str


class FunctionCallOutputPayload(BaseModel):
    """Codex tool/function result response item."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function_call_output"]
    call_id: str
    output: Any = ""


class ReasoningSummary(BaseModel):
    """Visible reasoning summary block."""

    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    summary: str | None = None


class ReasoningPayload(BaseModel):
    """Codex reasoning response item."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["reasoning"]
    summary: list[ReasoningSummary | dict[str, Any] | str] = Field(default_factory=list)
    content: Any = None
    encrypted_content: str | None = None


ResponseItemPayload = Annotated[
    MessagePayload | FunctionCallPayload | FunctionCallOutputPayload | ReasoningPayload,
    Field(discriminator="type"),
]


class TokenUsagePayload(BaseModel):
    """Token usage payload from Codex `token_count` events."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0


class TokenCountInfo(BaseModel):
    """Info block for Codex `token_count` events."""

    model_config = ConfigDict(extra="ignore")

    total_token_usage: TokenUsagePayload | None = None
    last_token_usage: TokenUsagePayload | None = None
    model_context_window: int | None = None


class SessionSourceThreadSpawn(BaseModel):
    """Subagent thread-spawn metadata from Codex session headers."""

    model_config = ConfigDict(extra="ignore")

    parent_thread_id: str | None = None
    depth: int | None = None
    agent_path: str | None = None
    agent_nickname: str | None = None
    agent_role: str | None = None


class SessionSourceSubagent(BaseModel):
    """Subagent metadata wrapper from Codex session headers."""

    model_config = ConfigDict(extra="ignore")

    thread_spawn: SessionSourceThreadSpawn | None = None


class SessionSourcePayload(BaseModel):
    """Structured session source metadata."""

    model_config = ConfigDict(extra="ignore")

    subagent: SessionSourceSubagent | None = None


class SessionMetaPayload(BaseModel):
    """Codex session metadata."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    forked_from_id: str | None = None
    model_provider: str | None = None
    agent_nickname: str | None = None
    agent_role: str | None = None
    source: str | SessionSourcePayload | None = None

    def parent_thread_id(self) -> str | None:
        """Return the parent thread id if this session is a subagent."""
        if self.forked_from_id:
            return self.forked_from_id
        if isinstance(self.source, SessionSourcePayload):
            subagent = self.source.subagent
            if subagent and subagent.thread_spawn:
                return subagent.thread_spawn.parent_thread_id
        return None


class TurnContextPayload(BaseModel):
    """Codex turn context metadata."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None


class EventMsgPayload(BaseModel):
    """Generic Codex operational event payload."""

    model_config = ConfigDict(extra="ignore")

    type: str


class TokenCountPayload(EventMsgPayload):
    """Codex operational token_count event."""

    type: Literal["token_count"]
    info: TokenCountInfo | None = None


class CollabAgentSpawnEndPayload(EventMsgPayload):
    """Codex subagent-spawn linkage event."""

    type: Literal["collab_agent_spawn_end"]
    call_id: str
    sender_thread_id: str | None = None
    new_thread_id: str | None = None
    new_agent_nickname: str | None = None
    new_agent_role: str | None = None
    prompt: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    status: str | None = None


class BaseRecord(BaseModel):
    """Base model for Codex rollout records."""

    model_config = ConfigDict(extra="ignore")

    timestamp: str | None = None
    type: str


class SessionMetaRecord(BaseRecord):
    """Session metadata record."""

    type: Literal["session_meta"] = "session_meta"
    payload: SessionMetaPayload


class TurnContextRecord(BaseRecord):
    """Turn context record."""

    type: Literal["turn_context"] = "turn_context"
    payload: TurnContextPayload


class ResponseItemRecord(BaseRecord):
    """Response item record."""

    type: Literal["response_item"] = "response_item"
    payload: ResponseItemPayload


class EventMsgRecord(BaseRecord):
    """Operational event record."""

    type: Literal["event_msg"] = "event_msg"
    payload: EventMsgPayload | TokenCountPayload | CollabAgentSpawnEndPayload


Record = SessionMetaRecord | TurnContextRecord | ResponseItemRecord | EventMsgRecord


def parse_event(raw: dict[str, Any]) -> Record | None:
    """Parse a raw Codex rollout record."""
    record_type = raw.get("type")

    if record_type == "session_meta":
        return SessionMetaRecord.model_validate(raw)
    if record_type == "turn_context":
        return TurnContextRecord.model_validate(raw)
    if record_type == "response_item":
        return ResponseItemRecord.model_validate(raw)
    if record_type == "event_msg":
        payload_type = raw.get("payload", {}).get("type", "")
        if payload_type == "token_count":
            payload = TokenCountPayload.model_validate(raw.get("payload", {}))
        elif payload_type == "collab_agent_spawn_end":
            payload = CollabAgentSpawnEndPayload.model_validate(raw.get("payload", {}))
        else:
            payload = EventMsgPayload.model_validate(raw.get("payload", {}))
        return EventMsgRecord(timestamp=raw.get("timestamp"), type="event_msg", payload=payload)
    return None


def parse_events(raw_events: list[dict[str, Any]]) -> list[Record]:
    """Parse a list of raw Codex rollout records."""
    parsed: list[Record] = []
    for raw_event in raw_events:
        event = parse_event(raw_event)
        if event is not None:
            parsed.append(event)
    return parsed


def extract_session_metadata(events: Sequence[Record]) -> dict[str, Any]:
    """Extract session metadata from parsed Codex rollout records."""
    metadata: dict[str, Any] = {}
    for event in events:
        if not isinstance(event, SessionMetaRecord):
            continue
        payload = event.payload
        if payload.id and "session_id" not in metadata:
            metadata["session_id"] = payload.id
        if payload.model_provider and "model_provider" not in metadata:
            metadata["model_provider"] = payload.model_provider
        parent_thread_id = payload.parent_thread_id()
        if parent_thread_id and "parent_thread_id" not in metadata:
            metadata["parent_thread_id"] = parent_thread_id
        if payload.agent_nickname and "agent_nickname" not in metadata:
            metadata["agent_nickname"] = payload.agent_nickname
        if payload.agent_role and "agent_role" not in metadata:
            metadata["agent_role"] = payload.agent_role
        if "session_id" in metadata and "model_provider" in metadata:
            break
    return metadata


def is_subagent_session(events: Sequence[Record]) -> bool:
    """Return whether the parsed Codex rollout belongs to a subagent thread."""
    return extract_session_metadata(events).get("parent_thread_id") is not None
