"""Pydantic models for the Codex CLI rollout JSONL format.

Codex persists each thread as a rollout file
(``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread-id>.jsonl``) where
every line is an envelope ``{"timestamp", "type", "payload"}`` (plus an
optional ``ordinal`` for paginated threads). The ``type``/``payload`` pair is
an adjacently-tagged ``RolloutItem``; ``response_item`` and ``event_msg``
payloads are further discriminated on ``payload["type"]``.

The rollout format is a moving target (there is no format-version field), so
parsing is deliberately lenient: models use ``extra="ignore"``, unknown item
and event types are silently dropped, and validation failures skip the line
with a warning. Ground truth: ``codex-rs/protocol/src/protocol.rs`` and
``codex-rs/protocol/src/models.rs`` in openai/codex.
"""

from logging import getLogger
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = getLogger(__name__)


class RolloutEvent(BaseModel):
    """Base class for parsed rollout lines.

    ``timestamp`` is taken from the line envelope (RFC3339 string).
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    timestamp: str | None = None


# ── session_meta ─────────────────────────────────────────────────────────


class GitInfo(BaseModel):
    """Git metadata collected at session start."""

    model_config = ConfigDict(extra="ignore")

    commit_hash: str | None = None
    branch: str | None = None
    repository_url: str | None = None


class HistoryBase(BaseModel):
    """Exclusive prefix of another rollout inherited by reference."""

    model_config = ConfigDict(extra="ignore")

    thread_id: str
    end_ordinal_exclusive: int
    end_byte_offset: int | None = None


class SessionMetaEvent(RolloutEvent):
    """The ``session_meta`` line (first line of every rollout file).

    Fork-copied files may contain a second ``session_meta`` mid-file (the
    source thread's); readers treat the first as the file's identity.
    """

    id: str | None = None
    session_id: str | None = None
    forked_from_id: str | None = None
    parent_thread_id: str | None = None
    cwd: str | None = None
    originator: str | None = None
    cli_version: str | None = None
    # "cli" | "vscode" | "exec" | "mcp" | {"custom": ...} | {"subagent": ...} | ...
    source: str | dict[str, Any] | None = None
    agent_nickname: str | None = None
    agent_path: str | None = None
    agent_role: str | None = None
    model_provider: str | None = None
    # Modern field is base_instructions ({"text": ...}); very old files used
    # a plain-string instructions field.
    base_instructions: dict[str, Any] | None = None
    instructions: str | None = None
    history_mode: str = "legacy"
    history_base: HistoryBase | None = None
    git: GitInfo | None = None

    @model_validator(mode="after")
    def _backfill_ids(self) -> "SessionMetaEvent":
        # Older files have only id; newer files carry both id and session_id.
        if self.id is None:
            self.id = self.session_id
        if self.session_id is None:
            self.session_id = self.id
        return self

    @property
    def thread_id(self) -> str | None:
        """The thread id (rollout identity)."""
        return self.id

    def subagent_source(self) -> str | dict[str, Any] | None:
        """The subagent classification, if this is a subagent thread.

        Returns e.g. ``"review"``, ``"compact"`` or ``{"thread_spawn": {...}}``.
        """
        if isinstance(self.source, dict):
            subagent = self.source.get("subagent")
            if isinstance(subagent, (str, dict)):
                return subagent
        return None


# ── response_item variants ───────────────────────────────────────────────


class ResponseMessage(RolloutEvent):
    """A ``message`` response item (user / assistant / developer)."""

    id: str | None = None
    role: str = "user"
    content: list[dict[str, Any]] | str = Field(default_factory=list)
    phase: str | None = None  # assistant only: "commentary" | "final_answer"


class ResponseReasoning(RolloutEvent):
    """A ``reasoning`` response item.

    ``summary`` holds readable summaries; ``content`` holds plaintext
    reasoning (rare — only some providers); ``encrypted_content`` is opaque.
    """

    id: str | None = None
    summary: list[dict[str, Any]] = Field(default_factory=list)
    content: list[dict[str, Any]] | None = None
    encrypted_content: str | None = None


class ResponseFunctionCall(RolloutEvent):
    """A ``function_call`` response item (shell, update_plan, MCP, ...)."""

    id: str | None = None
    name: str = ""
    namespace: str | None = None
    arguments: str = ""  # JSON-encoded string
    call_id: str = ""


class ResponseFunctionCallOutput(RolloutEvent):
    """A ``function_call_output`` response item.

    ``output`` is polymorphic on the wire: a plain string, an array of
    content items, or (in very old files) a JSON-encoded string of
    ``{"output": ..., "metadata": {"exit_code": ...}}``.
    """

    id: str | None = None
    call_id: str = ""
    output: Any = None


class ResponseLocalShellCall(RolloutEvent):
    """A ``local_shell_call`` item (models with the hosted local-shell tool)."""

    id: str | None = None
    call_id: str | None = None
    status: str | None = None
    action: dict[str, Any] = Field(default_factory=dict)


class ResponseCustomToolCall(RolloutEvent):
    """A ``custom_tool_call`` item (freeform tools, e.g. apply_patch)."""

    id: str | None = None
    call_id: str = ""
    name: str = ""
    input: str = ""


class ResponseCustomToolCallOutput(RolloutEvent):
    """A ``custom_tool_call_output`` item."""

    id: str | None = None
    call_id: str = ""
    name: str | None = None
    output: Any = None


class ResponseWebSearchCall(RolloutEvent):
    """A ``web_search_call`` hosted-tool item (no separate output item)."""

    id: str | None = None
    status: str | None = None
    action: dict[str, Any] | None = None


class ResponseCompaction(RolloutEvent):
    """A remote/server-side compaction artifact.

    Covers ``compaction`` (alias ``compaction_summary``) and
    ``context_compaction`` items — both carry only encrypted content.
    """

    id: str | None = None
    encrypted_content: str | None = None


# ── other rollout item variants ──────────────────────────────────────────


class CompactedEvent(RolloutEvent):
    """A ``compacted`` item (local compaction checkpoint).

    ``message`` is the plaintext summary; ``replacement_history`` is the
    complete post-compaction in-context history (raw response items).
    """

    message: str = ""
    replacement_history: list[dict[str, Any]] | None = None


class TurnContextEvent(RolloutEvent):
    """A ``turn_context`` item — the model/cwd/policies in effect for a turn."""

    turn_id: str | None = None
    cwd: str | None = None
    approval_policy: str | dict[str, Any] | None = None
    sandbox_policy: str | dict[str, Any] | None = None
    model: str | None = None
    effort: str | None = None


# ── persisted event_msg variants we consume ──────────────────────────────


class TokenCountEvent(RolloutEvent):
    """A ``token_count`` event — cumulative and last-turn usage."""

    info: dict[str, Any] | None = None
    rate_limits: dict[str, Any] | None = None


class TurnAbortedEvent(RolloutEvent):
    """A ``turn_aborted`` event (interrupt, replacement, budget, ...)."""

    turn_id: str | None = None
    reason: str | None = None


class TurnCompleteEvent(RolloutEvent):
    """A ``turn_complete`` event — carries terminal turn errors."""

    turn_id: str | None = None
    last_agent_message: str | None = None
    error: str | dict[str, Any] | None = None


class ThreadRolledBackEvent(RolloutEvent):
    """A ``thread_rolled_back`` event — the user undid ``num_turns`` turns.

    Rollout files are append-only, so rolled-back lines remain in the file
    before this marker.
    """

    num_turns: int = 0


class SubAgentActivityEvent(RolloutEvent):
    """A sub-agent lifecycle event linking a spawn call to its child thread.

    The wire payload also carries ``kind`` ("started" | "interacted" |
    "interrupted"); it is not declared because nothing consumes it — only
    "started" events reuse the spawn call's id, which is what the
    ``event_id`` correlation relies on.
    """

    event_id: str
    agent_thread_id: str
    agent_path: str | None = None


class ReviewModeEvent(RolloutEvent):
    """An ``entered_review_mode`` / ``exited_review_mode`` event."""

    entered: bool = True
    review: dict[str, Any] | None = None


# ── parsing ──────────────────────────────────────────────────────────────

_RESPONSE_ITEM_TYPES: dict[str, type[RolloutEvent]] = {
    "message": ResponseMessage,
    "reasoning": ResponseReasoning,
    "function_call": ResponseFunctionCall,
    "function_call_output": ResponseFunctionCallOutput,
    "local_shell_call": ResponseLocalShellCall,
    "custom_tool_call": ResponseCustomToolCall,
    "custom_tool_call_output": ResponseCustomToolCallOutput,
    "web_search_call": ResponseWebSearchCall,
    "compaction": ResponseCompaction,
    "compaction_summary": ResponseCompaction,
    "context_compaction": ResponseCompaction,
}

_EVENT_MSG_TYPES: dict[str, type[RolloutEvent]] = {
    "token_count": TokenCountEvent,
    "turn_aborted": TurnAbortedEvent,
    "turn_complete": TurnCompleteEvent,
    "thread_rolled_back": ThreadRolledBackEvent,
    "sub_agent_activity": SubAgentActivityEvent,
}


def parse_rollout_event(raw: dict[str, Any]) -> RolloutEvent | None:
    """Parse a single raw rollout line into a typed model.

    Returns None for unknown/unpersisted item types (deliberate — the format
    adds new types over time) and for lines that fail validation.
    """
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return None
    line_type = raw.get("type")

    model_cls: type[RolloutEvent] | None = None
    if line_type == "session_meta":
        model_cls = SessionMetaEvent
    elif line_type == "response_item":
        payload_type = payload.get("type")
        if isinstance(payload_type, str):
            model_cls = _RESPONSE_ITEM_TYPES.get(payload_type)
    elif line_type == "compacted":
        model_cls = CompactedEvent
    elif line_type == "turn_context":
        model_cls = TurnContextEvent
    elif line_type == "event_msg":
        payload_type = payload.get("type")
        if payload_type == "entered_review_mode":
            return ReviewModeEvent(
                entered=True,
                review=payload if payload else None,
                timestamp=_envelope_timestamp(raw),
            )
        elif payload_type == "exited_review_mode":
            return ReviewModeEvent(
                entered=False,
                review=payload if payload else None,
                timestamp=_envelope_timestamp(raw),
            )
        elif isinstance(payload_type, str):
            model_cls = _EVENT_MSG_TYPES.get(payload_type)

    if model_cls is None:
        return None

    try:
        event = model_cls.model_validate(payload)
    except ValidationError as ex:
        logger.warning(f"Failed to parse rollout {line_type} payload: {ex}")
        return None

    # Envelope timestamp wins when present (session_meta payloads carry their
    # own timestamp field which serves as a fallback).
    envelope_ts = _envelope_timestamp(raw)
    if envelope_ts is not None:
        event.timestamp = envelope_ts
    elif event.timestamp is None and isinstance(payload.get("timestamp"), str):
        event.timestamp = payload["timestamp"]
    return event


def _envelope_timestamp(raw: dict[str, Any]) -> str | None:
    ts = raw.get("timestamp")
    return ts if isinstance(ts, str) else None


def parse_rollout_events(raw_lines: list[dict[str, Any]]) -> list[RolloutEvent]:
    """Parse raw rollout lines to typed models, dropping unknown types."""
    events: list[RolloutEvent] = []
    for raw in raw_lines:
        event = parse_rollout_event(raw)
        if event is not None:
            events.append(event)
    return events
