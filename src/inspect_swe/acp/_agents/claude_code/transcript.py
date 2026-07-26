"""Build synthetic Claude Code session transcripts for resume.

Claude Code persists each session as a JSONL transcript under
``$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-id>.jsonl``. The
``claude-agent-acp`` adapter resolves a resumed session by reading that file
(via the Agent SDK's ``getSessionMessages`` for replay, and ``resume`` for the
model's own history), so writing a synthetic transcript there and then calling
ACP ``session/load`` with its ``session_id`` makes Claude Code resume from that
prior conversation on the next prompt turn.

The layout is what Claude Code itself writes: one row per *content block* — not
per message — chained through ``parentUuid``, where ``message`` holds a raw
Anthropic API message. Blocks from one assistant turn share a ``message.id``.

Build a :class:`TranscriptSpec` with :func:`build_transcript`, then write
``spec.content`` to ``<config_dir>/<spec.relative_path>`` — inside a sandbox via
``sandbox.write_file`` (see ``ClaudeCode`` resume), or on the host via
:meth:`TranscriptSpec.write_to`.

``build_transcript`` takes either Inspect ``ChatMessage`` (the ergonomic form)
or :data:`TranscriptItem` (the Claude-Code-native form); see
:func:`items_from_messages` for the differences that matter.

Truncating a session does *not* need this module: the Agent SDK's
``resumeSessionAt`` option resumes an existing session up to a chosen row uuid
(``ClaudeCode(resume_session_id=..., resume_message_uuid=...)``).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError
from pydantic import BaseModel, Field

# Recorded in every row. Claude Code reads it for backwards compatibility
# decisions, so it should look like a plausible recent CLI.
_CLAUDE_CODE_VERSION = "2.1.220"

# The SDK truncates a project slug at 200 chars and appends a hash of the full
# cwd; we can't reproduce that hash, so refuse rather than write to a path the
# SDK won't look in.
_MAX_SLUG_LENGTH = 200


class UserText(BaseModel):
    kind: Literal["user_text"] = "user_text"
    text: str


class AssistantText(BaseModel):
    kind: Literal["assistant_text"] = "assistant_text"
    text: str


class Thinking(BaseModel):
    kind: Literal["thinking"] = "thinking"
    thinking: str
    # Anthropic validates thinking signatures, and a synthetic conversation has
    # no valid one to offer: a fabricated signature makes the next turn fail
    # once the provider sees the replayed block. Leave it None only for a prior
    # that will never be sent to an Anthropic endpoint, and prefer dropping
    # thinking from synthetic priors entirely (see items_from_messages).
    signature: str | None = None


class ToolUse(BaseModel):
    kind: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    kind: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    # A string, or Anthropic content blocks when the tool returned structured
    # or image output.
    content: str | list[dict[str, Any]]
    is_error: bool = False


class RawBlock(BaseModel):
    """A content block this module doesn't model, preserved verbatim.

    Round-trips block types with no typed equivalent here (``image``,
    ``document``, ``server_tool_use``, a future type) so parsing a real
    transcript neither drops rows nor raises, and truncate-and-rebuild stays
    faithful.
    """

    kind: Literal["raw"] = "raw"
    role: Literal["user", "assistant"]
    block: dict[str, Any]


TranscriptItem = Annotated[
    UserText | AssistantText | Thinking | ToolUse | ToolResult | RawBlock,
    Field(discriminator="kind"),
]


class TranscriptSpec(BaseModel):
    """A built transcript, ready to write to a host fs or a sandbox fs."""

    session_id: str
    relative_path: str  # "projects/<cwd-slug>/<session-id>.jsonl"
    content: str  # complete jsonl, newline-terminated
    cwd: str  # the transcript's cwd (also encoded in relative_path)
    item_uuids: list[str]  # row uuid per item, in order; branch points for resume

    def write_to(self, config_dir: Path) -> Path:
        """Write under a host filesystem ``CLAUDE_CONFIG_DIR`` and return the path."""
        path = config_dir / self.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content)
        return path


class ParsedTranscript(BaseModel):
    """A transcript parsed back into typed items + session metadata.

    The result of :func:`parse_transcript`; feed ``items`` (optionally sliced)
    plus ``cwd`` back into :func:`build_transcript` to resume.
    """

    session_id: str | None
    cwd: str
    version: str
    items: list[TranscriptItem]
    item_uuids: list[str]  # row uuid per item, in order
    skipped_rows: int  # non-message rows (mode, file-history-snapshot, ...)

    def as_messages(self) -> list[ChatMessage]:
        """This transcript as Inspect ``ChatMessage``, for reading and scoring.

        Convenience wrapper around :func:`messages_from_items`. To *resume*,
        pass ``items`` back to :func:`build_transcript` rather than
        round-tripping through messages.
        """
        return messages_from_items(self.items)


def project_slug(cwd: str) -> str:
    """The ``projects/`` subdirectory Claude Code stores a cwd's sessions under.

    Mirrors the Agent SDK: every non-alphanumeric character becomes ``-``.

    Pass a path with symlinks already resolved. The SDK resolves the directory
    before slugging it, so a transcript written under the unresolved spelling
    (``/tmp/x`` where ``/tmp`` is a symlink to ``/private/tmp``) lands where the
    SDK never looks, and resume then silently starts a fresh conversation.
    :class:`ClaudeCode` resolves the sandbox cwd before placing the file.
    """
    slug = re.sub(r"[^a-zA-Z0-9]", "-", cwd)
    if len(slug) > _MAX_SLUG_LENGTH:
        raise ValueError(
            f"cwd is too long to address a Claude Code transcript: its slug is "
            f"{len(slug)} chars, and past {_MAX_SLUG_LENGTH} the SDK appends a hash "
            f"of the path that this module cannot reproduce. Use a shorter cwd."
        )
    return slug


def build_transcript(
    *,
    cwd: str,
    items: list[TranscriptItem] | list[ChatMessage],
    model: str,
    session_id: str | None = None,
    version: str = _CLAUDE_CODE_VERSION,
    git_branch: str = "",
    timestamp: datetime | None = None,
) -> TranscriptSpec:
    """Build a synthetic transcript without writing it.

    Returns a :class:`TranscriptSpec` whose ``content`` can be written to
    ``<config_dir>/<spec.relative_path>``, after which
    ``load_session(session_id=spec.session_id)`` resumes from this prior.

    ``items`` is either Inspect ``ChatMessage`` (converted via
    :func:`items_from_messages`, which drops thinking blocks) or native
    :data:`TranscriptItem` — all of one or all of the other, not a mix.

    Args:
        cwd: The session's working directory; also determines the on-disk path,
            and must match the ``cwd`` the resuming agent passes to
            ``session/load``.
        items: The prior conversation.
        model: Model name recorded on assistant rows.
        session_id: Session id to write under (a fresh uuid4 by default). Must
            be a uuid — the CLI rejects other ids.
        version: Claude Code version recorded on each row.
        git_branch: Branch recorded on each row.
        timestamp: Timestamp recorded on each row (now by default).
    """
    transcript_items = _as_transcript_items(items)
    session = session_id or str(uuid.uuid4())
    now = timestamp or datetime.now(UTC)
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    item_uuids = [str(uuid.uuid4()) for _ in transcript_items]
    rows: list[dict[str, Any]] = []
    parent_uuid: str | None = None
    # Blocks from one assistant turn share a message id, as they do when Claude
    # Code writes a streamed response; a user row between them ends the turn.
    assistant_message_id: str | None = None

    for item, row_uuid in zip(transcript_items, item_uuids, strict=True):
        role = _item_role(item)
        if role == "user":
            assistant_message_id = None
        elif assistant_message_id is None:
            assistant_message_id = f"msg_{uuid.uuid4().hex}"

        row: dict[str, Any] = {
            "parentUuid": parent_uuid,
            "isSidechain": False,
            "userType": "external",
            "cwd": cwd,
            "sessionId": session,
            "version": version,
            "gitBranch": git_branch,
            "type": role,
            "message": _message_payload(
                item, model=model, message_id=assistant_message_id
            ),
            "uuid": row_uuid,
            "timestamp": ts_iso,
        }
        if role == "assistant":
            row["requestId"] = f"req_{uuid.uuid4().hex}"
        rows.append(row)
        parent_uuid = row_uuid

    content = "".join(json.dumps(row) + "\n" for row in rows)
    return TranscriptSpec(
        session_id=session,
        relative_path=f"projects/{project_slug(cwd)}/{session}.jsonl",
        content=content,
        cwd=cwd,
        item_uuids=item_uuids,
    )


def parse_transcript(content: str) -> ParsedTranscript:
    """Parse a Claude Code transcript JSONL back into typed items + metadata.

    The inverse of :func:`build_transcript`. Use it to read a saved/real
    transcript, truncate ``items`` at a chosen point, and rebuild::

        parsed = parse_transcript(saved_content)
        spec = build_transcript(
            cwd=parsed.cwd, items=parsed.items[:n], model="claude-opus-5"
        )

    Round-trips losslessly for the modelled block types; blocks this module
    doesn't model come back as :class:`RawBlock` carrying the verbatim payload.
    Claude Code's own state rows (``mode``, ``file-history-snapshot``,
    ``attachment``, ``summary``, ...) are *not* preserved — they describe CLI
    state rather than conversation, and replaying stale state is worse than
    dropping it. ``skipped_rows`` counts them.
    """
    # Split on "\n" only — NOT str.splitlines(), which also breaks on the
    # unicode line separators that can appear inside JSON string values.
    rows = [json.loads(line) for line in content.split("\n") if line.strip()]
    items: list[TranscriptItem] = []
    item_uuids: list[str] = []
    skipped = 0
    session_id: str | None = None
    cwd = ""
    version = _CLAUDE_CODE_VERSION

    for row in rows:
        session_id = session_id or row.get("sessionId")
        cwd = cwd or row.get("cwd") or ""
        version = row.get("version") or version
        role = row.get("type")
        if role not in ("user", "assistant") or "message" not in row:
            skipped += 1
            continue
        blocks = _content_blocks(row["message"])
        row_uuid = row.get("uuid") or str(uuid.uuid4())
        for block in blocks:
            items.append(_block_to_item(block, role))
            # Every block of a multi-block row resolves to the same row uuid;
            # resuming at it resumes through the whole row.
            item_uuids.append(row_uuid)

    return ParsedTranscript(
        session_id=session_id,
        cwd=cwd,
        version=version,
        items=items,
        item_uuids=item_uuids,
        skipped_rows=skipped,
    )


# ---------------------------------------------------------------------------
# ChatMessage <-> TranscriptItem
# ---------------------------------------------------------------------------


def items_from_messages(
    messages: list[ChatMessage], include_thinking: bool = False
) -> list[TranscriptItem]:
    """Convert Inspect messages into Claude Code transcript items.

    Thinking is **dropped by default**: Anthropic validates thinking-block
    signatures, so replaying a block whose signature didn't come from the same
    provider fails the next turn. Pass ``include_thinking=True`` only when the
    reasoning came from an Anthropic endpoint with its signature intact
    (``ContentReasoning.signature``).

    Also lossy in that system messages have no transcript row — Claude Code
    takes its system prompt from the CLI, not the session file — so they are
    skipped; pass them via ``system_prompt`` on the agent instead.
    """
    items: list[TranscriptItem] = []
    for message in messages:
        if isinstance(message, ChatMessageSystem):
            continue
        if isinstance(message, ChatMessageUser):
            items.extend(_user_items(message))
        elif isinstance(message, ChatMessageAssistant):
            items.extend(_assistant_items(message, include_thinking))
        elif isinstance(message, ChatMessageTool):
            items.append(
                ToolResult(
                    tool_use_id=message.tool_call_id or "",
                    content=message.error.message if message.error else message.text,
                    is_error=message.error is not None,
                )
            )
    return items


def messages_from_items(items: list[TranscriptItem]) -> list[ChatMessage]:
    """Convert Claude Code transcript items into Inspect messages.

    Not a lossless inverse of :func:`items_from_messages` — resume from the
    items, not from these messages. Consecutive assistant-side items (thinking,
    text, tool uses) fold into one ``ChatMessageAssistant``; a tool result that
    Claude Code stored as content blocks is JSON-encoded into the message text;
    :class:`RawBlock` items are dropped.
    """
    messages: list[ChatMessage] = []
    content: list[Content] = []
    tool_calls: list[ToolCall] = []
    call_names: dict[str, str] = {}

    def flush_assistant() -> None:
        if content or tool_calls:
            messages.append(
                ChatMessageAssistant(
                    content=list(content), tool_calls=list(tool_calls) or None
                )
            )
            content.clear()
            tool_calls.clear()

    for item in items:
        if isinstance(item, UserText):
            flush_assistant()
            messages.append(ChatMessageUser(content=item.text))
        elif isinstance(item, AssistantText):
            content.append(ContentText(text=item.text))
        elif isinstance(item, Thinking):
            content.append(
                ContentReasoning(reasoning=item.thinking, signature=item.signature)
            )
        elif isinstance(item, ToolUse):
            call_names[item.id] = item.name
            tool_calls.append(
                ToolCall(id=item.id, function=item.name, arguments=item.input)
            )
        elif isinstance(item, ToolResult):
            flush_assistant()
            messages.append(
                ChatMessageTool(
                    content=item.content
                    if isinstance(item.content, str)
                    else json.dumps(item.content),
                    tool_call_id=item.tool_use_id,
                    function=call_names.get(item.tool_use_id),
                    error=ToolCallError("unknown", _error_text(item))
                    if item.is_error
                    else None,
                )
            )
        else:
            flush_assistant()  # RawBlock: no ChatMessage equivalent

    flush_assistant()
    return messages


def _error_text(item: ToolResult) -> str:
    return item.content if isinstance(item.content, str) else json.dumps(item.content)


def _as_transcript_items(
    items: list[TranscriptItem] | list[ChatMessage],
) -> list[TranscriptItem]:
    message_types = (
        ChatMessageSystem,
        ChatMessageUser,
        ChatMessageAssistant,
        ChatMessageTool,
    )
    is_message = [isinstance(i, message_types) for i in items]
    if not any(is_message):
        return cast(list[TranscriptItem], items)
    if not all(is_message):
        raise ValueError(
            "`items` must be all ChatMessage or all TranscriptItem, not a mix of both."
        )
    return items_from_messages(cast(list[ChatMessage], items))


def _user_items(message: ChatMessageUser) -> list[TranscriptItem]:
    if isinstance(message.content, str):
        return [UserText(text=message.content)]
    items: list[TranscriptItem] = []
    for block in message.content:
        if isinstance(block, ContentText):
            items.append(UserText(text=block.text))
        else:
            # Images and other modalities are valid Anthropic blocks; pass the
            # model_dump through rather than inventing a typed shape for each.
            items.append(RawBlock(role="user", block=block.model_dump()))
    return items


def _assistant_items(
    message: ChatMessageAssistant, include_thinking: bool
) -> list[TranscriptItem]:
    blocks: list[Content] = (
        [ContentText(text=message.content)]
        if isinstance(message.content, str)
        else list(message.content)
    )
    items: list[TranscriptItem] = []
    for block in blocks:
        if isinstance(block, ContentReasoning):
            if include_thinking:
                items.append(
                    Thinking(thinking=block.reasoning, signature=block.signature)
                )
        elif isinstance(block, ContentText):
            if block.text:
                items.append(AssistantText(text=block.text))
        else:
            items.append(RawBlock(role="assistant", block=block.model_dump()))
    for tool_call in message.tool_calls or []:
        items.append(
            ToolUse(id=tool_call.id, name=tool_call.function, input=tool_call.arguments)
        )
    return items


# ---------------------------------------------------------------------------
# Row <-> item
# ---------------------------------------------------------------------------


def _item_role(item: TranscriptItem) -> Literal["user", "assistant"]:
    if isinstance(item, UserText | ToolResult):
        return "user"
    if isinstance(item, RawBlock):
        return item.role
    return "assistant"


def _message_payload(
    item: TranscriptItem, *, model: str, message_id: str | None
) -> dict[str, Any]:
    if isinstance(item, UserText):
        # Claude Code writes plain-string content for a typed user turn.
        return {"role": "user", "content": item.text}
    if isinstance(item, ToolResult):
        return {"role": "user", "content": [_tool_result_block(item)]}
    if isinstance(item, RawBlock) and item.role == "user":
        return {"role": "user", "content": [item.block]}
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [_assistant_block(item)],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _tool_result_block(item: ToolResult) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": item.tool_use_id,
        "content": item.content,
        "is_error": item.is_error,
    }


def _assistant_block(item: TranscriptItem) -> dict[str, Any]:
    if isinstance(item, AssistantText):
        return {"type": "text", "text": item.text}
    if isinstance(item, Thinking):
        block: dict[str, Any] = {"type": "thinking", "thinking": item.thinking}
        if item.signature is not None:
            block["signature"] = item.signature
        return block
    if isinstance(item, ToolUse):
        return {
            "type": "tool_use",
            "id": item.id,
            "name": item.name,
            "input": item.input,
        }
    if isinstance(item, RawBlock):
        return item.block
    raise ValueError(f"{type(item).__name__} is not an assistant content block")


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in (content or []) if isinstance(b, dict)]


def _block_to_item(
    block: dict[str, Any], role: Literal["user", "assistant"]
) -> TranscriptItem:
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text") or ""
        return AssistantText(text=text) if role == "assistant" else UserText(text=text)
    if block_type == "thinking" and role == "assistant":
        return Thinking(
            thinking=block.get("thinking") or "", signature=block.get("signature")
        )
    if block_type == "tool_use" and role == "assistant":
        tool_input = block.get("input")
        if isinstance(tool_input, dict) and isinstance(block.get("id"), str):
            return ToolUse(
                id=block["id"], name=block.get("name") or "", input=tool_input
            )
    if block_type == "tool_result" and role == "user":
        tool_use_id = block.get("tool_use_id")
        content = block.get("content")
        if isinstance(tool_use_id, str) and isinstance(content, str | list):
            return ToolResult(
                tool_use_id=tool_use_id,
                content=content,
                is_error=bool(block.get("is_error")),
            )
    return RawBlock(role=role, block=block)
