"""Build synthetic codex ``rollout-*.jsonl`` session files for resume.

Codex persists each session as a JSONL rollout under
``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl``. Writing a
synthetic rollout there and then calling ACP ``session/load`` with its
``session_id`` makes codex resume from that prior conversation on the next
prompt turn — the prior turns become real session history rather than a
prefill of the live model context.

The schema is reverse-engineered from real codex rollouts: a minimal rollout
(a ``session_meta`` row, a ``turn_context`` row, then the prior response
items) is enough for the ``codex-acp`` adapter's ``load_session`` to accept it
and feed the prior context to the model.

Build a :class:`RolloutSpec` with :func:`build_rollout`, then write
``spec.content`` to ``<codex_home>/<spec.relative_path>`` — inside a sandbox
via ``sandbox.write_file`` (see ``CodexCli`` resume), or on the host via
:meth:`RolloutSpec.write_to`.

``build_rollout`` takes either a list of Inspect ``ChatMessage`` (the ergonomic
form — most callers already have messages) or a list of :data:`PriorItem` (the
codex-native form). ``ChatMessage`` is a lossy input for a few codex constructs,
so the typed items remain available as the full-fidelity layer; see
:func:`prior_from_messages` for what does not survive the conversion.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
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
    ContentImage,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall
from pydantic import BaseModel, Field, ValidationError

# Key under which Inspect's OpenAI Responses provider stashes the reasoning
# ciphertext for non-redacted reasoning (mirrors the private
# `inspect_ai.model._openai_responses.REASONING_ENCRYPTED_CONTENT`; importing it
# would pull the `openai` SDK into a runtime path where it isn't a dependency).
_REASONING_ENCRYPTED_CONTENT = "reasoning_encrypted_content"


class UserText(BaseModel):
    role: Literal["user"] = "user"
    text: str


class AssistantText(BaseModel):
    role: Literal["assistant"] = "assistant"
    text: str


class DeveloperText(BaseModel):
    role: Literal["developer"] = "developer"
    text: str


class FunctionCall(BaseModel):
    kind: Literal["function_call"] = "function_call"
    name: str
    arguments: str
    call_id: str


class FunctionCallOutput(BaseModel):
    kind: Literal["function_call_output"] = "function_call_output"
    call_id: str
    # Usually a string, but codex writes a list of content blocks when a tool
    # returns structured/image output (e.g. [{"type": "input_image", ...}]).
    output: str | list[Any]


class Reasoning(BaseModel):
    kind: Literal["reasoning"] = "reasoning"
    # Plaintext reasoning. Codex's own captures leave this empty (the backend
    # withholds plaintext and stores the encrypted signature in
    # ``encrypted_content``), but synthetic priors can carry plaintext too.
    # The "redacted" state is simply ``text=""`` with ``encrypted_content`` set
    # — exactly what codex writes — so there is no separate redacted flag to
    # desync from the content.
    text: str = ""
    # Optional human-readable summary. Codex stores ``summary`` as a list of
    # {type,text} blobs; collapsed to one string here, written back as a
    # single-element list only when set.
    summary: str | None = None
    # Opaque server-side ciphertext. Needed to replay reasoning for
    # signed-reasoning / visible-CoT model families without invalidation on
    # round-trip. ``None`` means "no signature" — codex still accepts the row
    # but treats the reasoning as plaintext-only.
    encrypted_content: str | None = None


class CustomToolCall(BaseModel):
    # Codex emits ``custom_tool_call`` (instead of ``function_call``) for tools
    # registered as Responses-API "custom" tools — apply_patch is the one that
    # hits this codepath in practice. ``input`` is a free-form string, NOT a
    # JSON-encoded args dict.
    kind: Literal["custom_tool_call"] = "custom_tool_call"
    name: str
    input: str
    call_id: str


class CustomToolCallOutput(BaseModel):
    kind: Literal["custom_tool_call_output"] = "custom_tool_call_output"
    call_id: str
    output: str | list[Any]


class RawResponseItem(BaseModel):
    # Round-trips any codex ``response_item`` type this module doesn't model
    # explicitly (e.g. ``web_search_call``, ``tool_search_call``). Real codex
    # rollouts contain such rows; preserving the raw payload verbatim lets
    # parse_rollout read real sessions without dropping rows or raising, and
    # keeps truncate-and-rebuild faithful.
    kind: Literal["raw"] = "raw"
    payload: dict[str, Any]


# Two discriminated sub-unions joined with a plain union: the message variants
# carry `role`, the response-item variants carry `kind`. A single
# `Field(discriminator="role")` over all variants is invalid (most have no
# `role`) and makes `TypeAdapter(PriorItem)` — and any model with a
# `list[PriorItem]` field — fail to build at schema time. A payload carrying
# BOTH `role` and `kind` is resolved by Pydantic's smart-union (payload-
# dependent, not a fixed arm precedence); the parser never feeds such payloads
# in (it dispatches on `type`), so this only matters for hand-constructed
# `list[PriorItem]` values, where it's a caller error to begin with.
_MessageItem = Annotated[
    UserText | AssistantText | DeveloperText, Field(discriminator="role")
]
_ResponseItem = Annotated[
    FunctionCall
    | FunctionCallOutput
    | Reasoning
    | CustomToolCall
    | CustomToolCallOutput
    | RawResponseItem,
    Field(discriminator="kind"),
]
PriorItem = _MessageItem | _ResponseItem


_CODEX_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent based on GPT-5. "
    "Answer the user's questions concisely."
)


class RolloutSpec(BaseModel):
    """A built rollout, ready to write to a host fs or a sandbox fs."""

    session_id: str
    relative_path: str  # "sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl"
    content: str  # complete jsonl, newline-terminated
    model: str  # the model recorded in the rollout (resume must serve the same)

    def write_to(self, codex_home: Path) -> Path:
        """Write under a host filesystem ``codex_home`` and return the path."""
        path = codex_home / self.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content)
        return path


class ParsedRollout(BaseModel):
    """A rollout parsed back into typed items + session metadata.

    The result of :func:`parse_rollout`; feed ``prior`` (optionally sliced)
    plus the metadata fields back into :func:`build_rollout` to resume.
    """

    session_id: str | None
    cwd: str
    model: str
    base_instructions: str
    model_provider: str
    cli_version: str
    prior: list[PriorItem]

    def as_messages(self) -> list[ChatMessage]:
        """This rollout as Inspect ``ChatMessage``, for reading and scoring.

        Convenience wrapper around :func:`messages_from_prior`. To *resume* from
        a rollout, pass ``prior`` back to :func:`build_rollout` rather than
        round-tripping through messages — the item form is lossless, messages
        are not (see :func:`messages_from_prior`). ``base_instructions`` is
        codex's own system prompt and is not included.
        """
        return messages_from_prior(self.prior)


def build_rollout(
    *,
    cwd: str,
    prior: Sequence[PriorItem | ChatMessage],
    model: str,
    base_instructions: str = _CODEX_BASE_INSTRUCTIONS,
    cli_version: str = "0.130.0",
    model_provider: str = "openai",
    originator: str = "codex_exec",
    timestamp: datetime | None = None,
) -> RolloutSpec:
    """Build a synthetic rollout without writing it.

    Returns a :class:`RolloutSpec` whose ``content`` can be written to
    ``<codex_home>/<spec.relative_path>`` (e.g. via ``sandbox.write_file`` for
    a sandboxed codex), after which ``load_session(session_id=spec.session_id)``
    resumes from this prior. For host-fs writes use :func:`synthesize_rollout`.

    ``prior`` is either Inspect ``ChatMessage`` (converted via
    :func:`prior_from_messages`) or codex-native :data:`PriorItem` — all of one
    or all of the other, not a mix.

    ``model`` is required and must match the model the resuming agent serves —
    a mismatch makes codex splice a ``<model_switch>`` banner into the resumed
    conversation (when resuming a parsed rollout, pass ``parsed.model``).
    """
    prior_items = _as_prior_items(prior)
    now = timestamp or datetime.now(UTC)
    session_id = str(uuid.uuid4())
    ts_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    relative_path = (
        f"sessions/{now.strftime('%Y')}/{now.strftime('%m')}/{now.strftime('%d')}/"
        f"rollout-{ts_str}-{session_id}.jsonl"
    )

    rows: list[dict[str, Any]] = [
        {
            "timestamp": ts_iso,
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": ts_iso,
                "cwd": cwd,
                "originator": originator,
                "cli_version": cli_version,
                "source": "exec",
                "model_provider": model_provider,
                "base_instructions": {"text": base_instructions},
                "git": None,
            },
        },
        {
            "timestamp": ts_iso,
            "type": "turn_context",
            "payload": _make_turn_context(cwd=cwd, model=model, current_date=now),
        },
    ]
    for item in prior_items:
        rows.append(
            {
                "timestamp": ts_iso,
                "type": "response_item",
                "payload": _item_payload(item),
            }
        )

    content = "".join(json.dumps(row) + "\n" for row in rows)
    return RolloutSpec(
        session_id=session_id,
        relative_path=relative_path,
        content=content,
        model=model,
    )


def synthesize_rollout(
    *,
    cwd: str,
    prior: Sequence[PriorItem | ChatMessage],
    codex_home: Path,
    model: str,
    base_instructions: str = _CODEX_BASE_INSTRUCTIONS,
    cli_version: str = "0.130.0",
    model_provider: str = "openai",
    originator: str = "codex_exec",
    timestamp: datetime | None = None,
) -> tuple[Path, str]:
    """Build and write a synthetic rollout to a host fs; return ``(path, session_id)``.

    Convenience for host-side round-trip tests. For sandbox-targeted writes use
    :func:`build_rollout` and route ``spec.content`` through
    ``sandbox.write_file``.
    """
    spec = build_rollout(
        cwd=cwd,
        prior=prior,
        model=model,
        base_instructions=base_instructions,
        cli_version=cli_version,
        model_provider=model_provider,
        originator=originator,
        timestamp=timestamp,
    )
    return spec.write_to(codex_home), spec.session_id


def parse_rollout(content: str) -> ParsedRollout:
    """Parse a codex rollout JSONL back into typed items + session metadata.

    The inverse of :func:`build_rollout`. Use it to read a saved/real rollout,
    truncate ``prior`` at a chosen point, and rebuild a resumable rollout::

        parsed = parse_rollout(saved_content)
        spec = build_rollout(
            cwd=parsed.cwd, prior=parsed.prior[:n], model=parsed.model
        )

    Round-trips losslessly for the modelled item types. Reasoning whose
    plaintext codex withheld comes back as ``text=""`` with its
    ``encrypted_content`` signature preserved. Rows this module doesn't model
    (or a modelled row whose shape drifted) come back as :class:`RawResponseItem`
    carrying the verbatim payload, so they too rebuild faithfully.
    """
    # Split on "\n" only — NOT str.splitlines(), which also breaks on the
    # unicode line separators (U+2028/U+2029, VT, FF, NEL) that appear inside
    # JSON string values in real rollouts and would corrupt a row mid-string.
    rows = [json.loads(line) for line in content.split("\n") if line.strip()]
    meta: dict[str, Any] = next(
        (r["payload"] for r in rows if r.get("type") == "session_meta"), {}
    )
    turn: dict[str, Any] = next(
        (r["payload"] for r in rows if r.get("type") == "turn_context"), {}
    )
    base = meta.get("base_instructions")
    base_instructions = base.get("text", "") if isinstance(base, dict) else (base or "")
    prior = [
        _payload_to_item(r["payload"]) for r in rows if r.get("type") == "response_item"
    ]
    return ParsedRollout(
        session_id=meta.get("id"),
        cwd=meta.get("cwd") or turn.get("cwd") or "",
        model=turn.get("model") or meta.get("model") or "gpt-5.5",
        base_instructions=base_instructions,
        model_provider=meta.get("model_provider") or "openai",
        cli_version=meta.get("cli_version") or "0.130.0",
        prior=prior,
    )


# ---------------------------------------------------------------------------
# ChatMessage <-> PriorItem
#
# Codex rollout ``response_item`` payloads are OpenAI Responses API items, so
# this is the same mapping Inspect's OpenAI Responses provider performs — done
# here rather than imported from it because that module imports the `openai`
# SDK, which inspect_swe does not depend on at runtime.
# ---------------------------------------------------------------------------


def prior_from_messages(messages: Sequence[ChatMessage]) -> list[PriorItem]:
    """Convert Inspect messages into codex prior items.

    The ergonomic way to build a prior, and lossy in ways the typed items are
    not:

    - tool-call arguments are re-serialized from ``ToolCall.arguments`` (a dict),
      so a rollout built this way won't be byte-identical to codex's own;
    - codex ``custom_tool_call`` items (``apply_patch``) take free-form text
      rather than JSON arguments and have no ``ToolCall`` representation, so
      they can only be expressed as :class:`CustomToolCall`;
    - rows codex writes that Inspect has no equivalent for (``web_search_call``,
      ``tool_search_call``) can only be expressed as :class:`RawResponseItem`.

    Reasoning is carried across in full: plaintext into ``text``, the ciphertext
    Inspect stashes for signed reasoning into ``encrypted_content``, and a
    redacted block as ``text=""`` plus its ciphertext (exactly how codex
    persists reasoning whose plaintext the backend withheld).

    Raises ``ValueError`` on content with no codex equivalent (audio, video)
    rather than dropping it silently.
    """
    prior: list[PriorItem] = []
    for message in messages:
        if isinstance(message, ChatMessageSystem):
            prior.append(DeveloperText(text=message.text))
        elif isinstance(message, ChatMessageUser):
            prior.extend(_user_items(message))
        elif isinstance(message, ChatMessageAssistant):
            prior.extend(_assistant_items(message))
        elif isinstance(message, ChatMessageTool):
            # codex has no error flag on a call output; fold the error text in.
            output = message.error.message if message.error else message.text
            prior.append(
                FunctionCallOutput(call_id=message.tool_call_id or "", output=output)
            )
    return prior


def messages_from_prior(prior: Sequence[PriorItem]) -> list[ChatMessage]:
    """Convert codex prior items into Inspect messages, for reading and scoring.

    Not a lossless inverse of :func:`prior_from_messages` — resume from the
    items, not from these messages. Specifically: consecutive assistant-side
    items (reasoning, text, tool calls) are folded into one
    ``ChatMessageAssistant``; ``custom_tool_call`` text is wrapped as
    ``{"input": ...}`` because ``ToolCall.arguments`` is a dict; a call output
    that codex stored as content blocks is JSON-encoded into the message text;
    and :class:`RawResponseItem` rows are dropped (there is nothing to map them
    to).
    """
    messages: list[ChatMessage] = []
    content: list[Content] = []
    tool_calls: list[ToolCall] = []
    # function name per call_id, so a tool result can name the tool it answers
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

    for item in prior:
        if isinstance(item, UserText):
            flush_assistant()
            messages.append(ChatMessageUser(content=item.text))
        elif isinstance(item, DeveloperText):
            flush_assistant()
            messages.append(ChatMessageSystem(content=item.text))
        elif isinstance(item, AssistantText):
            content.append(ContentText(text=item.text))
        elif isinstance(item, Reasoning):
            content.append(_reasoning_content(item))
        elif isinstance(item, FunctionCall):
            call_names[item.call_id] = item.name
            tool_calls.append(_tool_call(item))
        elif isinstance(item, CustomToolCall):
            call_names[item.call_id] = item.name
            tool_calls.append(
                ToolCall(
                    id=item.call_id,
                    function=item.name,
                    arguments={"input": item.input},
                )
            )
        elif isinstance(item, FunctionCallOutput | CustomToolCallOutput):
            flush_assistant()
            messages.append(
                ChatMessageTool(
                    content=item.output
                    if isinstance(item.output, str)
                    else json.dumps(item.output),
                    tool_call_id=item.call_id,
                    function=call_names.get(item.call_id),
                )
            )
        else:
            flush_assistant()  # RawResponseItem: no ChatMessage equivalent

    flush_assistant()
    return messages


def _as_prior_items(
    prior: Sequence[PriorItem | ChatMessage],
) -> list[PriorItem]:
    items = list(prior)
    message_types = ChatMessageSystem | ChatMessageUser | ChatMessageAssistant
    is_message = [isinstance(i, message_types | ChatMessageTool) for i in items]
    if not any(is_message):
        return cast(list[PriorItem], items)
    if not all(is_message):
        raise ValueError(
            "`prior` must be all ChatMessage or all PriorItem, not a mix of both."
        )
    return prior_from_messages(cast(Sequence[ChatMessage], items))


def _user_items(message: ChatMessageUser) -> list[PriorItem]:
    if isinstance(message.content, str):
        return [UserText(text=message.content)]
    items: list[PriorItem] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        if text_parts:
            items.append(UserText(text="".join(text_parts)))
            text_parts.clear()

    for block in message.content:
        if isinstance(block, ContentText):
            text_parts.append(block.text)
        elif isinstance(block, ContentImage):
            # Codex models images as an input_image block on a user message; we
            # don't model that shape, so emit the payload verbatim.
            flush_text()
            items.append(
                RawResponseItem(
                    payload={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": block.image,
                                "detail": block.detail,
                            }
                        ],
                    }
                )
            )
        else:
            raise ValueError(
                f"prior_from_messages: user content of type {block.type!r} has no "
                f"codex rollout equivalent; pass a RawResponseItem instead."
            )
    flush_text()
    return items


def _assistant_items(message: ChatMessageAssistant) -> list[PriorItem]:
    blocks: list[Content] = (
        [ContentText(text=message.content)]
        if isinstance(message.content, str)
        else list(message.content)
    )
    items: list[PriorItem] = []
    for block in blocks:
        if isinstance(block, ContentReasoning):
            items.append(_reasoning_item(block))
        elif isinstance(block, ContentText):
            # An empty text block is what Inspect emits alongside tool calls;
            # codex has no row for it.
            if block.text:
                items.append(AssistantText(text=block.text))
        else:
            raise ValueError(
                f"prior_from_messages: assistant content of type {block.type!r} has "
                f"no codex rollout equivalent; pass a RawResponseItem instead."
            )
    for tool_call in message.tool_calls or []:
        items.append(
            FunctionCall(
                name=tool_call.function,
                arguments=json.dumps(tool_call.arguments),
                call_id=tool_call.id,
            )
        )
    return items


def _reasoning_item(block: ContentReasoning) -> Reasoning:
    if block.redacted:
        # Inspect keeps the ciphertext in `reasoning` when the plaintext was
        # withheld; codex keeps it in `encrypted_content` with no plaintext.
        return Reasoning(
            text="", summary=block.summary, encrypted_content=block.reasoning
        )
    encrypted = (
        block.internal.get(_REASONING_ENCRYPTED_CONTENT)
        if isinstance(block.internal, dict)
        else None
    )
    return Reasoning(
        text=block.reasoning,
        summary=block.summary,
        encrypted_content=encrypted if isinstance(encrypted, str) else None,
    )


def _reasoning_content(item: Reasoning) -> ContentReasoning:
    if not item.text and item.encrypted_content is not None:
        return ContentReasoning(
            reasoning=item.encrypted_content, summary=item.summary, redacted=True
        )
    return ContentReasoning(
        reasoning=item.text,
        summary=item.summary,
        internal={_REASONING_ENCRYPTED_CONTENT: item.encrypted_content}
        if item.encrypted_content is not None
        else None,
    )


def _tool_call(item: FunctionCall) -> ToolCall:
    try:
        arguments = json.loads(item.arguments)
    except json.JSONDecodeError as ex:
        return ToolCall(
            id=item.call_id, function=item.name, arguments={}, parse_error=str(ex)
        )
    if not isinstance(arguments, dict):
        return ToolCall(
            id=item.call_id,
            function=item.name,
            arguments={},
            parse_error=f"arguments are {type(arguments).__name__}, not an object",
        )
    return ToolCall(id=item.call_id, function=item.name, arguments=arguments)


def _payload_to_item(payload: dict[str, Any]) -> PriorItem:
    # Anything we don't model explicitly (web_search_call, an unknown message
    # role, a future row type) — OR a modelled type whose shape has drifted
    # (a missing key, a dict where a str was expected) — is preserved verbatim
    # as a RawResponseItem rather than dropped or raised on, so parse survives
    # real codex rollouts and rebuilds them faithfully.
    try:
        typed = _typed_item(payload)
    except (KeyError, TypeError, AttributeError, ValidationError):
        typed = None
    return typed if typed is not None else RawResponseItem(payload=payload)


def _typed_item(payload: dict[str, Any]) -> PriorItem | None:
    """Map a payload to a modelled item, or ``None`` if it isn't one we model."""
    ptype = payload.get("type")
    if ptype == "function_call":
        return FunctionCall(
            name=payload["name"],
            arguments=payload["arguments"],
            call_id=payload["call_id"],
        )
    if ptype == "function_call_output":
        return FunctionCallOutput(call_id=payload["call_id"], output=payload["output"])
    if ptype == "custom_tool_call":
        return CustomToolCall(
            name=payload["name"], input=payload["input"], call_id=payload["call_id"]
        )
    if ptype == "custom_tool_call_output":
        return CustomToolCallOutput(
            call_id=payload["call_id"], output=payload["output"]
        )
    if ptype == "reasoning":
        content = payload.get("content")
        # Content we can't represent as plaintext (a drifted/multimodal block
        # type) must NOT be silently dropped — bail to RawResponseItem so the
        # row round-trips verbatim. content=None (codex withheld plaintext) and
        # all-reasoning_text content both stay typed.
        if content and any(b.get("type") != "reasoning_text" for b in content):
            return None
        text = "".join(b.get("text", "") for b in content) if content else ""
        summary_blocks = payload.get("summary") or []
        summary = (
            "".join(
                b.get("text", "")
                for b in summary_blocks
                if b.get("type") == "summary_text"
            )
            or None
        )
        return Reasoning(
            text=text,
            summary=summary,
            encrypted_content=payload.get("encrypted_content"),
        )
    if ptype == "message":
        role = payload.get("role")
        blocks = payload.get("content") or []
        # Only claim a typed text message when EVERY block is plain text; a
        # non-text block (e.g. an input_image) would be lost by the join, so
        # fall through to RawResponseItem and preserve the row verbatim.
        if role in ("user", "assistant", "developer") and all(
            b.get("type") in ("input_text", "output_text") for b in blocks
        ):
            text = "".join(b.get("text", "") for b in blocks)
            if role == "assistant":
                return AssistantText(text=text)
            if role == "developer":
                return DeveloperText(text=text)
            return UserText(text=text)
    return None


def _make_turn_context(
    *, cwd: str, model: str, current_date: datetime
) -> dict[str, Any]:
    return {
        "turn_id": str(uuid.uuid4()),
        "cwd": cwd,
        "current_date": current_date.strftime("%Y-%m-%d"),
        "timezone": "UTC",
        "approval_policy": "never",
        "sandbox_policy": {
            "type": "workspace-write",
            "writable_roots": [],
            "network_access": False,
            "exclude_tmpdir_env_var": False,
            "exclude_slash_tmp": False,
        },
        "permission_profile": {
            "type": "managed",
            "file_system": {"type": "restricted", "entries": []},
            "network": "restricted",
        },
        "file_system_sandbox_policy": {"kind": "restricted", "entries": []},
        "model": model,
        "personality": "pragmatic",
        "collaboration_mode": {
            "mode": "default",
            "settings": {
                "model": model,
                "reasoning_effort": "medium",
                "developer_instructions": None,
            },
        },
        "realtime_active": False,
        "effort": "medium",
        "summary": "none",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
    }


def _item_payload(item: PriorItem) -> dict[str, Any]:
    if isinstance(item, RawResponseItem):
        return item.payload
    if isinstance(item, FunctionCall):
        return {
            "type": "function_call",
            "name": item.name,
            "arguments": item.arguments,
            "call_id": item.call_id,
        }
    if isinstance(item, FunctionCallOutput):
        return {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": item.output,
        }
    if isinstance(item, CustomToolCall):
        return {
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": item.call_id,
            "name": item.name,
            "input": item.input,
        }
    if isinstance(item, CustomToolCallOutput):
        return {
            "type": "custom_tool_call_output",
            "call_id": item.call_id,
            "output": item.output,
        }
    if isinstance(item, Reasoning):
        # No plaintext (text == "") -> content is null, matching how codex
        # persists reasoning whose plaintext was withheld (signature only).
        payload: dict[str, Any] = {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": item.summary}]
            if item.summary
            else [],
            "content": [{"type": "reasoning_text", "text": item.text}]
            if item.text
            else None,
        }
        if item.encrypted_content is not None:
            payload["encrypted_content"] = item.encrypted_content
        return payload
    # message item — distinguish assistant (output_text) from user/developer (input_text)
    content_type = "output_text" if isinstance(item, AssistantText) else "input_text"
    return {
        "type": "message",
        "role": item.role,
        "content": [{"type": content_type, "text": item.text}],
    }
