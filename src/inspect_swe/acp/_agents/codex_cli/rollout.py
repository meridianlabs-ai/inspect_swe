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
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


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
    output: str


class Reasoning(BaseModel):
    kind: Literal["reasoning"] = "reasoning"
    # Plaintext reasoning. Codex's own captures leave this empty (the backend
    # stores reasoning encrypted in ``encrypted_content``), but synthetic
    # priors can carry plaintext too.
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
    # Whether the original reasoning was redacted by the provider (plaintext
    # withheld, only summary + signature surfaced). Preserved through
    # round-trip so consumers can distinguish "no plaintext available" from
    # "plaintext is the empty string".
    redacted: bool = False


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
    output: str


# Two discriminated sub-unions joined with a plain union: the message variants
# carry `role`, the response-item variants carry `kind`. A single
# `Field(discriminator="role")` over all 8 is invalid (5 variants have no
# `role`) and makes `TypeAdapter(PriorItem)` — and any model with a
# `list[PriorItem]` field — fail to build at schema time.
_MessageItem = Annotated[
    UserText | AssistantText | DeveloperText, Field(discriminator="role")
]
_ResponseItem = Annotated[
    FunctionCall
    | FunctionCallOutput
    | Reasoning
    | CustomToolCall
    | CustomToolCallOutput,
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


def build_rollout(
    *,
    cwd: str,
    prior: list[PriorItem],
    model: str = "gpt-5.5",
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
    """
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
    for item in prior:
        rows.append(
            {
                "timestamp": ts_iso,
                "type": "response_item",
                "payload": _item_payload(item),
            }
        )

    content = "".join(json.dumps(row) + "\n" for row in rows)
    return RolloutSpec(
        session_id=session_id, relative_path=relative_path, content=content
    )


def synthesize_rollout(
    *,
    cwd: str,
    prior: list[PriorItem],
    codex_home: Path,
    model: str = "gpt-5.5",
    base_instructions: str = _CODEX_BASE_INSTRUCTIONS,
    cli_version: str = "0.130.0",
    model_provider: str = "openai",
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

    Round-trips losslessly except redacted reasoning: codex withholds redacted
    reasoning plaintext on write, so a ``Reasoning`` whose plaintext was dropped
    comes back as ``text=""`` with ``redacted=True``.
    """
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
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


def _payload_to_item(payload: dict[str, Any]) -> PriorItem:
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
        text = (
            "".join(
                b.get("text", "") for b in content if b.get("type") == "reasoning_text"
            )
            if content
            else ""
        )
        summary_blocks = payload.get("summary") or []
        summary = (
            "".join(
                b.get("text", "")
                for b in summary_blocks
                if b.get("type") == "summary_text"
            )
            or None
        )
        encrypted = payload.get("encrypted_content")
        return Reasoning(
            text=text,
            summary=summary,
            encrypted_content=encrypted,
            redacted=content is None and encrypted is not None,
        )
    if ptype == "message":
        role = payload.get("role")
        blocks = payload.get("content") or []
        text = "".join(
            b.get("text", "")
            for b in blocks
            if b.get("type") in ("input_text", "output_text")
        )
        if role == "assistant":
            return AssistantText(text=text)
        if role == "developer":
            return DeveloperText(text=text)
        return UserText(text=text)
    raise ValueError(f"Unrecognised rollout response_item payload type: {ptype!r}")


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
        payload: dict[str, Any] = {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": item.summary}]
            if item.summary
            else [],
            "content": [{"type": "reasoning_text", "text": item.text}]
            if item.text and not item.redacted
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
