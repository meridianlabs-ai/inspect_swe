"""Pure helpers for interpreting Codex bridge `ModelEvent`s.

Everything here operates purely on the inspect_ai chat messages / tool calls the
bridge `ModelEventSink` already receives — there is no parsing of Codex's
`--json` stdout stream. This is what lets the consumer reconstruct sub-agent
spans (and detect compaction) bridge-only.
"""

import json
from dataclasses import dataclass
from typing import Any

from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

# Codex built-in multi-agent tool names (the `multi_agent_v1` namespace).
SPAWN_AGENT = "spawn_agent"
CLOSE_AGENT = "close_agent"
WAIT_AGENT = "wait_agent"

# Marker injected as a user message when Codex performs *local* compaction. Our
# custom bridge provider always forces the local path (remote compaction is gated
# to the real "OpenAI"/Azure providers), so this normal `/v1/responses` call is
# the only compaction signal — and it is what Codex's own tests key on.
# Source: codex-rs/core/templates/compact/prompt.md (injected at compact.rs:70-82).
COMPACTION_MARKER = "You are performing a CONTEXT CHECKPOINT COMPACTION."


@dataclass
class SpawnedAgent:
    """A `spawn_agent` tool-call extracted from a parent's model output."""

    call_id: str
    agent_type: str
    message: str
    reasoning_effort: str | None


def find_spawned_agents(tool_calls: list[ToolCall] | None) -> list[SpawnedAgent]:
    """Spawn_agent tool-calls in a parent's output, with their spawn prompts."""
    result: list[SpawnedAgent] = []
    for tc in tool_calls or []:
        if tc.function != SPAWN_AGENT:
            continue
        args = tc.arguments or {}
        message = args.get("message")
        if not isinstance(message, str) or not message:
            continue
        reasoning = args.get("reasoning_effort")
        result.append(
            SpawnedAgent(
                call_id=tc.id,
                agent_type=str(args.get("agent_type") or "agent"),
                message=message,
                reasoning_effort=str(reasoning) if reasoning else None,
            )
        )
    return result


def find_close_targets(tool_calls: list[ToolCall] | None) -> list[str]:
    """Thread ids targeted by `close_agent` tool-calls in a parent's output."""
    targets: list[str] = []
    for tc in tool_calls or []:
        if tc.function != CLOSE_AGENT:
            continue
        target = (tc.arguments or {}).get("target")
        if isinstance(target, str) and target:
            targets.append(target)
    return targets


@dataclass
class SpawnResult:
    """The `{agent_id, nickname}` returned by a `spawn_agent` tool result."""

    agent_id: str
    nickname: str | None


def spawn_result(message: ChatMessageTool) -> SpawnResult | None:
    """The `agent_id` (thread id) + `nickname` from a `spawn_agent` tool result.

    The result is correlated to its spawn call by `message.tool_call_id`, so the
    caller can bind thread_id → span without any ordering assumptions. The
    `nickname` (Codex's friendly per-agent name) is surfaced for tool views.
    """
    if message.function != SPAWN_AGENT:
        return None
    data = _loads(message.text)
    if isinstance(data, dict):
        agent_id = data.get("agent_id")
        if isinstance(agent_id, str) and agent_id:
            nickname = data.get("nickname")
            return SpawnResult(
                agent_id=agent_id,
                nickname=nickname if isinstance(nickname, str) and nickname else None,
            )
    return None


def completed_thread_ids(input_messages: list[ChatMessage]) -> set[str]:
    """Thread ids reported `completed`, from wait/close results and notifications.

    Two carriers, both seen in a parent's `input`:
      - `wait_agent`/`close_agent` tool results: `{"status": {"<tid>": {"completed": ...}}}`
      - `<subagent_notification>` user messages: `{"agent_path": "<tid>", "status": {"completed": ...}}`
    """
    completed: set[str] = set()
    for msg in input_messages:
        if isinstance(msg, ChatMessageTool):
            if msg.function in (WAIT_AGENT, CLOSE_AGENT):
                _collect_status_completed(_loads(msg.text), completed)
        elif isinstance(msg, ChatMessageUser):
            if "<subagent_notification>" in msg.text:
                _collect_notification_completed(msg.text, completed)
    return completed


def is_compaction_request(input_messages: list[ChatMessage]) -> bool:
    """Whether this request is a (local) compaction summarization call."""
    return any(
        isinstance(msg, ChatMessageUser)
        and msg.text.lstrip().startswith(COMPACTION_MARKER)
        for msg in input_messages
    )


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _collect_status_completed(data: Any, out: set[str]) -> None:
    # {"status": {"<thread_id>": {"completed": ...}, ...}}
    if not isinstance(data, dict):
        return
    status = data.get("status")
    if isinstance(status, dict):
        for thread_id, value in status.items():
            if (
                isinstance(thread_id, str)
                and isinstance(value, dict)
                and "completed" in value
            ):
                out.add(thread_id)


def _collect_notification_completed(text: str, out: set[str]) -> None:
    # <subagent_notification>{"agent_path": "<tid>", "status": {"completed": ...}}</...>
    payload = (
        text.replace("<subagent_notification>", "")
        .replace("</subagent_notification>", "")
        .strip()
    )
    data = _loads(payload)
    if not isinstance(data, dict):
        return
    thread_id = data.get("agent_path")
    status = data.get("status")
    if (
        isinstance(thread_id, str)
        and isinstance(status, dict)
        and "completed" in status
    ):
        out.add(thread_id)
