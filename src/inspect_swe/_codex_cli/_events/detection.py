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
    task_name: str | None = None
    """Multi-Agent V2 task name (e.g. "write_fizzbuzz"); None under V1."""

    @property
    def name(self) -> str:
        """Display name for the agent's span (V2 task_name, else V1 agent_type)."""
        return self.task_name or self.agent_type


def agent_message_recipients(input_messages: list[ChatMessage]) -> set[str]:
    """Recipients of the agent_message items in a request's input.

    Multi-Agent V2 delivers inter-agent messages as `agent_message` input items;
    the bridge preserves each raw item (author/recipient) on ContentText.internal.
    Every agent_message in a request is inbound to the requester, so the
    recipient path (e.g. "/root/write_fizzbuzz") identifies the calling agent.
    """
    return {
        recipient
        for item in _agent_message_items(input_messages)
        if isinstance(recipient := item.get("recipient"), str) and recipient
    }


def final_answer_authors(input_messages: list[ChatMessage]) -> set[str]:
    """Authors of FINAL_ANSWER agent_message items in a request's input.

    A FINAL_ANSWER is a sub-agent's terminal return under Multi-Agent V2 (the
    `wait_agent` result no longer carries per-thread completion status), so its
    author path marks that agent's thread as completed.
    """
    authors: set[str] = set()
    for item in _agent_message_items(input_messages):
        author = item.get("author")
        if not (isinstance(author, str) and author):
            continue
        for part in item.get("content") or []:
            if (
                isinstance(part, dict)
                and part.get("type") == "input_text"
                and isinstance(text := part.get("text"), str)
                and text.lstrip().startswith("Message Type: FINAL_ANSWER")
            ):
                authors.add(author)
                break
    return authors


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
        task_name = args.get("task_name")
        result.append(
            SpawnedAgent(
                call_id=tc.id,
                agent_type=str(args.get("agent_type") or "agent"),
                message=message,
                reasoning_effort=str(reasoning) if reasoning else None,
                task_name=str(task_name) if task_name else None,
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

    Multi-Agent V2 results carry `{"task_name": "/root/<name>"}` instead of
    `agent_id`/`nickname`; the absolute task path serves as the thread id.
    """
    if message.function != SPAWN_AGENT:
        return None
    data = _loads(message.text)
    if isinstance(data, dict):
        agent_id = data.get("agent_id") or data.get("task_name")
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


def _agent_message_items(input_messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Raw agent_message items stashed on user-message content by the bridge."""
    items: list[dict[str, Any]] = []
    for msg in input_messages:
        if not isinstance(msg, ChatMessageUser) or isinstance(msg.content, str):
            continue
        for content in msg.content:
            internal = getattr(content, "internal", None)
            if isinstance(internal, dict):
                item = internal.get("agent_message")
                if isinstance(item, dict):
                    items.append(item)
    return items


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
