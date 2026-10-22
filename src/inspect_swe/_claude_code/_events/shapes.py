"""What Claude Code's messages look like, for tracking its sub-agents.

Everything here depends on undocumented, version-specific text from Claude
Code (checked against 2.1.303 and 2.1.304). When a new version changes one of
these, re-record the fixtures in `tests/fixtures/claude_code_traces` and
update this module.
"""

import re
from typing import Any

from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)

# Start of a background Task/Agent tool result (the launch acknowledgement).
ASYNC_LAUNCHED = "Async agent launched"

# Start of the tool result each tool_use in a fork's launching message gets.
FORK_STARTED = "Fork started"

# Start of the first user message after Claude Code compacts a history.
COMPACTION_CONTINUATION = "This session is being continued from a previous conversation"

# What precedes the summary in a compaction continuation (whitespace-
# normalised): "This session is being continued from a previous conversation
# that ran out of context. The summary below covers the earlier portion of the
# conversation.", then the formatted summary.
_CONTINUATION_PREAMBLE = re.compile(
    rf"^{COMPACTION_CONTINUATION}[^.]*\. ?(?:The summary below covers [^.]*\. ?)?"
)
# Metadata Claude Code (2.1.304) puts between the preamble and the summary,
# e.g. <tools-used-before-summary>Agent, Bash</tools-used-before-summary>
_CONTINUATION_METADATA = re.compile(r"^(?:<([a-z-]+)>[^<]*</\1> ?)+")

# The request that asks the model for a compaction summary.
_COMPACTION_REQUEST = (
    "Your task is to create a detailed summary of the conversation so far"
)

# Starts of the user turns Claude Code adds to keep a run going: its own
# classifier (exact messages and prefixes) for retry, recovery and
# hook-feedback turns, plus hook blocking errors.
_IN_RUN_USER_PREFIXES = (
    # exact messages (matched by their opening words)
    "The previous response failed to produce a valid tool call.",
    "Your tool call was malformed and could not be parsed.",
    "[Your previous response had no visible output.",
    "The PermissionDenied hook indicated you may retry this tool call.",
    "The turn did not end: a tool call in your last message asked to end it",
    "Your response above was stopped by a safety classifier",
    "Output token limit hit",
    "[Your previous response reached the output limit with no visible output.",
    "Your response was cut off because it exceeded the output token limit.",
    "Your response above was cut off mid-stream.",
    # prefixes
    "[structured-output-enforce]",
    "[projects-reply-gate]",
    "You ended the turn without calling SendUserMessage.",
    "Goal check-in: \u00ab",
    # auto-compaction's summary request (a separate call within the run)
    _COMPACTION_REQUEST,
    # hook feedback
    "Stop hook feedback:",
    "TeammateIdle hook feedback:",
    "TaskCreated hook feedback:",
    "TaskCompleted hook feedback:",
)
_HOOK_BLOCKING_ERROR = re.compile(r"^\w+ hook blocking error from command")

_LEADING_REMINDERS = re.compile(
    r"^(?:\s*<system-reminder>.*?</system-reminder>)+", re.S
)
_AGENT_ID_LINE = re.compile(r"^agentId: (\S+)", re.MULTILINE)
_TASK_NOTIFICATION = re.compile(r"<task-notification>(.*?)</task-notification>", re.S)


# ---------------------------------------------------------------------------
# histories
# ---------------------------------------------------------------------------


def first_user_message(messages: list[ChatMessage]) -> ChatMessageUser | None:
    """The first user message, past leading system messages."""
    for msg in messages:
        if isinstance(msg, ChatMessageSystem):
            continue
        return msg if isinstance(msg, ChatMessageUser) else None
    return None


def strip_reminders(text: str) -> str:
    """Text without leading `<system-reminder>` blocks or surrounding whitespace."""
    return _LEADING_REMINDERS.sub("", text).strip()


def leading_text(message: ChatMessageUser | None) -> str:
    """A user message's text, skipping leading `<system-reminder>` blocks.

    Claude Code puts its context reminders before a sub-agent's prompt.
    """
    if message is None:
        return ""
    contents = (
        [message.content] if isinstance(message.content, str) else message.content
    )
    texts = [c if isinstance(c, str) else getattr(c, "text", None) for c in contents]
    return strip_reminders("\n".join(t for t in texts if isinstance(t, str)))


def run_boundaries(history: list[ChatMessage]) -> int:
    """Finished runs visible in a sub-agent's history.

    A run ends with a text-only assistant turn; a resumed run continues the
    same history with a new user turn after it. User turns Claude Code adds
    to keep one run going (max-output-tokens recovery, a blocking Stop or
    SubagentStop hook) are not boundaries.
    """
    return sum(
        1
        for prev, nxt in zip(history, history[1:], strict=False)
        if isinstance(prev, ChatMessageAssistant)
        and not prev.tool_calls
        and isinstance(nxt, ChatMessageUser)
        and not _is_in_run_user_turn(nxt)
    )


def _is_in_run_user_turn(message: ChatMessageUser) -> bool:
    """Whether a user turn is one Claude Code adds within a run.

    Like Claude Code, matches the start of the first text block, skipping
    blocks that are only a `<system-reminder>`.
    """
    contents = (
        [message.content] if isinstance(message.content, str) else message.content
    )
    for content in contents:
        text = content if isinstance(content, str) else getattr(content, "text", None)
        if not isinstance(text, str):
            continue
        text = text.lstrip()
        if text.startswith("<system-reminder>") and text.rstrip().endswith(
            "</system-reminder>"
        ):
            continue
        return text.startswith(_IN_RUN_USER_PREFIXES) or bool(
            _HOOK_BLOCKING_ERROR.match(text)
        )
    return False


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------


def is_compaction_request(messages: list[ChatMessage]) -> bool:
    """Whether a call asks the model for a compaction summary."""
    last = messages[-1] if messages else None
    return isinstance(last, ChatMessageUser) and _COMPACTION_REQUEST in last.text


def summary_body(reply: str) -> str | None:
    """A compaction reply as the next history quotes it (whitespace-normalised).

    Formatted like Claude Code: the `<analysis>` block dropped, and
    `<summary>X</summary>` replaced by "Summary: X". None if too short to
    identify anything.
    """
    reply = re.sub(r"<analysis>.*?</analysis>", "", reply, flags=re.S)
    reply = re.sub(r"<summary>(.*?)</summary>", r"Summary: \1", reply, flags=re.S)
    body = " ".join(reply.split())
    return body if len(body) >= 32 else None


def continued_summary(text: str) -> str | None:
    """What follows Claude Code's preamble in a compaction continuation.

    Whitespace-normalised. None if `text` is not a continuation.
    """
    if not text.startswith(COMPACTION_CONTINUATION):
        return None
    rest = _CONTINUATION_PREAMBLE.sub("", " ".join(text.split()), count=1)
    return _CONTINUATION_METADATA.sub("", rest, count=1)


# ---------------------------------------------------------------------------
# tool results and notifications
# ---------------------------------------------------------------------------


def block_text(block: dict[str, Any]) -> str:
    """Text of a stream-json tool_result block."""
    content = block.get("content")
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else ""


def agent_id_in(text: str) -> str | None:
    """The `agentId: …` line Claude Code puts in Task/Agent results."""
    m = _AGENT_ID_LINE.search(text)
    return m.group(1) if m else None


def async_launch_task_id(
    structured: Any, block: dict[str, Any], prompt: str | None
) -> str | None:
    """Task id ("" if unknown) if a Task/Agent tool_result is a background launch ack.

    Returns None if it is not one: the sub-agent has finished. Only
    `async_launched` agents keep running locally; `remote_launched` agents run
    in the cloud and never call the bridge, so their span closes at launch.
    Prefers the structured `tool_use_result` Claude Code attaches to the JSONL
    `user` line (checked against the prompt it echoes); falls back to the
    result text.
    """
    if (
        isinstance(structured, dict)
        and "status" in structured
        and structured.get("prompt") == prompt
    ):
        if structured["status"] != "async_launched":
            return None
        return str(structured.get("agentId") or "")
    content = block.get("content")
    if isinstance(content, list):
        content = next(
            (str(c.get("text", "")) for c in content if isinstance(c, dict)), ""
        )
    if isinstance(content, str) and content.lstrip().startswith(ASYNC_LAUNCHED):
        return agent_id_in(block_text(block)) or ""
    return None


def send_message_succeeded(structured: Any, block: dict[str, Any]) -> bool:
    """Whether a SendMessage result says the agent is being resumed."""
    if block.get("is_error"):
        return False
    if isinstance(structured, dict):
        return structured.get("success") is True or any(
            k in structured for k in ("resumedAgentId", "revivedAgentId")
        )
    text = block_text(block)
    return '"success":true' in text.replace(" ", "") or "Resuming agent" in text


def task_notifications(text: str) -> list[tuple[str | None, str | None]]:
    """(tool-use-id, task-id) of each `<task-notification>` in a message."""
    return [
        (_xml_field(note, "tool-use-id"), _xml_field(note, "task-id"))
        for note in _TASK_NOTIFICATION.findall(text)
    ]


def _xml_field(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else None
