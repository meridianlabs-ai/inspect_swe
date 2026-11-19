"""Request shapes of DeepSeek Harness (dsh) the bridge classifies and places by.

Everything here reads text dsh's own runtime writes, at the position it writes
it (dsh 0.2.2rc2; `tests/test_deepseek_harness_drift.py` checks the strings
against upstream):

- **Session reminder.** dsh tells every agent who it is in a user-role
  `<system-reminder>` block of its own (`runtime-context.ts`, identity section
  from `tool-subagent-control`): `You are the root agent. Your name is "lead".`
  or `You are a subagent delegated by agent "lead".` / `Your name is "alpha".`
  A later reminder replaces earlier ones (after compaction, or in a forked
  child, whose history starts with its parent's), so the last one in a request
  names the agent that sent it.
- **Delegation.** `subagent` and `subagent_fork` start a child and return
  `started subagent <name>` at once; the child's answer comes back later as a
  user-role settlement notice whose first block is
  `Background subagent <name> finished ...` (`continuation-messages.ts`).
- **Compaction.** dsh's summarizer request is the span of conversation being
  condensed followed by one final user block opening with
  `COMPACTION_INSTRUCTION_PREAMBLE` (`compaction-basic/src/summarizer.ts`).
  The conversation then continues from a checkpoint message holding the
  summary, with a fresh session reminder appended.

- **Web search.** dsh's `web_search` tool makes a model request of its own:
  no system prompt and a single user message,
  `Perform a web search for the query: <query>`
  (`web-search-deepseek/src/provider.ts`).

Only user messages are read, and only whole text blocks: tool output quoting
any of this is a tool message, and text inside a larger block (the task, a
delegated prompt) never starts one.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass

from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
)
from inspect_ai.tool import ToolCall

ROOT_AGENT_NAME = "lead"
"""dsh's name for the root of a subagent tree (no child can take it)."""

DELEGATION_TOOLS = ("subagent", "subagent_fork")
"""dsh's delegation tools in the headless profile: a fresh child, and one
seeded with the caller's completed turns."""

SESSION_REMINDER_OPEN = (
    "<system-reminder>\nThe following information describes your current session"
)
SESSION_REMINDER_CLOSE = "</system-reminder>"

# names are JSON string literals (`JSON.stringify`)
_JSON_STRING = r'"(?:[^"\\]|\\.)*"'
_ROOT_IDENTITY_RE = re.compile(
    rf"^You are the root agent\. Your name is ({_JSON_STRING})\.$", re.MULTILINE
)
_CHILD_IDENTITY_RE = re.compile(
    rf"^You are a subagent delegated by agent ({_JSON_STRING})\.\n"
    rf"Your name is ({_JSON_STRING})\.$",
    re.MULTILINE,
)

_STARTED_RE = re.compile(r"started subagent (\S+)")

# every ending `settlementSummary` writes: a notice means the child's run is
# over however it ended
_SETTLED_RE = re.compile(
    r"Background subagent (\S+) (?:"
    r"finished and will do no further work unless you send it more\."
    r"|finished\. It cannot receive follow-up messages\."
    r"|was stopped before it finished\."
    r"|ran out of room before it finished\."
    r"|declined the task\."
    r"|failed before it finished\."
    r"|ended abnormally \(.*\) before it finished\."
    r")"
)

COMPACTION_INSTRUCTION_PREAMBLE = (
    "You are now acting as a compaction engine for this AI coding assistant."
)
COMPACTION_INSTRUCTION_MARKER = "Output EXACTLY the Markdown structure below"


WEB_SEARCH_REQUEST_PREFIX = "Perform a web search for the query: "


@dataclass(frozen=True)
class Identity:
    """Who dsh told the requesting agent it is."""

    name: str
    parent: str | None
    """Name of the agent that started it; `None` for the root."""

    @property
    def is_root(self) -> bool:
        return self.parent is None


@dataclass(frozen=True)
class Delegation:
    """A `subagent` / `subagent_fork` call."""

    call_id: str
    tool: str
    name: str | None
    """The call's `name` argument; dsh picks `agent-N` when it is omitted."""
    prompt: str
    description: str | None


def _text_blocks(message: ChatMessageUser) -> list[str]:
    if isinstance(message.content, str):
        return [message.content]
    return [item.text for item in message.content if isinstance(item, ContentText)]


def _identity_of(block: str) -> Identity | None:
    if not block.startswith(SESSION_REMINDER_OPEN):
        return None
    if not block.rstrip().endswith(SESSION_REMINDER_CLOSE):
        return None
    try:
        child = _CHILD_IDENTITY_RE.search(block)
        if child is not None:
            return Identity(
                name=str(json.loads(child.group(2))),
                parent=str(json.loads(child.group(1))),
            )
        root = _ROOT_IDENTITY_RE.search(block)
        if root is not None:
            return Identity(name=str(json.loads(root.group(1))), parent=None)
    except json.JSONDecodeError:
        return None
    return None


def requester_identity(messages: list[ChatMessage]) -> Identity | None:
    """The agent that sent a request, from its latest session reminder.

    `None` when no reminder with an identity is found (dsh then gave the agent
    none, or its wording changed).
    """
    for message in reversed(messages):
        if not isinstance(message, ChatMessageUser):
            continue
        for block in reversed(_text_blocks(message)):
            identity = _identity_of(block)
            if identity is not None:
                return identity
    return None


def initial_prompt(messages: list[ChatMessage]) -> str | None:
    """The task a child was started with: what opens the turn its reminder ends.

    dsh sends a new child its delegated prompt, the result-delivery guidance
    and its session reminder as one user turn (for a fork, after the inherited
    history).
    """
    for message in reversed(messages):
        if not isinstance(message, ChatMessageUser):
            continue
        blocks = _text_blocks(message)
        if any(_identity_of(block) is not None for block in blocks):
            return blocks[0] if _identity_of(blocks[0]) is None else None
    return None


def find_delegations(tool_calls: list[ToolCall] | None) -> list[Delegation]:
    """The delegation calls in an assistant message."""
    found: list[Delegation] = []
    for call in tool_calls or []:
        if call.function not in DELEGATION_TOOLS:
            continue
        arguments = call.arguments or {}
        name = arguments.get("name")
        prompt = arguments.get("prompt")
        description = arguments.get("description")
        found.append(
            Delegation(
                call_id=call.id,
                tool=call.function,
                name=name if isinstance(name, str) and name else None,
                prompt=prompt if isinstance(prompt, str) else "",
                description=(
                    description
                    if isinstance(description, str) and description
                    else None
                ),
            )
        )
    return found


def started_name(message: ChatMessageTool) -> str | None:
    """The child's name from a delegation call's result, `None` if it failed.

    dsh answers an accepted start with exactly `started subagent <name>`; a
    rejected one (duplicate name, depth or capacity limit) is an error result.
    """
    if message.error is not None:
        return None
    match = _STARTED_RE.fullmatch(message.text.strip())
    return match.group(1) if match is not None else None


def settled_children(messages: list[ChatMessage]) -> Counter[str]:
    """How many settlement notices a request carries, per child name.

    A notice is delivered to the child's direct parent and stays in its
    history, so only a count above the last one seen is a new settlement.
    """
    settled: Counter[str] = Counter()
    for message in messages:
        if not isinstance(message, ChatMessageUser):
            continue
        for block in _text_blocks(message):
            match = _SETTLED_RE.fullmatch(block.strip())
            if match is not None:
                settled[match.group(1)] += 1
    return settled


def is_compaction_request(messages: list[ChatMessage]) -> bool:
    """Whether a request is dsh's own compaction summarizer call.

    Only when the conversation's last block is the instruction: whitespace is
    collapsed so upstream re-wrapping the template doesn't break the match.
    """
    if not messages or not isinstance(messages[-1], ChatMessageUser):
        return False
    blocks = _text_blocks(messages[-1])
    if not blocks:
        return False
    text = " ".join(blocks[-1].split())
    return (
        text.startswith(COMPACTION_INSTRUCTION_PREAMBLE)
        and COMPACTION_INSTRUCTION_MARKER in text
    )


def is_web_search_request(messages: list[ChatMessage]) -> bool:
    """Whether a request is the model call behind dsh's `web_search` tool.

    The whole request is one user message; an agent's own requests always
    carry dsh's system prompt, so a task that starts with the same words
    isn't one.
    """
    if len(messages) != 1 or not isinstance(messages[0], ChatMessageUser):
        return False
    blocks = _text_blocks(messages[0])
    return len(blocks) == 1 and blocks[0].startswith(WEB_SEARCH_REQUEST_PREFIX)
