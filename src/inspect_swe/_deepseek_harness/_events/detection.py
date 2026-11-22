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
  `Background subagent <name> finished ...`, followed by
  `Its closing message:` and the child's final text blocks
  (`continuation-messages.ts`). `send_message` gives a child more work; the
  recipient sees `Agent <sender> sent a message: ` and the message blocks.
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
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
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

SEND_MESSAGE_TOOL = "send_message"

# every ending `settlementSummary` writes: a notice means the child's run is
# over however it ended. The first two are a run that completed.
_SETTLED_RE = re.compile(
    r"Background subagent (\S+) (?:"
    r"(?P<completed>finished and will do no further work unless you send it more\."
    r"|finished\. It cannot receive follow-up messages\.)"
    r"|was stopped before it finished\."
    r"|ran out of room before it finished\."
    r"|declined the task\."
    r"|failed before it finished\."
    r"|ended abnormally \(.*\) before it finished\."
    r")"
)

_CLOSING_MESSAGE = "Its closing message:"
_CLOSING_OMITTED_PREFIX = "It sent you a message just before finishing"
_AGENT_MESSAGE_RE = re.compile(r"Agent (\S+) sent a message:")

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
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Settlement:
    """A settlement notice: one run of a child, over."""

    name: str
    """The child's name in dsh's subagent tree."""
    result: str
    """What the run produced, as its parent was told: the child's closing
    message (or, when dsh left that out because the child had just messaged
    its parent, that message). Never empty: a run that left nothing, or
    didn't complete, is described by the notice's own sentences."""


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
                arguments=dict(arguments),
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


def _starts_harness_text(block: str) -> bool:
    """Whether a block opens text of dsh's own, ending whatever came before it."""
    stripped = block.strip()
    return (
        _SETTLED_RE.fullmatch(stripped) is not None
        or _AGENT_MESSAGE_RE.fullmatch(stripped) is not None
        or block.startswith("<system-reminder>")
        or block.startswith(COMPACTION_INSTRUCTION_PREAMBLE)
    )


def settlements(messages: list[ChatMessage]) -> list[Settlement]:
    """The settlement notices a request carries, in order.

    A notice is delivered to the child's direct parent and stays in its
    history until compaction cuts it, so the same notice turns up in request
    after request (the consumer takes each once). dsh merges adjacent user-role messages, so a notice's blocks
    run up to the next block of dsh's own (another notice, a message from an
    agent, a session reminder, the compaction instruction) or the end of the
    user message.
    """
    found: list[Settlement] = []
    # sender → its latest message to the requester, for a notice that omits
    # the closing message because the child had just sent one
    last_message: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, ChatMessageUser):
            continue
        blocks = _text_blocks(message)
        index = 0
        while index < len(blocks):
            header = blocks[index].strip()
            index += 1
            settled = _SETTLED_RE.fullmatch(header)
            sent = _AGENT_MESSAGE_RE.fullmatch(header)
            if settled is None and sent is None:
                continue
            body: list[str] = []
            while index < len(blocks) and not _starts_harness_text(blocks[index]):
                body.append(blocks[index])
                index += 1
            if sent is not None:
                last_message[sent.group(1)] = "\n\n".join(body)
                continue
            assert settled is not None
            name = settled.group(1)
            if body[:1] == [_CLOSING_MESSAGE]:
                body = body[1:]
            elif body and body[0].startswith(_CLOSING_OMITTED_PREFIX):
                body = [last_message.get(name) or body[0], *body[1:]]
            result = "\n\n".join(body)
            if settled.group("completed") is None:
                # stopped, failed, declined, out of room: say so first
                result = f"{header}\n\n{result}" if result else header
            found.append(Settlement(name=name, result=result or header))
    return found


@dataclass(frozen=True)
class Restart:
    """A child's request that is the first of a new run."""

    closing: str
    """The final answer its previous run ended with."""
    key: tuple[int, str]
    """Where the new run begins in the child's history, so a repeat of the
    same request isn't taken for another restart."""


def restarted(messages: list[ChatMessage], parent: str | None) -> Restart | None:
    """Whether a child's request begins a new run, after its last one ended.

    A run ends with an answer that calls no tools. dsh restarts a finished
    child when it is sent a message, which arrives as a user turn opening
    with `Agent <sender> sent a message:` straight after that answer. Only a
    message from the child's parent counts here: an agent waiting on children
    of its own can also write an answer without tool calls and then hear from
    one of them, with its run still going.
    """
    if parent is None:
        return None
    for index in range(len(messages) - 1, -1, -1):
        answer = messages[index]
        if isinstance(answer, ChatMessageAssistant):
            break
    else:
        return None
    if answer.tool_calls:
        return None
    restart = False
    for message in messages[index + 1 :]:
        if not isinstance(message, ChatMessageUser):
            continue
        for block in _text_blocks(message):
            if _SETTLED_RE.fullmatch(block.strip()) is not None:
                # one of its own children finished: it was waiting on them
                return None
            sent = _AGENT_MESSAGE_RE.fullmatch(block.strip())
            restart = restart or (sent is not None and sent.group(1) == parent)
    return Restart(closing=answer.text, key=(index, answer.text)) if restart else None


def settled_children(messages: list[ChatMessage]) -> Counter[str]:
    """How many settlement notices a request carries, per child name."""
    return Counter(settlement.name for settlement in settlements(messages))


def find_messages(tool_calls: list[ToolCall] | None) -> list[ToolCall]:
    """The `send_message` calls in an assistant message that name a recipient."""
    return [
        call
        for call in tool_calls or []
        if call.function == SEND_MESSAGE_TOOL
        and isinstance((call.arguments or {}).get("agent"), str)
    ]


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
