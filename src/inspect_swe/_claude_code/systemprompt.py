"""Pin Claude Code's system prompt across ``--resume`` launches.

The agent launches a fresh ``claude`` subprocess for every user turn /
attempt / crash retry (see claude_code.py), continuing the conversation
with ``--resume``. Each launch rebuilds the system prompt from the current
environment while replaying the conversation itself byte-identically
(live-verified against Claude Code 2.1.257). Observed differences between
a launch and its resume:

- the gitStatus section ("the git status at the start of the
  conversation ... will not update during the conversation") is
  regenerated from the CURRENT repo state, so the agent sees its own
  mid-episode edits presented as the starting state -- agents have used
  the contradiction to recognize they are running inside an evaluation;
- ``--append-system-prompt`` text is only passed on the first launch, so
  it disappears on resume and the identity line flips.

The system prompt heads every request and carries a cache_control
breakpoint, so any changed byte invalidates the prompt cache for the whole
conversation prefix on every resumed turn.

``pin_system_prompt_filter`` wraps the bridge filter to restore the
semantics of a single interactive process: the session's first bridged
request records its system prompt (the leading system messages) together
with its first user message; a later request whose first user message has identical
content belongs to the same conversation and gets the recorded system prompt
written back in place. Conversations with a different first user message
-- Task-tool sub-agents, utility calls -- carry their own system prompts
and pass through untouched. Nothing in Claude Code's system prompt is
parsed; the first user message (system reminders + task prompt) is the
only anchor, and Claude Code replays it verbatim from its session file.

Recorded state lives in the sample store keyed by session id, so it
survives a checkpoint restore. Messages are rewritten in place rather than
via ``GenerateInput`` so the pinned text is what the bridge records into
``bridge.state.messages`` (and so the eval log), not just what reaches the
model. Anything unexpected -- no leading system messages, no user message
following them, a different system message count, non-text system content,
or an exception -- fails open to today's behavior (each distinct problem
warned once).
"""

import inspect
import warnings
from logging import getLogger
from typing import Awaitable, Callable, cast

from inspect_ai._util.logger import warn_once
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    GenerateFilter,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import store

logger = getLogger(__name__)

# GenerateFilter is a union of Model-first and (deprecated) str-first
# callables; the bridge dispatches on the user filter's first-parameter
# annotation. Our wrapper hides the user filter from that dispatch, so it
# replicates it: Model-first filters get the Model, str-first get model.name.
_ModelFilter = Callable[
    [Model, list[ChatMessage], list[ToolInfo], "ToolChoice | None", GenerateConfig],
    Awaitable["ModelOutput | GenerateInput | None"],
]
_StrFilter = Callable[
    [str, list[ChatMessage], list[ToolInfo], "ToolChoice | None", GenerateConfig],
    Awaitable["ModelOutput | GenerateInput | None"],
]


def pin_system_prompt_filter(
    session_id: Callable[[], str], user_filter: GenerateFilter | None
) -> GenerateFilter:
    """Bridge filter that pins the root conversation's system prompt.

    Rewrites qualifying requests in place (see module docstring) before
    delegating to `user_filter`, so the user's filter and the model both see
    the pinned messages. `session_id` is read per-request so the pin tracks
    a session id restored from a checkpoint.
    """
    user_is_legacy = user_filter is not None and _is_legacy_str_filter(user_filter)
    if user_is_legacy:
        # the bridge emits this for str-first filters it dispatches; our
        # wrapper is all it sees, so replicate the migration signal
        warnings.warn(
            "GenerateFilter with 'str' as the first parameter is deprecated. "
            "Update your filter to accept a 'Model' instance instead.",
            DeprecationWarning,
            # attribute to the claude_code(filter=...) caller: this function,
            # the claude_code() body, and inspect's @agent wrapper sit between
            stacklevel=4,
        )

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        try:
            problem = _pin_messages(messages, session_id())
        except Exception as ex:
            problem = f"error pinning claude code system prompt: {ex}"
        if problem is not None:
            # never let pinning break generation -- fall back to the
            # unpinned request (cache misses, but correct output), and say
            # so once so an inert pin is visible in the log
            warn_once(logger, problem)

        if user_filter is None:
            return None
        if user_is_legacy:
            return await cast(_StrFilter, user_filter)(
                model.name, messages, tools, tool_choice, config
            )
        return await cast(_ModelFilter, user_filter)(
            model, messages, tools, tool_choice, config
        )

    return _filter


def _pin_messages(messages: list[ChatMessage], session_id: str) -> str | None:
    """Record the session's system prompt on first sight; write it back after.

    Returns a warning when the request is recognizably the pinned
    conversation but cannot be pinned (the pin has gone inert), else None.
    """
    # the system prompt is the leading run of system messages (the request's
    # `system` blocks, hoisted by the bridge). Claude Code also injects
    # role="system" messages INTO the conversation -- a skills reminder after
    # the first prompt, a per-turn "<total_tokens>" budget marker -- which the
    # bridge maps to ChatMessageSystem in place; those are replayed history
    # and are left alone.
    system: list[ChatMessageSystem] = []
    for message in messages:
        if not isinstance(message, ChatMessageSystem):
            break
        system.append(message)
    anchor = messages[len(system)] if len(system) < len(messages) else None
    if not system or not isinstance(anchor, ChatMessageUser):
        return None
    if not all(isinstance(m.content, str) for m in system):
        # the bridge only produces str-content system messages; rewriting a
        # list-content message as a str would collapse it
        return None
    texts = [cast(str, m.content) for m in system]
    # compare the anchor's full content (not `.text`, which drops non-text
    # parts and joins text parts) so distinct first messages never collide
    anchor_key = anchor.model_dump_json(include={"content"})

    key = f"claude_code_pinned_system_prompt:{session_id}"
    stored = store().get(key, None)
    if stored is None:
        store().set(key, {"anchor": anchor_key, "system": texts})
        return None
    if not isinstance(stored, dict) or stored.get("anchor") != anchor_key:
        return None
    pinned = stored.get("system")
    if (
        not isinstance(pinned, list)
        or len(pinned) != len(system)
        or not all(isinstance(t, str) for t in pinned)
    ):
        return (
            f"claude code system prompt not pinned for session {session_id}: "
            "the conversation matches but its system prompt layout changed "
            f"({len(system)} leading system messages vs "
            f"{len(pinned) if isinstance(pinned, list) else '?'} recorded)"
        )
    for message, text in zip(system, pinned, strict=True):
        if message.content != text:
            message.content = text
    return None


def _is_legacy_str_filter(fn: GenerateFilter) -> bool:
    """True when `fn`'s first parameter is annotated `str` (deprecated dispatch)."""
    first = next(iter(inspect.signature(fn).parameters.values()), None)
    return first is not None and first.annotation is str
