"""Pin Claude Code's git status snapshot across ``--resume`` launches.

Claude Code renders a gitStatus section into its system prompt describing
"the git status at the start of the conversation". Because the agent
launches a fresh ``claude`` subprocess for every user turn / attempt /
crash retry (see claude_code.py), each ``--resume`` rebuilds the system
prompt and regenerates that section from the CURRENT repo state
(live-verified against Claude Code 2.1.215). Two problems:

- The system prompt heads every request, so the changed section
  invalidates the prompt cache for the entire conversation prefix
  (system prompt + full replayed history) on every resumed turn.
- The section still claims to be the status "at the start of the
  conversation" that "will not update during the conversation", while
  now reflecting the agent's own mid-episode edits and commits. Agents
  have been observed using the contradiction to recognize they are
  running inside an evaluation.

``pin_git_status_filter`` wraps the bridge filter to restore the
single-process semantics of an interactive Claude Code session: the
first bridged request records the root conversation's system prompt
shape and its git status section; a later request whose system prompt is
otherwise identical gets the recorded section spliced back in. Requests
with any other system prompt shape (Task-tool sub-agents carry their own,
much shorter system prompt with their own gitStatus section) pass through
untouched — as does everything if the section is absent or Claude Code
changes its format (fail open to today's behavior).
"""

import inspect
from hashlib import sha256
from logging import getLogger
from typing import Awaitable, Callable, cast

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
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

GIT_STATUS_SENTINEL = (
    "gitStatus: This is the git status at the start of the conversation."
)
"""Opening line of the system prompt section Claude Code regenerates on --resume."""


def split_git_status(text: str) -> tuple[str, str] | None:
    """Split system text into (prefix, git status section).

    The section runs from the sentinel to the end of the text — it is the
    final section of the system prompt in every observed layout. Returns
    None when the sentinel is absent.
    """
    index = text.find(GIT_STATUS_SENTINEL)
    if index == -1:
        return None
    return text[:index], text[index:]


def pin_git_status_filter(
    session_id: Callable[[], str], user_filter: GenerateFilter | None
) -> GenerateFilter:
    """Bridge filter that pins the git status section to its first-seen value.

    Rewrites qualifying requests (see module docstring) before delegating
    to `user_filter`, so the user's filter and the model both see the
    pinned messages. `session_id` is read per-request so the pin tracks a
    session id restored from a checkpoint.
    """
    user_is_legacy = user_filter is not None and _is_legacy_str_filter(user_filter)
    warned = False

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        nonlocal warned
        pinned: list[ChatMessage] | None = None
        try:
            pinned = _pin_messages(messages, session_id())
        except Exception as ex:
            # never let pinning break generation -- fall back to the
            # unpinned request (cache misses, but correct output)
            if not warned:
                warned = True
                logger.warning(f"error pinning claude code git status: {ex}")
        if pinned is not None:
            messages = pinned

        result: ModelOutput | GenerateInput | None = None
        if user_filter is not None:
            # replicate the bridge's legacy str-first dispatch, which our
            # wrapper otherwise hides (it sees only the Model-first wrapper)
            if user_is_legacy:
                result = await cast(_StrFilter, user_filter)(
                    model.name, messages, tools, tool_choice, config
                )
            else:
                result = await cast(_ModelFilter, user_filter)(
                    model, messages, tools, tool_choice, config
                )
        if result is None and pinned is not None:
            return GenerateInput(
                input=messages, tools=tools, tool_choice=tool_choice, config=config
            )
        return result

    return _filter


def _pin_messages(
    messages: list[ChatMessage], session_id: str
) -> list[ChatMessage] | None:
    """Return messages with the git status section pinned, or None to pass through."""
    # locate the (single) system message carrying the git status section,
    # accumulating the surrounding system prompt shape as we go
    target_index: int | None = None
    target_split: tuple[str, str] | None = None
    shape_parts: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ChatMessageSystem):
            continue
        split = split_git_status(message.text) if target_index is None else None
        if split is not None:
            target_index = index
            target_split = split
            shape_parts.append(split[0])
        else:
            shape_parts.append(message.text)
    if target_index is None or target_split is None:
        return None
    prefix, section = target_split

    # the shape hash identifies the conversation this request belongs to:
    # the full system prompt minus the volatile section. Sub-agent and
    # utility conversations have their own system prompts and never match
    # the root baseline.
    shape = sha256("\x1e".join(shape_parts).encode()).hexdigest()

    key = f"claude_code_pinned_git_status:{session_id}"
    stored = cast("dict[str, str] | None", store().get(key, None))
    if stored is None:
        # baseline: first bridged request of the session is the root
        # conversation's first request -- record and pass through
        store().set(key, {"shape": shape, "git_status": section})
        return None
    if stored.get("shape") != shape or stored.get("git_status") == section:
        return None

    pinned = list(messages)
    system_message = cast(ChatMessageSystem, messages[target_index])
    pinned[target_index] = system_message.model_copy(
        update={"content": prefix + stored["git_status"]}
    )
    return pinned


def _is_legacy_str_filter(fn: GenerateFilter) -> bool:
    """True when `fn`'s first parameter is annotated `str` (deprecated dispatch)."""
    first = next(iter(inspect.signature(fn).parameters.values()), None)
    return first is not None and first.annotation is str
