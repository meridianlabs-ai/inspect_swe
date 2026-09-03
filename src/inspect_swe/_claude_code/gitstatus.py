"""Pin Claude Code's git status snapshot across ``--resume`` launches.

Claude Code renders a gitStatus section into its system prompt describing
"the git status at the start of the conversation". A single interactive
process renders it once, at startup, and shares that text with every
conversation it runs -- the root and any sub-agents -- for the life of the
process (live-verified against Claude Code 2.1.257: three commits made
mid-process changed nothing in any request's system prompt).

The agent instead launches a fresh ``claude`` subprocess for every user
turn / attempt / crash retry (see claude_code.py), and each ``--resume``
rebuilds the system prompt with the section regenerated from the CURRENT
repo state. Two problems:

- The system prompt heads every request, so the changed section
  invalidates the prompt cache for the entire conversation prefix
  (system prompt + full replayed history) on every resumed turn.
- The section still claims to be the status "at the start of the
  conversation" that "will not update during the conversation", while
  now reflecting the agent's own mid-episode edits and commits. Agents
  have been observed using the contradiction to recognize they are
  running inside an evaluation.

``pin_git_status_filter`` wraps the bridge filter to restore the
single-process semantics: the first bridged request that carries the
section records it in the sample store, keyed by session id (so it
survives a checkpoint restore); every later request whose section differs
gets the recorded text spliced back in, whatever system prompt surrounds
it (root or sub-agent, with or without ``--append-system-prompt`` text,
before or after a date rollover). Messages are rewritten in place rather
than via ``GenerateInput`` so the pinned text is what the bridge records
into ``bridge.state.messages`` (and so the eval log's ``sample.messages``),
not just what reaches the model and the ModelEvent.

Layout assumptions (verified for 2.1.257, on both the per-block system
messages inspect_ai >= 0.3.262 produces and the single flattened message
older versions produce): the section is the tail of the system block that
carries it, so the pin covers everything from the sentinel to the end of
that block's text; and root and sub-agent prompts render the same section,
so pinning one session-wide value is faithful (if a future Claude Code
rendered a sub-agent's status differently, e.g. for another cwd, the pin
would hand it the root's). Requests without the sentinel pass through
untouched, as does everything if pinning raises (fail open to today's
behavior, warned once per agent instance).
"""

import inspect
import warnings
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

    The section runs from the sentinel to the end of the text (see the
    layout assumption in the module docstring). Returns None when the
    sentinel is absent.
    """
    index = text.find(GIT_STATUS_SENTINEL)
    if index == -1:
        return None
    return text[:index], text[index:]


def pin_git_status_filter(
    session_id: Callable[[], str], user_filter: GenerateFilter | None
) -> GenerateFilter:
    """Bridge filter that pins the git status section to its first-seen value.

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
            stacklevel=3,  # attribute to the claude_code(filter=...) caller
        )
    warned = False

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        nonlocal warned
        try:
            _pin_messages(messages, session_id())
        except Exception as ex:
            # never let pinning break generation -- fall back to the
            # unpinned request (cache misses, but correct output)
            if not warned:
                warned = True
                logger.warning(f"error pinning claude code git status: {ex}")

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


def _pin_messages(messages: list[ChatMessage], session_id: str) -> None:
    """Record the session's git status section on first sight; rewrite it after.

    Only str-content system messages are considered (the bridge produces no
    other kind); rewriting a list-content message as a str would collapse it.
    """
    for message in messages:
        if not isinstance(message, ChatMessageSystem) or not isinstance(
            message.content, str
        ):
            continue
        split = split_git_status(message.content)
        if split is None:
            continue
        prefix, section = split

        key = f"claude_code_pinned_git_status:{session_id}"
        pinned = cast("str | None", store().get(key, None))
        if pinned is None:
            # first request of the session carrying the section: record it
            store().set(key, section)
        elif pinned != section:
            message.content = prefix + pinned
        return


def _is_legacy_str_filter(fn: GenerateFilter) -> bool:
    """True when `fn`'s first parameter is annotated `str` (deprecated dispatch)."""
    first = next(iter(inspect.signature(fn).parameters.values()), None)
    return first is not None and first.annotation is str
