"""Tell a `GenerateFilter` whether the call it is inspecting belongs to a sub-agent.

WHY THIS EXISTS
---------------
A coding CLI that delegates makes its sub-agents' model calls through the same agent bridge as
the top-level agent's, so a caller-supplied ``GenerateFilter`` fires on all of them. Filter
state is per-EPISODE while the message lists are per-THREAD, which breaks an un-gated filter in
ways that are silent rather than loud:

* steering meant for the top-level agent is injected into a sub-agent's forked conversation;
* cadence logic compares turn counts across unrelated histories;
* an early stop returned as a ``ModelOutput`` ends that SUB-AGENT's turn, not the episode;
* an expensive model-backed check runs once per concurrent sub-agent instead of once.

Nothing in the filter's arguments distinguishes the threads, and spans do not help: the
consumers tag emitted events rather than entering a span around the call, so ``current_span_id()``
is identical for the top-level agent and every sub-agent, and ``agent_name`` is ``None``
throughout.

But the consumers ALREADY COMPUTE the answer. Each one attributes every bridge call to an agent
span (``_attribute``) in order to build the transcript's span tree, taking the same
``list[ChatMessage]`` a filter receives. This module exposes that existing verdict rather than
inventing a second mechanism: :func:`sub_agent_scope` wraps the caller's filter at the one place
that already pairs a filter with its consumer, so a filter can only ever see its own episode's
attribution, and :func:`is_sub_agent` reads it.

USAGE
-----
Call :func:`is_sub_agent` as the FIRST statement of a filter, before any state is read or
mutated::

    async def my_filter(model, messages, tools, tool_choice, config):
        if is_sub_agent():
            return None
        ...

WHAT IT DOES NOT COVER
----------------------
``is_sub_agent()`` returns ``False`` — the safe default, equivalent to no gating at all —
outside a bridged model call, for an agent that installs no event consumer (the ACP agents,
``antigravity``), and whenever attribution is uncertain. Two inherent uncertainties are worth
naming because they are properties of attribution itself, not of this wrapper: a sub-agent
prompt shorter than the matcher's minimum length is never matched, and two concurrent
sub-agents whose prompts overlap resolve to the outer span. Both under-detect, which restores
un-gated behaviour rather than stripping the real agent of its steering.

A SIDE CALL IS NOT A SUB-AGENT. Some CLIs make private bridged calls that are neither the
episode's conversation nor a delegated one (inspect_ai's own bridge notes that "claude code does
bash path detection with a side call"). Those attribute to the outer span, so ``is_sub_agent()``
correctly reports ``False`` and a filter still fires on them. Distinguishing a side call needs
knowledge of what the episode's own conversation is, which the caller has and this module does
not.
"""

import inspect
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Protocol, runtime_checkable

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    GenerateFilter,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo

_sub_agent: ContextVar[bool] = ContextVar("inspect_swe_sub_agent", default=False)


def is_sub_agent() -> bool:
    """Whether the model call currently being filtered belongs to a sub-agent.

    Meaningful only inside a `GenerateFilter` invoked by an inspect_swe agent's bridge; anywhere
    else it is ``False``. See the module docstring for what is and is not covered — in
    particular, ``False`` means "the top-level agent, OR we could not tell", never "definitely
    not a sub-agent".
    """
    return _sub_agent.get()


@runtime_checkable
class SubAgentAttribution(Protocol):
    """A model-event consumer that can say which thread a bridge call belongs to."""

    def is_sub_agent_call(self, input_messages: list[ChatMessage]) -> bool:
        """Whether these messages are a sub-agent's conversation rather than the outer one."""
        ...


@contextmanager
def sub_agent_scope(value: bool) -> Iterator[None]:
    """Answer :func:`is_sub_agent` with *value* for the duration of the block."""
    token = _sub_agent.set(value)
    try:
        yield
    finally:
        _sub_agent.reset(token)


def _takes_model_first(fn: object) -> bool:
    """Whether *fn* is a modern (``Model``-first) filter rather than a legacy ``str``-first one.

    Mirrors the test inspect_ai's bridge applies to the filter it is handed. We must apply it
    ourselves because the bridge will see OUR wrapper, not the caller's function, and would
    otherwise hand a legacy filter a ``Model``.
    """
    try:
        first = next(iter(inspect.signature(fn).parameters.values()), None)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return first is None or first.annotation is not str


def with_sub_agent_attribution(
    user_filter: GenerateFilter | None,
    consumer: object | None,
) -> GenerateFilter | None:
    """Wrap *user_filter* so :func:`is_sub_agent` answers for its own episode.

    Binds the filter to the consumer STRUCTURALLY, at the single call site that already
    constructs both, rather than through any ambient registry — so under concurrent samples a
    filter cannot observe another sample's attribution. Returns *user_filter* unchanged when
    there is nothing to bind (no filter, or a consumer that cannot attribute), so agents without
    an attributing consumer are unaffected.

    Attribution is purely observational: it reads the message list and the consumer's own
    already-open span registry, and mutates nothing.
    """
    if user_filter is None or not isinstance(consumer, SubAgentAttribution):
        return user_filter

    attributing = consumer
    inner_takes_model = _takes_model_first(user_filter)

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        try:
            sub_agent = attributing.is_sub_agent_call(messages)
        except Exception:
            # Fail SAFE and in one direction only. Over-reporting would strip the real agent of
            # its steering; under-reporting merely restores un-gated behaviour.
            sub_agent = False
        with sub_agent_scope(sub_agent):
            if inner_takes_model:
                model_filter: Callable[
                    [
                        Model,
                        list[ChatMessage],
                        list[ToolInfo],
                        ToolChoice | None,
                        GenerateConfig,
                    ],
                    Awaitable[ModelOutput | GenerateInput | None],
                ] = user_filter  # type: ignore[assignment]
                return await model_filter(model, messages, tools, tool_choice, config)
            str_filter: Callable[
                [
                    str,
                    list[ChatMessage],
                    list[ToolInfo],
                    ToolChoice | None,
                    GenerateConfig,
                ],
                Awaitable[ModelOutput | GenerateInput | None],
            ] = user_filter  # type: ignore[assignment]
            return await str_filter(model.name, messages, tools, tool_choice, config)

    return _filter
