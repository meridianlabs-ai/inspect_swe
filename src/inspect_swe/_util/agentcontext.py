"""Shared plumbing for populating inspect_ai's ambient AgentBridgeContext.

Each inspect_swe agent wraps its bridge filter with `classify_filter`, passing
a per-agent `classify` callable. The wrapper stamps the context (visible to
the user's filter and everything downstream in the request task) and then
delegates to the user's filter unchanged.
"""

import inspect
import warnings
from logging import getLogger
from typing import Awaitable, Callable, Literal, Mapping, Set, cast

from inspect_ai.agent import (
    AgentBridgeContext,
    current_bridge_request,
    set_agent_bridge_context,
)
from inspect_ai.event import ModelEvent
from inspect_ai.log._transcript import transcript
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    GenerateFilter,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.model._model import ModelEventSink, use_model_event_sink
from inspect_ai.tool import ToolChoice, ToolInfo

from .model_event_sink import filter_generation, preserve_decision

logger = getLogger(__name__)

# GenerateFilter is a union of Model-first and (deprecated) str-first callables;
# the bridge dispatches on the user filter's first-parameter annotation. Our
# classify_filter wrapper hides the user filter from that dispatch, so it
# replicates it: Model-first filters get the Model, str-first get model.name.
ModelFilter = Callable[
    [Model, list[ChatMessage], list[ToolInfo], "ToolChoice | None", GenerateConfig],
    Awaitable["ModelOutput | GenerateInput | None"],
]
StrFilter = Callable[
    [str, list[ChatMessage], list[ToolInfo], "ToolChoice | None", GenerateConfig],
    Awaitable["ModelOutput | GenerateInput | None"],
]

AgentContextClassifier = Callable[
    [Model, list[ChatMessage], list[ToolInfo]], AgentBridgeContext
]
"""Per-agent callable that attributes a bridged model request to an agent context.

Called with the resolved `Model`, the (agent-specific, possibly already
transformed) message history, and the available tools; returns the
`AgentBridgeContext` that should be in effect for the remainder of the
request.
"""


def static_root_classifier(
    model: Model, messages: list[ChatMessage], tools: list[ToolInfo]
) -> AgentBridgeContext:
    """Classifier for agents with no delegation capability.

    Every bridged request belongs to the agent's own (root) thread.
    """
    return AgentBridgeContext("root")


def slug_map_classifier(
    root_slugs: Set[str],
    kind_by_slug: Mapping[str, Literal["subagent", "utility"]],
) -> AgentContextClassifier:
    """Classifier keyed on the requested model slug (pre-alias-resolution).

    For agents with no live consumer to drive attribution (ACP adapters
    with no JSONL/event stream to parse, or delegation-free variants that
    only need to pick out a handful of known utility/subagent slugs), this
    is the simplest thing that can work: read the raw slug the inner agent
    requested the model under from `current_bridge_request()` — the same
    structural signal `LiveConsumer.classify` and `CodexConsumer.classify`
    check first, minus the stateful signals (pending sub-agents, prompt
    substring matching) a consumer layers on top.

    `slug in root_slugs` -> `"root"`; `slug in kind_by_slug` -> the mapped
    kind; no bridged request info, or a slug recognized by neither map ->
    `"unknown"`.

    A slug present in both maps is the root slug on the wire, so it
    classifies `"root"`: its `kind_by_slug` entry is dropped here (with a
    warning, once, at construction) rather than left to shadow silently.
    Callers that can present distinct slugs should do so upstream (see
    `resolve_claude_code_models`); this guard is the backstop for the ones
    that can't.
    """
    colliding = sorted(slug for slug in kind_by_slug if slug in root_slugs)
    if colliding:
        logger.warning(
            f"agent context classifier: slug(s) {colliding} mapped to a kind "
            f"are also root slugs; classifying them as root"
        )
    kind_by_slug = {
        slug: kind for slug, kind in kind_by_slug.items() if slug not in root_slugs
    }

    def _classify(
        model: Model, messages: list[ChatMessage], tools: list[ToolInfo]
    ) -> AgentBridgeContext:
        request = current_bridge_request()
        slug = request.model if request is not None else None
        if slug in root_slugs:
            return AgentBridgeContext("root")
        if slug is not None and slug in kind_by_slug:
            return AgentBridgeContext(kind_by_slug[slug])
        return AgentBridgeContext("unknown")

    return _classify


class _FilterGenerationSink:
    """Route a user filter's own generation to the bridge's sink.

    The bridge installs its `ModelEventSink` only around its own generation,
    after the filter, so a filter that generates (and returns that output)
    would record its `ModelEvent`, and give a live sentinel its conversation,
    in the outer span rather than the agent the filter was told about.
    Installed around the user filter, this forwards that generation to the
    sink, which places it like the bridge's own (`on_start`, from the
    filter-time decision), and marks its `on_complete` as a filter's
    generation (`filter_generation`): only an output the filter returns
    reaches the agent, and `on_filter_output` tracks delegation in that one.

    A nested bridged request the filter makes is its own request: one that
    installs its own sink is unaffected, and any other generation under
    another request is recorded in the current span as without a sink.
    """

    def __init__(self, inner: ModelEventSink, request: object) -> None:
        self.inner = inner
        self.request = request

    def _ours(self) -> bool:
        return current_bridge_request() is self.request

    def on_start(self, input: list[ChatMessage]) -> str | None:
        on_start = getattr(self.inner, "on_start", None)
        if not self._ours() or on_start is None:
            return None
        return cast(str | None, on_start(input))

    def on_pending(self, event: ModelEvent) -> None:
        if self._ours():
            self.inner.on_pending(event)
        else:
            transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        if self._ours():
            with filter_generation(self.inner):
                self.inner.on_complete(event)
        else:
            transcript()._event_updated(event)


def is_legacy_str_filter(fn: GenerateFilter) -> bool:
    """True when `fn`'s first parameter is annotated `str` (legacy dispatch).

    Inherited limitation (mirrors inspect_ai's own dispatch intentionally):
    modules using `from __future__ import annotations` stringize all
    annotations, so a legacy filter's `str` annotation reads back as the
    string `"str"` rather than the `str` type and is classified Model-first.
    """
    first = next(iter(inspect.signature(fn).parameters.values()), None)
    return first is not None and first.annotation is str


def classify_filter(
    user_filter: GenerateFilter | None,
    classify: AgentContextClassifier,
    on_filter_output: Callable[[list[ChatMessage], ModelOutput], None] | None = None,
    sink: ModelEventSink | None = None,
) -> ModelFilter:
    """Wrap a user-supplied bridge filter so it runs with the agent context set.

    Classifies the request with `classify` and stamps the resulting
    `AgentBridgeContext` via `set_agent_bridge_context` before delegating to
    `user_filter` (if any). The classification happens first so that the
    context is visible to `user_filter` and to anything downstream in the
    request task. If `classify` raises, the failure is logged (once per
    distinct error, to avoid a broken classifier warning on every request)
    and swallowed rather than propagated: the request proceeds under
    whatever `AgentBridgeContext` the bridge's ambient scope already has in
    place (`"unknown"`, absent any earlier classification) rather than
    breaking generation.

    When `user_filter` returns a `ModelOutput`, the bridge skips its own
    generation, so a `ModelEventSink` never sees that output.
    `on_filter_output` (if given) is called with the request's messages and
    the output before it is returned, for agents whose attribution tracks
    state from model outputs. Its failures are logged and swallowed the same
    way.

    `sink` is the bridge's `ModelEventSink`, if it has one: a generation the
    user filter runs itself is routed to it (`_FilterGenerationSink`), so its
    `ModelEvent` is placed in the span the filter's context names, as the
    bridge's own generation would be.
    """
    user_is_legacy = user_filter is not None and is_legacy_str_filter(user_filter)
    if user_is_legacy:
        # The bridge only ever sees our Model-first wrapper, so re-emit the
        # deprecation it would otherwise have raised for the user's filter.
        warnings.warn(
            "GenerateFilter with 'str' as the first parameter is deprecated. "
            "Update your filter to accept a 'Model' instance instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    warned: set[str] = set()

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        try:
            set_agent_bridge_context(classify(model, messages, tools))
        except Exception as ex:
            # key on exception type only: messages may embed per-request data
            # (ids, paths), which would defeat the dedupe and grow the set
            key = type(ex).__name__
            if key not in warned:
                warned.add(key)
                logger.warning(f"agent context classification failed: {ex}")

        if user_filter is None:
            return None

        # a user filter may run a nested bridged request in this task, whose
        # classifier records its own placement: keep this request's
        request = current_bridge_request()
        filter_sink = (
            _FilterGenerationSink(sink, request)
            if sink is not None and request is not None
            else None
        )
        with preserve_decision(), use_model_event_sink(filter_sink):
            if user_is_legacy:
                result = await cast(StrFilter, user_filter)(
                    model.name, messages, tools, tool_choice, config
                )
            else:
                result = await cast(ModelFilter, user_filter)(
                    model, messages, tools, tool_choice, config
                )
        if on_filter_output is not None and isinstance(result, ModelOutput):
            try:
                on_filter_output(messages, result)
            except Exception as ex:
                key = f"on_filter_output:{type(ex).__name__}"
                if key not in warned:
                    warned.add(key)
                    logger.warning(f"agent context output tracking failed: {ex}")
        return result

    return _filter
