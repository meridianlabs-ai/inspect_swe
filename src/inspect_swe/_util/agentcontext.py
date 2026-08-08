"""Shared plumbing for populating inspect_ai's ambient AgentBridgeContext.

Each inspect_swe agent wraps its bridge filter with `classify_filter`, passing
a per-agent `classify` callable. The wrapper stamps the context (visible to
the user's filter and everything downstream in the request task) and then
delegates to the user's filter unchanged.
"""

import inspect
from logging import getLogger
from typing import Awaitable, Callable, cast

from inspect_ai.agent import AgentBridgeContext, set_agent_bridge_context
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    GenerateFilter,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo

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


def is_legacy_str_filter(fn: GenerateFilter) -> bool:
    first = next(iter(inspect.signature(fn).parameters.values()), None)
    return first is not None and first.annotation is str


def classify_filter(
    user_filter: GenerateFilter | None, classify: AgentContextClassifier
) -> ModelFilter:
    """Wrap a user-supplied bridge filter so it runs with the agent context set.

    Classifies the request with `classify` and stamps the resulting
    `AgentBridgeContext` via `set_agent_bridge_context` before delegating to
    `user_filter` (if any). The classification happens first so that the
    context is visible to `user_filter` and to anything downstream in the
    request task. Classification failures are logged and swallowed rather
    than propagated, since a broken classifier should never break generation.
    """
    user_is_legacy = user_filter is not None and is_legacy_str_filter(user_filter)

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
            logger.warning(f"agent context classification failed: {ex}")

        if user_filter is None:
            return None

        if user_is_legacy:
            return await cast(StrFilter, user_filter)(
                model.name, messages, tools, tool_choice, config
            )
        else:
            return await cast(ModelFilter, user_filter)(
                model, messages, tools, tool_choice, config
            )

    return _filter
