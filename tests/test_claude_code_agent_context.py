"""Tests for LiveConsumer.classify() (claude_code's filter-time agent context).

Covers the truth table documented on `LiveConsumer.classify` -- structural
slug checks (subagent / small-fast) ahead of the inferred pending-prompt
substring match, with the presented-slug and no-signal fallbacks -- and
depends on the slug-multiplexing invariant `resolve_claude_code_models`
enforces (`models.subagent != models.presented` always; see
tests/test_claude_code_model.py for that invariant's own coverage).
"""

from typing import Any

from inspect_ai.agent import AgentBridgeContext
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_swe._claude_code._events import live_consumer as live_consumer_module
from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._claude_code.model import resolve_claude_code_models

# Long enough to clear _MIN_PROMPT_LENGTH's substring-match guard.
_TASK_PROMPT = "Investigate the failing integration test and report the root cause."


class _TranscriptStub:
    """Swallows the SpanBeginEvent/SpanEndEvent traffic on_complete emits."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)

    def _event_updated(self, event: Any) -> None:
        pass


def _consumer(monkeypatch: Any, **model_kwargs: Any) -> LiveConsumer:
    monkeypatch.setattr(live_consumer_module, "transcript", lambda: _TranscriptStub())
    models = resolve_claude_code_models("mockllm/model", None, **model_kwargs)
    return LiveConsumer(models)


def _task_call_event(tool_call_id: str, prompt: str) -> ModelEvent:
    """A completed ModelEvent whose output spawns a Task sub-agent."""
    message = ChatMessageAssistant(
        content="ok",
        tool_calls=[
            ToolCall(id=tool_call_id, function="Task", arguments={"prompt": prompt})
        ],
    )
    return ModelEvent(
        model="m",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput.from_message(message),
    )


def _spawn_pending(consumer: LiveConsumer, tool_call_id: str, prompt: str) -> None:
    """Register a pending sub-agent via the real on_complete flow.

    Not a direct state poke -- mirrors how the bridge actually drives this.
    """
    consumer.on_complete(_task_call_event(tool_call_id, prompt))


def _classify(consumer: LiveConsumer, text: str = "hi") -> AgentBridgeContext:
    return consumer.classify(
        get_model("mockllm/model"), [ChatMessageUser(content=text)], []
    )


# ---------------------------------------------------------------------------
# truth table
# ---------------------------------------------------------------------------


def test_classify_subagent_slug_is_subagent(monkeypatch: Any) -> None:
    consumer = _consumer(monkeypatch)
    with bridged_request_scope(consumer._models.subagent):
        result = _classify(consumer)
    assert result == AgentBridgeContext("subagent")


def test_classify_small_fast_slug_distinct_from_presented_is_utility(
    monkeypatch: Any,
) -> None:
    consumer = _consumer(monkeypatch, haiku_model="mockllm/haiku")
    assert consumer._models.haiku != consumer._models.presented
    with bridged_request_scope(consumer._models.haiku):
        result = _classify(consumer)
    assert result == AgentBridgeContext("utility")


def test_classify_small_fast_slug_equal_presented_is_not_utility(
    monkeypatch: Any,
) -> None:
    """Default config: haiku_model unset -> haiku inherits presented verbatim.

    That slug is indistinguishable from main-thread traffic, so it must fall
    through to the remaining checks rather than register as utility -- here
    that lands on "root" (no pending sub-agents, slug == presented).
    """
    consumer = _consumer(monkeypatch)
    assert consumer._models.haiku == consumer._models.presented
    with bridged_request_scope(consumer._models.haiku):
        result = _classify(consumer)
    assert result != AgentBridgeContext("utility")
    assert result == AgentBridgeContext("root")


def test_classify_presented_slug_no_pending_is_root(monkeypatch: Any) -> None:
    consumer = _consumer(monkeypatch)
    with bridged_request_scope(consumer._models.presented):
        result = _classify(consumer)
    assert result == AgentBridgeContext("root")


def test_classify_presented_slug_with_pending_is_still_root(monkeypatch: Any) -> None:
    """A presented-slug call is main-thread even while sub-agents are open.

    Structural per P1: Claude Code never sends the presented slug for
    subagent traffic -- so this is not the ambiguous "unknown" case.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    with bridged_request_scope(consumer._models.presented):
        result = _classify(consumer, text="a completely unrelated main-thread turn")
    assert result == AgentBridgeContext("root")


def test_classify_unrecognized_slug_with_pending_is_unknown(monkeypatch: Any) -> None:
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    with bridged_request_scope("some-unrecognized-slug"):
        result = _classify(consumer, text="a completely unrelated main-thread turn")
    assert result == AgentBridgeContext("unknown")


def test_classify_unrecognized_slug_no_pending_is_root(monkeypatch: Any) -> None:
    consumer = _consumer(monkeypatch)
    with bridged_request_scope("some-unrecognized-slug"):
        result = _classify(consumer)
    assert result == AgentBridgeContext("root")


def test_classify_no_request_info_no_pending_is_root(monkeypatch: Any) -> None:
    """No bridged_request_scope active -> current_bridge_request() is None."""
    consumer = _consumer(monkeypatch)
    assert _classify(consumer) == AgentBridgeContext("root")


def test_classify_prompt_match_is_subagent_even_with_unrecognized_slug(
    monkeypatch: Any,
) -> None:
    """A call carrying a pending Task prompt is a sub-agent call regardless of slug.

    Covers slug-bypass drift: the inferred substring-match signal fires even
    when the slug matches nothing structural.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    with bridged_request_scope("some-unrecognized-slug"):
        result = _classify(consumer, text=f"<task>\n{_TASK_PROMPT}\n</task>")
    assert result == AgentBridgeContext("subagent")


def test_classify_prompt_match_is_subagent_with_no_request_info(
    monkeypatch: Any,
) -> None:
    """Same as above, but with no bridge request scope at all (slug=None).

    The prompt-match signal doesn't depend on slug being present.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    result = _classify(consumer, text=f"<task>\n{_TASK_PROMPT}\n</task>")
    assert result == AgentBridgeContext("subagent")


def test_classify_agrees_with_span_attribution(monkeypatch: Any) -> None:
    """Shared-resolver invariant: classify() agrees with _attribute()'s span.

    classify() reports "subagent" exactly when _attribute() resolved a
    sub-agent span (rather than the outer span) -- both are backed by the
    same `_match_pending_prompt`.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    messages: list[ChatMessage] = [
        ChatMessageUser(content=f"<task>\n{_TASK_PROMPT}\n</task>")
    ]

    span_id = consumer._attribute(messages)
    with bridged_request_scope("some-unrecognized-slug"):
        result = consumer.classify(get_model("mockllm/model"), messages, [])

    assert span_id != consumer.outer_span_id
    assert result == AgentBridgeContext("subagent")
