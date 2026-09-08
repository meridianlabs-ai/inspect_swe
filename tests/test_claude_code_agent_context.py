"""Tests for LiveConsumer.classify() (claude_code's filter-time agent context).

Covers the truth table documented on `LiveConsumer.classify` -- structural
slug checks (subagent / small-fast) ahead of the inferred pending-prompt
substring match, with the presented-slug and no-signal fallbacks -- and
depends on the slug-multiplexing invariant `resolve_claude_code_models`
enforces (`models.subagent` distinct from all four role names -- presented,
opus, sonnet, and haiku -- always; see tests/test_claude_code_model.py for
that invariant's own coverage).
"""

from typing import Any

import pytest
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


def test_classify_opus_tier_slug_with_pending_is_root(monkeypatch: Any) -> None:
    """A distinct opus/sonnet tier slug is main-thread traffic, not "unknown".

    env.py exports `models.opus`/`models.sonnet` as
    ANTHROPIC_DEFAULT_OPUS_MODEL/ANTHROPIC_DEFAULT_SONNET_MODEL, so Claude
    Code's own tier swap legitimately sends main-thread requests under those
    slugs. With a sub-agent pending they must still classify as "root".
    """
    consumer = _consumer(monkeypatch, opus_model="mockllm/opus")
    assert consumer._models.opus != consumer._models.presented
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    with bridged_request_scope(consumer._models.opus):
        result = _classify(consumer, text="a completely unrelated main-thread turn")
    assert result == AgentBridgeContext("root")


def test_classify_sonnet_tier_slug_with_pending_is_root(monkeypatch: Any) -> None:
    consumer = _consumer(monkeypatch, sonnet_model="mockllm/sonnet")
    assert consumer._models.sonnet != consumer._models.presented
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    with bridged_request_scope(consumer._models.sonnet):
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


# ---------------------------------------------------------------------------
# subagent-slug drift canary
# ---------------------------------------------------------------------------
#
# The synthetic subagent slug shape ("<presented>-subagent") was never
# live-verified against a real CC build the way P1's real catalog-name slugs
# were. If some CC version rejects/ignores it, subagent requests would
# silently stop carrying it and attribution would quietly degrade to
# prompt-match-only. The canary counts exactly that evidence: a request
# `classify` attributes to a sub-agent via the pending-prompt match (its step
# 4) while carrying anything other than the subagent slug. These tests drive
# `_check_subagent_slug_drift` (called from `classify`) through its public
# entry point.

_DRIFT_LOGGER = "inspect_swe._claude_code._events.live_consumer"


def _warnings(caplog: Any) -> list[Any]:
    return [r for r in caplog.records if r.levelname == "WARNING"]


def _classify_as_subagent_by_prompt(
    consumer: LiveConsumer, slug: str | None
) -> AgentBridgeContext:
    """A request only the pending-prompt match can attribute to a sub-agent."""
    text = f"<task>\n{_TASK_PROMPT}\n</task>"
    if slug is None:
        return _classify(consumer, text=text)
    with bridged_request_scope(slug):
        return _classify(consumer, text=text)


def test_subagent_slug_drift_ignores_main_thread_requests(
    monkeypatch: Any, caplog: Any
) -> None:
    """Main-thread traffic while a Task is pending is not evidence of drift.

    A Task launched in the background (or several in parallel) lets the main
    thread keep issuing requests before the sub-agent's first call arrives,
    and a refused main-thread request is re-run through the filter once per
    retry attempt. None of those say anything about the subagent slug.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)

    with caplog.at_level("WARNING", logger=_DRIFT_LOGGER):
        with bridged_request_scope(consumer._models.presented):
            for _ in range(10):
                assert _classify(consumer) == AgentBridgeContext("root")

    assert _warnings(caplog) == []


def test_subagent_slug_drift_ignores_unmatched_unknown_slug(
    monkeypatch: Any, caplog: Any
) -> None:
    """An unrecognized slug that matches no pending prompt is "unknown", not drift."""
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)

    with caplog.at_level("WARNING", logger=_DRIFT_LOGGER):
        with bridged_request_scope("some-unrecognized-slug"):
            for _ in range(10):
                assert _classify(consumer) == AgentBridgeContext("unknown")

    assert _warnings(caplog) == []


@pytest.mark.parametrize("slug", ["some-unrecognized-slug", None])
def test_subagent_slug_drift_warns_once(
    monkeypatch: Any, caplog: Any, slug: str | None
) -> None:
    """Prompt-matched sub-agent requests without the subagent slug warn exactly once.

    Well past the threshold -- the warning must not repeat.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)

    with caplog.at_level("WARNING", logger=_DRIFT_LOGGER):
        for _ in range(10):
            result = _classify_as_subagent_by_prompt(consumer, slug)
            assert result == AgentBridgeContext("subagent")

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert consumer._models.subagent in warnings[0].message


def test_subagent_slug_drift_tolerates_one_missing_slug(
    monkeypatch: Any, caplog: Any
) -> None:
    """A single slug-less sub-agent request is the documented env-propagation race.

    Exactly `_SUBAGENT_SLUG_DRIFT_WARN_THRESHOLD - 1` such requests stay
    silent; the threshold-th one warns.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    threshold = live_consumer_module._SUBAGENT_SLUG_DRIFT_WARN_THRESHOLD

    with caplog.at_level("WARNING", logger=_DRIFT_LOGGER):
        for _ in range(threshold - 1):
            _classify_as_subagent_by_prompt(consumer, "some-unrecognized-slug")
        assert _warnings(caplog) == []
        _classify_as_subagent_by_prompt(consumer, "some-unrecognized-slug")

    assert len(_warnings(caplog)) == 1


def test_subagent_slug_seen_suppresses_drift_warning(
    monkeypatch: Any, caplog: Any
) -> None:
    """Ever observing the subagent slug permanently suppresses the warning.

    Even a single sighting suppresses it, however many slug-less
    prompt-matched sub-agent requests follow.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)

    with caplog.at_level("WARNING", logger=_DRIFT_LOGGER):
        with bridged_request_scope(consumer._models.subagent):
            _classify(consumer)
        for _ in range(10):
            _classify_as_subagent_by_prompt(consumer, "some-unrecognized-slug")

    assert _warnings(caplog) == []


def _tool_result_line(tool_use_id: str) -> dict[str, Any]:
    """The JSONL `user` line Claude Code prints when a Task's tool_result lands."""
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id}],
        },
    }


def test_subagent_slug_drift_not_counted_once_pending_cleared(
    monkeypatch: Any, caplog: Any
) -> None:
    """A Task that dies before any sub-agent request reaches the bridge is not drift.

    on_complete registers the pending entry when the Task tool_call appears
    in output; if the tool_result then arrives with no sub-agent request in
    between, the window in which sub-agent traffic was expected has closed.
    Requests that happen to carry the dead Task's prompt after that cannot
    be prompt-matched to it, so they must not count toward the canary.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    consumer.process_jsonl_line(_tool_result_line("call_1"))
    assert not consumer._pending_subagents

    with caplog.at_level("WARNING", logger=_DRIFT_LOGGER):
        for _ in range(10):
            result = _classify_as_subagent_by_prompt(consumer, "some-unrecognized-slug")
            assert result == AgentBridgeContext("root")

    assert _warnings(caplog) == []


def test_subagent_slug_drift_requires_a_first_span(
    monkeypatch: Any, caplog: Any
) -> None:
    """No Task/Agent span ever opened -> nothing to warn about.

    An agent that never delegates isn't drift, whatever its requests carry.
    """
    consumer = _consumer(monkeypatch)

    with caplog.at_level("WARNING", logger=_DRIFT_LOGGER):
        for _ in range(10):
            _classify_as_subagent_by_prompt(consumer, "some-unrecognized-slug")
        with bridged_request_scope(consumer._models.presented):
            for _ in range(10):
                _classify(consumer)

    assert _warnings(caplog) == []


# ---------------------------------------------------------------------------
# structural slug beats the prompt heuristic
# ---------------------------------------------------------------------------


def test_classify_root_slug_wins_over_prompt_match(monkeypatch: Any) -> None:
    """A root request whose task prompt contains the delegated prompt is root.

    The root thread's first user message is its own task prompt, and a Task
    prompt is often a verbatim excerpt of it. Under the presented slug that
    request is structurally main-thread (P1), so the content match must not
    stamp it "subagent" -- that would switch an `is_root_agent()` gate off
    for the root thread for the whole delegation window.
    """
    consumer = _consumer(monkeypatch)
    root_task = f"{_TASK_PROMPT} Then update the CHANGELOG."
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    with bridged_request_scope(consumer._models.presented):
        result = _classify(consumer, text=root_task)
    assert result == AgentBridgeContext("root")


def test_attribute_root_slug_lands_on_outer_span(monkeypatch: Any) -> None:
    """Span attribution agrees: a root-slug request is never put on a sub-agent span."""
    consumer = _consumer(monkeypatch)
    root_task = f"{_TASK_PROMPT} Then update the CHANGELOG."
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    with bridged_request_scope(consumer._models.presented):
        span_id = consumer._attribute([ChatMessageUser(content=root_task)])
    assert span_id == consumer.outer_span_id
    # ...while the sub-agent's own request (no root slug) still resolves its span
    with bridged_request_scope(consumer._models.subagent):
        span_id = consumer._attribute([ChatMessageUser(content=_TASK_PROMPT)])
    assert span_id == "agent-call_1"


def test_classify_small_fast_slug_equal_to_a_tier_is_root(monkeypatch: Any) -> None:
    """haiku_model resolving to the same name as opus_model is a main-thread slug.

    Mirrors the ACP variant: small-fast traffic sharing ANY main-thread slug
    is indistinguishable from it and must not be stamped "utility".
    """
    consumer = _consumer(
        monkeypatch, opus_model="mockllm/shared", haiku_model="mockllm/shared"
    )
    assert consumer._models.haiku == consumer._models.opus
    with bridged_request_scope(consumer._models.haiku):
        result = _classify(consumer)
    assert result == AgentBridgeContext("root")
