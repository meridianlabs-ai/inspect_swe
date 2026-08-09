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

from inspect_ai.agent import AgentBridgeContext
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
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

# Second prompt for the two-subagent async replay (mirrors the real recorded
# log's two concurrent Task spawns).
_TASK_PROMPT_2 = "Run the second background command and report its output."


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


def _consumer_with_transcript(
    monkeypatch: Any, **model_kwargs: Any
) -> tuple[LiveConsumer, _TranscriptStub]:
    """Like `_consumer`, but returns the *same* stub instance every call.

    `_consumer`'s `lambda: _TranscriptStub()` hands back a fresh (and
    therefore immediately-discarded) stub on every `transcript()` call --
    fine for tests that only assert on consumer state, but the async-replay
    tests below need to see the actual sequence of SpanBeginEvent /
    SpanEndEvent traffic emitted across multiple calls.
    """
    stub = _TranscriptStub()
    monkeypatch.setattr(live_consumer_module, "transcript", lambda: stub)
    models = resolve_claude_code_models("mockllm/model", None, **model_kwargs)
    return LiveConsumer(models), stub


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


# ---------------------------------------------------------------------------
# subagent-slug drift canary
# ---------------------------------------------------------------------------
#
# The synthetic subagent slug shape ("<presented>-subagent") was never
# live-verified against a real CC build the way P1's real catalog-name slugs
# were. If some CC version rejects/ignores it, subagent requests would
# silently stop carrying it and attribution would quietly degrade to
# prompt-match-only. These tests drive `_check_subagent_slug_drift` (called
# from `classify`) directly through its public entry point.


def test_subagent_slug_drift_warns_once(monkeypatch: Any, caplog: Any) -> None:
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)

    with caplog.at_level(
        "WARNING", logger="inspect_swe._claude_code._events.live_consumer"
    ):
        with bridged_request_scope(consumer._models.presented):
            # well past the threshold -- must still warn exactly once
            for _ in range(10):
                _classify(consumer)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert consumer._models.subagent in warnings[0].message


def test_subagent_slug_seen_suppresses_drift_warning(
    monkeypatch: Any, caplog: Any
) -> None:
    """Ever observing the subagent slug permanently suppresses the warning.

    Even a single sighting suppresses it, however many non-subagent-slug
    requests follow.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)

    with caplog.at_level(
        "WARNING", logger="inspect_swe._claude_code._events.live_consumer"
    ):
        with bridged_request_scope(consumer._models.subagent):
            _classify(consumer)
        with bridged_request_scope(consumer._models.presented):
            for _ in range(10):
                _classify(consumer)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 0


def test_subagent_slug_drift_requires_a_first_span(
    monkeypatch: Any, caplog: Any
) -> None:
    """No Task/Agent span ever opened -> nothing to warn about.

    An agent that never delegates isn't drift.
    """
    consumer = _consumer(monkeypatch)

    with caplog.at_level(
        "WARNING", logger="inspect_swe._claude_code._events.live_consumer"
    ):
        with bridged_request_scope(consumer._models.presented):
            for _ in range(10):
                _classify(consumer)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 0


# ---------------------------------------------------------------------------
# async Task span lifecycle
# ---------------------------------------------------------------------------
#
# Shapes below are replayed verbatim (structurally) from a real recorded CC
# 2.1.220 log -- two concurrent background Task spawns -- at
# ~/Development/test_evals/logs/agent-context/claude_code/*.eval. See
# `live_consumer._ASYNC_LAUNCH_ACK_PREFIX` / `_TASK_NOTIFICATION_MARKER` /
# `_TASK_NOTIFICATION_TOOL_USE_ID_RE` for the provenance notes on each shape.
#
# The bug this covers: `_handle_user` used to treat *any* tool_result on a
# pending Task tool_use_id as completion, closing the span and clearing the
# registry. For an async/background Task, that tool_result is only the
# launch acknowledgment -- the real work (and real bridged model calls)
# arrives seconds later, by which point the registry was already empty, so
# every subsequent subagent event fell through to the outer span (the
# "utility agents" / "tail-only messages" viewer artifacts).


def _span_events(stub: "_TranscriptStub", event_type: type) -> list[Any]:
    return [e for e in stub.events if isinstance(e, event_type)]


def _tool_result_jsonl(tool_use_id: str, text: str) -> dict[str, Any]:
    """A raw "user" JSONL line carrying one tool_result content block."""
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": text}],
                }
            ]
        },
    }


def _async_launch_ack_jsonl(
    tool_use_id: str, agent_id: str = "a56bbf04021ac8018"
) -> dict[str, Any]:
    """An async Task's launch-acknowledgment tool_result.

    Text prefix verified live against CC 2.1.220 (recorded log msg[3]/
    msg[4]) -- see `live_consumer._ASYNC_LAUNCH_ACK_PREFIX`.
    """
    text = (
        "Async agent launched successfully. (This tool result is internal "
        "metadata — never quote or paste any part of it, including the "
        f"agentId below, into a user-facing reply.)\nagentId: {agent_id} "
        "(internal ID - do not mention to user. Use SendMessage to "
        "continue this agent.)\nThe agent is working in the background. "
        "You will be notified automatically when it completes."
    )
    return _tool_result_jsonl(tool_use_id, text)


def _sync_task_result_jsonl(
    tool_use_id: str, result_text: str = "the command finished; here is the output"
) -> dict[str, Any]:
    """A synchronous Task's real-result tool_result (no async-ack prefix)."""
    return _tool_result_jsonl(tool_use_id, result_text)


def _task_notification_jsonl(
    tool_use_id: str, task_id: str = "a56bbf04021ac8018"
) -> dict[str, Any]:
    """An async Task's genuine completion signal.

    Shape verified live against CC 2.1.220 (recorded log msg[6]/msg[8]):
    `message.content` is a plain string (not a tool_result block), and
    correlates to the *original* Task tool_use_id via `<tool-use-id>` --
    not the CC-internal `<task-id>` also present in the block -- see
    `live_consumer._TASK_NOTIFICATION_TOOL_USE_ID_RE`.
    """
    text = (
        "[SYSTEM NOTIFICATION - NOT USER INPUT]\n"
        "This is an automated background-task event, NOT a message from "
        "the user.\n\n"
        "<task-notification>\n"
        f"<task-id>{task_id}</task-id>\n"
        f"<tool-use-id>{tool_use_id}</tool-use-id>\n"
        "<status>completed</status>\n"
        "<summary>Agent finished</summary>\n"
        "<result>the result text</result>\n"
        "</task-notification>"
    )
    return {"type": "user", "message": {"content": text}}


def _subagent_bridge_event(prompt: str) -> ModelEvent:
    """An in-flight ModelEvent from a sub-agent's own bridge call.

    Re-sends the sub-agent's full history with the original Task prompt at
    `input[0]` -- see `LiveConsumer`'s class docstring for why that's what
    drives substring-match attribution.
    """
    return ModelEvent(
        model="m",
        input=[ChatMessageUser(content=f"<task>\n{prompt}\n</task>")],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput.from_content("m", "working on it"),
    )


def test_async_task_full_replay(monkeypatch: Any) -> None:
    """Full async Task lifecycle, replayed from the real recorded log's shapes.

    spawn -> launch-ack (span stays OPEN, prompt still registered) ->
    subagent model events (attributed to the agent span) -> completion
    notification (span closes) -> reset() closes any stragglers.
    """
    consumer, stub = _consumer_with_transcript(monkeypatch)

    # 1. spawn: on_complete with two concurrent Task tool_calls (mirrors the
    #    real log's two background Task spawns).
    message = ChatMessageAssistant(
        content="Spawning two background agents.",
        tool_calls=[
            ToolCall(id="call_1", function="Task", arguments={"prompt": _TASK_PROMPT}),
            ToolCall(
                id="call_2", function="Task", arguments={"prompt": _TASK_PROMPT_2}
            ),
        ],
    )
    consumer.on_complete(
        ModelEvent(
            model="m",
            input=[],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(),
            output=ModelOutput.from_message(message),
        )
    )
    span_1 = consumer._open_agents["call_1"].span_id
    span_2 = consumer._open_agents["call_2"].span_id
    assert len(_span_events(stub, SpanBeginEvent)) == 2

    # 2. launch-ack tool_results for both -- spans must stay OPEN, prompts
    #    still registered. This is the bug: previously this closed the span
    #    and cleared the registry, orphaning every later subagent event onto
    #    the outer span.
    consumer.process_jsonl_line(_async_launch_ack_jsonl("call_1"))
    consumer.process_jsonl_line(_async_launch_ack_jsonl("call_2"))
    assert "call_1" in consumer._open_agents
    assert "call_2" in consumer._open_agents
    assert "call_1" in consumer._pending_subagents
    assert "call_2" in consumer._pending_subagents
    assert consumer._open_agents["call_1"].launched_async is True
    assert consumer._open_agents["call_2"].launched_async is True
    assert _span_events(stub, SpanEndEvent) == []  # nothing closed yet

    # 3. subagent model events -- attributed to their own agent span, not
    #    dumped on the outer span.
    event_1 = _subagent_bridge_event(_TASK_PROMPT)
    consumer.on_pending(event_1)
    assert event_1.span_id == span_1

    event_2 = _subagent_bridge_event(_TASK_PROMPT_2)
    consumer.on_pending(event_2)
    assert event_2.span_id == span_2

    # 4. completion notification for agent 1 only -- its span closes; agent
    #    2 stays open (not yet notified).
    consumer.process_jsonl_line(_task_notification_jsonl("call_1"))
    assert "call_1" not in consumer._open_agents
    assert "call_1" not in consumer._pending_subagents
    assert "call_2" in consumer._open_agents
    span_ends = _span_events(stub, SpanEndEvent)
    assert len(span_ends) == 1
    assert span_ends[0].id == span_1

    # 5. reset() closes any stragglers (agent 2, never notified).
    consumer.reset()
    assert consumer._open_agents == {}
    span_ends = _span_events(stub, SpanEndEvent)
    assert len(span_ends) == 2
    assert {e.id for e in span_ends} == {span_1, span_2}


def test_sync_task_result_closes_span_immediately(monkeypatch: Any) -> None:
    """Regression check: sync Task lifecycle is unchanged by this fix.

    A synchronous Task's tool_result (no async-ack shape) still closes the
    span and clears the registry right away.
    """
    consumer, stub = _consumer_with_transcript(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    span_id = consumer._open_agents["call_1"].span_id

    consumer.process_jsonl_line(_sync_task_result_jsonl("call_1"))

    assert "call_1" not in consumer._open_agents
    assert "call_1" not in consumer._pending_subagents
    span_ends = _span_events(stub, SpanEndEvent)
    assert len(span_ends) == 1
    assert span_ends[0].id == span_id


def test_task_notification_unresolvable_id_is_noop(monkeypatch: Any) -> None:
    """An unresolvable notification is a no-op, not a guess.

    A notification whose <tool-use-id> names no open agent leaves the span
    open until `reset()` -- conservative by design.
    """
    consumer, stub = _consumer_with_transcript(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    consumer.process_jsonl_line(_async_launch_ack_jsonl("call_1"))

    consumer.process_jsonl_line(_task_notification_jsonl("does-not-exist"))

    assert "call_1" in consumer._open_agents
    assert _span_events(stub, SpanEndEvent) == []


def test_plain_user_text_without_notification_is_noop(monkeypatch: Any) -> None:
    """Only the <task-notification> marker triggers a lookup.

    An ordinary string-content user turn without it must not close
    anything.
    """
    consumer, stub = _consumer_with_transcript(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    consumer.process_jsonl_line(_async_launch_ack_jsonl("call_1"))

    consumer.process_jsonl_line(
        {"type": "user", "message": {"content": "hello, unrelated user text"}}
    )

    assert "call_1" in consumer._open_agents
    assert _span_events(stub, SpanEndEvent) == []


def test_attribute_safety_net_single_async_agent_by_slug(monkeypatch: Any) -> None:
    """Safety net: slug + a single open async agent resolves unambiguously.

    Prompt-match fails, but the slug is the subagent slug and exactly one
    async agent is open, so attribution falls back to it.
    """
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    consumer.process_jsonl_line(_async_launch_ack_jsonl("call_1"))
    span_id = consumer._open_agents["call_1"].span_id

    messages: list[ChatMessage] = [
        ChatMessageUser(content="totally unrelated text, no prompt substring")
    ]
    with bridged_request_scope(consumer._models.subagent):
        resolved = consumer._attribute(messages)
    assert resolved == span_id


def test_attribute_safety_net_no_op_with_multiple_async_agents(
    monkeypatch: Any,
) -> None:
    """Two async agents open -- ambiguous, safety net must not guess."""
    consumer = _consumer(monkeypatch)
    _spawn_pending(consumer, "call_1", _TASK_PROMPT)
    _spawn_pending(consumer, "call_2", _TASK_PROMPT_2)
    consumer.process_jsonl_line(_async_launch_ack_jsonl("call_1"))
    consumer.process_jsonl_line(_async_launch_ack_jsonl("call_2"))

    messages: list[ChatMessage] = [
        ChatMessageUser(content="totally unrelated text, no prompt substring")
    ]
    with bridged_request_scope(consumer._models.subagent):
        resolved = consumer._attribute(messages)
    assert resolved == consumer.outer_span_id
