"""Codex Multi-Agent V2 sub-agent span reconstruction (bridge-only).

Multi-Agent V2 (codex >= 0.146, forced for gpt-5.6-terra/sol) changed every
signal the CodexConsumer used to build the agent-span tree:

- `spawn_agent` calls carry `task_name` (not `agent_type`) and an *encrypted*
  `message`, so spans were named the generic "agent" and prompt-substring
  attribution matched nothing (all model events landed on the outer span).
- `spawn_agent` results carry `{"task_name": "/root/<name>"}` (not
  `agent_id`/`nickname`).
- completion arrives as a FINAL_ANSWER `agent_message` from the child (the
  `wait_agent` result no longer carries per-thread status).

The agent bridge (inspect_ai >= 0.3.253) converts each `agent_message` input
item to a ChatMessageUser and preserves the raw item (author/recipient) on
ContentText.internal — every agent_message in a request is inbound to the
requester, so `recipient` identifies the calling agent exactly.

Fixture shapes below are copied from a real gpt-5.6-sol eval log
(codex 0.147.0, 2026-08-07).
"""

from typing import Any

from inspect_ai._util.content import ContentText
from inspect_ai.agent import AgentBridgeContext
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_swe._codex_cli._events import consumer as consumer_module
from inspect_swe._codex_cli._events.consumer import CodexConsumer
from inspect_swe._codex_cli._events.detection import (
    COMPACTION_MARKER,
    agent_message_recipients,
    final_answer_authors,
    find_spawned_agents,
    spawn_result,
)

# ---------------------------------------------------------------------------
# fixtures (shapes from a real codex 0.147.0 / gpt-5.6-sol run)
# ---------------------------------------------------------------------------


def _v2_spawn_call(call_id: str, task_name: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={
            "task_name": task_name,
            "fork_turns": "all",
            "message": "gAAAAABqdjXI1bVTsv-encrypted-payload",
        },
    )


def _v1_spawn_call(call_id: str, prompt: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={"agent_type": "explore", "message": prompt},
    )


def _agent_message_user(
    author: str, recipient: str, message_type: str = "MESSAGE", payload: str = ""
) -> ChatMessageUser:
    """A ChatMessageUser as the bridge produces it from an agent_message item."""
    envelope = (
        f"Message Type: {message_type}\n"
        f"Task name: {recipient}\n"
        f"Sender: {author}\n"
        f"Payload:\n{payload}"
    )
    raw_item: dict[str, Any] = {
        "type": "agent_message",
        "author": author,
        "recipient": recipient,
        "content": [{"type": "input_text", "text": envelope}],
    }
    return ChatMessageUser(
        content=[
            ContentText(
                text=f"Agent message from {author}:\n{envelope}",
                internal={"agent_message": raw_item},
            )
        ]
    )


def _spawn_result_tool_message(call_id: str, text: str) -> ChatMessageTool:
    return ChatMessageTool(content=text, tool_call_id=call_id, function="spawn_agent")


def _model_event(
    input: list[ChatMessage], tool_calls: list[ToolCall] | None = None
) -> ModelEvent:
    message = ChatMessageAssistant(content="ok", tool_calls=tool_calls)
    return ModelEvent(
        model="openai/gpt-5.6-sol",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(message),
    )


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)

    def _event_updated(self, event: Any) -> None:
        pass

    def span_begins(self) -> list[SpanBeginEvent]:
        return [e for e in self.events if isinstance(e, SpanBeginEvent)]

    def span_ends(self) -> list[SpanEndEvent]:
        return [e for e in self.events if isinstance(e, SpanEndEvent)]


def _consumer(monkeypatch: Any) -> tuple[CodexConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    return CodexConsumer(), stub


# ---------------------------------------------------------------------------
# 1. detection: V2 spawn calls and results
# ---------------------------------------------------------------------------


def test_find_spawned_agents_uses_v2_task_name() -> None:
    spawned = find_spawned_agents([_v2_spawn_call("call_1", "write_fizzbuzz")])
    assert len(spawned) == 1
    assert spawned[0].name == "write_fizzbuzz"


def test_find_spawned_agents_keeps_v1_agent_type() -> None:
    spawned = find_spawned_agents(
        [_v1_spawn_call("call_1", "write a fizzbuzz program to /tmp/fizzbuzz.py")]
    )
    assert len(spawned) == 1
    assert spawned[0].name == "explore"


def test_spawn_result_parses_v2_task_name() -> None:
    result = spawn_result(
        _spawn_result_tool_message("call_1", '{"task_name":"/root/write_fizzbuzz"}')
    )
    assert result is not None
    assert result.agent_id == "/root/write_fizzbuzz"
    assert result.nickname is None


def test_spawn_result_still_parses_v1_agent_id() -> None:
    result = spawn_result(
        _spawn_result_tool_message(
            "call_1", '{"agent_id":"thread_abc","nickname":"Explorer"}'
        )
    )
    assert result is not None
    assert result.agent_id == "thread_abc"
    assert result.nickname == "Explorer"


# ---------------------------------------------------------------------------
# 2. detection: agent_message identity and completion signals
# ---------------------------------------------------------------------------


def test_agent_message_recipients_identifies_requester() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_fizzbuzz"),
        _agent_message_user(
            "/root/write_fizzbuzz/implement_file", "/root/write_fizzbuzz"
        ),
    ]
    assert agent_message_recipients(input) == {"/root/write_fizzbuzz"}


def test_agent_message_recipients_empty_without_agent_messages() -> None:
    assert agent_message_recipients([ChatMessageUser(content="plain task")]) == set()


def test_final_answer_authors_detects_completion() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_primes"),
        _agent_message_user(
            "/root/write_primes", "/root", "FINAL_ANSWER", "primes written"
        ),
    ]
    assert final_answer_authors(input) == {"/root/write_primes"}


def test_final_answer_authors_ignores_plain_messages() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root/write_primes", "/root", "MESSAGE")
    ]
    assert final_answer_authors(input) == set()


# ---------------------------------------------------------------------------
# 3. consumer: spans are named after V2 task_name
# ---------------------------------------------------------------------------


def test_consumer_names_v2_spans_after_task_name(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="spawn two agents")],
        tool_calls=[
            _v2_spawn_call("call_fb", "write_fizzbuzz"),
            _v2_spawn_call("call_pr", "write_primes"),
        ],
    )
    consumer.on_complete(parent)

    names = [e.name for e in stub.span_begins()]
    assert names == ["write_fizzbuzz", "write_primes"]


# ---------------------------------------------------------------------------
# 4. consumer: attribution by agent_message recipient
# ---------------------------------------------------------------------------


def test_consumer_attributes_subagent_call_by_recipient(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    # parent spawns two agents (V2: encrypted prompts, so no substring match)
    parent = _model_event(
        [ChatMessageUser(content="spawn two agents")],
        tool_calls=[
            _v2_spawn_call("call_fb", "write_fizzbuzz"),
            _v2_spawn_call("call_pr", "write_primes"),
        ],
    )
    consumer.on_complete(parent)
    fb_span, pr_span = (e.id for e in stub.span_begins())

    # each sub-agent's first call carries its inbound spawn agent_message
    fb_call = _model_event([_agent_message_user("/root", "/root/write_fizzbuzz")])
    consumer.on_pending(fb_call)
    assert fb_call.span_id == fb_span

    pr_call = _model_event([_agent_message_user("/root", "/root/write_primes")])
    consumer.on_pending(pr_call)
    assert pr_call.span_id == pr_span

    # the parent's own calls (inbound messages addressed to /root) stay outer
    parent_call = _model_event(
        [
            ChatMessageUser(content="spawn two agents"),
            _agent_message_user("/root/write_primes", "/root"),
        ]
    )
    consumer.on_pending(parent_call)
    assert parent_call.span_id not in (fb_span, pr_span)


def test_consumer_nests_grandchild_span_under_child(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_fb", "write_fizzbuzz")],
    )
    consumer.on_complete(parent)
    fb_span = stub.span_begins()[0].id

    # child call spawns its own sub-agent
    child_call = _model_event(
        [_agent_message_user("/root", "/root/write_fizzbuzz")],
        tool_calls=[_v2_spawn_call("call_impl", "implement_file")],
    )
    consumer.on_pending(child_call)
    consumer.on_complete(child_call)

    impl_begin = stub.span_begins()[1]
    assert impl_begin.name == "implement_file"
    assert impl_begin.parent_id == fb_span


# ---------------------------------------------------------------------------
# 5. consumer: FINAL_ANSWER closes the child's span
# ---------------------------------------------------------------------------


def test_consumer_closes_span_on_final_answer(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_pr", "write_primes")],
    )
    consumer.on_complete(parent)
    pr_span = stub.span_begins()[0].id

    # child identifies itself (binds /root/write_primes to the span) ...
    child_call = _model_event([_agent_message_user("/root", "/root/write_primes")])
    consumer.on_pending(child_call)
    assert child_call.span_id == pr_span

    # ... then the parent sees the child's FINAL_ANSWER -> span closes
    parent_call = _model_event(
        [
            ChatMessageUser(content="go"),
            _agent_message_user(
                "/root/write_primes", "/root", "FINAL_ANSWER", "all primes written"
            ),
        ]
    )
    consumer.on_pending(parent_call)

    assert [e.id for e in stub.span_ends()] == [pr_span]


# ---------------------------------------------------------------------------
# 6. consumer: V1 prompt-substring attribution still works
# ---------------------------------------------------------------------------


def test_consumer_v1_prompt_attribution_unchanged(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    prompt = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"
    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v1_spawn_call("call_1", prompt)],
    )
    consumer.on_complete(parent)
    span = stub.span_begins()[0].id

    # V1: codex re-sends the spawn prompt as the sub-agent's user message
    child_call = _model_event([ChatMessageUser(content=prompt)])
    consumer.on_pending(child_call)
    assert child_call.span_id == span


# ---------------------------------------------------------------------------
# 7. classify(): filter-time agent context from the same attribution maps
# ---------------------------------------------------------------------------


def test_classify_root_before_any_spawn(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)

    result = consumer.classify(
        get_model("mockllm/model"), [ChatMessageUser(content="plain task")], []
    )

    assert result == AgentBridgeContext("root")


def test_classify_bound_subagent(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_pr", "write_primes")],
    )
    consumer.on_complete(parent)
    pr_span = stub.span_begins()[0].id

    # first call from the sub-agent's thread: binds by recipient basename
    # ("write_primes") the same way on_pending's _attribute() would.
    child_messages: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_primes")
    ]
    result = consumer.classify(get_model("mockllm/model"), child_messages, [])

    assert result == AgentBridgeContext("subagent")
    # the binding stuck (not a one-off side effect of classify alone) —
    # a later call resolves to the same span without re-deriving it.
    assert consumer._attribute(child_messages) == pr_span


def test_classify_compaction_as_utility(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)

    messages: list[ChatMessage] = [
        ChatMessageUser(
            content=f"{COMPACTION_MARKER}\nSummarize the conversation so far."
        )
    ]
    result = consumer.classify(get_model("mockllm/model"), messages, [])

    assert result == AgentBridgeContext("utility")


def test_classify_guardian_slug_as_utility(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)

    messages: list[ChatMessage] = [ChatMessageUser(content="review this diff")]
    with bridged_request_scope("codex-auto-review"):
        result = consumer.classify(get_model("mockllm/model"), messages, [])

    assert result == AgentBridgeContext("utility")


def test_classify_ambiguous_falls_to_unknown(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)

    # two unbound spawns sharing the same task name -> a recipient basename
    # match can't disambiguate between them.
    parent = _model_event(
        [ChatMessageUser(content="spawn two same-named agents")],
        tool_calls=[
            _v2_spawn_call("call_1", "write_fizzbuzz"),
            _v2_spawn_call("call_2", "write_fizzbuzz"),
        ],
    )
    consumer.on_complete(parent)

    messages: list[ChatMessage] = [_agent_message_user("/root", "/root/write_fizzbuzz")]
    result = consumer.classify(get_model("mockllm/model"), messages, [])

    assert result == AgentBridgeContext("unknown")


def test_classify_root_while_subagents_open(monkeypatch: Any) -> None:
    """A genuine root-thread request while a sub-agent is open.

    Fixture evidence (see `test_consumer_attributes_subagent_call_by_recipient`
    and `test_consumer_closes_span_on_final_answer` above): when a sub-agent
    sends the *parent* a message, the bridge produces a ChatMessageUser whose
    agent_message item has recipient "/root" — that is the only positive
    "this call belongs to root" signal these fixtures carry once agents are
    open (root's own plain-text turns look identical whether or not
    sub-agents are open, so they can't serve as that signal). `classify`
    must special-case recipients == {"/root"} to "root"; without it, an
    open sub-agent set would otherwise push this call to "unknown".
    """
    consumer, _ = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_pr", "write_primes")],
    )
    consumer.on_complete(parent)

    root_messages: list[ChatMessage] = [
        ChatMessageUser(content="go"),
        _agent_message_user("/root/write_primes", "/root", "MESSAGE", "status update"),
    ]
    result = consumer.classify(get_model("mockllm/model"), root_messages, [])

    assert result == AgentBridgeContext("root")


def test_classify_matches_span_attribution(monkeypatch: Any) -> None:
    """Shared-resolver invariant: classify() agrees with _attribute()'s span.

    For any request, classify() reports "subagent" exactly when _attribute()
    resolved a sub-agent span (rather than the outer span) — classify is a
    thin reclassification of the same maps _attribute uses, not a second
    source of truth.
    """
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="spawn two agents")],
        tool_calls=[
            _v2_spawn_call("call_fb", "write_fizzbuzz"),
            _v2_spawn_call("call_pr", "write_primes"),
        ],
    )
    consumer.on_complete(parent)

    requests: list[list[ChatMessage]] = [
        [_agent_message_user("/root", "/root/write_fizzbuzz")],
        [_agent_message_user("/root", "/root/write_primes")],
        [
            ChatMessageUser(content="spawn two agents"),
            _agent_message_user("/root/write_primes", "/root"),
        ],
    ]
    for messages in requests:
        is_subagent_span = consumer._attribute(messages) != consumer.outer_span_id
        kind = consumer.classify(get_model("mockllm/model"), messages, []).kind
        assert (kind == "subagent") == is_subagent_span


def test_classify_then_on_pending_is_idempotent(monkeypatch: Any) -> None:
    """The bridge runs classify() then on_pending() for the same request.

    This is the actual production call sequence, not just each method
    exercised in isolation. Both derive the request's span
    via `_attribute`; the first call from a sub-agent's thread binds it
    (mutating `_thread_index` and the matched `_OpenAgent.thread_id`) as a
    side effect of `_attribute_by_recipient`'s basename match. `on_pending`'s
    own `_attribute` pass over the *same* event must resolve to that already-
    bound span (not re-derive, and potentially disagree with, the binding)
    and must not mutate that state any further.
    """
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_pr", "write_primes")],
    )
    consumer.on_complete(parent)
    pr_span = stub.span_begins()[0].id

    # first call from the sub-agent's thread -- unbound until classify()
    # resolves it via recipient basename matching.
    child_call = _model_event([_agent_message_user("/root", "/root/write_primes")])

    classified = consumer.classify(get_model("mockllm/model"), child_call.input, [])
    assert classified == AgentBridgeContext("subagent")

    thread_index_after_classify = dict(consumer._thread_index)
    agents_after_classify = {
        call_id: (agent.thread_id, agent.span_id)
        for call_id, agent in consumer._agents.items()
    }

    consumer.on_pending(child_call)

    # on_pending's own _attribute() pass agreed with classify()'s implied span.
    assert child_call.span_id == pr_span

    # ... and the second pass left the binding state exactly as classify()
    # left it -- no re-derivation, no drift.
    assert dict(consumer._thread_index) == thread_index_after_classify
    assert {
        call_id: (agent.thread_id, agent.span_id)
        for call_id, agent in consumer._agents.items()
    } == agents_after_classify
