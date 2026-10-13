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

import json
from typing import Any

import pytest
from inspect_ai._util.content import ContentText
from inspect_ai.agent import (
    AgentBridgeContext,
    AgentState,
    current_agent_bridge_context,
)
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.agent._bridge.types import AgentBridge
from inspect_ai.agent._bridge.util import bridge_generate
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import (
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_swe._codex_cli._events import consumer as consumer_module
from inspect_swe._codex_cli._events.consumer import CodexConsumer
from inspect_swe._codex_cli._events.detection import (
    COMPACTION_MARKER,
    agent_message_recipients,
    final_answer_authors,
    find_spawned_agents,
    spawn_result,
)
from inspect_swe._util.agentcontext import classify_filter

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


def test_classify_ambiguous_span_is_still_subagent(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)

    # two unbound spawns sharing the same task name -> a recipient basename
    # match can't disambiguate between them, so the span stays unresolved ...
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

    # ... but a non-root recipient still identifies the kind
    assert consumer._attribute(messages) == consumer.outer_span_id
    assert result == AgentBridgeContext("subagent")


def test_classify_v2_followup_after_final_answer_is_subagent(
    monkeypatch: Any,
) -> None:
    """V2: a child resumed by followup_task after its FINAL_ANSWER.

    The FINAL_ANSWER closes the child's span, so no span state identifies the
    child's next request; its inbound agent_message recipient still does.
    """
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_w", "worker")],
    )
    consumer.on_complete(parent)
    consumer.on_pending(_model_event([_agent_message_user("/root", "/root/worker")]))

    final = _agent_message_user("/root/worker", "/root", "FINAL_ANSWER", "done")
    root_messages: list[ChatMessage] = [ChatMessageUser(content="go"), final]
    assert consumer.classify(get_model("mockllm/model"), root_messages, []) == (
        AgentBridgeContext("root")
    )
    assert not consumer._agents

    followup: list[ChatMessage] = [
        _agent_message_user("/root", "/root/worker"),
        _agent_message_user("/root", "/root/worker", "MESSAGE", "one more thing"),
    ]
    result = consumer.classify(get_model("mockllm/model"), followup, [])

    assert result == AgentBridgeContext("subagent")

    # same with another sub-agent still open (was "unknown")
    other = _model_event(
        root_messages, tool_calls=[_v2_spawn_call("call_o", "other_task")]
    )
    consumer.on_complete(other)
    assert consumer._agents
    result = consumer.classify(get_model("mockllm/model"), followup, [])

    assert result == AgentBridgeContext("subagent")


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

    For any request _attribute() resolves to a sub-agent span, classify()
    reports "subagent", and here (every call attributable) the converse
    holds too — classify is a thin reclassification of the same maps
    _attribute uses. The one addition is a non-root recipient with no
    resolvable span (see the follow-up and ambiguous tests), where the kind
    is known but the span isn't.
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


# ---------------------------------------------------------------------------
# 8. classify(): thread-close signals in the request itself
# ---------------------------------------------------------------------------


def _v1_completion_carrier(kind: str, thread_id: str) -> ChatMessage:
    """The two V1 carriers of a `completed` status for a thread (detection.py)."""
    if kind == "wait_agent":
        return ChatMessageTool(
            content=json.dumps({"status": {thread_id: {"completed": "done"}}}),
            tool_call_id="call_wait",
            function="wait_agent",
        )
    return ChatMessageUser(
        content="<subagent_notification>"
        + json.dumps({"agent_path": thread_id, "status": {"completed": "done"}})
        + "</subagent_notification>"
    )


@pytest.mark.parametrize("carrier", ["wait_agent", "subagent_notification"])
def test_classify_root_after_v1_completion_in_request(
    monkeypatch: Any, carrier: str
) -> None:
    """V1: the last open sub-agent completes; root's next request proves it.

    The bridge runs classify() before on_pending(), so classify must apply
    the request's own completion signal before deciding whether any
    sub-agent is still open. Without that, a genuine root request whose
    input closes the last thread is stamped "unknown" (no agent_message
    recipient under V1, `_agents` still non-empty).
    """
    consumer, stub = _consumer(monkeypatch)

    prompt = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"
    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v1_spawn_call("call_1", prompt)],
    )
    consumer.on_complete(parent)
    span = stub.span_begins()[0].id

    child_call = _model_event([ChatMessageUser(content=prompt)])
    consumer.on_pending(child_call)
    assert child_call.span_id == span

    # root's next request: full history (spawn result binds the thread id)
    # plus the completion of that thread.
    root_messages: list[ChatMessage] = [
        ChatMessageUser(content="go"),
        _spawn_result_tool_message(
            "call_1", '{"agent_id":"thread_abc","nickname":"Explorer"}'
        ),
        _v1_completion_carrier(carrier, "thread_abc"),
    ]
    result = consumer.classify(get_model("mockllm/model"), root_messages, [])

    assert result == AgentBridgeContext("root")
    assert [e.id for e in stub.span_ends()] == [span]

    # on_pending over the same request must not close the span twice
    consumer.on_pending(_model_event(root_messages))
    assert [e.id for e in stub.span_ends()] == [span]
    assert not consumer._agents and not consumer._thread_index


def test_classify_closes_span_on_v2_final_answer(monkeypatch: Any) -> None:
    """V2: a FINAL_ANSWER in root's request closes the child's span at filter time.

    The following on_pending over the same request must not close it again.
    (Classification itself was already "root" here via the `/root`
    recipient rescue; this pins the span-close timing and single emission.)
    """
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_pr", "write_primes")],
    )
    consumer.on_complete(parent)
    pr_span = stub.span_begins()[0].id

    child_call = _model_event([_agent_message_user("/root", "/root/write_primes")])
    consumer.on_pending(child_call)
    assert child_call.span_id == pr_span

    root_messages: list[ChatMessage] = [
        ChatMessageUser(content="go"),
        _agent_message_user(
            "/root/write_primes", "/root", "FINAL_ANSWER", "all primes written"
        ),
    ]
    result = consumer.classify(get_model("mockllm/model"), root_messages, [])

    assert result == AgentBridgeContext("root")
    assert [e.id for e in stub.span_ends()] == [pr_span]
    assert not consumer._agents

    consumer.on_pending(_model_event(root_messages))
    assert [e.id for e in stub.span_ends()] == [pr_span]


# ---------------------------------------------------------------------------
# 9. V1: a spawner whose own prompt contains the delegated text
# ---------------------------------------------------------------------------


def test_classify_v1_spawner_is_not_its_own_subagent(monkeypatch: Any) -> None:
    """V1: root delegates a sentence copied from its own task prompt.

    Root's next request carries that sentence (in its task prompt) and so
    substring-matches the open spawn. It also carries the spawn call and its
    result, which a V1 child's thread never does: the requester is the
    spawner, so the spawn is no candidate. With nothing else open root
    classifies "unknown" for the delegation window, never "subagent".
    """
    consumer, stub = _consumer(monkeypatch)

    delegated = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"
    task = ChatMessageUser(content=f"Two steps. First, {delegated}. Then test it.")
    spawn = _v1_spawn_call("call_1", delegated)
    consumer.on_complete(_model_event([task], tool_calls=[spawn]))
    span = stub.span_begins()[0].id

    root_messages: list[ChatMessage] = [
        task,
        ChatMessageAssistant(content="", tool_calls=[spawn]),
        _spawn_result_tool_message(
            "call_1", '{"agent_id":"thread_abc","nickname":"Explorer"}'
        ),
    ]
    result = consumer.classify(get_model("mockllm/model"), root_messages, [])

    assert result == AgentBridgeContext("unknown")
    root_call = _model_event(root_messages)
    consumer.on_pending(root_call)
    assert root_call.span_id == consumer.outer_span_id

    # the child itself still matches
    child_messages: list[ChatMessage] = [ChatMessageUser(content=delegated)]
    assert consumer.classify(get_model("mockllm/model"), child_messages, []) == (
        AgentBridgeContext("subagent")
    )
    assert consumer._attribute(child_messages) == span


# ---------------------------------------------------------------------------
# 10. a filter that returns its own ModelOutput
# ---------------------------------------------------------------------------


async def test_filter_returned_spawn_registers_subagent(monkeypatch: Any) -> None:
    """A filter that generates itself bypasses the bridge's ModelEventSink.

    Driven through the real bridge_generate with the same filter wiring
    codex_cli() installs: the spawn_agent call in the filter's own output
    must still open the sub-agent, so the child's V1 request classifies
    "subagent" rather than "root".
    """
    consumer, stub = _consumer(monkeypatch)
    prompt = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"
    model = get_model("mockllm/model")

    kinds: list[str] = []

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        context = current_agent_bridge_context()
        kinds.append(context.kind if context is not None else "none")
        if messages[-1].text == "go":
            spawn = _v1_spawn_call("call_1", prompt)
            return ModelOutput.from_message(
                ChatMessageAssistant(content="", tool_calls=[spawn])
            )
        return ModelOutput.from_content(model.name, "done")

    bridge = AgentBridge(
        AgentState(messages=[]),
        filter=classify_filter(
            user_filter, consumer.classify, consumer.on_filter_output
        ),
        model_event_sink=consumer,
    )
    for text in ("go", prompt):
        messages: list[ChatMessage] = [ChatMessageUser(content=text)]
        await bridge_generate(bridge, model, messages, [], None, GenerateConfig(), None)

    assert kinds == ["root", "subagent"]
    assert [e.id for e in stub.span_begins()] == ["agent-call_1"]
