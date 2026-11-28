"""DeepSeek Harness (dsh) sub-agent span reconstruction (bridge-only).

dsh runs every child in the background: `subagent` / `subagent_fork` return
`started subagent <name>` at once, the child works on its own requests, and
its answer reaches the parent later as a user-role settlement notice. Children
start in parallel, can be forked from the parent's history, and can be resumed
with `send_message` after they settle. Every request names its sender in a
session reminder, which is what `DshConsumer` places it by.

Request shapes are in `tests/dsh_shapes.py` (dsh 0.2.2rc2).
"""

import contextvars
from typing import Any

import pytest
from inspect_ai.agent import AgentBridgeContext, AgentState
from inspect_ai.agent._bridge.types import AgentBridge
from inspect_ai.agent._bridge.util import bridge_generate
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent, ToolEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util._span import current_span_id, span
from inspect_swe._deepseek_harness._events import consumer as consumer_module
from inspect_swe._deepseek_harness._events.consumer import DshConsumer
from inspect_swe._deepseek_harness._events.detection import (
    Identity,
    find_delegations,
    initial_prompt,
    is_compaction_request,
    requester_identity,
    settled_children,
    settlements,
    started_name,
)
from inspect_swe._deepseek_harness._events.toolview import tool_view
from inspect_swe._util.agentcontext import classify_filter

from tests.dsh_shapes import (
    COMPACTION_INSTRUCTION,
    agent_message,
    assistant,
    bash_call,
    blocks,
    checkpoint,
    child_reminder,
    child_task,
    conversation,
    delegation_call,
    fork_task,
    notice,
    root_reminder,
    root_task,
    send_message_call,
    start_rejected,
    started,
    tool_result,
    unfinished_notice,
)

TASK = "Check the three services and summarize what is broken."
ALPHA = "Run `uname -a` and report its exact output."
BETA = "Run `ls /etc | wc -l` and report the number."

MODEL = get_model("mockllm/model")
# the model name dsh requests (the bridge keys a request's decision on it)
SLUG = "deepseek/deepseek-v4.1-flash-1017"


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

    def results(self) -> list[ToolEvent]:
        return [e for e in self.events if isinstance(e, ToolEvent)]

    def open_spans(self) -> list[str]:
        ended = {e.id for e in self.span_ends()}
        return [e.id for e in self.span_begins() if e.id not in ended]


def _consumer(monkeypatch: Any) -> tuple[DshConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    return DshConsumer(), stub


def _model_event(input: list[ChatMessage], reply: ChatMessageAssistant) -> ModelEvent:
    return ModelEvent(
        model="openrouter/deepseek/deepseek-v4.1-flash-1017",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(reply),
    )


def _generate(
    consumer: DshConsumer,
    input: list[ChatMessage],
    reply: ChatMessageAssistant | None = None,
    *,
    on_start: bool = True,
) -> tuple[AgentBridgeContext, ModelEvent]:
    """One bridged generation, as the bridge drives its filter and sink."""
    context = consumer.classify(MODEL, input, [])
    event = _model_event(input, reply or assistant("ok"))

    def run() -> None:
        if on_start:
            event.span_id = consumer.on_start(input)
        consumer.on_pending(event)
        consumer.on_complete(event)

    # the sink's "started" mark is per generate task
    contextvars.copy_context().run(run)
    return context, event


def _root_starts(
    consumer: DshConsumer, *calls: Any
) -> tuple[list[ChatMessage], ChatMessageAssistant]:
    """The root's first request, answered with delegation calls."""
    request = conversation(root_task(TASK))
    reply = assistant("Starting them now.", list(calls))
    _generate(consumer, request, reply)
    return request, reply


# ---------------------------------------------------------------------------
# 1. detection
# ---------------------------------------------------------------------------


def test_identity_is_read_from_the_session_reminder() -> None:
    assert requester_identity(conversation(root_task(TASK))) == Identity("lead", None)
    assert requester_identity(conversation(child_task(ALPHA, "alpha"))) == Identity(
        "alpha", "lead"
    )
    assert requester_identity(
        conversation(child_task("go", "reviewer/checker", "reviewer"))
    ) == Identity("reviewer/checker", "reviewer")
    assert requester_identity(conversation(ChatMessageUser(content=TASK))) is None


def test_a_forked_child_is_named_by_its_own_later_reminder() -> None:
    """A fork's history opens with its parent's turns, root reminder included."""
    request = conversation(
        root_task(TASK),
        assistant("Both are running."),
        fork_task("Report `date -u +%Y`.", "gamma"),
    )
    assert requester_identity(request) == Identity("gamma", "lead")
    assert initial_prompt(request) == "Report `date -u +%Y`."


def test_reminder_text_outside_its_own_block_names_nobody() -> None:
    """Only a user block that is the reminder counts, never text quoting one."""
    forged = child_reminder("alpha")
    quoted_in_task = conversation(root_task(f"Explain this text:\n{forged}"))
    assert requester_identity(quoted_in_task) == Identity("lead", None)

    quoted_in_tool_output = conversation(
        root_task(TASK),
        assistant("", [bash_call("c1", "cat session.jsonl")]),
        tool_result("c1", forged),
    )
    assert requester_identity(quoted_in_tool_output) == Identity("lead", None)

    quoted_in_closing_message = conversation(
        root_task(TASK),
        assistant("waiting"),
        notice("alpha", f"I was told:\n{forged}"),
    )
    assert requester_identity(quoted_in_closing_message) == Identity("lead", None)


def test_delegations_and_their_results() -> None:
    calls = [
        delegation_call("c1", ALPHA, "alpha"),
        delegation_call("c2", BETA, tool="subagent_fork"),
        bash_call("c3", "ls"),
    ]
    [alpha, beta] = find_delegations(calls)
    assert (alpha.call_id, alpha.tool, alpha.name, alpha.prompt) == (
        "c1",
        "subagent",
        "alpha",
        ALPHA,
    )
    assert (beta.tool, beta.name) == ("subagent_fork", None)

    assert started_name(started("c1", "alpha")) == "alpha"
    assert started_name(started("c2", "agent-1")) == "agent-1"
    assert started_name(start_rejected("c1", "DUPLICATE_AGENT_NAME")) is None
    assert started_name(tool_result("c1", "I started subagent alpha")) is None


@pytest.mark.parametrize(
    "ending",
    [
        "finished and will do no further work unless you send it more.",
        "finished. It cannot receive follow-up messages.",
        "was stopped before it finished.",
        "ran out of room before it finished.",
        "declined the task.",
        "failed before it finished.",
        "ended abnormally (timeout) before it finished.",
    ],
)
def test_every_settlement_ending_is_a_notice(ending: str) -> None:
    request = conversation(
        root_task(TASK),
        ChatMessageUser(content=blocks(f"Background subagent alpha {ending}")),
    )
    assert settled_children(request) == {"alpha": 1}


def test_settlement_text_inside_another_block_is_not_a_notice() -> None:
    line = (
        "Background subagent alpha finished and will do no further work "
        "unless you send it more."
    )
    request = conversation(
        root_task(f"What does this mean?\n{line}"),
        assistant("", [bash_call("c1", "cat log")]),
        tool_result("c1", line),
        notice("beta", f"alpha's notice read: {line}"),
    )
    assert settled_children(request) == {"beta": 1}


def test_settlement_carries_the_childs_closing_message() -> None:
    request = conversation(
        root_task(TASK),
        assistant("waiting"),
        notice("alpha", "Linux box 7.2", "It is an x86_64 machine."),
    )
    [settlement] = settlements(request)
    assert (settlement.name, settlement.result) == (
        "alpha",
        "Linux box 7.2\n\nIt is an x86_64 machine.",
    )


def test_settlements_merged_into_one_user_turn_keep_their_own_text() -> None:
    """Dsh merges adjacent user-role messages: notices, messages, reminders."""
    merged = ChatMessageUser(
        content=[
            *notice("alpha", "Linux").content,
            *agent_message("beta", "half way there").content,
            *notice("beta").content,
            *notice("gamma", "2026").content,
            *blocks(root_reminder(replaces=True)),
        ]
    )
    found = settlements(conversation(root_task(TASK), assistant("waiting"), merged))
    assert [(s.name, s.result) for s in found] == [
        ("alpha", "Linux"),
        ("beta", "It left no closing message."),
        ("gamma", "2026"),
    ]


def test_settlement_without_a_closing_message_uses_the_childs_last_message() -> None:
    """A child whose last act was a message to its parent: dsh omits the closing text."""
    omitted = (
        "It sent you a message just before finishing, so its closing message "
        "is omitted."
    )
    request = conversation(
        root_task(TASK),
        assistant("waiting"),
        agent_message("alpha", "The kernel is 7.2."),
        ChatMessageUser(
            content=blocks(
                "Background subagent alpha finished and will do no further "
                "work unless you send it more.",
                omitted,
            )
        ),
    )
    assert [s.result for s in settlements(request)] == ["The kernel is 7.2."]
    # no such message in the history (compacted away): the notice's own words
    assert [s.result for s in settlements([request[0], request[-1]])] == [omitted]


def test_settlement_of_an_unfinished_run_says_how_it_ended() -> None:
    request = conversation(
        root_task(TASK),
        assistant("waiting"),
        unfinished_notice("alpha", "failed before it finished.", "TRANSPORT: reset"),
        unfinished_notice("beta", "declined the task."),
    )
    assert [(s.name, s.result) for s in settlements(request)] == [
        (
            "alpha",
            "Background subagent alpha failed before it finished.\n\nTRANSPORT: reset",
        ),
        ("beta", "Background subagent beta declined the task."),
    ]


def test_compaction_request_is_only_the_last_block() -> None:
    history: list[ChatMessage] = [root_task(TASK), assistant("working")]
    assert is_compaction_request(
        conversation(*history, ChatMessageUser(content=COMPACTION_INSTRUCTION))
    )
    # dsh merges the instruction into a preceding user turn
    assert is_compaction_request(
        conversation(
            root_task(TASK),
            assistant("waiting"),
            ChatMessageUser(
                content=[
                    *notice("alpha", "done").content,
                    *blocks(COMPACTION_INSTRUCTION),
                ]
            ),
        )
    )
    # quoted earlier, in a task, or by a tool: an ordinary request
    assert not is_compaction_request(
        conversation(
            ChatMessageUser(content=COMPACTION_INSTRUCTION),
            assistant("That is dsh's compaction prompt."),
            ChatMessageUser(content="Thanks, carry on."),
        )
    )
    assert not is_compaction_request(
        conversation(root_task(f"Explain this prompt:\n{COMPACTION_INSTRUCTION}"))
    )
    assert not is_compaction_request(
        conversation(
            root_task(TASK),
            assistant("", [bash_call("c1", "cat summarizer.ts")]),
            tool_result("c1", COMPACTION_INSTRUCTION),
        )
    )


def test_tool_views_render_commands_and_prompts() -> None:
    bash = tool_view("bash", {"command": "ls", "description": "list"})
    assert bash is not None and "{{command}}" in bash.content
    delegated = tool_view("subagent", {"name": "alpha", "prompt": ALPHA})
    assert delegated is not None
    assert (delegated.title, delegated.content) == ("subagent: alpha", "{{prompt}}")
    assert tool_view("read", {"path": "a.py"}) is None


# ---------------------------------------------------------------------------
# 2. a background child: open at the call, closed by its settlement notice
# ---------------------------------------------------------------------------


def test_background_child_span_outlives_its_launch_result(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))

    # open before the child's first request, named after the child
    [begin] = stub.span_begins()
    assert (begin.id, begin.type, begin.name) == ("agent-c1", "agent", "alpha")
    assert begin.parent_id is None  # the root's (outer) span

    # the call's result is only a launch acknowledgement: the root's next
    # request carries it and the span stays open
    root = [*root, reply, started("c1", "alpha")]
    context, event = _generate(consumer, root, assistant("Waiting for alpha."))
    assert context == AgentBridgeContext("root")
    assert event.span_id is None
    assert stub.open_spans() == ["agent-c1"]

    child = conversation(child_task(ALPHA, "alpha"))
    context, event = _generate(
        consumer, child, assistant("", [bash_call("b1", "uname -a")])
    )
    assert context == AgentBridgeContext(
        "subagent", subagent_id="alpha", subagent_type="subagent"
    )
    assert event.span_id == "agent-c1"

    child = [
        *child,
        assistant("", [bash_call("b1", "uname -a")]),
        tool_result("b1", "Linux"),
    ]
    _, event = _generate(consumer, child, assistant("Linux"))
    assert event.span_id == "agent-c1"
    assert stub.open_spans() == ["agent-c1"]

    # the settlement notice in the root's history ends the run
    root = [*root, assistant("Waiting for alpha."), notice("alpha", "Linux")]
    context, event = _generate(consumer, root, assistant("alpha says Linux."))
    assert context == AgentBridgeContext("root")
    assert event.span_id is None
    assert stub.open_spans() == []

    # and stays in the history without closing anything again
    root = [*root, assistant("alpha says Linux."), ChatMessageUser(content="thanks")]
    _generate(consumer, root)
    assert len(stub.span_ends()) == 1


def test_child_request_before_the_parents_next_request(monkeypatch: Any) -> None:
    """The child's first request can beat the request carrying its launch result."""
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    context, event = _generate(consumer, conversation(child_task(ALPHA, "alpha")))
    assert context.subagent_id == "alpha"
    assert event.span_id == "agent-c1"


# ---------------------------------------------------------------------------
# 3. parallel children
# ---------------------------------------------------------------------------


def test_parallel_children_keep_their_own_spans(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(
        consumer,
        delegation_call("c1", ALPHA, "alpha"),
        delegation_call("c2", BETA, "beta"),
    )
    assert [e.id for e in stub.span_begins()] == ["agent-c1", "agent-c2"]

    alpha = conversation(child_task(ALPHA, "alpha"))
    beta = conversation(child_task(BETA, "beta"))
    # interleaved, beta first
    spans = [
        _generate(consumer, request)[1].span_id
        for request in (beta, alpha, beta, alpha)
    ]
    assert spans == ["agent-c2", "agent-c1", "agent-c2", "agent-c1"]

    # beta settles first: only its span closes
    root = [
        *root,
        reply,
        started("c1", "alpha"),
        started("c2", "beta"),
        assistant("Both are running."),
        notice("beta", "121"),
    ]
    _generate(consumer, root, assistant("beta is done."))
    assert stub.open_spans() == ["agent-c1"]
    assert _generate(consumer, alpha)[1].span_id == "agent-c1"


def test_unnamed_children_are_told_apart_by_their_prompt(monkeypatch: Any) -> None:
    """Without a `name` dsh picks `agent-N`, which only the call's result reports."""
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA), delegation_call("c2", BETA))
    # dsh numbered them in the other order
    context, event = _generate(consumer, conversation(child_task(BETA, "agent-1")))
    assert context == AgentBridgeContext(
        "subagent", subagent_id="agent-1", subagent_type="subagent"
    )
    assert event.span_id == "agent-c2"
    assert (
        _generate(consumer, conversation(child_task(ALPHA, "agent-2")))[1].span_id
        == "agent-c1"
    )


def test_unnamed_child_is_bound_by_its_launch_result(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(
        consumer, delegation_call("c1", ALPHA), delegation_call("c2", ALPHA)
    )
    # same prompt twice: the results say which call got which name
    _generate(
        consumer, [*root, reply, started("c1", "agent-2"), started("c2", "agent-1")]
    )
    assert (
        _generate(consumer, conversation(child_task(ALPHA, "agent-1")))[1].span_id
        == "agent-c2"
    )
    assert (
        _generate(consumer, conversation(child_task(ALPHA, "agent-2")))[1].span_id
        == "agent-c1"
    )


# ---------------------------------------------------------------------------
# 4. a resumed child
# ---------------------------------------------------------------------------


def test_resumed_child_continues_in_a_new_span(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))

    root = [
        *root,
        reply,
        started("c1", "alpha"),
        assistant("Waiting."),
        notice("alpha", "Linux"),
    ]
    send = assistant(
        "",
        [
            delegation_call("unrelated", "x", tool="send_message"),
        ],
    )
    _generate(consumer, root, send)
    assert stub.open_spans() == []

    # the parent's send_message restarts the settled child
    child = [*child, assistant("Linux"), agent_message("lead", "Also run `id -u`.")]
    context, event = _generate(consumer, child, assistant("0"))
    assert context == AgentBridgeContext(
        "subagent", subagent_id="alpha", subagent_type="subagent"
    )
    assert event.span_id == "agent-c1-resumed-1"
    continuation = stub.span_begins()[-1]
    assert (continuation.id, continuation.name) == ("agent-c1-resumed-1", "alpha")
    assert continuation.metadata is not None
    assert continuation.metadata["continues"] == "agent-c1"

    # the first notice is still in the root's history: only a second closes it
    root = [*root, send, tool_result("unrelated", "message delivered to agent alpha")]
    _generate(consumer, root, assistant("Waiting again."))
    assert stub.open_spans() == ["agent-c1-resumed-1"]

    root = [*root, assistant("Waiting again."), notice("alpha", "0")]
    _generate(consumer, root, assistant("All done."))
    assert stub.open_spans() == []


def test_message_to_a_running_child_stays_in_its_span(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("", [bash_call("b1", "uname -a")]))
    child = [
        *child,
        assistant("", [bash_call("b1", "uname -a")]),
        tool_result("b1", "Linux"),
        agent_message("lead", "Also run `id -u`."),
    ]
    assert _generate(consumer, child)[1].span_id == "agent-c1"
    assert len(stub.span_begins()) == 1


# ---------------------------------------------------------------------------
# 4b. each run's answer is recorded against its span
# ---------------------------------------------------------------------------


def test_each_run_gets_its_closing_message_as_its_result(monkeypatch: Any) -> None:
    """The launch result is an acknowledgement; the answer comes with the notice.

    It is recorded as the run's call (a tool event beside the span, naming it),
    which is where the timeline takes an agent span's result from.
    """
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))
    # nothing to record while the child runs, or from its launch result
    _generate(consumer, [*root, reply, started("c1", "alpha")], assistant("Waiting."))
    assert stub.results() == []

    root = [
        *root,
        reply,
        started("c1", "alpha"),
        assistant("Waiting."),
        notice("alpha", "Linux"),
    ]
    send = assistant("", [send_message_call("m1", "alpha", "Also run `id -u`.")])
    _generate(consumer, root, send)
    [first] = stub.results()
    assert (first.id, first.function, first.result) == ("c1", "subagent", "Linux")
    assert first.arguments["prompt"] == ALPHA
    assert (first.agent_span_id, first.span_id) == ("agent-c1", None)
    assert first.pending is None
    # recorded before the span closes, and once: the notice stays in history
    assert stub.events.index(first) < stub.events.index(stub.span_ends()[0])

    # the resumed run answers the send_message call
    _generate(
        consumer,
        [*child, assistant("Linux"), agent_message("lead", "Also run `id -u`.")],
        assistant("0"),
    )
    root = [*root, send, tool_result("m1", "message delivered to agent alpha")]
    _generate(consumer, root, assistant("Waiting again."))
    assert len(stub.results()) == 1

    _generate(
        consumer,
        [*root, assistant("Waiting again."), notice("alpha", "uid is 0")],
        assistant("All done."),
    )
    [_, second] = stub.results()
    assert (second.id, second.function, second.result) == (
        "m1",
        "send_message",
        "uid is 0",
    )
    assert second.arguments == {"agent": "alpha", "message": "Also run `id -u`."}
    assert second.agent_span_id == "agent-c1-resumed-1"


def test_result_sits_beside_a_nested_childs_span(monkeypatch: Any) -> None:
    """A child's child: the result goes in the span the nested span sits in."""
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    alpha = conversation(child_task(ALPHA, "alpha"))
    start = assistant("", [delegation_call("c2", "sub", "checker")])
    _generate(consumer, alpha, start)
    _generate(consumer, conversation(child_task("sub", "alpha/checker", "alpha")))
    _generate(
        consumer,
        [*alpha, start, started("c2", "alpha/checker"), notice("alpha/checker", "ok")],
    )
    [result] = stub.results()
    assert (result.agent_span_id, result.span_id) == ("agent-c2", "agent-c1")
    assert result.result == "ok"
    assert stub.open_spans() == ["agent-c1"]


def test_resumed_run_without_a_seen_message_still_gets_a_result(
    monkeypatch: Any,
) -> None:
    """A child a later dsh process resumes: its start and the message are history."""
    consumer, stub = _consumer(monkeypatch)
    child = conversation(
        child_task(ALPHA, "alpha"),
        assistant("Linux"),
        agent_message("lead", "Also run `id -u`."),
    )
    _, event = _generate(consumer, child, assistant("0"))
    _generate(
        consumer,
        conversation(root_task(TASK), assistant("w"), notice("alpha", "uid is 0")),
    )
    [result] = stub.results()
    assert result.agent_span_id == event.span_id
    assert (result.function, result.arguments) == ("send_message", {"agent": "alpha"})
    assert result.result == "uid is 0"


def test_run_still_going_at_reset_says_so(monkeypatch: Any) -> None:
    """Not the launch acknowledgement the timeline would otherwise show."""
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    _generate(
        consumer,
        conversation(child_task(ALPHA, "alpha")),
        assistant("", [bash_call("b1", "uname -a")]),
    )
    consumer.reset()
    assert stub.open_spans() == []
    [result] = stub.results()
    assert (result.id, result.agent_span_id) == ("c1", "agent-c1")
    assert result.result == consumer_module.NO_RESULT


def test_run_that_finished_unannounced_at_reset_gets_its_answer(
    monkeypatch: Any,
) -> None:
    """The child answered, but dsh exited before the parent heard of it."""
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    _generate(consumer, conversation(child_task(ALPHA, "alpha")), assistant("Linux"))
    consumer.reset()
    [result] = stub.results()
    assert (result.id, result.result) == ("c1", "Linux")


def test_child_restarted_before_its_parent_saw_the_notice(monkeypatch: Any) -> None:
    """The parent messages a child that has just finished, unaware of it.

    dsh restarts the child, so the child's own next request shows a new run
    before the parent's history shows the first one ended. The first run's
    span closes then, and each notice answers its own run, in order.
    """
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))  # run 1 ends
    root = [*root, reply, started("c1", "alpha")]
    send = assistant(
        "", [send_message_call("m1", "alpha", "id -u?"), bash_call("b1", "make")]
    )
    _generate(consumer, root, send)  # no notice in the parent's history yet

    restart = [*child, assistant("Linux"), agent_message("lead", "id -u?")]
    _, event = _generate(consumer, restart, assistant("", [bash_call("b2", "id -u")]))
    assert event.span_id == "agent-c1-resumed-1"
    assert stub.open_spans() == ["agent-c1-resumed-1"]
    # the same request again (a retry) is no new restart
    assert _generate(consumer, restart)[1].span_id == "agent-c1-resumed-1"
    assert stub.results() == []

    root = [
        *root,
        send,
        tool_result("m1", "message delivered to agent alpha", "send_message"),
        tool_result("b1", "ok"),
        notice("alpha", "Linux"),
    ]
    _generate(consumer, root, assistant("w"))
    [first] = stub.results()
    assert (first.id, first.result, first.agent_span_id) == ("c1", "Linux", "agent-c1")
    assert stub.open_spans() == ["agent-c1-resumed-1"]

    _generate(consumer, [*root, assistant("w"), notice("alpha", "uid 0")])
    [_, second] = stub.results()
    assert (second.id, second.function, second.result, second.agent_span_id) == (
        "m1",
        "send_message",
        "uid 0",
        "agent-c1-resumed-1",
    )
    assert stub.open_spans() == []


def test_two_notices_for_one_child_in_one_request(monkeypatch: Any) -> None:
    """Both runs end while the parent is busy: each notice goes to its own run."""
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))
    root = [*root, reply, started("c1", "alpha")]
    send = assistant(
        "", [send_message_call("m1", "alpha", "id -u?"), bash_call("b1", "sleep 60")]
    )
    _generate(consumer, root, send)
    _generate(
        consumer,
        [*child, assistant("Linux"), agent_message("lead", "id -u?")],
        assistant("uid 0"),
    )
    _generate(
        consumer,
        [
            *root,
            send,
            tool_result("m1", "message delivered to agent alpha", "send_message"),
            tool_result("b1", ""),
            notice("alpha", "Linux"),
            notice("alpha", "uid 0"),
        ],
    )
    assert [(r.id, r.result, r.agent_span_id) for r in stub.results()] == [
        ("c1", "Linux", "agent-c1"),
        ("m1", "uid 0", "agent-c1-resumed-1"),
    ]
    assert stub.open_spans() == []


def test_message_from_its_own_child_is_no_restart(monkeypatch: Any) -> None:
    """An agent waiting on a child of its own may answer without tool calls."""
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    alpha = conversation(child_task(ALPHA, "alpha"))
    start = assistant("", [delegation_call("c2", "sub", "checker")])
    _generate(consumer, alpha, start)
    alpha = [*alpha, start, started("c2", "alpha/checker")]
    _generate(consumer, alpha, assistant("Waiting."))
    alpha = [*alpha, assistant("Waiting.")]
    for sender in ("alpha/checker", "lead"):
        request = [*alpha, agent_message(sender, "progress")]
        assert _generate(consumer, request)[1].span_id == "agent-c1"
    assert stub.results() == []
    assert sorted(stub.open_spans()) == ["agent-c1", "agent-c2"]


def test_waiting_answer_then_its_childs_notice_and_a_message(
    monkeypatch: Any,
) -> None:
    """Its child's notice and its parent's message reach a waiting agent together.

    The notice closes the child's span before the request is placed, so
    whether the agent was waiting is decided when it wrote the answer.
    """
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    alpha = conversation(child_task(ALPHA, "alpha"))
    start = assistant("", [delegation_call("c2", "sub", "checker")])
    _generate(consumer, alpha, start)
    alpha = [*alpha, start, started("c2", "alpha/checker")]
    _generate(consumer, alpha, assistant("Waiting for checker."))
    _generate(consumer, conversation(child_task("sub", "alpha/checker", "alpha")))
    request = [
        *alpha,
        assistant("Waiting for checker."),
        ChatMessageUser(
            content=[
                *notice("alpha/checker", "ok").content,
                *agent_message("lead", "also check disk").content,
            ]
        ),
    ]
    assert _generate(consumer, request)[1].span_id == "agent-c1"
    assert [(r.id, r.result) for r in stub.results()] == [("c2", "ok")]
    assert stub.open_spans() == ["agent-c1"]


# ---------------------------------------------------------------------------
# 5. a forked child
# ---------------------------------------------------------------------------


def test_forked_child_is_not_taken_for_its_parent(monkeypatch: Any) -> None:
    """A fork's requests open with the root's own turns, settlement notices included."""
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    _generate(consumer, conversation(child_task(ALPHA, "alpha")), assistant("Linux"))
    root = [
        *root,
        reply,
        started("c1", "alpha"),
        assistant("Waiting."),
        notice("alpha", "Linux"),
    ]
    resend = assistant("", [delegation_call("m1", "x", tool="send_message")])
    _generate(consumer, root, resend)
    root = [*root, resend, tool_result("m1", "message delivered to agent alpha")]
    fork = assistant(
        "", [delegation_call("c2", "Report the year.", "gamma", "subagent_fork")]
    )
    _generate(consumer, root, fork)

    # alpha is running again when the fork, which inherited alpha's first
    # notice with the root's history, makes its request
    alpha = conversation(
        child_task(ALPHA, "alpha"),
        assistant("Linux"),
        agent_message("lead", "Also run `id -u`."),
    )
    assert _generate(consumer, alpha)[1].span_id == "agent-c1-resumed-1"

    gamma = [*root[:-2], fork_task("Report the year.", "gamma")]
    context, event = _generate(consumer, gamma, assistant("2026"))
    assert context == AgentBridgeContext(
        "subagent", subagent_id="gamma", subagent_type="subagent_fork"
    )
    assert event.span_id == "agent-c2"
    # the inherited notice is the root's, not news about alpha
    assert sorted(stub.open_spans()) == ["agent-c1-resumed-1", "agent-c2"]


# ---------------------------------------------------------------------------
# 6. starts dsh rejects, and children whose start was never seen
# ---------------------------------------------------------------------------


def test_rejected_start_closes_its_span(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    _generate(
        consumer, [*root, reply, start_rejected("c1", "ACTIVATION_LIMIT_REACHED")]
    )
    assert stub.open_spans() == []
    # nothing is left to attribute to
    assert consumer.classify(MODEL, conversation(root_task(TASK)), []).kind == "root"


def test_duplicate_name_start_does_not_take_the_name(monkeypatch: Any) -> None:
    """Dsh rejects a second child named like a live one; the first keeps its span."""
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    again = assistant("", [delegation_call("c2", BETA, "alpha")])
    root = [*root, reply, started("c1", "alpha")]
    _generate(consumer, root, again)

    assert _generate(consumer, conversation(child_task(ALPHA, "alpha")))[1].span_id == (
        "agent-c1"
    )
    _generate(consumer, [*root, again, start_rejected("c2", "DUPLICATE_AGENT_NAME")])
    assert stub.open_spans() == ["agent-c1"]


def test_child_whose_start_was_not_seen_gets_a_span(monkeypatch: Any) -> None:
    """A `workflow` script's child, or one resumed in a later dsh process."""
    consumer, stub = _consumer(monkeypatch)
    _generate(consumer, conversation(root_task(TASK)))
    context, event = _generate(consumer, conversation(child_task(ALPHA, "agent-7")))
    assert context == AgentBridgeContext("subagent", subagent_id="agent-7")
    [begin] = stub.span_begins()
    assert (begin.type, begin.name) == ("agent", "agent-7")
    assert begin.id.startswith("agent-agent-7-")
    assert event.span_id == begin.id
    # the same child again: same span
    assert (
        _generate(consumer, conversation(child_task(ALPHA, "agent-7")))[1].span_id
        == begin.id
    )
    assert len(stub.span_begins()) == 1


def test_unseen_child_gets_a_new_span_id_in_each_dsh_process(monkeypatch: Any) -> None:
    """Span ids are unique in a transcript; the same child returns turn after turn."""
    consumer, stub = _consumer(monkeypatch)
    request = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, request)
    consumer.reset()
    _generate(consumer, request)
    consumer.reset()
    # a later turn of the sample runs under a new consumer
    other = DshConsumer()
    _generate(other, request)
    other.reset()

    ids = [e.id for e in stub.span_begins()]
    assert len(ids) == len(set(ids)) == 3
    assert sorted(ids) == sorted(e.id for e in stub.span_ends())


def test_request_without_a_reminder(monkeypatch: Any) -> None:
    """Root while nothing was delegated; unknown once a sub-agent could be the sender."""
    consumer, stub = _consumer(monkeypatch)
    bare = conversation(ChatMessageUser(content=TASK))
    assert consumer.classify(MODEL, bare, []) == AgentBridgeContext("root")
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    assert consumer.classify(MODEL, bare, []) == AgentBridgeContext("unknown")


# ---------------------------------------------------------------------------
# 7. compaction
# ---------------------------------------------------------------------------


def test_compaction_is_marked_in_the_compacting_agents_span(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))

    child = conversation(
        child_task(ALPHA, "alpha"),
        assistant("", [bash_call("b1", "uname -a")]),
        tool_result("b1", "Linux"),
        ChatMessageUser(content=COMPACTION_INSTRUCTION),
    )
    context, event = _generate(consumer, child, assistant("## Primary Request"))
    assert context == AgentBridgeContext("utility")
    assert event.span_id == "agent-c1"
    [marker] = [e for e in stub.events if isinstance(e, CompactionEvent)]
    assert (marker.source, marker.span_id) == ("deepseek_harness", "agent-c1")

    root = conversation(
        root_task(TASK),
        assistant("working"),
        ChatMessageUser(content=COMPACTION_INSTRUCTION),
    )
    context, event = _generate(consumer, root, assistant("## Primary Request"))
    assert context == AgentBridgeContext("utility")
    assert event.span_id is None
    assert [e.span_id for e in stub.events if isinstance(e, CompactionEvent)] == [
        "agent-c1",
        None,
    ]


def test_compaction_of_a_span_without_a_reminder(monkeypatch: Any) -> None:
    """A later compaction condenses a span that starts at the last checkpoint.

    It carries no session reminder, so it names nobody: still "utility", in
    the root's span, and not mistaken for a request of unknown origin (which
    would drop the root's latest generation).
    """
    consumer, stub = _consumer(monkeypatch)
    _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    answer = assistant("Still working.")
    _generate(consumer, conversation(root_task(TASK), assistant("w")), answer)
    assert consumer.root_generation is not None

    request = conversation(
        ChatMessageUser(
            content=[
                *checkpoint("## Current Work\n- step 3").content,
                *blocks(COMPACTION_INSTRUCTION),
            ]
        )
    )
    assert requester_identity(request) is None
    context, event = _generate(consumer, request, assistant("## Current Work"))
    assert context == AgentBridgeContext("utility")
    assert event.span_id is None
    assert consumer.root_generation is not None
    assert consumer.root_generation.output.message is answer


def test_compaction_request_does_not_close_a_running_child(monkeypatch: Any) -> None:
    """The condensed span is an older part of the history, short of its notices.

    The root holds a notice for alpha, resumes it, then compacts a span that
    ends before that notice. The request after it still carries the notice,
    which must not look new and close the running child's span.
    """
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))
    root = [
        *root,
        reply,
        started("c1", "alpha"),
        assistant("Waiting."),
        notice("alpha", "Linux"),
    ]
    _generate(consumer, root, assistant("resuming alpha"))
    _generate(consumer, [*child, assistant("Linux"), agent_message("lead", "more")])
    assert stub.open_spans() == ["agent-c1-resumed-1"]

    # the root compacts everything up to (not including) the notice
    context, _ = _generate(
        consumer,
        [*root[:4], ChatMessageUser(content=COMPACTION_INSTRUCTION)],
        assistant("## Current Work"),
    )
    assert context == AgentBridgeContext("utility")
    _generate(consumer, [*root, assistant("resuming alpha")], assistant("waiting"))
    assert stub.open_spans() == ["agent-c1-resumed-1"]
    assert [e.id for e in stub.span_begins()] == ["agent-c1", "agent-c1-resumed-1"]


def test_compaction_by_a_settled_child_does_not_reopen_its_span(
    monkeypatch: Any,
) -> None:
    """A compaction request is placed, never used to open or resume a span."""
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))
    _generate(
        consumer,
        [*root, reply, started("c1", "alpha"), assistant("w"), notice("alpha", "ok")],
    )
    assert stub.open_spans() == []

    for name in ("alpha", "never-seen"):
        request = conversation(
            child_task(ALPHA, name), ChatMessageUser(content=COMPACTION_INSTRUCTION)
        )
        context, event = _generate(consumer, request, assistant("## Current Work"))
        assert context == AgentBridgeContext("utility")
        assert event.span_id is None
    assert [e.id for e in stub.span_begins()] == ["agent-c1"]


def test_compacted_history_keeps_closed_children_closed(monkeypatch: Any) -> None:
    """Compaction drops old notices; a later one for the same child is still new."""
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))
    _generate(
        consumer,
        [
            *root,
            reply,
            started("c1", "alpha"),
            assistant("w"),
            notice("alpha", "Linux"),
        ],
    )
    assert stub.open_spans() == []

    # the root's history now continues from a checkpoint, with a fresh
    # reminder appended: no notice left
    compacted = conversation(
        checkpoint("## Current Work\n- alpha reported Linux"),
        assistant("", [bash_call("b9", "true")]),
        tool_result("b9", ""),
        ChatMessageUser(content=blocks(root_reminder(replaces=True))),
    )
    assert _generate(consumer, compacted)[0] == AgentBridgeContext("root")

    # alpha is resumed and settles again
    _generate(consumer, [*child, assistant("Linux"), agent_message("lead", "again")])
    assert stub.open_spans() == ["agent-c1-resumed-1"]
    _generate(consumer, [*compacted, assistant("ok"), notice("alpha", "0")])
    assert stub.open_spans() == []


def _alpha_resumed_after_one_run(
    consumer: DshConsumer,
) -> list[ChatMessage]:
    """The state after alpha's first run settled ("Linux") and it was resumed."""
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))
    root = [
        *root,
        reply,
        started("c1", "alpha"),
        assistant("w"),
        notice("alpha", "Linux"),
    ]
    send = assistant("", [send_message_call("m1", "alpha", "id -u?")])
    _generate(consumer, root, send)
    _generate(
        consumer,
        [*child, assistant("Linux"), agent_message("lead", "id -u?")],
        assistant("", [bash_call("b1", "id -u")]),
    )
    return [*root, send, tool_result("m1", "delivered", "send_message")]


def test_compaction_keeping_only_the_new_notice(monkeypatch: Any) -> None:
    """The old notice is condensed away and the new one kept: as many as before."""
    consumer, stub = _consumer(monkeypatch)
    _alpha_resumed_after_one_run(consumer)
    assert stub.open_spans() == ["agent-c1-resumed-1"]

    compacted = conversation(
        checkpoint("## Current Work\n- alpha reported Linux; asked for id -u"),
        assistant("waiting"),
        notice("alpha", "uid 0"),
        ChatMessageUser(content=blocks(root_reminder(replaces=True))),
    )
    _generate(consumer, compacted, assistant("done"))
    assert [(r.id, r.result) for r in stub.results()] == [
        ("c1", "Linux"),
        ("m1", "uid 0"),
    ]
    assert stub.open_spans() == []


def test_notice_seen_only_in_a_compaction_request(monkeypatch: Any) -> None:
    """The condensed span holds the only copy of a new notice."""
    consumer, stub = _consumer(monkeypatch)
    root = _alpha_resumed_after_one_run(consumer)
    _generate(
        consumer,
        [
            *root,
            assistant("w"),
            notice("alpha", "uid 0"),
            ChatMessageUser(content=COMPACTION_INSTRUCTION),
        ],
        assistant("## Current Work"),
    )
    assert [(r.id, r.result) for r in stub.results()] == [
        ("c1", "Linux"),
        ("m1", "uid 0"),
    ]
    assert stub.open_spans() == []
    # the history after it, without either notice, changes nothing
    _generate(
        consumer,
        conversation(
            checkpoint("## Current Work\n- alpha said uid 0"),
            ChatMessageUser(content=blocks(root_reminder(replaces=True))),
        ),
    )
    assert len(stub.results()) == 2


# ---------------------------------------------------------------------------
# 8. reset() balance, on_start and its fallback
# ---------------------------------------------------------------------------


def test_reset_closes_every_open_span_innermost_first(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    _root_starts(
        consumer,
        delegation_call("c1", ALPHA, "alpha"),
        delegation_call("c2", BETA, "beta"),
    )
    # a nested child, opened under alpha's span
    alpha = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, alpha, assistant("", [delegation_call("c3", "sub", "checker")]))
    nested = stub.span_begins()[-1]
    assert (nested.id, nested.parent_id) == ("agent-c3", "agent-c1")
    assert (
        _generate(consumer, conversation(child_task("sub", "alpha/checker", "alpha")))[
            1
        ].span_id
        == "agent-c3"
    )

    consumer.reset()
    assert [e.id for e in stub.span_ends()] == ["agent-c3", "agent-c2", "agent-c1"]
    assert stub.open_spans() == []

    # a second reset, and a fresh attempt, add nothing
    consumer.reset()
    assert len(stub.span_ends()) == 3
    assert consumer.classify(MODEL, conversation(root_task(TASK)), []).kind == "root"
    assert consumer.root_generation is None


def test_reset_after_a_resumed_child_keeps_spans_balanced(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    root, reply = _root_starts(consumer, delegation_call("c1", ALPHA, "alpha"))
    child = conversation(child_task(ALPHA, "alpha"))
    _generate(consumer, child, assistant("Linux"))
    _generate(
        consumer,
        [
            *root,
            reply,
            started("c1", "alpha"),
            assistant("w"),
            notice("alpha", "Linux"),
        ],
    )
    _generate(consumer, [*child, assistant("Linux"), agent_message("lead", "more")])
    consumer.reset()
    assert [e.id for e in stub.span_begins()] == ["agent-c1", "agent-c1-resumed-1"]
    assert [e.id for e in stub.span_ends()] == ["agent-c1", "agent-c1-resumed-1"]


# ---------------------------------------------------------------------------
# 8b. a generation that outlives its dsh process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_generation_finishing_after_reset_starts_nothing() -> None:
    """Dsh can exit with a request still in flight at the bridge.

    Its event is completed in the transcript, but the delegation in its reply
    reaches no dsh process: it must not open a span (which, after the last
    reset, nothing would close) or become the root's latest generation.
    """
    init_transcript(Transcript())
    consumer = DshConsumer()

    def reply_after_dsh_exited(*_: Any) -> ModelOutput:
        consumer.reset()
        return ModelOutput.from_message(
            assistant("", [delegation_call("c1", ALPHA, "alpha")])
        )

    model = get_model("mockllm/model", custom_outputs=reply_after_dsh_exited)
    bridge = _bridge(consumer)
    async with span("outer", type="agent"):
        outer = current_span_id()
        await bridge_generate(
            bridge,
            model,
            conversation(root_task(TASK)),
            [],
            None,
            GenerateConfig(),
            SLUG,
        )
        consumer.reset()

    events = transcript().events
    [event] = [e for e in events if isinstance(e, ModelEvent)]
    assert event.pending is None and event.output.message.tool_calls
    assert [e.id for e in events if isinstance(e, SpanBeginEvent)] == [outer]
    assert consumer.root_generation is None


@pytest.mark.parametrize("on_start", [True, False])
def test_on_start_and_the_fallback_place_events_alike(
    monkeypatch: Any, on_start: bool
) -> None:
    """With `on_start` the generate runs under its span; without, `on_pending` places it."""
    consumer, stub = _consumer(monkeypatch)
    request = conversation(root_task(TASK))
    reply = assistant("", [delegation_call("c1", ALPHA, "alpha")])
    _, event = _generate(consumer, request, reply, on_start=on_start)
    assert event.span_id is None

    child = conversation(child_task(ALPHA, "alpha"))
    if on_start:
        assert consumer.on_start(child) == "agent-c1"
    _, event = _generate(consumer, child, on_start=on_start)
    assert event.span_id == "agent-c1"
    assert [e for e in stub.events if isinstance(e, ModelEvent)][-1] is event


# ---------------------------------------------------------------------------
# 9. through the real bridge: one decision for filter and transcript
# ---------------------------------------------------------------------------


def _bridge(consumer: DshConsumer, user_filter: Any = None) -> AgentBridge:
    return AgentBridge(
        AgentState(messages=[]),
        filter=classify_filter(
            user_filter, consumer.classify, consumer.on_filter_output, consumer
        ),
        model_event_sink=consumer,
    )


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_filter_rewrite_cannot_move_a_childs_generation() -> None:
    """A filter that replaces the messages leaves the event in the child's span."""
    init_transcript(Transcript())
    consumer = DshConsumer()

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        if requester_identity(messages) == Identity("alpha", "lead"):
            return GenerateInput(
                input=[ChatMessageUser(content="summarized")],
                tools=tools,
                tool_choice=tool_choice,
                config=config,
            )
        return None

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_message(
                assistant("", [delegation_call("c1", ALPHA, "alpha")])
            ),
            ModelOutput.from_content("mockllm/model", "Linux"),
        ],
    )
    bridge = _bridge(consumer, user_filter)
    async with span("outer", type="agent"):
        outer = current_span_id()
        for request in (
            conversation(root_task(TASK)),
            conversation(child_task(ALPHA, "alpha")),
        ):
            await bridge_generate(
                bridge, model, request, [], None, GenerateConfig(), SLUG
            )
        consumer.reset()

    events = transcript().events
    [root_event, child_event] = [e for e in events if isinstance(e, ModelEvent)]
    assert root_event.span_id == outer
    assert child_event.span_id == "agent-c1"
    assert child_event.input[-1].text == "summarized"
    [begin] = [
        e
        for e in events
        if isinstance(e, SpanBeginEvent) and e.type == "agent" and e.id != outer
    ]
    assert (begin.id, begin.parent_id) == ("agent-c1", outer)


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_filter_returned_delegation_opens_a_span() -> None:
    """An output the user's filter returns never reaches the sink as an event."""
    init_transcript(Transcript())
    consumer = DshConsumer()

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        if requester_identity(messages) == Identity("lead", None):
            return ModelOutput.from_message(
                assistant("", [delegation_call("c1", ALPHA, "alpha")])
            )
        return None

    bridge = _bridge(consumer, user_filter)
    async with span("outer", type="agent"):
        for request in (
            conversation(root_task(TASK)),
            conversation(child_task(ALPHA, "alpha")),
        ):
            await bridge_generate(
                bridge, MODEL, request, [], None, GenerateConfig(), SLUG
            )
        root = consumer.root_generation
        consumer.reset()

    [child_event] = [e for e in transcript().events if isinstance(e, ModelEvent)]
    assert child_event.span_id == "agent-c1"
    # the filter's output is the root's generation all the same
    assert root is not None and root.output.message.tool_calls


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_root_generation_keeps_the_conversation_the_bridge_received() -> None:
    """A filter's rewrite changes what the model is sent, not the sample's conversation."""
    init_transcript(Transcript())
    consumer = DshConsumer()

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        return GenerateInput(
            input=[ChatMessageUser(content="summarized")],
            tools=tools,
            tool_choice=tool_choice,
            config=config,
        )

    bridge = _bridge(consumer, user_filter)
    request = conversation(root_task(TASK))
    async with span("outer", type="agent"):
        await bridge_generate(bridge, MODEL, request, [], None, GenerateConfig(), SLUG)
        root = consumer.root_generation
        consumer.reset()

    [event] = [e for e in transcript().events if isinstance(e, ModelEvent)]
    assert event.input[-1].text == "summarized"
    assert root is not None
    assert [m.text for m in root.messages] == [m.text for m in request]
