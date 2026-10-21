"""LiveConsumer sub-agent spans from bridge traffic alone (no stream-json).

The ACP Claude Code agent has no JSONL output, so spans must close from what
the parent sees in its next request: the Task/Agent tool result (foreground)
or a `<task-notification>` (background). Message text is from real Claude
Code 2.1.303 runs.
"""

import pytest
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.live_consumer import (
    BridgeOnlyLiveConsumer,
    LiveConsumer,
)

PROMPT = "Run the command `wc -l /etc/passwd` and report the number of lines."
MAIN = "Count lines in /etc/passwd using a sub-agent."
AGENT_ID = "a66bab0b3f5f438a5"

ACK = (
    "Async agent launched successfully. (This tool result is internal metadata "
    "— never quote or paste any part of it, including the agentId below, into a "
    "user-facing reply.)\n"
    f"agentId: {AGENT_ID} (internal ID - do not mention to user. Use SendMessage "
    f"with to: '{AGENT_ID}', summary: '<5-10 word recap>' to continue this agent.)\n"
    "The agent is working in the background. You will be notified automatically "
    "when it completes."
)


def _notification(tool_use_id: str | None) -> str:
    tool_use = f"<tool-use-id>{tool_use_id}</tool-use-id>\n" if tool_use_id else ""
    return (
        "<system-reminder>\n<task-notification>\n"
        f"<task-id>{AGENT_ID}</task-id>\n{tool_use}"
        f"<output-file>/tmp/claude-0/tasks/{AGENT_ID}.output</output-file>\n"
        "<status>completed</status>\n"
        '<summary>Agent "Count lines" finished</summary>\n'
        "<result>21 lines.</result>\n"
        "</task-notification>\n</system-reminder>"
    )


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[object] = []

    def _event(self, event: object) -> None:
        self.events.append(event)

    def _event_updated(self, event: object) -> None:
        pass

    def begins(self) -> list[str]:
        return [e.id for e in self.events if isinstance(e, SpanBeginEvent)]

    def ends(self) -> list[str]:
        return [e.id for e in self.events if isinstance(e, SpanEndEvent)]


@pytest.fixture
def setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LiveConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    return BridgeOnlyLiveConsumer(), stub


def _call(
    consumer: LiveConsumer,
    input: list[ChatMessage],
    tool_calls: list[ToolCall] | None = None,
    final: bool = False,
) -> str | None:
    # a mid-run call ends in a tool call; a run's final reply is text only
    if tool_calls is None and not final:
        tool_calls = [ToolCall(id="toolu_bash", function="Bash", arguments={})]
    event = ModelEvent(
        model="anthropic/claude-haiku-4-5",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(
            ChatMessageAssistant(content="ok", tool_calls=tool_calls)
        ),
    )
    consumer.on_pending(event)
    consumer.on_complete(event)
    return event.span_id


LAUNCH = ChatMessageAssistant(
    content="",
    tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
)


def _launch(consumer: LiveConsumer, subagent_type: str = "general-purpose") -> None:
    args = {"prompt": PROMPT, "subagent_type": subagent_type}
    _call(
        consumer,
        [ChatMessageUser(content=MAIN)],
        [ToolCall(id="toolu_1", function="Agent", arguments=args)],
    )


def _parent(consumer: LiveConsumer, *tail: ChatMessage) -> str | None:
    return _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *tail])


def _subagent(consumer: LiveConsumer) -> str | None:
    return _call(consumer, [ChatMessageUser(content=PROMPT)])


def test_foreground_closes_on_tool_result(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    assert _subagent(consumer) == "agent-toolu_1"
    _parent(consumer, ChatMessageTool(content="21 lines.", tool_call_id="toolu_1"))
    assert stub.ends() == ["agent-toolu_1"]


def test_background_closes_on_notification(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    assert _parent(consumer, ack) == "outer"
    assert stub.ends() == []
    assert _subagent(consumer) == "agent-toolu_1"

    waiting = ChatMessageAssistant(content="Waiting for the agent.")
    _parent(consumer, ack, waiting, ChatMessageUser(content=_notification("toolu_1")))
    assert stub.ends() == ["agent-toolu_1"]


def test_resumed_run_closes_by_task_id_and_ignores_old_notification(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    first_done = ChatMessageUser(content=_notification("toolu_1"))
    _parent(consumer, ack, ChatMessageAssistant(content="waiting"), first_done)

    send = [
        ToolCall(id="toolu_send", function="SendMessage", arguments={"to": AGENT_ID})
    ]
    history: list[ChatMessage] = [
        ack,
        ChatMessageAssistant(content="waiting"),
        first_done,
    ]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history], send)
    assert stub.begins()[-1] == "agent-toolu_1-run2"

    # the old notification is still in the parent's history; it must not
    # close the resumed run
    resuming = ChatMessageTool(content="Resuming agent", tool_call_id="toolu_send")
    history += [ChatMessageAssistant(content="", tool_calls=send), resuming]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history])
    assert stub.ends() == ["agent-toolu_1"]

    # the resumed run's notification may carry only the task id
    history += [
        ChatMessageAssistant(content="waiting again"),
        ChatMessageUser(content=_notification(None)),
    ]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history])
    assert stub.ends() == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_fork_started_result_does_not_close(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer, "fork")
    fork_input: list[ChatMessage] = [
        ChatMessageUser(content=MAIN),
        LAUNCH,
        ChatMessageTool(
            content="Fork started — processing in background", tool_call_id="toolu_1"
        ),
        ChatMessageUser(content=f"Your directive: {PROMPT}"),
    ]
    assert _call(consumer, fork_input) == "agent-toolu_1"
    assert stub.ends() == []


def _send(consumer: LiveConsumer, history: list[ChatMessage], send_id: str) -> None:
    send = [ToolCall(id=send_id, function="SendMessage", arguments={"to": AGENT_ID})]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history], send)


def test_resumed_run_closes_on_its_own_tool_use_id(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    # real shape: the resumed run's notification names the SendMessage id
    consumer, stub = setup
    _launch(consumer)
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    first_done = ChatMessageUser(content=_notification("toolu_1"))
    history: list[ChatMessage] = [ack, ChatMessageAssistant(content="w"), first_done]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history])
    _send(consumer, history, "toolu_send")
    history += [
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(id="toolu_send", function="SendMessage", arguments={})
            ],
        ),
        ChatMessageTool(content="Resuming agent", tool_call_id="toolu_send"),
        ChatMessageAssistant(content="waiting again"),
        ChatMessageUser(content=_notification("toolu_send")),
    ]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *history])
    assert stub.ends() == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_stale_notification_after_rollover_keeps_new_run(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, ack])
    _send(consumer, [ack], "toolu_send")  # queued: run 1 still open
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21 lines."),
        ChatMessageUser(content="The coordinator sent a message: more"),
    ]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"
    # run 1's notification reaches the parent after the rollover
    late: list[ChatMessage] = [ack, ChatMessageAssistant(content="w")]
    _call(
        consumer,
        [
            ChatMessageUser(content=MAIN),
            LAUNCH,
            *late,
            ChatMessageUser(content=_notification("toolu_1")),
        ],
    )
    assert stub.ends() == ["agent-toolu_1"]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"


def test_cli_consumer_does_not_close_from_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    consumer = LiveConsumer()
    _launch(consumer)
    _parent(consumer, ChatMessageTool(content="21 lines.", tool_call_id="toolu_1"))
    assert stub.ends() == []


def test_final_reply_closes_background_run_without_parent_request(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    _parent(consumer, ChatMessageTool(content=ACK, tool_call_id="toolu_1"))
    assert _subagent(consumer) == "agent-toolu_1"
    final = _call(consumer, [ChatMessageUser(content=PROMPT)], final=True)
    assert final == "agent-toolu_1"
    assert stub.ends() == ["agent-toolu_1"]


def test_resume_after_final_reply_opens_next_run(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    _call(consumer, [ChatMessageUser(content=PROMPT)], final=True)
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21 lines."),
        ChatMessageUser(content="The coordinator sent a message: more"),
    ]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"


def test_parent_final_reply_closes_nothing(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH], final=True)
    assert stub.ends() == []


def _final(
    consumer: LiveConsumer, input: list[ChatMessage], stop_reason: str = "stop"
) -> str | None:
    event = ModelEvent(
        model="anthropic/claude-haiku-4-5",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(
            ChatMessageAssistant(content="done"),
            stop_reason=stop_reason,  # type: ignore[arg-type]
        ),
    )
    consumer.on_pending(event)
    consumer.on_complete(event)
    return event.span_id


def _background(consumer: LiveConsumer) -> ChatMessageTool:
    _launch(consumer)
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    _parent(consumer, ack)
    return ack


@pytest.mark.parametrize(
    "continuation",
    [
        "Stop hook feedback:\nnot done yet",
        "Your task is to create a detailed summary of the conversation so far",
    ],
)
def test_run_continued_after_its_final_reply_stays_in_its_span(
    setup: tuple[LiveConsumer, _TranscriptStub], continuation: str
) -> None:
    consumer, stub = setup
    _background(consumer)
    history: list[ChatMessage] = [ChatMessageUser(content=PROMPT)]
    assert _final(consumer, history) == "agent-toolu_1"
    history += [
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content=continuation),
    ]
    assert _call(consumer, history) == "agent-toolu_1"
    assert stub.begins() == ["agent-toolu_1"]


def test_compaction_summary_reply_does_not_close(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _background(consumer)
    summary_request: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageUser(
            content="Your task is to create a detailed summary of the "
            "conversation so far, paying close attention to the user's requests"
        ),
    ]
    assert _final(consumer, summary_request) == "agent-toolu_1"
    assert stub.ends() == []


def test_notification_after_reply_close_does_not_close_next_run(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    # 2.1.303 lets a sub-agent end its turn while its own children run, and
    # resume when they finish; run 1's notification can then arrive in run 2
    consumer, stub = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="waiting on my agents"),
        ChatMessageUser(
            content=_notification("toolu_child").replace(AGENT_ID, "a-child")
        ),
    ]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"
    _parent(
        consumer,
        ack,
        ChatMessageAssistant(content="w"),
        ChatMessageUser(content=_notification("toolu_1")),
    )
    assert stub.ends() == ["agent-toolu_1"]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"


def test_notification_after_reply_close_is_not_a_second_end(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    _parent(
        consumer,
        ack,
        ChatMessageAssistant(content="w"),
        ChatMessageUser(content=_notification("toolu_1")),
    )
    assert stub.ends() == ["agent-toolu_1"]


@pytest.mark.parametrize("stop_reason", ["max_tokens", "content_filter", "unknown"])
def test_text_reply_that_is_not_an_end_turn_does_not_close(
    setup: tuple[LiveConsumer, _TranscriptStub], stop_reason: str
) -> None:
    consumer, stub = setup
    _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)], stop_reason)
    assert stub.ends() == []


def test_cli_consumer_does_not_close_on_final_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    consumer = LiveConsumer()
    _launch(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    assert stub.ends() == []


def test_foreground_result_after_reply_close_is_noop(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    _parent(consumer, ChatMessageTool(content="21 lines.", tool_call_id="toolu_1"))
    assert stub.ends() == ["agent-toolu_1"]


def test_nested_final_reply_closes_only_its_own_span(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _launch(consumer)
    child_prompt = "Count the groups in /etc/group and report the number."
    _call(
        consumer,
        [ChatMessageUser(content=PROMPT)],
        [ToolCall(id="toolu_2", function="Agent", arguments={"prompt": child_prompt})],
    )
    assert stub.begins()[-1] == "agent-toolu_2"
    _final(consumer, [ChatMessageUser(content=child_prompt)])
    assert stub.ends() == ["agent-toolu_2"]


# ---------------------------------------------------------------------------
# same prompts, late acknowledgements, retries and failed resumes
# ---------------------------------------------------------------------------


def test_same_prompt_agents_do_not_close_each_other(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    args = {"prompt": PROMPT, "subagent_type": "general-purpose"}
    _call(
        consumer,
        [ChatMessageUser(content=MAIN)],
        [
            ToolCall(id="toolu_1", function="Agent", arguments=args),
            ToolCall(id="toolu_2", function="Agent", arguments=args),
        ],
    )
    # identical first calls: whichever replies first is the first agent
    assert _final(consumer, [ChatMessageUser(content=PROMPT)]) == "agent-toolu_1"
    assert stub.ends() == ["agent-toolu_1"]
    assert _subagent(consumer) == "agent-toolu_2"
    assert stub.ends() == ["agent-toolu_1"]


def test_late_ack_after_reply_close_does_not_close_next_run(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    # the parent's request carrying the ack waits on a slow Bash call while
    # the agent finishes run 1 and is resumed by its own child's notification
    consumer, stub = setup
    _launch(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="waiting on my agents"),
        ChatMessageUser(content=_notification("toolu_c").replace(AGENT_ID, "a-c")),
    ]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"
    ack = ChatMessageTool(content=ACK, tool_call_id="toolu_1")
    _parent(consumer, ack)
    _parent(
        consumer,
        ack,
        ChatMessageAssistant(content="w"),
        ChatMessageUser(content=_notification("toolu_1")),
    )
    assert stub.ends() == ["agent-toolu_1"]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"


def test_retried_parent_request_handles_notification_once(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    run2: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="child completed"),
    ]
    assert _call(consumer, run2) == "agent-toolu_1-run2"
    parent: list[ChatMessage] = [
        ChatMessageUser(content=MAIN),
        LAUNCH,
        ack,
        ChatMessageAssistant(content="waiting"),
        ChatMessageUser(content=_notification("toolu_1")),
    ]
    _final(consumer, parent, stop_reason="content_filter")
    # retry_refusals repeats the request unchanged
    _call(consumer, parent)
    assert stub.ends() == ["agent-toolu_1"]


def test_failed_send_message_starts_no_run(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    history: list[ChatMessage] = [
        ack,
        ChatMessageAssistant(content="waiting"),
        ChatMessageUser(content=_notification("toolu_1")),
    ]
    _send(consumer, history, "s1")
    history += [
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="s1", function="SendMessage", arguments={})],
        ),
        ChatMessageTool(content="Agent not found", tool_call_id="s1"),
    ]
    _send(consumer, history, "s2")
    run2: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    # (its last run ended with its reply, so the sends wait for a run to
    # show up rather than open one)
    assert _final(consumer, run2) == "agent-toolu_1-run2"
    run3 = run2 + [ChatMessageAssistant(content="done"), ChatMessageUser(content="m")]
    span = _call(consumer, run3)
    assert span == "agent-toolu_1-run3"
    assert span not in stub.ends()


def test_same_message_at_same_index_after_compaction_is_handled(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    # run 2's notification can read exactly like run 1's, and land at the
    # same index once the parent has compacted
    consumer, stub = setup
    ack = _background(consumer)
    note = ChatMessageUser(content=_notification("toolu_1"))
    system = ChatMessageSystem(content="You are Claude Code.")
    first: list[ChatMessage] = [system, ChatMessageUser(content=MAIN), LAUNCH, ack]
    _call(consumer, [*first, ChatMessageAssistant(content="w"), note])
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21 lines."),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, resumed) == "agent-toolu_1-run2"
    compacted: list[ChatMessage] = [
        ChatMessageUser(
            content="This session is being continued from a previous "
            "conversation that ran out of context."
        ),
        ChatMessageAssistant(content="w"),
        ChatMessageUser(content="go on"),
        ChatMessageAssistant(content="w"),
        note,
    ]
    # the bridge puts the system prompt first, unchanged by compaction
    _call(consumer, [ChatMessageSystem(content="You are Claude Code."), *compacted])
    assert stub.ends() == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_failed_send_message_keeps_reply_closed_run_continuable(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    _send(consumer, [ack], "s1")
    failed: list[ChatMessage] = [
        ack,
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="s1", function="SendMessage", arguments={})],
        ),
        ChatMessageTool(content="Agent not found", tool_call_id="s1"),
    ]
    _call(consumer, [ChatMessageUser(content=MAIN), LAUNCH, *failed])
    hook: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="Stop hook feedback:\nnot done yet"),
    ]
    assert _call(consumer, hook) == "agent-toolu_1"


HOOK_CONTINUATION: list[ChatMessage] = [
    ChatMessageUser(content=PROMPT),
    ChatMessageAssistant(content="done"),
    ChatMessageUser(content="Stop hook feedback:\nnot done yet"),
]


def test_hook_continuation_does_not_take_a_same_prompt_siblings_span(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    # while A's Stop hook runs, the parent launches B with the same prompt
    _call(
        consumer,
        [ChatMessageUser(content=MAIN), LAUNCH],
        [ToolCall(id="toolu_2", function="Agent", arguments={"prompt": PROMPT})],
    )
    assert _final(consumer, HOOK_CONTINUATION) in ("agent-toolu_1", "outer")
    assert "agent-toolu_2" not in stub.ends()


def test_send_message_during_a_blocking_hook_keeps_the_run(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    _send(consumer, [ack], "s1")
    assert _call(consumer, HOOK_CONTINUATION) == "agent-toolu_1"


SUMMARY = (
    "The user asked for the number of lines in /etc/passwd. I ran wc -l and "
    "am about to report the count."
)


def test_send_during_a_blocking_hook_waits_for_a_run_to_show(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    # delivered within run 1 (its Stop hook kept it going): no run 2
    consumer, _ = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    _send(consumer, [ack], "s1")
    send = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="s1", function="SendMessage", arguments={})],
    )
    queued = ChatMessageTool(
        content='{"success":true,"message":"Message queued"}', tool_call_id="s1"
    )
    _parent(consumer, ack, send, queued)
    delivered = [*HOOK_CONTINUATION, ChatMessageUser(content="A message: more")]
    assert _final(consumer, delivered) == "agent-toolu_1"
    note = ChatMessageUser(content=_notification("toolu_1"))
    _parent(consumer, ack, send, queued, ChatMessageAssistant(content="w"), note)
    assert consumer._agents.open_runs() == []


def test_hook_continuation_with_a_send_pending_keeps_its_own_span(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, stub = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    _send(consumer, [ack], "s1")
    _call(
        consumer,
        [ChatMessageUser(content=MAIN), LAUNCH],
        [ToolCall(id="toolu_2", function="Agent", arguments={"prompt": PROMPT})],
    )
    assert _final(consumer, HOOK_CONTINUATION) in ("agent-toolu_1", "outer")
    assert "agent-toolu_2" not in stub.ends()


def test_summary_written_in_a_hook_continuation_is_that_runs(
    setup: tuple[LiveConsumer, _TranscriptStub],
) -> None:
    consumer, _ = setup
    ack = _background(consumer)
    _final(consumer, [ChatMessageUser(content=PROMPT)])
    _send(consumer, [ack], "s1")
    request = ChatMessageUser(
        content="Your task is to create a detailed summary of the conversation so far"
    )
    event = ModelEvent(
        model="anthropic/claude-haiku-4-5",
        input=[*HOOK_CONTINUATION, request],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(
            ChatMessageAssistant(content=f"<summary>{SUMMARY}</summary>")
        ),
    )
    consumer.on_pending(event)
    consumer.on_complete(event)
    assert event.span_id == "agent-toolu_1"
    continued = ChatMessageUser(
        content="This session is being continued from a previous conversation "
        f"that ran out of context.\n\nSummary:\n{SUMMARY}"
    )
    assert _call(consumer, [continued]) == "agent-toolu_1"
