"""Claude Code sub-agent span reconstruction in LiveConsumer.

Recent Claude Code runs Agent/Task sub-agents in the background by default
(`run_in_background`). The parent's `tool_result` is then an immediate
`async_launched` acknowledgement while the sub-agent keeps making model
calls; completion arrives later as a `task_notification` system event.
"""

from typing import Any

import pytest
from inspect_ai._util.content import ContentText
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.live_consumer import LiveConsumer

PROMPT = "Find every TODO comment in the repository and summarise them."


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


def _consumer(monkeypatch: pytest.MonkeyPatch) -> tuple[LiveConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer")
    return LiveConsumer(), stub


def _model_event(
    input: list[ChatMessage], tool_calls: list[ToolCall] | None = None
) -> ModelEvent:
    return ModelEvent(
        model="anthropic/claude-sonnet-4-5",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(
            ChatMessageAssistant(content="ok", tool_calls=tool_calls)
        ),
    )


def _call(consumer: LiveConsumer, event: ModelEvent) -> ModelEvent:
    consumer.on_pending(event)
    consumer.on_complete(event)
    return event


PROMPT_2 = "Run the test suite and report which tests fail and why."
MAIN_TASK = "main task"


# Background vs foreground is decided by the tool_result Claude Code returns,
# not by the `run_in_background` argument, so launches don't set it.
def _launch(
    consumer: LiveConsumer,
    prompts: dict[str, str] | None = None,
    function: str = "Agent",
    main_task: str = MAIN_TASK,
    subagent_type: str = "Explore",
) -> None:
    prompts = prompts or {"toolu_1": PROMPT}
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=main_task)],
            [
                ToolCall(
                    id=tool_use_id,
                    function=function,
                    arguments={"prompt": prompt, "subagent_type": subagent_type},
                )
                for tool_use_id, prompt in prompts.items()
            ],
        ),
    )


def _subagent_call(consumer: LiveConsumer, prompt: str = PROMPT) -> ModelEvent:
    return _call(consumer, _model_event([ChatMessageUser(content=prompt)]))


def _result_block(text: str, tool_use_id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}


def _user_line(blocks: list[dict[str, Any]], tool_use_result: Any) -> dict[str, Any]:
    line: dict[str, Any] = {
        "type": "user",
        "message": {"role": "user", "content": blocks},
    }
    if tool_use_result is not None:
        line["tool_use_result"] = tool_use_result
    return line


def _tool_result_line(
    text: str, tool_use_result: Any, tool_use_id: str = "toolu_1"
) -> dict[str, Any]:
    return _user_line([_result_block(text, tool_use_id)], tool_use_result)


def _async_ack(
    tool_use_id: str = "toolu_1",
    prompt: str = PROMPT,
    agent_id: str = "a1b2c3",
    status: str = "async_launched",
) -> dict[str, Any]:
    return _tool_result_line(
        f"Async agent launched successfully.\nagentId: {agent_id}",
        {
            "status": status,
            "agentId": agent_id,
            "description": "Find TODOs",
            "prompt": prompt,
            "outputFile": f"/tmp/{agent_id}.output",
        },
        tool_use_id,
    )


def _notification(
    tool_use_id: str | None = "toolu_1",
    task_id: str = "a1b2c3",
    status: str = "completed",
) -> dict[str, Any]:
    line: dict[str, Any] = {
        "type": "system",
        "subtype": "task_notification",
        "task_id": task_id,
        "status": status,
        "output_file": f"/tmp/{task_id}.output",
        "summary": "done",
    }
    if tool_use_id is not None:
        line["tool_use_id"] = tool_use_id
    return line


ASYNC_ACK = _async_ack()
TASK_NOTIFICATION = _notification()


def test_foreground_subagent_span_closes_on_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    [begin] = stub.span_begins()
    assert (begin.type, begin.name, begin.parent_id) == ("agent", "Explore", "outer")

    assert _subagent_call(consumer).span_id == begin.id

    consumer.process_jsonl_line(
        _tool_result_line("Found 3 TODOs", {"status": "completed", "prompt": PROMPT})
    )
    assert [e.id for e in stub.span_ends()] == [begin.id]
    assert _subagent_call(consumer).span_id == "outer"


def test_background_subagent_stays_in_own_span_after_launch_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    [begin] = stub.span_begins()

    consumer.process_jsonl_line(ASYNC_ACK)
    assert stub.span_ends() == []

    # sub-agent keeps working after the parent got its acknowledgement
    assert _subagent_call(consumer).span_id == begin.id
    # the parent's own calls still land on the outer span
    parent = _call(consumer, _model_event([ChatMessageUser(content=MAIN_TASK)]))
    assert parent.span_id == "outer"

    consumer.process_jsonl_line(TASK_NOTIFICATION)
    assert [e.id for e in stub.span_ends()] == [begin.id]
    assert _subagent_call(consumer).span_id == "outer"


def test_background_ack_detected_from_text_without_structured_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    line = dict(ASYNC_ACK)
    del line["tool_use_result"]
    consumer.process_jsonl_line(line)
    assert stub.span_ends() == []


def test_reset_closes_background_subagent_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.reset()
    [begin] = stub.span_begins()
    assert [e.id for e in stub.span_ends()] == [begin.id]


def _ends(stub: _TranscriptStub) -> list[str]:
    return [e.id for e in stub.span_ends()]


def test_task_tool_name_opens_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, function="Task")
    consumer.process_jsonl_line(ASYNC_ACK)
    assert _subagent_call(consumer).span_id == "agent-toolu_1"


@pytest.mark.parametrize("status", ["completed", "failed", "stopped"])
def test_any_notification_status_closes(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_notification(status=status))
    assert _ends(stub) == ["agent-toolu_1"]


def test_notification_before_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(TASK_NOTIFICATION)
    consumer.process_jsonl_line(ASYNC_ACK)
    assert _ends(stub) == ["agent-toolu_1"]
    assert _subagent_call(consumer).span_id == "outer"


def test_unknown_notification_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_notification(tool_use_id="toolu_x", task_id="zzz"))
    consumer.process_jsonl_line(_notification(tool_use_id=None, task_id="zzz"))
    assert _ends(stub) == []


def test_notification_without_tool_use_id_closes_by_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_notification(tool_use_id=None))
    assert _ends(stub) == ["agent-toolu_1"]


@pytest.mark.parametrize("drop", ["structured", "prompt"])
def test_text_ack_then_task_id_only_notification_closes(
    monkeypatch: pytest.MonkeyPatch, drop: str
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    ack = _async_ack(agent_id="aid")
    if drop == "structured":
        del ack["tool_use_result"]
    else:
        ack["tool_use_result"]["prompt"] = "something else"
    consumer.process_jsonl_line(ack)
    consumer.process_jsonl_line(_notification(None, "aid"))
    assert _ends(stub) == ["agent-toolu_1"]


@pytest.mark.parametrize("as_blocks", [False, True])
def test_user_line_task_notification_closes(
    monkeypatch: pytest.MonkeyPatch, as_blocks: bool
) -> None:
    # Claude Code versions with no `system` task_notification events
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    text = (
        "<task-notification>\n<task-id>a1b2c3</task-id>\n"
        "<tool-use-id>toolu_1</tool-use-id>\n<status>completed</status>\n"
        "</task-notification>"
    )
    content: Any = [{"type": "text", "text": text}] if as_blocks else text
    consumer.process_jsonl_line(
        {"type": "user", "message": {"role": "user", "content": content}}
    )
    assert _ends(stub) == ["agent-toolu_1"]


def test_concurrent_background_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, {"toolu_1": PROMPT, "toolu_2": PROMPT_2})
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(_async_ack("toolu_2", PROMPT_2, "d4e5f6"))
    assert _subagent_call(consumer, PROMPT).span_id == "agent-toolu_1"
    assert _subagent_call(consumer, PROMPT_2).span_id == "agent-toolu_2"

    consumer.process_jsonl_line(_notification("toolu_2", "d4e5f6"))
    assert _ends(stub) == ["agent-toolu_2"]
    assert _subagent_call(consumer, PROMPT).span_id == "agent-toolu_1"
    assert _subagent_call(consumer, PROMPT_2).span_id == "outer"


def test_remote_launched_closes_at_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(status="remote_launched"))
    assert _ends(stub) == ["agent-toolu_1"]


def test_multi_block_line_ignores_line_level_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, {"toolu_1": PROMPT, "toolu_2": PROMPT_2})
    # toolu_1 is acknowledged as background, toolu_2 finished in the
    # foreground; the line-level result describes only one of them
    consumer.process_jsonl_line(
        _user_line(
            [
                _result_block("Async agent launched successfully.", "toolu_1"),
                _result_block("All tests pass.", "toolu_2"),
            ],
            {"status": "async_launched", "agentId": "a1b2c3", "prompt": PROMPT},
        )
    )
    assert _ends(stub) == ["agent-toolu_2"]


def test_parent_quoting_prompt_stays_on_outer_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    main_task = f"Please do this. {PROMPT} Then fix them."
    _launch(consumer, main_task=main_task)
    consumer.process_jsonl_line(ASYNC_ACK)
    launch = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
    )
    parent = _call(consumer, _model_event([ChatMessageUser(content=main_task), launch]))
    assert parent.span_id == "outer"
    assert _subagent_call(consumer).span_id == "agent-toolu_1"


def test_foreground_error_result_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(
        _tool_result_line("Error: agent failed", "Error: agent failed")
    )
    assert _ends(stub) == ["agent-toolu_1"]


def test_foreground_report_mentioning_launch_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(
        _tool_result_line("Done. Async agent launched for the docs earlier.", None)
    )
    assert _ends(stub) == ["agent-toolu_1"]


def test_verbatim_delegation_attributes_to_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, main_task=PROMPT)
    assert _subagent_call(consumer).span_id == "agent-toolu_1"


# Shapes below are from real Claude Code 2.1.303 runs (claude-haiku-4-5).
AGENT_ID = "a6b8c22231413918c"


def _resume_call(
    consumer: LiveConsumer, send_id: str, to: str = AGENT_ID
) -> ModelEvent:
    return _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [
                ToolCall(
                    id=send_id,
                    function="SendMessage",
                    arguments={"to": to, "message": "Now count /etc/group."},
                )
            ],
        ),
    )


def _resumed_subagent_call(consumer: LiveConsumer) -> ModelEvent:
    return _call(
        consumer,
        _model_event(
            [
                ChatMessageUser(content=PROMPT),
                ChatMessageAssistant(content="21 lines."),
                ChatMessageUser(
                    content="The coordinator sent a message while you were "
                    "working:\nNow count /etc/group."
                ),
            ]
        ),
    )


def _finished_background_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LiveConsumer, _TranscriptStub]:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id=AGENT_ID))
    consumer.process_jsonl_line(_notification(task_id=AGENT_ID))
    return consumer, stub


def test_send_message_resume_opens_new_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _finished_background_agent(monkeypatch)
    _resume_call(consumer, "toolu_send")
    begin = stub.span_begins()[-1]
    assert (begin.id, begin.name, begin.parent_id) == (
        "agent-toolu_1-run2",
        "Explore",
        "outer",
    )
    assert begin.metadata == {"resumes": "agent-toolu_1"}

    # the SendMessage result ("Resuming agent …") does not close it
    consumer.process_jsonl_line(
        _tool_result_line(
            '{"success":true,"message":"Resuming agent a6b8c22"}',
            {"success": True, "resumedAgentId": AGENT_ID},
            "toolu_send",
        )
    )
    assert _resumed_subagent_call(consumer).span_id == "agent-toolu_1-run2"
    # the duplicate task_started for this run does not open another span
    consumer.process_jsonl_line(
        {
            "type": "system",
            "subtype": "task_started",
            "task_id": AGENT_ID,
            "tool_use_id": "toolu_send",
        }
    )
    assert len(stub.span_begins()) == 2

    consumer.process_jsonl_line(_notification("toolu_send", AGENT_ID))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_send_message_by_name_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [
                ToolCall(
                    id="toolu_1",
                    function="Agent",
                    arguments={"prompt": PROMPT, "name": "counter"},
                )
            ],
        ),
    )
    consumer.process_jsonl_line(_async_ack(agent_id=AGENT_ID))
    consumer.process_jsonl_line(_notification(task_id=AGENT_ID))
    _resume_call(consumer, "toolu_send", to="counter")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run2"


def test_send_message_to_running_agent_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id=AGENT_ID))
    _resume_call(consumer, "toolu_send")
    assert len(stub.span_begins()) == 1


def test_task_started_reopens_resumed_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # a message queued while the agent ran makes Claude Code resume it after
    # its notification, under the original tool_use_id
    consumer, stub = _finished_background_agent(monkeypatch)
    consumer.process_jsonl_line(
        {
            "type": "system",
            "subtype": "task_started",
            "task_id": AGENT_ID,
            "tool_use_id": "toolu_1",
        }
    )
    assert stub.span_begins()[-1].id == "agent-toolu_1-run2"
    assert _resumed_subagent_call(consumer).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_notification("toolu_1", AGENT_ID))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_fork_attributed_by_fork_started_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, subagent_type="fork")
    consumer.process_jsonl_line(ASYNC_ACK)
    launch = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
    )
    fork = _call(
        consumer,
        _model_event(
            [
                ChatMessageUser(content=MAIN_TASK),
                launch,
                ChatMessageTool(
                    content="Fork started — processing in background",
                    tool_call_id="toolu_1",
                ),
                ChatMessageUser(
                    content="<fork-boilerplate>…</fork-boilerplate>\n\n"
                    f"Your directive: {PROMPT}"
                ),
            ]
        ),
    )
    assert fork.span_id == "agent-toolu_1"
    parent = _call(
        consumer,
        _model_event(
            [
                ChatMessageUser(content=MAIN_TASK),
                launch,
                ChatMessageTool(
                    content="Async agent launched successfully.",
                    tool_call_id="toolu_1",
                ),
            ]
        ),
    )
    assert parent.span_id == "outer"


# ---------------------------------------------------------------------------
# forks launched together, resume edge cases (independent review findings)
# ---------------------------------------------------------------------------


def _launch_many(
    consumer: LiveConsumer, agents: dict[str, tuple[str, str]], extra: bool = False
) -> None:
    calls = [
        ToolCall(id=i, function="Agent", arguments={"prompt": p, "subagent_type": t})
        for i, (p, t) in agents.items()
    ]
    if extra:
        calls.append(ToolCall(id="toolu_bash", function="Bash", arguments={}))
    _call(consumer, _model_event([ChatMessageUser(content=MAIN_TASK)], calls))


def _fork_input(ids: list[str], directive: str) -> list[ChatMessage]:
    launch = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id=i, function="Agent", arguments={}) for i in ids],
    )
    return [
        ChatMessageUser(content=MAIN_TASK),
        launch,
        *[
            ChatMessageTool(
                content="Fork started — processing in background", tool_call_id=i
            )
            for i in ids
        ],
        ChatMessageUser(content=f"<fork-boilerplate/>\nYour directive: {directive}"),
    ]


def _send(consumer: LiveConsumer, send_id: str, to: str) -> None:
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [ToolCall(id=send_id, function="SendMessage", arguments={"to": to})],
        ),
    )


def _started(task_id: str, tool_use_id: str, run_id: str) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "task_started",
        "task_id": task_id,
        "tool_use_id": tool_use_id,
        "run_id": run_id,
    }


def test_parallel_forks_pick_their_own_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch_many(consumer, {"t1": (PROMPT, "fork"), "t2": (PROMPT_2, "fork")})
    for directive, span in ((PROMPT, "agent-t1"), (PROMPT_2, "agent-t2")):
        call = _call(consumer, _model_event(_fork_input(["t1", "t2"], directive)))
        assert call.span_id == span


def test_fork_beside_regular_agent_and_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch_many(
        consumer, {"t1": (PROMPT, "fork"), "t2": (PROMPT_2, "Explore")}, extra=True
    )
    call = _call(
        consumer, _model_event(_fork_input(["t1", "t2", "toolu_bash"], PROMPT))
    )
    assert call.span_id == "agent-t1"


def test_nested_fork_goes_to_innermost(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_many(consumer, {"t1": (PROMPT, "fork")})
    outer_fork = _fork_input(["t1"], PROMPT)
    _call(
        consumer,
        _model_event(
            outer_fork,
            [
                ToolCall(
                    id="t2",
                    function="Agent",
                    arguments={"prompt": PROMPT_2, "subagent_type": "fork"},
                )
            ],
        ),
    )
    assert stub.span_begins()[-1].parent_id == "agent-t1"
    inner = outer_fork + _fork_input(["t2"], PROMPT_2)[1:]
    assert _call(consumer, _model_event(inner)).span_id == "agent-t2"


def test_resumed_fork_uses_new_run(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_many(consumer, {"t1": (PROMPT, "fork")})
    consumer.process_jsonl_line(_async_ack("t1", agent_id="aid"))
    consumer.process_jsonl_line(_notification("t1", "aid"))
    _send(consumer, "s1", "aid")
    history = _fork_input(["t1"], PROMPT) + [
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-t1-run2"


def test_failed_send_message_closes_its_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    _send(consumer, "s1", "aid")
    line = _tool_result_line(
        "<tool_use_error>Error: no agent</tool_use_error>", None, "s1"
    )
    line["message"]["content"][0]["is_error"] = True
    consumer.process_jsonl_line(line)
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]
    # a later, valid resume still gets a span
    _send(consumer, "s2", "aid")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run3"


def test_resumed_run_closes_on_task_id_only_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(None, "aid"))
    _send(consumer, "s1", "aid")
    consumer.process_jsonl_line(_notification(None, "aid"))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_foreground_agent_resumable_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(
        _tool_result_line(
            "21 lines\nagentId: aid", {"status": "completed", "prompt": PROMPT}
        )
    )
    _send(consumer, "s1", "aid")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run2"


def test_resumed_call_before_notification_rolls_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # seen in a real run: the resumed run's first call reaches the bridge
    # before the previous run's notification reaches the stdout reader
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21"),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"

    # the late notification for run 1 must not close run 2
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    assert _ends(stub) == ["agent-toolu_1"]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_lagging_send_message_keys_rolled_over_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")  # run 1's notification not read yet
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21"),
        ChatMessageUser(content="more"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_notification("s1", "aid"))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_reset_closes_resumed_run_and_keeps_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    _send(consumer, "s1", "aid")
    consumer.reset()
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]
    # a retry resumes the session, which can still message the agent
    _send(consumer, "s2", "aid")
    assert stub.span_begins()[-1].id == "agent-toolu_1-run3"


def test_same_name_prefers_callers_own_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [
                ToolCall(
                    id="t1", function="Agent", arguments={"prompt": PROMPT, "name": "w"}
                )
            ],
        ),
    )
    # sub-agent t1 launches its own "w"
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=PROMPT)],
            [
                ToolCall(
                    id="t2",
                    function="Agent",
                    arguments={"prompt": PROMPT_2, "name": "w"},
                )
            ],
        ),
    )
    consumer.process_jsonl_line(_tool_result_line("done", None, "t2"))
    consumer.process_jsonl_line(_tool_result_line("done", None, "t1"))
    _send(consumer, "s1", "w")  # from the main agent
    assert stub.span_begins()[-1].id == "agent-t1-run2"


def test_queued_message_resume_after_close(monkeypatch: pytest.MonkeyPatch) -> None:
    # real 2.1.303 trace: SendMessage while the agent runs (queued), run 1's
    # notification closes it, then Claude Code resumes it under the launch id
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="21"),
        ChatMessageUser(content="The coordinator sent a message: count /etc/group"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    assert len(stub.span_begins()) == 2
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def _launch_one(
    consumer: LiveConsumer, tool_use_id: str, prompt: str, name: str | None = None
) -> None:
    args: dict[str, Any] = {"prompt": prompt, "subagent_type": "Explore"}
    if name:
        args["name"] = name
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=MAIN_TASK)],
            [ToolCall(id=tool_use_id, function="Agent", arguments=args)],
        ),
    )


def test_relaunch_with_same_prompt_goes_to_new_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch_one(consumer, "t1", PROMPT)
    consumer.process_jsonl_line(_async_ack("t1", agent_id="a1"))
    consumer.process_jsonl_line(_notification("t1", "a1"))
    _launch_one(consumer, "t2", PROMPT)
    assert _subagent_call(consumer).span_id == "agent-t2"


def test_longer_prompt_wins_over_the_prompt_it_contains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    short = "Run the test suite and report."
    _launch_one(consumer, "t1", short)
    _launch_one(consumer, "t2", short + " Then fix the first failure.")
    longer = _subagent_call(consumer, short + " Then fix the first failure.")
    assert longer.span_id == "agent-t2"
    assert _subagent_call(consumer, short).span_id == "agent-t1"


def test_finished_agent_not_matched_without_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # e.g. the main agent after compaction, its summary quoting a prompt
    consumer, stub = _consumer(monkeypatch)
    _launch_one(consumer, "t1", PROMPT)
    consumer.process_jsonl_line(_tool_result_line("done", None, "t1"))
    assert _subagent_call(consumer).span_id == "outer"
    assert len(stub.span_begins()) == 1


def test_name_scoped_to_launching_agent_across_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_one(consumer, "t1", PROMPT, name="w")
    # sub-agent t1 launches its own "w" from its span
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=PROMPT)],
            [
                ToolCall(
                    id="t2",
                    function="Agent",
                    arguments={"prompt": PROMPT_2, "name": "w"},
                )
            ],
        ),
    )
    assert stub.span_begins()[-1].parent_id == "agent-t1"
    consumer.process_jsonl_line(_tool_result_line("done", None, "t2"))
    consumer.process_jsonl_line(_tool_result_line("done", None, "t1"))
    # main agent's checkpoint span rotated since t1 was launched
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer-2")
    _send(consumer, "s1", "w")
    assert stub.span_begins()[-1].id == "agent-t1-run2"


def test_task_started_for_new_run_rolls_over(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    assert _ends(stub) == ["agent-toolu_1"]
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    assert _ends(stub) == ["agent-toolu_1"]
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


@pytest.mark.parametrize(
    "injected",
    [
        # verbatim openings from Claude Code 2.1.303
        "Output token limit hit. Resume directly — no apology, no recap.",
        "Output token limit hit while you were still reasoning, before this "
        "response produced any text or tool call.",
        "Your response was cut off because it exceeded the output token limit. "
        "Continue from where you left off.",
        "[Your previous response reached the output limit with no visible "
        "output. Please continue and produce a brief user-visible response.]",
        "Your response above was cut off mid-stream. Resume directly.",
        "[Your previous response had no visible output. Please continue and "
        "produce a user-visible response.]",
        "Your response above was stopped by a safety classifier — this is not "
        "a tool or API error.",
        "The previous response failed to produce a valid tool call. Please "
        "retry the tool call now.",
        "Goal check-in: «finish the report»",
        "Your task is to create a detailed summary of the conversation so far, "
        "paying close attention to the user's explicit requests.",
        "[structured-output-enforce] call the StructuredOutput tool",
        "You ended the turn without calling SendUserMessage.",
        "Stop hook feedback:\nkeep going",
        "TaskCompleted hook feedback:\nnot done",
        'SubagentStop hook blocking error from command: "check.sh": not done',
        'Stop hook blocking error from command "check.sh": out',
    ],
)
def test_in_run_user_turn_is_not_a_new_run(
    monkeypatch: pytest.MonkeyPatch, injected: str
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    history: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="partial"),
        ChatMessageUser(content=injected),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-toolu_1"
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    assert _ends(stub) == ["agent-toolu_1"]
    assert len(stub.span_begins()) == 1


def test_resume_quoting_an_in_run_message_is_still_a_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # only the start of the turn counts, as in Claude Code
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    history: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(
            content="The coordinator sent a message: the log says "
            "'Output token limit hit', please retry"
        ),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-toolu_1-run2"


def test_in_run_turn_after_system_reminder_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    history: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="partial"),
        ChatMessageUser(
            content=[
                ContentText(text="<system-reminder>\nnote\n</system-reminder>"),
                ContentText(text="Output token limit hit. Resume directly."),
            ]
        ),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-toolu_1"
    assert len(stub.span_begins()) == 1


# ---------------------------------------------------------------------------
# ambiguous prompts, compaction, lagging and failed resumes
# ---------------------------------------------------------------------------


def test_same_prompt_agents_are_told_apart_by_their_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})

    def first_call(tool_call_id: str) -> str | None:
        bash = ToolCall(id=tool_call_id, function="Bash", arguments={})
        return _call(
            consumer, _model_event([ChatMessageUser(content=PROMPT)], [bash])
        ).span_id

    # identical first calls go to each agent in turn
    assert first_call("b1") == "agent-t1"
    assert first_call("b2") == "agent-t2"

    def next_call(tool_call_id: str) -> str | None:
        history: list[ChatMessage] = [
            ChatMessageUser(content=PROMPT),
            ChatMessageAssistant(
                content="",
                tool_calls=[ToolCall(id=tool_call_id, function="Bash", arguments={})],
            ),
            ChatMessageTool(content="21 /etc/passwd", tool_call_id=tool_call_id),
        ]
        return _call(consumer, _model_event(history)).span_id

    # later calls hold their own earlier tool calls
    assert next_call("b2") == "agent-t2"
    assert next_call("b1") == "agent-t1"


def test_a_third_first_call_for_two_same_prompt_agents_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    assert _subagent_call(consumer).span_id == "agent-t1"
    assert _subagent_call(consumer).span_id == "agent-t2"
    # (a retry of one of them)
    assert _subagent_call(consumer).span_id == "outer"


@pytest.mark.parametrize(
    "summary",
    [
        "This session is being continued from a previous conversation that ran "
        f"out of context.\n\nSummary:\nI launched an agent with prompt '{PROMPT}'",
        f"Earlier I asked an agent: {PROMPT}",
    ],
)
def test_main_agent_quoting_finished_agent_prompt_stays_outer(
    monkeypatch: pytest.MonkeyPatch, summary: str
) -> None:
    # e.g. the main agent after compaction: the launching tool call is gone
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    assert _subagent_call(consumer).span_id == "agent-toolu_1"
    consumer.process_jsonl_line(TASK_NOTIFICATION)
    history: list[ChatMessage] = [
        ChatMessageUser(content=summary),
        ChatMessageAssistant(content="Continuing."),
        ChatMessageUser(content="<task-notification>…</task-notification>"),
    ]
    assert _call(consumer, _model_event(history)).span_id == "outer"
    assert len(stub.span_begins()) == 1


def test_main_agent_compacted_beside_running_agent_stays_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    continued = ChatMessageUser(
        content="This session is being continued from a previous conversation "
        f"that ran out of context.\n\nSummary:\nAn agent is running: {PROMPT}"
    )
    assert _call(consumer, _model_event([continued])).span_id == "outer"


def test_prompt_after_system_reminder_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    consumer.process_jsonl_line(TASK_NOTIFICATION)
    history: list[ChatMessage] = [
        ChatMessageUser(
            content=[
                ContentText(text="<system-reminder>\ncontext\n</system-reminder>"),
                ContentText(text=PROMPT),
            ]
        ),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, _model_event(history)).span_id == "agent-toolu_1-run2"


def test_user_line_notification_ignored_with_system_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification("toolu_1", "aid"))
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    # the parent receiving run 1's notification must not close run 2
    text = "<task-notification><tool-use-id>toolu_1</tool-use-id></task-notification>"
    consumer.process_jsonl_line(
        {"type": "user", "message": {"role": "user", "content": text}}
    )
    assert _ends(stub) == ["agent-toolu_1"]


def test_parallel_forks_with_nested_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    longer = PROMPT + " Also implement fixes."
    _launch_many(consumer, {"t1": (PROMPT, "fork"), "t2": (longer, "fork")})
    for directive, span in ((longer, "agent-t2"), (PROMPT, "agent-t1")):
        call = _call(consumer, _model_event(_fork_input(["t1", "t2"], directive)))
        assert call.span_id == span


def test_lagging_first_task_started_after_lazy_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_one(consumer, "toolu_1", PROMPT, name="worker")
    # the bridge sees the resumed run before stdout shows run 1's start
    _send(consumer, "s1", "worker")
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    assert len(stub.span_begins()) == 2
    consumer.process_jsonl_line({**_notification("s1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_failed_send_message_starts_no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    _send(consumer, "s1", "aid")
    line = _tool_result_line("<tool_use_error>Error</tool_use_error>", None, "s1")
    line["message"]["content"][0]["is_error"] = True
    consumer.process_jsonl_line(line)
    _send(consumer, "s2", "aid")
    run2: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    assert _call(consumer, _model_event(run2)).span_id == "agent-toolu_1-run3"
    consumer.process_jsonl_line(_notification("s2", "aid"))
    # the agent's next run is its third, whatever the span numbering
    run3 = run2 + [ChatMessageAssistant(content="done"), ChatMessageUser(content="m")]
    assert _call(consumer, _model_event(run3)).span_id == "agent-toolu_1-run4"


def test_failed_queued_send_message_keys_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")  # while run 1 is open
    line = _tool_result_line("<tool_use_error>Error</tool_use_error>", None, "s1")
    line["message"]["content"][0]["is_error"] = True
    consumer.process_jsonl_line(line)
    agent = consumer._agents.get("toolu_1")
    assert agent is not None and agent.pending_send is None


def test_reset_drops_pending_send_message(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")  # while run 1 is open
    consumer.reset()
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    assert [run.key for run in consumer._agents.open_runs()] == ["toolu_1"]


SUMMARY = (
    "1. Primary Request and Intent: find every TODO comment in the repository "
    "and summarise them. 2. Progress: searched src/ and tests/, found 12 TODOs "
    "so far, still to check docs/."
)


def _compaction(consumer: LiveConsumer, history: list[ChatMessage]) -> ModelEvent:
    event = _model_event(
        history
        + [
            ChatMessageUser(
                content="Your task is to create a detailed summary of the "
                "conversation so far"
            )
        ]
    )
    event.output = ModelOutput.from_message(
        ChatMessageAssistant(
            content=f"<analysis>thinking</analysis>\n<summary>\n{SUMMARY}\n</summary>"
        )
    )
    return _call(consumer, event)


def _continued(*tail: ChatMessage) -> list[ChatMessage]:
    return [
        ChatMessageUser(
            content="This session is being continued from a previous conversation "
            "that ran out of context. The summary below covers the earlier "
            f"portion of the conversation.\n\nSummary:\n{SUMMARY}"
        ),
        *tail,
    ]


def test_compacted_subagent_keeps_its_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    history: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(
            content="", tool_calls=[ToolCall(id="g", function="Grep", arguments={})]
        ),
        ChatMessageTool(content="12 matches", tool_call_id="g"),
    ]
    assert _compaction(consumer, history).span_id == "agent-toolu_1"
    assert _call(consumer, _model_event(_continued())).span_id == "agent-toolu_1"
    # a resume after the compaction is the agent's second run
    consumer.process_jsonl_line(TASK_NOTIFICATION)
    resumed = _continued(
        ChatMessageAssistant(content="12 TODOs."), ChatMessageUser(content="more")
    )
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"


def test_compacted_fork_keeps_its_span(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch_many(consumer, {"t1": (PROMPT, "fork")})
    assert _compaction(consumer, _fork_input(["t1"], PROMPT)).span_id == "agent-t1"
    assert _call(consumer, _model_event(_continued())).span_id == "agent-t1"


def test_main_agent_compaction_is_not_a_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    assert _compaction(consumer, [ChatMessageUser(content=MAIN_TASK)]).span_id == (
        "outer"
    )
    assert _call(consumer, _model_event(_continued())).span_id == "outer"


def test_user_line_notification_for_rolled_over_run_keeps_next_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # no run ids: a queued message resumed the agent (same key) before the
    # parent received run 1's notification
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    note = (
        "<task-notification><task-id>aid</task-id>"
        "<tool-use-id>toolu_1</tool-use-id></task-notification>"
    )
    line = {"type": "user", "message": {"role": "user", "content": note}}
    consumer.process_jsonl_line(line)
    assert _ends(stub) == ["agent-toolu_1"]
    consumer.process_jsonl_line(line)  # run 2's own notification
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_summary_without_summary_tags_drops_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    event = _model_event(
        [
            ChatMessageUser(content=PROMPT),
            ChatMessageUser(
                content="Your task is to create a detailed summary of the "
                "conversation so far"
            ),
        ]
    )
    event.output = ModelOutput.from_message(
        ChatMessageAssistant(content=f"<analysis>thinking</analysis>\n{SUMMARY}")
    )
    assert _call(consumer, event).span_id == "agent-toolu_1"
    assert _call(consumer, _model_event(_continued())).span_id == "agent-toolu_1"


def test_quoting_a_summary_outside_a_continuation_stays_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    _compaction(consumer, [ChatMessageUser(content=PROMPT)])
    quoting = ChatMessageUser(content=f"{MAIN_TASK}. Notes: {SUMMARY}")
    assert _call(consumer, _model_event([quoting])).span_id == "outer"


def test_launcher_quoting_a_summary_after_compacting_stays_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    _compaction(consumer, [ChatMessageUser(content=PROMPT)])
    launch = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
    )
    history = _continued(launch, ChatMessageTool(content="ok", tool_call_id="toolu_1"))
    assert _call(consumer, _model_event(history)).span_id == "outer"


def test_queued_run_opened_before_ids_closes_on_launch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch_one(consumer, "toolu_1", PROMPT, name="worker")
    _send(consumer, "s1", "worker")
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


@pytest.mark.parametrize("finished", [False, True])
def test_launcher_whose_task_quotes_a_summary_stays_outer(
    monkeypatch: pytest.MonkeyPatch, finished: bool
) -> None:
    consumer, _ = _consumer(monkeypatch)
    main_task = f"Evaluate this progress report using a worker: {SUMMARY}"
    _launch(consumer, main_task=main_task)
    consumer.process_jsonl_line(ASYNC_ACK)
    _compaction(consumer, [ChatMessageUser(content=PROMPT)])
    if finished:
        consumer.process_jsonl_line(TASK_NOTIFICATION)
    history: list[ChatMessage] = [
        ChatMessageUser(content=main_task),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="toolu_1", function="Agent", arguments={})],
        ),
        ChatMessageTool(content="Async agent launched", tool_call_id="toolu_1"),
    ]
    assert _call(consumer, _model_event(history)).span_id == "outer"


def test_new_agent_given_a_summary_keeps_its_own_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    _compaction(consumer, [ChatMessageUser(content=PROMPT)])
    consumer.process_jsonl_line(TASK_NOTIFICATION)
    second = f"Independently verify the following progress report: {SUMMARY}"
    _launch_one(consumer, "t2", second)
    assert _subagent_call(consumer, second).span_id == "agent-t2"


def test_summaries_sharing_a_long_start_are_told_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT_2})
    shared = (
        "The user asked for a review of the repository. We must inspect source "
        "and tests, preserve existing behaviour, give file and line references, "
        "and report correctness findings first. "
    ) * 2
    ends = {"t1": "Reviewed package alpha.", "t2": "Reviewed package beta."}
    for origin, prompt in (("t1", PROMPT), ("t2", PROMPT_2)):
        event = _model_event(
            [
                ChatMessageUser(content=prompt),
                ChatMessageUser(
                    content="Your task is to create a detailed summary of the "
                    "conversation so far"
                ),
            ]
        )
        event.output = ModelOutput.from_message(
            ChatMessageAssistant(content=f"<summary>{shared}{ends[origin]}</summary>")
        )
        assert _call(consumer, event).span_id == f"agent-{origin}"
    for origin, end in ends.items():
        continued = ChatMessageUser(
            content="This session is being continued from a previous "
            f"conversation. Summary: {shared}{end}"
        )
        assert _call(consumer, _model_event([continued])).span_id == f"agent-{origin}"


def test_prompt_starting_with_a_reminder_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    prompt = f"<system-reminder>Work read-only.</system-reminder>\n{PROMPT}"
    _launch_one(consumer, "t1", prompt)
    assert _subagent_call(consumer, prompt).span_id == "agent-t1"


def test_task_started_without_tool_use_id_keeps_run_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the Agent SDK types declare task_started's tool_use_id optional
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    started = _started("aid", "toolu_1", "r1")
    del started["tool_use_id"]
    consumer.process_jsonl_line(started)
    consumer.process_jsonl_line({**_notification("toolu_1", "aid"), "run_id": "r1"})
    _send(consumer, "s1", "aid")
    consumer.process_jsonl_line(_started("aid", "s1", "r2"))
    consumer.process_jsonl_line({**_notification("s1", "aid"), "run_id": "r2"})
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_reset_forgets_notifications_owed_by_the_old_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="queued message"),
    ]
    _call(consumer, _model_event(resumed))  # run 2; run 1's notification owed
    consumer.reset()
    again = resumed + [
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="another queued message"),
    ]
    assert _call(consumer, _model_event(again)).span_id == "agent-toolu_1-run3"
    note = (
        "<task-notification><task-id>aid</task-id>"
        "<tool-use-id>toolu_1</tool-use-id></task-notification>"
    )
    consumer.process_jsonl_line(
        {"type": "user", "message": {"role": "user", "content": note}}
    )
    assert _ends(stub)[-1] == "agent-toolu_1-run3"


def test_compacted_parent_quoting_a_child_summary_stays_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the launch is gone from the parent's compacted history
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    _compaction(consumer, [ChatMessageUser(content=PROMPT)])
    parent = ChatMessageUser(
        content="This session is being continued from a previous conversation "
        "that ran out of context.\n\nSummary:\n1. Primary Request: count TODOs. "
        f"2. The worker reported its progress: {SUMMARY}"
    )
    assert _call(consumer, _model_event([parent])).span_id == "outer"


def test_untagged_summary_after_preamble_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    event = _model_event(
        [
            ChatMessageUser(content=PROMPT),
            ChatMessageUser(
                content="Your task is to create a detailed summary of the "
                "conversation so far"
            ),
        ]
    )
    event.output = ModelOutput.from_message(ChatMessageAssistant(content=SUMMARY))
    _call(consumer, event)
    continued = ChatMessageUser(
        content="This session is being continued from a previous conversation "
        "that ran out of context. The summary below covers the earlier portion "
        f"of the conversation.\n\n{SUMMARY}"
    )
    assert _call(consumer, _model_event([continued])).span_id == "agent-toolu_1"


def test_late_notification_after_rollover_to_send_message_keeps_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # no run ids: SendMessage while run 1 still looked open, then run 2's
    # first call, then run 1's notification (in the parent's user line)
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"

    def note(tool_use_id: str) -> dict[str, Any]:
        text = (
            "<task-notification><task-id>aid</task-id>"
            f"<tool-use-id>{tool_use_id}</tool-use-id></task-notification>"
        )
        return {"type": "user", "message": {"role": "user", "content": text}}

    consumer.process_jsonl_line(note("toolu_1"))
    assert _ends(stub) == ["agent-toolu_1"]
    consumer.process_jsonl_line(note("s1"))
    assert _ends(stub) == ["agent-toolu_1", "agent-toolu_1-run2"]


def test_summary_with_text_outside_its_tags_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Claude Code keeps text outside <analysis> and <summary> in the
    # continuation, before "Summary:"
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    event = _model_event(
        [
            ChatMessageUser(content=PROMPT),
            ChatMessageUser(
                content="Your task is to create a detailed summary of the "
                "conversation so far"
            ),
        ]
    )
    event.output = ModelOutput.from_message(
        ChatMessageAssistant(
            content="I'll summarise.\n<analysis>x</analysis>\n"
            f"<summary>\n{SUMMARY}\n</summary>"
        )
    )
    _call(consumer, event)
    continued = ChatMessageUser(
        content="This session is being continued from a previous conversation "
        "that ran out of context. The summary below covers the earlier portion "
        f"of the conversation.\n\nI'll summarise.\n\nSummary:\n{SUMMARY}"
    )
    assert _call(consumer, _model_event([continued])).span_id == "agent-toolu_1"


def test_late_ack_still_marks_rolled_over_run_notification_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # stdout lags: the bridge rolls over to run 2 before the launch ack, and
    # the user-line notification for run 1 has no run id
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="queued message"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    note = (
        "<task-notification><task-id>aid</task-id>"
        "<tool-use-id>toolu_1</tool-use-id></task-notification>"
    )
    consumer.process_jsonl_line(
        {"type": "user", "message": {"role": "user", "content": note}}
    )
    assert _ends(stub) == ["agent-toolu_1"]


def test_lagging_send_message_opens_run_under_its_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)  # parent P
    consumer.process_jsonl_line(_async_ack(agent_id="p"))
    # P launches child C, then finishes and is resumed
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=PROMPT)],
            [ToolCall(id="c", function="Agent", arguments={"prompt": PROMPT_2})],
        ),
    )
    consumer.process_jsonl_line(_async_ack("c", PROMPT_2, "c-id"))
    consumer.process_jsonl_line(_notification(task_id="p"))
    _send(consumer, "to-p", "p")
    p_run2: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    # P's run 2 messages C, whose notification stdout has not shown yet
    send = ToolCall(id="to-c", function="SendMessage", arguments={"to": "c-id"})
    _call(consumer, _model_event(p_run2, [send]))
    c_run2: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT_2),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="more"),
    ]
    _call(consumer, _model_event(c_run2))
    begin = stub.span_begins()[-1]
    assert (begin.id, begin.parent_id) == ("agent-c-run2", "agent-toolu_1-run2")


def test_failed_send_message_keeps_a_run_a_call_showed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the agent resumed on its own (a child's notification) while the
    # SendMessage that keyed the run was still failing
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    _send(consumer, "s1", "aid")
    resumed: list[ChatMessage] = [
        ChatMessageUser(content=PROMPT),
        ChatMessageAssistant(content="done"),
        ChatMessageUser(content="<task-notification>child</task-notification>"),
    ]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"
    failed = _tool_result_line("Error", None, "s1")
    failed["message"]["content"][0]["is_error"] = True
    consumer.process_jsonl_line(failed)
    assert _ends(stub) == ["agent-toolu_1"]
    assert _call(consumer, _model_event(resumed)).span_id == "agent-toolu_1-run2"


def test_run_opened_later_goes_under_the_launchers_current_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # with checkpointing, the main agent's span rotates after the launch
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line(_notification(task_id="aid"))
    monkeypatch.setattr(consumer_module, "current_span_id", lambda: "outer-2")
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    begin = stub.span_begins()[-1]
    assert (begin.id, begin.parent_id) == ("agent-toolu_1-run2", "outer-2")


def test_nested_run_opened_later_goes_under_its_launchers_latest_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)  # P
    consumer.process_jsonl_line(_async_ack(agent_id="p"))
    _call(
        consumer,
        _model_event(
            [ChatMessageUser(content=PROMPT)],
            [ToolCall(id="c", function="Agent", arguments={"prompt": PROMPT_2})],
        ),
    )
    consumer.process_jsonl_line(_started("c-id", "c", "r1"))
    consumer.process_jsonl_line(_async_ack("c", PROMPT_2, "c-id"))
    consumer.process_jsonl_line(_notification("c", "c-id"))
    consumer.process_jsonl_line(_notification(task_id="p"))
    _send(consumer, "to-p", "p")  # P's run 2
    consumer.process_jsonl_line(_started("c-id", "c", "r2"))  # C resumed
    begin = stub.span_begins()[-1]
    assert (begin.id, begin.parent_id) == ("agent-c-run2", "agent-toolu_1-run2")


def test_history_starting_with_a_prompt_beats_one_it_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"toolu_1": PROMPT, "toolu_2": PROMPT_2})
    consumer.process_jsonl_line(_async_ack("toolu_1", PROMPT, "a1"))
    consumer.process_jsonl_line(_async_ack("toolu_2", PROMPT_2, "a2"))
    quoting = f"{PROMPT_2}\n\nContext: another agent was asked: {PROMPT}"
    assert _subagent_call(consumer, quoting).span_id == "agent-toolu_2"


def test_task_started_confirms_a_run_a_send_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the agent resumed on its own (task_started) while the SendMessage that
    # opened the run was failing
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r1"))
    consumer.process_jsonl_line(_async_ack(agent_id="aid"))
    consumer.process_jsonl_line({**_notification(task_id="aid"), "run_id": "r1"})
    _send(consumer, "s1", "aid")
    consumer.process_jsonl_line(_started("aid", "toolu_1", "r2"))
    error = _tool_result_line("Error: message is required", None, "s1")
    error["message"]["content"][0]["is_error"] = True
    consumer.process_jsonl_line(error)
    assert _ends(stub) == ["agent-toolu_1"]


# (captured before conftest makes tracking failures raise)
_LOGGED_TRACKING = consumer_module._tracking


def test_a_tracking_failure_falls_back_to_the_outer_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(consumer_module, "_tracking", _LOGGED_TRACKING)
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer)

    def broken(*args: Any) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(consumer_module, "resolve", broken)
    event = _model_event([ChatMessageUser(content=PROMPT)])
    consumer.on_pending(event)
    consumer.on_complete(event)
    assert event.span_id == "outer"
    assert event in stub.events


def _bash_call(tool_call_id: str) -> ChatMessageAssistant:
    return ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id=tool_call_id, function="Bash", arguments={})],
    )


def test_retried_first_call_keeps_its_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # same-prompt agents: a failed first call's retry is not the other's first
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    failed = _model_event([ChatMessageUser(content=PROMPT)])
    failed.output = ModelOutput()
    failed.error = "API connection failed"
    assert _call(consumer, failed).span_id == "agent-t1"
    retry = _model_event([ChatMessageUser(content=PROMPT)], _bash_call("b1").tool_calls)
    assert _call(consumer, retry).span_id == "agent-t1"
    assert _subagent_call(consumer).span_id == "agent-t2"


def test_compacted_agent_found_by_its_tool_calls_keeps_its_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # two same-prompt agents write the same summary; A's own tool calls pick
    # it out, in the run it compacted in
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    a: list[ChatMessage] = [ChatMessageUser(content=PROMPT)]
    b: list[ChatMessage] = [ChatMessageUser(content=PROMPT)]
    assert _call(consumer, _model_event(a, _bash_call("ba").tool_calls)).span_id == (
        "agent-t1"
    )
    assert _call(consumer, _model_event(b, _bash_call("bb").tool_calls)).span_id == (
        "agent-t2"
    )
    a += [_bash_call("ba"), ChatMessageTool(content="ok", tool_call_id="ba")]
    b += [_bash_call("bb"), ChatMessageTool(content="ok", tool_call_id="bb")]
    consumer.process_jsonl_line(_notification("t1", "aid"))
    a += [ChatMessageAssistant(content="done"), ChatMessageUser(content="continue")]
    assert _compaction(consumer, a).span_id == "agent-t1-run2"
    continued = _model_event(_continued(), _bash_call("ba2").tool_calls)
    assert _call(consumer, continued).span_id == "agent-t1-run2"
    assert _compaction(consumer, b).span_id == "agent-t2"
    history = _continued(
        _bash_call("ba2"), ChatMessageTool(content="ok", tool_call_id="ba2")
    )
    assert _call(consumer, _model_event(history)).span_id == "agent-t1-run2"


def test_refused_first_call_keeps_its_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    refused = _model_event([ChatMessageUser(content=PROMPT)])
    refused.output = ModelOutput.from_message(
        ChatMessageAssistant(content="I can't help with that."),
        stop_reason="content_filter",
    )
    assert _call(consumer, refused).span_id == "agent-t1"

    def first(tool_call_id: str) -> str | None:
        event = _model_event(
            [ChatMessageUser(content=PROMPT)], _bash_call(tool_call_id).tool_calls
        )
        return _call(consumer, event).span_id

    def later(tool_call_id: str) -> str | None:
        history: list[ChatMessage] = [
            ChatMessageUser(content=PROMPT),
            _bash_call(tool_call_id),
            ChatMessageTool(content="out", tool_call_id=tool_call_id),
        ]
        return _call(consumer, _model_event(history)).span_id

    assert (first("b1"), first("b2")) == ("agent-t1", "agent-t2")
    assert (later("b1"), later("b2")) == ("agent-t1", "agent-t2")


NESTED = "Find every TODO comment in the repository and summarise them."
DELEGATE = "Delegate the TODO survey to a child agent and report back."


@pytest.mark.parametrize(
    "main_task", [f"Use a helper that delegates this: '{NESTED}'", NESTED]
)
def test_one_wrongly_attributed_main_call_does_not_take_the_rest(
    monkeypatch: pytest.MonkeyPatch, main_task: str
) -> None:
    # main's task quotes (or is) a running grandchild's prompt, so one main
    # call goes to the grandchild's span; its tool calls must not pull in
    # later ones
    consumer, _ = _consumer(monkeypatch)
    launch_a = ToolCall(id="tA", function="Agent", arguments={"prompt": DELEGATE})
    _call(consumer, _model_event([ChatMessageUser(content=main_task)], [launch_a]))
    consumer.process_jsonl_line(_async_ack("tA", DELEGATE, "aA"))
    launch_b = ToolCall(id="tB", function="Agent", arguments={"prompt": NESTED})
    _call(consumer, _model_event([ChatMessageUser(content=DELEGATE)], [launch_b]))
    main: list[ChatMessage] = [
        ChatMessageUser(content=main_task),
        ChatMessageAssistant(content="", tool_calls=[launch_a]),
        ChatMessageTool(
            content="Async agent launched successfully.", tool_call_id="tA"
        ),
    ]
    _call(consumer, _model_event(main, _bash_call("m_bash").tool_calls))
    consumer.process_jsonl_line(_user_line([_result_block("done", "tB")], None))
    later = main + [
        _bash_call("m_bash"),
        ChatMessageTool(content="ok", tool_call_id="m_bash"),
    ]
    assert _call(consumer, _model_event(later)).span_id == "outer"


@pytest.mark.parametrize(
    "a_runs",
    [
        # A's run 3 open
        [("a1", "a2"), ("a2", "a3")],
        # A's run 2 closed
        [("a1", "a2"), ("a2", None)],
    ],
)
def test_swapped_same_prompt_agents_keep_each_run_in_one_span(
    monkeypatch: pytest.MonkeyPatch, a_runs: list[tuple[str, str | None]]
) -> None:
    # A's first call fails and B's starts before A retries, so they swap
    # spans; A's later runs must not split B's one run across spans
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, {"t1": PROMPT, "t2": PROMPT})
    for tool_use_id, agent_id, run_id in [("t1", "aA", "a1"), ("t2", "aB", "b1")]:
        consumer.process_jsonl_line(_started(agent_id, tool_use_id, run_id))
        consumer.process_jsonl_line(_async_ack(tool_use_id, PROMPT, agent_id))
    failed = _model_event([ChatMessageUser(content=PROMPT)])
    failed.output = ModelOutput()
    failed.error = "API connection failed"
    _call(consumer, failed)
    first: list[ChatMessage] = [ChatMessageUser(content=PROMPT)]
    b_first = _call(consumer, _model_event(first, _bash_call("b").tool_calls))
    a_retry = _call(consumer, _model_event(first, _bash_call("a").tool_calls))
    assert (b_first.span_id, a_retry.span_id) == ("agent-t1", "agent-t2")
    a: list[ChatMessage] = [
        *first,
        _bash_call("a"),
        ChatMessageTool(content="ok", tool_call_id="a"),
    ]
    for run_id, next_run_id in a_runs:
        consumer.process_jsonl_line({**_notification("t1", "aA"), "run_id": run_id})
        if next_run_id is not None:
            consumer.process_jsonl_line(_started("aA", "t1", next_run_id))
            a += [
                ChatMessageAssistant(content="done"),
                ChatMessageUser(content="more"),
            ]
            _call(consumer, _model_event(a))
    b: list[ChatMessage] = [
        *first,
        _bash_call("b"),
        ChatMessageTool(content="ok", tool_call_id="b"),
    ]
    assert _call(consumer, _model_event(b)).span_id == b_first.span_id


def test_continuation_with_tools_used_metadata_keeps_its_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Claude Code 2.1.304 lists the tools used before the summary
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer)
    consumer.process_jsonl_line(ASYNC_ACK)
    assert _compaction(consumer, [ChatMessageUser(content=PROMPT)]).span_id == (
        "agent-toolu_1"
    )
    continued = ChatMessageUser(
        content="This session is being continued from a previous conversation "
        "that ran out of context. The summary below covers the earlier portion "
        "of the conversation.\n\n<tools-used-before-summary>Agent, Bash, Read"
        f"</tools-used-before-summary>\nSummary:\n{SUMMARY}"
    )
    assert _call(consumer, _model_event([continued])).span_id == "agent-toolu_1"
