"""Claude Code sub-agent span reconstruction in LiveConsumer.

Recent Claude Code runs Agent/Task sub-agents in the background by default
(`run_in_background`). The parent's `tool_result` is then an immediate
`async_launched` acknowledgement while the sub-agent keeps making model
calls; completion arrives later as a `task_notification` system event.
"""

from typing import Any

import pytest
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
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
                    arguments={"prompt": prompt, "subagent_type": "Explore"},
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
    consumer.process_jsonl_line(_notification(tool_use_id="toolu_other"))
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
