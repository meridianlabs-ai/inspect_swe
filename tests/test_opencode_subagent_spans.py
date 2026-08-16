"""OpenCode sub-agent span reconstruction (bridge-only).

OpenCode's `task` tool runs each sub-agent as a child session whose requests
arrive at the bridge interleaved with the parent's. The `OpenCodeConsumer`
(a bridge `ModelEventSink`) reconstructs the agent-span tree from the model
events alone:

- span *open* happens in `on_complete` when the parent's output contains
  `task` tool-calls (synchronous, ahead of the sub-agent's first call);
- sub-agent calls are *attributed* by substring-matching their first user
  message (the task prompt, re-sent on every request) against open tasks;
- span *close* happens in `on_pending` when the parent's next request carries
  the task tool_result (`ChatMessageTool` correlated by `tool_call_id`).
"""

from typing import Any

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
from inspect_swe._opencode._events import consumer as consumer_module
from inspect_swe._opencode._events.consumer import OpenCodeConsumer

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _task_call(call_id: str, prompt: str, subagent_type: str = "general") -> ToolCall:
    return ToolCall(
        id=call_id,
        function="task",
        arguments={
            "description": "Do a subtask",
            "prompt": prompt,
            "subagent_type": subagent_type,
        },
    )


def _task_result(call_id: str, text: str = "done") -> ChatMessageTool:
    return ChatMessageTool(content=text, tool_call_id=call_id, function="task")


def _model_event(
    input: list[ChatMessage], tool_calls: list[ToolCall] | None = None
) -> ModelEvent:
    message = ChatMessageAssistant(content="ok", tool_calls=tool_calls)
    return ModelEvent(
        model="anthropic/claude-sonnet-4-5",
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


def _consumer(monkeypatch: Any) -> tuple[OpenCodeConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    return OpenCodeConsumer(), stub


PROMPT_FB = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"
PROMPT_PR = "write a prime sieve and save it to /tmp/primes.py"


# ---------------------------------------------------------------------------
# 1. span open: task tool-calls in parent output
# ---------------------------------------------------------------------------


def test_task_calls_open_named_spans(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="delegate two subtasks")],
        tool_calls=[
            _task_call("call_fb", PROMPT_FB, "general"),
            _task_call("call_pr", PROMPT_PR, "explore"),
        ],
    )
    consumer.on_complete(parent)

    begins = stub.span_begins()
    assert [e.name for e in begins] == ["general", "explore"]
    assert [e.id for e in begins] == ["agent-call_fb", "agent-call_pr"]
    assert all(e.type == "agent" for e in begins)
    assert begins[0].metadata == {"description": "Do a subtask"}


def test_task_call_attaches_tool_view(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)
    call = _task_call("call_fb", PROMPT_FB)
    parent = _model_event([ChatMessageUser(content="go")], tool_calls=[call])
    consumer.on_complete(parent)
    assert call.view is not None
    assert "Task: general" in (call.view.title or "")


def test_non_task_calls_do_not_open_spans(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    bash = ToolCall(id="call_1", function="bash", arguments={"command": "ls"})
    parent = _model_event([ChatMessageUser(content="go")], tool_calls=[bash])
    consumer.on_complete(parent)
    assert stub.span_begins() == []


# ---------------------------------------------------------------------------
# 2. attribution: sub-agent requests match their task prompt
# ---------------------------------------------------------------------------


def test_subagent_calls_attributed_by_prompt(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="delegate two subtasks")],
        tool_calls=[
            _task_call("call_fb", PROMPT_FB),
            _task_call("call_pr", PROMPT_PR),
        ],
    )
    consumer.on_complete(parent)
    fb_span, pr_span = (e.id for e in stub.span_begins())

    # each sub-agent session starts with the task prompt as its first user
    # message (past the sub-agent system prompt)
    fb_call = _model_event(
        [
            ChatMessageSystem(content="You are a general sub-agent."),
            ChatMessageUser(content=PROMPT_FB),
        ]
    )
    consumer.on_pending(fb_call)
    assert fb_call.span_id == fb_span

    pr_call = _model_event([ChatMessageUser(content=PROMPT_PR)])
    consumer.on_pending(pr_call)
    assert pr_call.span_id == pr_span

    # the parent's own interleaved call stays on the outer span
    parent_call = _model_event([ChatMessageUser(content="delegate two subtasks")])
    consumer.on_pending(parent_call)
    assert parent_call.span_id not in (fb_span, pr_span)


def test_subagent_followup_calls_still_attributed(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="go")], tool_calls=[_task_call("call_fb", PROMPT_FB)]
    )
    consumer.on_complete(parent)
    fb_span = stub.span_begins()[0].id

    # later sub-agent requests re-send the full session (prompt still first)
    followup = _model_event(
        [
            ChatMessageUser(content=PROMPT_FB),
            ChatMessageAssistant(content="working on it"),
            ChatMessageTool(content="ok", tool_call_id="call_x", function="bash"),
            ChatMessageUser(content="continue"),
        ]
    )
    consumer.on_pending(followup)
    assert followup.span_id == fb_span


def test_short_prompts_never_match(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="go")], tool_calls=[_task_call("call_1", "fix it")]
    )
    consumer.on_complete(parent)

    call = _model_event([ChatMessageUser(content="fix it")])
    consumer.on_pending(call)
    assert call.span_id != "agent-call_1"


# ---------------------------------------------------------------------------
# 3. close: task tool_result in the parent's next request
# ---------------------------------------------------------------------------


def test_task_result_closes_span(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="go")], tool_calls=[_task_call("call_fb", PROMPT_FB)]
    )
    consumer.on_complete(parent)
    fb_span = stub.span_begins()[0].id

    # parent's next request carries the task tool_result → close, and the
    # request itself attributes to the outer span
    parent_next = _model_event(
        [
            ChatMessageUser(content="go"),
            ChatMessageAssistant(
                content="ok", tool_calls=[_task_call("call_fb", PROMPT_FB)]
            ),
            _task_result("call_fb", "fizzbuzz written"),
        ]
    )
    consumer.on_pending(parent_next)

    assert [e.id for e in stub.span_ends()] == [fb_span]
    assert parent_next.span_id != fb_span


def test_parallel_tasks_close_independently(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[
            _task_call("call_fb", PROMPT_FB),
            _task_call("call_pr", PROMPT_PR),
        ],
    )
    consumer.on_complete(parent)
    fb_span, pr_span = (e.id for e in stub.span_begins())

    # first task completes; second still running and still attributable
    parent_next = _model_event([ChatMessageUser(content="go"), _task_result("call_fb")])
    consumer.on_pending(parent_next)
    assert [e.id for e in stub.span_ends()] == [fb_span]

    pr_call = _model_event([ChatMessageUser(content=PROMPT_PR)])
    consumer.on_pending(pr_call)
    assert pr_call.span_id == pr_span


def test_nested_task_parented_under_subagent_span(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="go")], tool_calls=[_task_call("call_fb", PROMPT_FB)]
    )
    consumer.on_complete(parent)
    fb_span = stub.span_begins()[0].id

    # sub-agent call spawns its own task
    child_call = _model_event(
        [ChatMessageUser(content=PROMPT_FB)],
        tool_calls=[_task_call("call_nested", PROMPT_PR)],
    )
    consumer.on_pending(child_call)
    consumer.on_complete(child_call)

    nested_begin = stub.span_begins()[1]
    assert nested_begin.parent_id == fb_span


# ---------------------------------------------------------------------------
# 4. reset: orphaned spans are closed innermost-first
# ---------------------------------------------------------------------------


def test_reset_closes_orphans_innermost_first(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[
            _task_call("call_fb", PROMPT_FB),
            _task_call("call_pr", PROMPT_PR),
        ],
    )
    consumer.on_complete(parent)
    fb_span, pr_span = (e.id for e in stub.span_begins())

    consumer.reset()
    assert [e.id for e in stub.span_ends()] == [pr_span, fb_span]

    # idempotent
    consumer.reset()
    assert len(stub.span_ends()) == 2
