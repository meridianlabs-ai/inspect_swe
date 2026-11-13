"""Claude Code sub-agent span attribution against the request shapes of 2.1.3xx.

Each request below mirrors what Claude Code 2.1.311 sends through the bridge
(captured from live runs): a billing header system block that carries
`cc_is_subagent=true` on sub-agent requests, a sub-agent opening whose first
user text is a `<system-reminder>` with the Task prompt after it, and Agent
calls that run in the background, so their tool_result ("Async agent launched
successfully") arrives before the sub-agent's first request and the
`task_notification` system event marks the end of its work. The older shape
(no header, the prompt as the first user message, a foreground Task ended by
its tool_result) must keep working.

Each request is driven the way the bridge drives it: `classify` at filter time,
then `on_start` for the generate, in the request's own context.
"""

import contextvars
from typing import Any

from inspect_ai.agent import AgentBridgeContext
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall
from inspect_swe._claude_code._events import live_consumer as consumer_module
from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._claude_code.model import resolve_claude_code_models

UNAME = "Run `uname -s` with Bash and reply with its output only."
WHOAMI = "Run `whoami` with Bash and reply with its output only."
TASK = (
    "In a single message, launch TWO Agent tool calls in parallel. "
    f"Subagent A: '{UNAME}' Subagent B: '{WHOAMI}'"
)
REMINDER = (
    "<system-reminder>\nAttribution for git commits and pull requests you "
    "create from here on ...\n</system-reminder>\n"
)
HANDBACK_REMINDER = (
    "<system-reminder>\nYour final report is delivered through "
    "SubagentHandback ...\n</system-reminder>"
)


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)

    def _event_updated(self, event: Any) -> None:
        pass

    def begun(self) -> list[str]:
        return [e.id for e in self.events if isinstance(e, SpanBeginEvent)]

    def ended(self) -> list[str]:
        return [e.id for e in self.events if isinstance(e, SpanEndEvent)]


def _consumer(monkeypatch: Any) -> tuple[LiveConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    return LiveConsumer(resolve_claude_code_models("mockllm/model", None)), stub


def _header(key: str, subagent: bool) -> ChatMessageSystem:
    flag = " cc_is_subagent=true;" if subagent else ""
    return ChatMessageSystem(
        content=(
            f"x-anthropic-billing-header: cc_version=2.1.311.{key}; "
            f"cc_entrypoint=sdk-cli;{flag}"
        )
    )


def _subagent_request(
    prompt: str, key: str, turns: int = 0, header: bool = True
) -> list[ChatMessage]:
    """A 2.1.311 sub-agent request: reminder, prompt, then `turns` tool turns."""
    messages: list[ChatMessage] = [_header(key, subagent=True)] if header else []
    messages += [
        ChatMessageSystem(content="You are an agent for Claude Code."),
        ChatMessageUser(
            content=[
                ContentText(text=REMINDER),
                ContentText(text=prompt + "\n"),
                ContentText(text=HANDBACK_REMINDER),
            ]
        ),
        ChatMessageSystem(content="# Environment"),
    ]
    for turn in range(turns):
        call = ToolCall(id=f"bash_{key}_{turn}", function="Bash", arguments={})
        messages += [
            ChatMessageAssistant(content="", tool_calls=[call]),
            ChatMessageTool(content="Linux", tool_call_id=call.id),
        ]
    return messages


def _main_request(text: str = TASK, key: str = "40b") -> list[ChatMessage]:
    """A main-thread request whose opening (the user's task) quotes both prompts."""
    return [
        _header(key, subagent=False),
        ChatMessageSystem(content="You are Claude Code."),
        ChatMessageUser(content=[ContentText(text=REMINDER), ContentText(text=text)]),
    ]


def _classifier_request() -> list[ChatMessage]:
    """The auto-mode permission classifier: its input quotes the transcript."""
    return [
        _header("e93", subagent=False),
        ChatMessageSystem(content="You are a security classifier."),
        ChatMessageUser(content=f'<transcript>\n\n{{"user":"{TASK}"}}\n</transcript>'),
    ]


def _launch(consumer: LiveConsumer, *calls: tuple[str, str]) -> None:
    """The main thread's output launches one Agent call per (tool_use_id, prompt)."""
    message = ChatMessageAssistant(
        content="",
        tool_calls=[
            ToolCall(
                id=tool_use_id,
                function="Agent",
                arguments={"prompt": prompt, "subagent_type": "general-purpose"},
            )
            for tool_use_id, prompt in calls
        ],
    )
    consumer.on_complete(
        ModelEvent(
            model="m",
            input=[],
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
            output=ModelOutput.from_message(message),
        )
    )


def _launched_line(tool_use_id: str, prompt: str) -> dict[str, Any]:
    """The JSONL tool_result Claude Code 2.1.311 prints for a background Agent."""
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [
                        {
                            "type": "text",
                            "text": "Async agent launched successfully. (This "
                            "tool result is internal metadata ...)",
                        }
                    ],
                }
            ]
        },
        "tool_use_result": {
            "isAsync": True,
            "status": "async_launched",
            "agentId": f"agent_{tool_use_id}",
            "prompt": prompt,
        },
    }


def _notification_line(tool_use_id: str) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "task_notification",
        "task_id": f"agent_{tool_use_id}",
        "tool_use_id": tool_use_id,
        "status": "completed",
    }


def _tool_result_line(tool_use_id: str, text: str = "done") -> dict[str, Any]:
    return {
        "type": "user",
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
            ]
        },
    }


def _request(
    consumer: LiveConsumer, messages: list[ChatMessage], slug: str | None
) -> tuple[AgentBridgeContext, str | None]:
    """Classify a request at filter time, then start its generate."""

    def run() -> tuple[AgentBridgeContext, str | None]:
        with bridged_request_scope(slug):
            context = consumer.classify(get_model("mockllm/model"), messages, [])
            return context, consumer.on_start(messages)

    return contextvars.copy_context().run(run)


def _span(
    consumer: LiveConsumer, messages: list[ChatMessage], slug: str | None
) -> str | None:
    return _request(consumer, messages, slug)[1]


# ---------------------------------------------------------------------------
# foreground and background sub-agents
# ---------------------------------------------------------------------------


def test_foreground_subagent_owns_its_requests_until_its_tool_result(
    monkeypatch: Any,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    sub = consumer._models.subagent
    _launch(consumer, ("toolu_a", UNAME))

    assert _span(consumer, _subagent_request(UNAME, "6f9"), sub) == "agent-toolu_a"
    assert _span(consumer, _subagent_request(UNAME, "6f9", 1), sub) == "agent-toolu_a"
    assert stub.ended() == []

    consumer.process_jsonl_line(_tool_result_line("toolu_a", "Linux"))
    assert stub.ended() == ["agent-toolu_a"]
    assert _span(consumer, _subagent_request(UNAME, "6f9", 2), sub) is None


def test_background_subagent_stays_attributed_after_its_launch_result(
    monkeypatch: Any,
) -> None:
    """The launch tool_result arrives before the sub-agent's first request."""
    consumer, stub = _consumer(monkeypatch)
    sub = consumer._models.subagent
    _launch(consumer, ("toolu_a", UNAME))
    consumer.process_jsonl_line(_launched_line("toolu_a", UNAME))
    assert stub.ended() == []

    context, span_id = _request(consumer, _subagent_request(UNAME, "6f9"), sub)
    assert span_id == "agent-toolu_a"
    assert context.subagent_id == "toolu_a"
    assert _span(consumer, _subagent_request(UNAME, "6f9", 1), sub) == "agent-toolu_a"
    # the main thread keeps working meanwhile
    assert _span(consumer, _main_request(), consumer._models.presented) is None

    consumer.process_jsonl_line(_notification_line("toolu_a"))
    assert stub.ended() == ["agent-toolu_a"]
    assert _span(consumer, _subagent_request(UNAME, "6f9", 2), sub) is None


def test_parallel_background_subagents_keep_their_own_spans(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    sub = consumer._models.subagent
    _launch(consumer, ("toolu_a", UNAME), ("toolu_b", WHOAMI))
    consumer.process_jsonl_line(_launched_line("toolu_a", UNAME))
    consumer.process_jsonl_line(_launched_line("toolu_b", WHOAMI))
    assert stub.begun() == ["agent-toolu_a", "agent-toolu_b"]

    # interleaved requests, as the two threads run concurrently
    sequence = [
        (_subagent_request(UNAME, "6f9"), "agent-toolu_a"),
        (_subagent_request(WHOAMI, "f5b"), "agent-toolu_b"),
        (_subagent_request(WHOAMI, "f5b", 1), "agent-toolu_b"),
        (_subagent_request(UNAME, "6f9", 1), "agent-toolu_a"),
    ]
    for messages, expected in sequence:
        assert _span(consumer, messages, sub) == expected

    consumer.process_jsonl_line(_notification_line("toolu_b"))
    assert stub.ended() == ["agent-toolu_b"]
    assert _span(consumer, _subagent_request(UNAME, "6f9", 2), sub) == "agent-toolu_a"
    assert _span(consumer, _subagent_request(WHOAMI, "f5b", 2), sub) is None


def test_reset_closes_a_background_subagent_that_never_reported(
    monkeypatch: Any,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME))
    consumer.process_jsonl_line(_launched_line("toolu_a", UNAME))
    consumer.reset()
    assert stub.ended() == ["agent-toolu_a"]


def test_launch_text_marks_a_background_result_without_structured_data(
    monkeypatch: Any,
) -> None:
    """A line with several tool_results: the structured result is not per block."""
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME), ("toolu_b", WHOAMI))
    launched = _launched_line("toolu_a", UNAME)
    launched["message"]["content"].append(
        {"type": "tool_result", "tool_use_id": "toolu_b", "content": "root"}
    )
    consumer.process_jsonl_line(launched)
    assert stub.ended() == ["agent-toolu_b"]
    assert "toolu_a" in consumer._pending_subagents


# ---------------------------------------------------------------------------
# where the prompt is
# ---------------------------------------------------------------------------


def test_prompt_after_a_separate_reminder_message_is_matched(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME))
    messages: list[ChatMessage] = [
        _header("6f9", subagent=True),
        ChatMessageUser(content=REMINDER),
        ChatMessageUser(content=UNAME),
    ]
    assert _span(consumer, messages, consumer._models.subagent) == "agent-toolu_a"


def test_prompt_only_in_a_later_turn_is_not_matched(monkeypatch: Any) -> None:
    """Only the opening (before the first assistant turn) identifies the thread."""
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME))
    messages: list[ChatMessage] = [
        ChatMessageUser(content="something else entirely, long enough"),
        ChatMessageAssistant(content="ok"),
        ChatMessageUser(content=f"now: {UNAME}"),
    ]
    assert _span(consumer, messages, consumer._models.subagent) is None


def test_older_first_message_prompt_still_attributes(monkeypatch: Any) -> None:
    """No billing header, no subagent slug, the prompt as the first user message."""
    consumer, stub = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME))
    messages: list[ChatMessage] = [
        ChatMessageSystem(content="You are a sub-agent."),
        ChatMessageUser(content=UNAME),
    ]
    context, span_id = _request(consumer, messages, None)
    assert span_id == "agent-toolu_a"
    assert context == AgentBridgeContext(
        "subagent", subagent_id="toolu_a", subagent_type="general-purpose"
    )
    consumer.process_jsonl_line(_tool_result_line("toolu_a"))
    assert stub.ended() == ["agent-toolu_a"]
    assert _span(consumer, messages, None) is None


# ---------------------------------------------------------------------------
# other threads that quote a sub-agent's prompt
# ---------------------------------------------------------------------------


def test_main_thread_quoting_the_prompt_stays_on_the_outer_span(
    monkeypatch: Any,
) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME))
    consumer.process_jsonl_line(_launched_line("toolu_a", UNAME))

    # by slug
    assert _span(consumer, _main_request(), consumer._models.presented) is None
    # by the billing header once Claude Code has shown it sends the flag,
    # whatever the slug (or with none)
    assert _span(consumer, _subagent_request(UNAME, "6f9"), None) == "agent-toolu_a"
    for slug in ("some-unrecognized-slug", None):
        context, span_id = _request(consumer, _main_request(), slug)
        assert span_id is None
        assert context.kind != "subagent"
        context, span_id = _request(consumer, _classifier_request(), slug)
        assert span_id is None
        assert context.kind != "subagent"


def test_flag_seen_check_is_needed_before_the_header_rules_out(
    monkeypatch: Any,
) -> None:
    """A Claude Code that sends the header but never the flag keeps prompt matching."""
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME))
    messages = _subagent_request(UNAME, "6f9")
    messages[0] = _header("6f9", subagent=False)
    assert _span(consumer, messages, "some-unrecognized-slug") == "agent-toolu_a"


def test_header_flag_makes_an_unrecognized_slug_a_subagent(monkeypatch: Any) -> None:
    consumer, _ = _consumer(monkeypatch)
    _launch(consumer, ("toolu_a", UNAME))
    context, span_id = _request(
        consumer, _subagent_request(UNAME, "6f9"), "some-unrecognized-slug"
    )
    assert span_id == "agent-toolu_a"
    assert context == AgentBridgeContext(
        "subagent", subagent_id="toolu_a", subagent_type="general-purpose"
    )
    # a flagged request with no pending match is still a sub-agent's, unplaced
    context, span_id = _request(
        consumer, _subagent_request("an unrelated prompt of some length", "abc"), None
    )
    assert context == AgentBridgeContext("subagent")
    assert span_id is None
