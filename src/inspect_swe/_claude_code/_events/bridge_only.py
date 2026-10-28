"""`LiveConsumer` for Claude Code with no stream-json output (ACP)."""

from dataclasses import dataclass

from inspect_ai.event._model import ModelEvent
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)

from ..._util.model_event_sink import started
from . import live_consumer
from .live_consumer import LiveConsumer
from .shapes import (
    ASYNC_LAUNCHED,
    FORK_STARTED,
    agent_id_in,
    first_user_message,
    is_compaction_request,
    send_message_succeeded,
    summary_body,
    task_notifications,
)
from .subagents import SubAgent


@dataclass
class _Notifications:
    """Notifications a `BridgeOnlyLiveConsumer` expects for a sub-agent."""

    # still to come for runs their own reply closed
    pending: int = 0
    # runs closed that way before we knew the agent was in the background
    early_reply_closes: int = 0


class BridgeOnlyLiveConsumer(LiveConsumer):
    """`LiveConsumer` that closes spans from bridge traffic (no stream-json)."""

    def __init__(self) -> None:
        super().__init__()
        self._notifications: dict[str, _Notifications] = {}
        # (conversation, index, role, tool call id, text) of parent messages
        # already acted on, which a retried request repeats
        self._handled: set[tuple[int, int, str, str, int]] = set()

    def reset(self) -> None:
        super().reset()
        self._notifications.clear()
        self._handled.clear()

    def _expected(self, agent: SubAgent) -> _Notifications:
        return self._notifications.setdefault(agent.origin, _Notifications())

    def on_start(self, input: list[ChatMessage]) -> str | None:
        # close before attributing, as on_pending does without on_start
        with live_consumer._tracking("parent input"):
            self._close_from_input(input)
        return super().on_start(input)

    def on_pending(self, event: ModelEvent) -> None:
        if not started(self):
            with live_consumer._tracking("parent input"):
                self._close_from_input(event.input)
        super().on_pending(event)

    def on_complete(self, event: ModelEvent) -> None:
        super().on_complete(event)
        with live_consumer._tracking("final reply"):
            self._close_finished_run(event)

    def _compacted(self, event: ModelEvent) -> None:
        # nothing else reports the main agent's compaction
        if self._agents.run_of_span(event.span_id) is not None:
            super()._compacted(event)
        elif summary_body(event.output.message.text) is not None:
            self._compaction_event(event.span_id, event.output.usage)

    def _close_finished_run(self, event: ModelEvent) -> None:
        """Close a sub-agent's span at its text-only final reply."""
        msg = event.output.message if event.output and event.output.choices else None
        if (
            event.error is not None
            or msg is None
            or msg.tool_calls
            or event.output.stop_reason != "stop"
        ):
            return
        if is_compaction_request(event.input):
            return
        for run in self._agents.open_runs():
            if run.span_id == event.span_id:
                self._close(run)
                run.ended_by_reply = True
                if run.agent.background:
                    self._expected(run.agent).pending += 1
                else:
                    self._expected(run.agent).early_reply_closes += 1

    def _mark_background(self, agent: SubAgent) -> None:
        if self._learned_background(agent):
            expected = self._expected(agent)
            expected.pending += expected.early_reply_closes
            expected.early_reply_closes = 0

    def _close_from_input(self, input_messages: list[ChatMessage]) -> None:
        """Close sub-agents whose result or notification the caller now sees.

        Only messages after the caller's last assistant turn are new.
        """
        for msg in input_messages:
            if isinstance(msg, ChatMessageTool):
                agent_id = agent_id_in(msg.text)
                if agent_id:
                    self._bind_agent_id(self._agents.get(msg.tool_call_id), agent_id)

        last_assistant = max(
            (
                i
                for i, m in enumerate(input_messages)
                if isinstance(m, ChatMessageAssistant)
            ),
            default=-1,
        )
        # the first user message tells conversations apart (and changes when
        # one compacts, which restarts its indices)
        first_user = first_user_message(input_messages)
        first = hash(first_user.text) if first_user else 0
        for i in range(last_assistant + 1, len(input_messages)):
            msg = input_messages[i]
            handled = (
                first,
                i,
                msg.role,
                str(getattr(msg, "tool_call_id", "")),
                hash(msg.text),
            )
            if handled in self._handled:
                continue
            self._handled.add(handled)
            if isinstance(msg, ChatMessageTool):
                self._tool_result(str(msg.tool_call_id), msg)
            elif isinstance(msg, ChatMessageUser):
                for tool_use_id, task_id in task_notifications(msg.text):
                    self._input_notification(tool_use_id, task_id)

    def _tool_result(self, tool_use_id: str, msg: ChatMessageTool) -> None:
        text = msg.text.lstrip()
        launched = self._agents.get(tool_use_id)
        if launched is not None and text.startswith(ASYNC_LAUNCHED):
            # also when the run it acknowledges already ended
            self._mark_background(launched)
        block = {"content": msg.text, "is_error": msg.error is not None}
        run = self._agents.open_run(tool_use_id)
        if run is None or run.via_send:
            if not send_message_succeeded(None, block):
                self._send_failed(tool_use_id)
            if run is None and launched is not None and not launched.background:
                self._confirm_ended(launched)
        elif not text.startswith((FORK_STARTED, ASYNC_LAUNCHED)):
            self._close(run)

    def _confirm_ended(self, agent: SubAgent) -> None:
        """The parent saw a reply-closed run's result: it makes no more calls."""
        latest = agent.latest
        if latest is not None and not latest.open:
            latest.ended_by_reply = False

    def _input_notification(self, tool_use_id: str | None, task_id: str | None) -> None:
        agent = self._agents.by_agent_id(task_id)
        if agent is not None and self._expected(agent).pending > 0:
            # for a run its own reply already closed
            self._expected(agent).pending -= 1
            if self._expected(agent).pending == 0:
                self._confirm_ended(agent)
            return
        self._notified(tool_use_id, task_id, closed_is_stale=True)
