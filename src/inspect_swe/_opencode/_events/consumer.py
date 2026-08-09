"""Bridge `ModelEventSink` for OpenCode sub-agent spans (bridge-only).

Installed on the agent bridge so the bridge hands us every `ModelEvent` for
routing instead of emitting it to the transcript itself. From those events
alone (no OpenCode stdout parsing) we reconstruct the agent-span tree:

  1. **Open** (race-free) — when a parent's output contains `task` tool-calls
     (OpenCode's delegation tool), `on_complete` opens an agent
     `SpanBeginEvent` for each, keyed by the task tool-call id, and registers
     the task prompt for attribution. This happens synchronously before the
     bridge response is returned, so the span is open before the sub-agent
     can make its first call.

  2. **Attribute** — `on_pending` resolves each call's span by substring-
     matching its first user-message text against the open task prompts.
     OpenCode runs each sub-agent as a child session whose first user message
     is the task prompt, and re-sends the full session on every request, so
     this works for every sub-agent call (not just the first). Zero/multiple
     matches → outer span (defensive default).

  3. **Close** — the parent's next request carries the task tool *result* as
     a `ChatMessageTool` correlated by `tool_call_id`; seeing it in
     `on_pending` closes the span (before attribution, so the parent's own
     request lands on the outer span). `reset()` closes orphans between
     attempts and at the end.

Concurrency: attribution is per-request (keyed on the call's own prompt, not
wall-clock state), so parallel sub-agents are handled correctly — each call
routes to its own span regardless of interleaving, and each span closes
independently via its unique task `tool_call_id`.
"""

from dataclasses import dataclass
from logging import getLogger

from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.util._span import current_span_id

from .toolview import tool_view

logger = getLogger(__name__)


# Minimum task-prompt length to consider for substring matching, guarding
# against short prompts accidentally matching unrelated content.
_MIN_PROMPT_LENGTH = 16


@dataclass
class _OpenAgent:
    """A sub-agent span currently open (task tool_use seen, no tool_result yet)."""

    span_id: str
    prompt: str


class OpenCodeConsumer(ModelEventSink):
    def __init__(self) -> None:
        # task tool_call_id → open sub-agent span. Insertion order = open
        # order (used to close innermost-first in reset()).
        self._agents: dict[str, _OpenAgent] = {}

        # ModelEvents we've _event()'d, so on_complete knows to _event_updated.
        self._emitted_events: set[int] = set()

    @property
    def outer_span_id(self) -> str | None:
        """Span for main-agent attribution, resolved at emission time.

        Must not be captured once at construction: the enclosing span can
        change across attempts and a frozen id would pin every event to the
        span active at construction time.
        """
        return current_span_id()

    def reset(self) -> None:
        """Close any open spans and clear per-attempt state.

        Called between OpenCode attempts and after the attempt loop, so the
        span tree stays balanced even if OpenCode exited before a task's
        tool_result was produced.
        """
        for call_id in reversed(list(self._agents.keys())):
            agent = self._agents.pop(call_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))
        self._emitted_events.clear()

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        # close spans whose task tool_result appears in this request's input
        # (done before attribution so the parent's own request — which carries
        # the completed task's result — resolves to the outer span)
        self._close_completed(event.input)

        # attribute this call to a span
        event.span_id = self._attribute(event.input)

        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            # custom rendering for OpenCode's task tool (see toolview.py)
            for tc in msg.tool_calls:
                if tc.view is None:
                    custom = tool_view(tc.function, tc.arguments or {})
                    if custom is not None:
                        tc.view = custom

            # open a span for each spawned sub-agent — synchronously, before
            # the bridge response is returned, so the span is ready before the
            # sub-agent's first call arrives.
            parent_span_id = event.span_id or self.outer_span_id
            for tc in msg.tool_calls:
                if tc.function != "task":
                    continue
                args = tc.arguments or {}
                prompt = args.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    continue
                if tc.id in self._agents:
                    continue  # idempotent (defensive against retries)
                span_id = f"agent-{tc.id}"
                self._agents[tc.id] = _OpenAgent(span_id=span_id, prompt=prompt)
                span_name = args.get("subagent_type") or "agent"
                description = args.get("description") or ""
                transcript()._event(
                    SpanBeginEvent(
                        id=span_id,
                        parent_id=parent_span_id,
                        type="agent",
                        name=str(span_name),
                        metadata={"description": description} if description else None,
                    )
                )

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _close_completed(self, input_messages: list[ChatMessage]) -> None:
        """Close spans for any open task whose tool_result is in this input."""
        for msg in input_messages:
            if not isinstance(msg, ChatMessageTool) or msg.tool_call_id is None:
                continue
            agent = self._agents.pop(msg.tool_call_id, None)
            if agent is not None:
                transcript()._event(SpanEndEvent(id=agent.span_id))

    def _attribute(self, input_messages: list[ChatMessage]) -> str | None:
        """Resolve the span_id for an incoming bridge call.

        Substring-matches the first user message's text against currently-open
        task prompts (a sub-agent session's first user message is its task
        prompt, re-sent on every request). Exactly one match → that
        sub-agent's span; zero/multiple → outer span (defensive default).
        """
        if not self._agents:
            return self.outer_span_id

        user_text = self._first_user_text(input_messages)
        if not user_text:
            return self.outer_span_id

        matches = [
            agent
            for agent in self._agents.values()
            if len(agent.prompt) >= _MIN_PROMPT_LENGTH and agent.prompt in user_text
        ]
        if len(matches) == 1:
            return matches[0].span_id
        return self.outer_span_id

    @staticmethod
    def _first_user_text(input_messages: list[ChatMessage]) -> str | None:
        """Return the text of the first ChatMessageUser past leading system messages."""
        for msg in input_messages:
            if isinstance(msg, ChatMessageSystem):
                continue
            if isinstance(msg, ChatMessageUser):
                return msg.text
            break
        return None
