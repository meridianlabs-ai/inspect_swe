"""Real-time consumer of Claude Code JSONL output.

Two responsibilities, both driven from the same `LiveConsumer` instance:

1. **`ModelEventSink`** — installed on the agent bridge so the bridge hands
   us every `ModelEvent` for routing instead of emitting it to the transcript
   itself. We attribute each event to the correct agent span at `on_pending`
   time, then forward to the transcript with the attributed span_id.

   **Attribution mechanism (substring match against pending sub-agents)**:

   Claude Code 2.1.x makes sub-agents opaque in JSONL — sub-agent model
   calls have no `parent_tool_use_id`, no separate `session_id`, and no
   `isSidechain` marker. So we cannot use JSONL alone to drive span
   open/close for sub-agents.

   When a bridge call's output contains Task/Agent tool_calls (i.e. the
   parent agent is spawning sub-agents), `on_complete` does two things,
   synchronously, *before* the bridge response is sent back to Claude
   Code:

     1. Open an agent `SpanBeginEvent` for each Task/Agent tool_use, with
        `parent_id` = the parent call's span_id and span_id =
        `agent-{tool_use_id}`.
     2. Register the sub-agent in `_pending_subagents` (mapping
        `tool_use_id → prompt`).

   When sub-agent's first bridge call arrives, `_attribute` scans its
   first user message text for any pending sub-agent prompt as a
   substring. A single hit identifies the sub-agent and we look up its
   already-open span. Zero hits → main-agent call (outer span). Multiple
   hits (rare; concurrent sub-agents with substring-overlapping prompts)
   → outer span as defensive default.

   Doing both open + register in `on_complete` (rather than from JSONL)
   eliminates a race: the sub-agent's bridge call arrives at the bridge
   server ~1–2 seconds before our stdout reader processes the parent's
   `assistant` JSONL line, so a JSONL-driven span open would miss every
   first call.

   This works for every sub-agent call (not just the first) because
   Claude Code re-sends the sub-agent's full conversation history on
   each request, with the original Task prompt always at `input[0]`.
   `_pending_subagents` and the open span are cleared in `_handle_user`
   when the matching `tool_result` arrives.

2. **JSONL consumer** — `process_jsonl_line` reads each line printed by
   Claude Code's `--output-format stream-json` and emits agent
   `SpanEndEvent` (on `tool_result` for Task/Agent), and emits
   `CompactionEvent` for `compact_boundary` system events. Span
   *opening* is no longer driven from JSONL — see callback (1) above.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink

from .toolview import tool_view

logger = getLogger(__name__)


# Minimum prompt length to consider for substring matching. Short prompts
# could accidentally appear in unrelated content; this guards against false
# positives while still catching every plausible Task prompt (which are
# typically full sentences).
_MIN_PROMPT_LENGTH = 16


@dataclass
class _OpenAgent:
    """An agent span currently open (Task tool_use seen, no tool_result yet)."""

    span_id: str


class LiveConsumer(ModelEventSink):
    """Sink + JSONL consumer.

    The bridge calls `on_pending` / `on_complete` for every `ModelEvent`.
    The runner loop calls `process_jsonl_line` for every JSONL line printed
    by Claude Code.
    """

    def __init__(self, outer_span_id: str | None) -> None:
        self.outer_span_id = outer_span_id

        # tool_use_id → _OpenAgent for currently-open agent spans (Task/Agent
        # tool_use blocks we've SpanBegin'd, not yet SpanEnd'd).
        self._open_agents: dict[str, _OpenAgent] = {}

        # tool_use_id → Task prompt for sub-agents currently RUNNING. Populated
        # in `on_complete` when a parent's output contains Task/Agent tool_calls
        # (synchronously, before the response is sent back to Claude Code, so
        # the entry is ready before any sub-agent can make a bridge call).
        # Cleared in `_handle_user` when the matching tool_result arrives.
        self._pending_subagents: dict[str, str] = {}

        # Track which event objects we've already _event()'d so on_complete
        # knows whether to emit _event_updated (yes if we emitted) vs swallow
        # (no if we didn't — shouldn't happen with current logic but defensive).
        self._emitted_events: set[int] = set()

    def reset(self) -> None:
        """Close any open spans and clear per-attempt state.

        Called between Claude Code subprocess restarts (retry attempts) and
        in a `finally` after the retry loop. Emits `SpanEndEvent` for every
        still-open agent span (innermost first) so the transcript stays
        balanced even if Claude Code crashed before its tool_result blocks
        were written.
        """
        for tool_use_id in reversed(list(self._open_agents.keys())):
            agent = self._open_agents.pop(tool_use_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))
        self._pending_subagents.clear()
        self._emitted_events.clear()

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        event.span_id = self._attribute(event.input)
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            # Attach custom rendering for Claude Code's built-in tools.
            # inspect_ai's `tool_call_view` only handles tools registered as
            # ToolDefs; Claude Code's built-in tools (Write, Task, Agent,
            # ExitPlanMode, …) aren't, so we fill in our own views here
            # before `_event_updated` lets the viewer render the call.
            for tc in msg.tool_calls:
                if tc.view is None:
                    custom = tool_view(tc.function, tc.arguments or {})
                    if custom is not None:
                        tc.view = custom

            # If this call's response launched any Task/Agent sub-agents,
            # open their spans and register pending entries NOW —
            # synchronously, before the bridge response is sent back to
            # Claude Code. By the time any sub-agent makes its first bridge
            # call, both `_open_agents` and `_pending_subagents` are ready.
            parent_span_id = event.span_id or self.outer_span_id
            for tc in msg.tool_calls:
                if tc.function not in ("Task", "Agent"):
                    continue
                args = tc.arguments or {}
                prompt = args.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    continue
                if tc.id in self._open_agents:
                    # idempotent — defensive against retries
                    continue
                agent_span_id = f"agent-{tc.id}"
                self._open_agents[tc.id] = _OpenAgent(span_id=agent_span_id)
                self._pending_subagents[tc.id] = prompt
                span_name = args.get("subagent_type") or args.get("name") or "agent"
                description = args.get("description") or ""
                transcript()._event(
                    SpanBeginEvent(
                        id=agent_span_id,
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
    # JSONL consumer
    # ------------------------------------------------------------------

    def process_jsonl_line(self, raw: dict[str, Any]) -> None:
        """Process one raw JSONL event from Claude Code.

        Called from the runner loop as each JSONL line arrives.

        Note: sub-agent span *opening* is no longer driven from JSONL —
        it happens in `on_complete` (synchronous with the parent's bridge
        call, ahead of the race with stdout-buffered JSONL arrival). We
        only consume `user` (for tool_result → span close) and `system`
        (for compaction) events here.
        """
        event_type = raw.get("type")
        if event_type == "user":
            self._handle_user(raw)
        elif event_type == "system":
            self._handle_system(raw)

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def _attribute(self, input_messages: list[ChatMessage]) -> str | None:
        """Resolve the span_id for an incoming bridge call.

        Substring-matches the first user message's text against currently-
        pending sub-agent prompts. Exactly one match → that sub-agent's
        span. Zero or multiple matches → outer span.
        """
        if not self._pending_subagents:
            return self.outer_span_id

        user_text = self._first_user_text(input_messages)
        if not user_text:
            return self.outer_span_id

        matches: list[str] = []
        for tool_use_id, prompt in self._pending_subagents.items():
            if len(prompt) < _MIN_PROMPT_LENGTH:
                continue
            if prompt in user_text:
                matches.append(tool_use_id)

        if len(matches) == 1:
            agent = self._open_agents.get(matches[0])
            if agent is not None:
                return agent.span_id
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

    # ------------------------------------------------------------------
    # JSONL event handlers
    # ------------------------------------------------------------------

    def _handle_user(self, raw: dict[str, Any]) -> None:
        """Close agent spans and clear pending-sub-agent entries for tool_result blocks."""
        message = raw.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if not tool_use_id:
                continue
            # Clear pending-subagent entry (no-op for non-Task tool_results).
            self._pending_subagents.pop(tool_use_id, None)
            agent = self._open_agents.pop(tool_use_id, None)
            if agent is None:
                continue
            transcript()._event(SpanEndEvent(id=agent.span_id))

    def _handle_system(self, raw: dict[str, Any]) -> None:
        """Handle system events (only compaction boundaries today).

        Sub-agent lifecycle (`task_started`/`task_notification`) intentionally
        ignored: registration happens in `on_complete` (synchronous with the
        parent's bridge call, ahead of any race), and cleanup happens in
        `_handle_user` on the matching tool_result.
        """
        subtype = raw.get("subtype")
        if subtype == "compact_boundary":
            self._handle_compact_boundary(raw)

    def _handle_compact_boundary(self, raw: dict[str, Any]) -> None:
        parent_tool_use_id = raw.get("parent_tool_use_id")
        if parent_tool_use_id:
            parent_agent = self._open_agents.get(parent_tool_use_id)
            span_id = parent_agent.span_id if parent_agent else self.outer_span_id
        else:
            span_id = self.outer_span_id

        compact_meta = raw.get("compactMetadata") or {}
        transcript()._event(
            CompactionEvent(
                source="claude_code",
                tokens_before=compact_meta.get("preTokens"),
                span_id=span_id,
                metadata={
                    "trigger": compact_meta.get("trigger", "auto"),
                    "content": raw.get("content") or "Conversation compacted",
                },
            )
        )
