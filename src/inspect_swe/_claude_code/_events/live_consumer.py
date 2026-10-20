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
   when the matching `tool_result` arrives — except for background agents
   (`run_in_background`, the default in recent Claude Code), whose
   `tool_result` is an immediate `async_launched` acknowledgement while the
   sub-agent keeps running. Those stay open until the `task_notification`
   system event for the same `tool_use_id` (or `reset()`).

2. **JSONL consumer** — `process_jsonl_line` reads each line printed by
   Claude Code's `--output-format stream-json` and emits agent
   `SpanEndEvent` (on `tool_result` for foreground Task/Agent, on
   `task_notification` for background ones), and emits
   `CompactionEvent` for `compact_boundary` system events. Span
   *opening* is no longer driven from JSONL — see callback (1) above.
"""

import re
from dataclasses import dataclass
from logging import getLogger
from typing import Any

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.model._model_output import StopReason
from inspect_ai.util._span import current_span_id

from .toolview import tool_view

logger = getLogger(__name__)


# Minimum prompt length to consider for substring matching. Short prompts
# could accidentally appear in unrelated content; this guards against false
# positives while still catching every plausible Task prompt (which are
# typically full sentences).
_MIN_PROMPT_LENGTH = 16


@dataclass
class _OpenAgent:
    """An agent span currently open (Task tool_use seen, sub-agent not done)."""

    span_id: str


class LiveConsumer(ModelEventSink):
    """Sink + JSONL consumer.

    The bridge calls `on_pending` / `on_complete` for every `ModelEvent`.
    The runner loop calls `process_jsonl_line` for every JSONL line printed
    by Claude Code.
    """

    def __init__(self) -> None:
        # tool_use_id → _OpenAgent for currently-open agent spans (Task/Agent
        # tool_use blocks we've SpanBegin'd, not yet SpanEnd'd).
        self._open_agents: dict[str, _OpenAgent] = {}

        # tool_use_id → Task prompt for sub-agents currently RUNNING. Populated
        # in `on_complete` when a parent's output contains Task/Agent tool_calls
        # (synchronously, before the response is sent back to Claude Code, so
        # the entry is ready before any sub-agent can make a bridge call).
        # Cleared when the sub-agent finishes (see `_close_agent`).
        self._pending_subagents: dict[str, str] = {}

        # whether Claude Code reports background completions as `system`
        # task_notification events (2.1.303); older versions only show the
        # parent a `<task-notification>` in a `user` line
        self._system_notifications = False

        # background agent task_id → tool_use_id, for task_notifications that
        # carry no tool_use_id
        self._task_ids: dict[str, str] = {}

        # Track which event objects we've already _event()'d so on_complete
        # knows whether to emit _event_updated (yes if we emitted) vs swallow
        # (no if we didn't — shouldn't happen with current logic but defensive).
        self._emitted_events: set[int] = set()

        # Stop reason of the most recent completed ModelEvent. Used by the
        # runner loop to distinguish an Anthropic refusal (content_filter)
        # from a genuine scaffold crash when Claude Code exits non-zero.
        self._last_stop_reason: StopReason | None = None

    @property
    def last_stop_reason(self) -> StopReason | None:
        """Stop reason of the last completed model event this attempt."""
        return self._last_stop_reason

    @property
    def outer_span_id(self) -> str | None:
        """Span for main-agent attribution, resolved at emission time.

        Must not be captured once at construction: with checkpointing
        active, the enclosing checkpoint span rotates at each fire and a
        frozen id would pin every event to the first checkpoint.
        """
        return current_span_id()

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
        self._task_ids.clear()
        self._emitted_events.clear()
        self._last_stop_reason = None

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        event.span_id = self._attribute(event.input)
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        # Record the terminal stop reason (guard against empty choices, where
        # ModelOutput.stop_reason raises). This fires for sub-agent calls too,
        # but the runner only consults it on a non-zero exit.
        if event.output and event.output.choices:
            self._last_stop_reason = event.output.stop_reason

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

        # a parent working beside a background sub-agent may quote its prompt;
        # only the parent's history holds the launching tool call
        launched_here = {
            tc.id
            for msg in input_messages
            if isinstance(msg, ChatMessageAssistant)
            for tc in msg.tool_calls or []
        }
        matches: list[str] = []
        for tool_use_id, prompt in self._pending_subagents.items():
            if len(prompt) < _MIN_PROMPT_LENGTH:
                continue
            if prompt not in user_text:
                continue
            if tool_use_id in launched_here:
                continue
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
        """Close agent spans for tool_result blocks and `<task-notification>`s."""
        message = raw.get("message", {})
        content = message.get("content", [])
        if not self._system_notifications:
            self._handle_user_notifications(content)
        if not isinstance(content, list):
            return

        blocks = [
            b
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "tool_result"
            and b.get("tool_use_id")
        ]
        # `tool_use_result` describes one tool's output, so only trust it
        # when it can belong to just this block
        structured = raw.get("tool_use_result") if len(blocks) == 1 else None
        for block in blocks:
            tool_use_id = str(block["tool_use_id"])
            if tool_use_id in self._open_agents:
                task_id = _async_launch_task_id(
                    structured, block, self._pending_subagents.get(tool_use_id)
                )
                if task_id is not None:
                    # background agent: still running, close on task_notification
                    if task_id:
                        self._task_ids[task_id] = tool_use_id
                    continue
            self._close_agent(tool_use_id)

    def _handle_user_notifications(self, content: Any) -> None:
        """Close background agents from `<task-notification>`s in a user line.

        For Claude Code versions that print no `system` task_notification
        events, only the notification the parent receives.
        """
        if isinstance(content, list):
            content = "\n".join(
                str(c.get("text", ""))
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        if not isinstance(content, str):
            return
        for note in _TASK_NOTIFICATION.findall(content):
            self._notified(_xml_field(note, "tool-use-id"), _xml_field(note, "task-id"))

    def _notified(self, tool_use_id: str | None, task_id: str | None) -> None:
        """A background agent finished: close its span."""
        key = tool_use_id if tool_use_id in self._open_agents else None
        if key is None and tool_use_id is None and task_id:
            key = self._task_ids.get(task_id)
        if key is None:
            logger.debug(
                "task notification for unknown agent (tool_use_id=%s, task_id=%s)",
                tool_use_id,
                task_id,
            )
            return
        self._close_agent(key)

    def _close_agent(self, tool_use_id: str) -> None:
        # Clear pending-subagent entry (no-op for non-Task tool_results).
        self._pending_subagents.pop(tool_use_id, None)
        agent = self._open_agents.pop(tool_use_id, None)
        if agent is not None:
            self._task_ids = {
                k: v for k, v in self._task_ids.items() if v != tool_use_id
            }
            transcript()._event(SpanEndEvent(id=agent.span_id))

    def _handle_system(self, raw: dict[str, Any]) -> None:
        """Handle compaction boundaries and background-agent completion.

        `task_started` is ignored: registration happens in `on_complete`
        (synchronous with the parent's bridge call, ahead of any race).
        `task_notification` closes background agents, whose `tool_result`
        arrived at launch; foreground agents close in `_handle_user`.
        """
        subtype = raw.get("subtype")
        if subtype == "compact_boundary":
            self._handle_compact_boundary(raw)
        elif subtype == "task_notification":
            self._system_notifications = True
            tool_use_id = raw.get("tool_use_id")
            task_id = raw.get("task_id")
            self._notified(
                tool_use_id if isinstance(tool_use_id, str) else None,
                task_id if isinstance(task_id, str) else None,
            )

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


def _async_launch_task_id(
    structured: Any, block: dict[str, Any], prompt: str | None
) -> str | None:
    """Task id ("" if unknown) if a Task/Agent tool_result is a background launch ack.

    Returns None if it is not one: the sub-agent has finished. Only
    `async_launched` agents keep running locally; `remote_launched` agents run
    in the cloud and never call the bridge, so their span closes at launch.
    Prefers the structured `tool_use_result` Claude Code attaches to the JSONL
    `user` line (checked against the prompt it echoes); falls back to the
    result text.
    """
    if (
        isinstance(structured, dict)
        and "status" in structured
        and structured.get("prompt") == prompt
    ):
        if structured["status"] != "async_launched":
            return None
        return str(structured.get("agentId") or "")
    content = block.get("content")
    if isinstance(content, list):
        content = next(
            (str(c.get("text", "")) for c in content if isinstance(c, dict)), ""
        )
    if isinstance(content, str) and content.lstrip().startswith("Async agent launched"):
        return _text_agent_id(block) or ""
    return None


def _block_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else ""


_AGENT_ID_LINE = re.compile(r"^agentId: (\S+)", re.MULTILINE)


def _text_agent_id(block: dict[str, Any]) -> str | None:
    """The `agentId: …` line Claude Code puts in Task/Agent results."""
    m = _AGENT_ID_LINE.search(_block_text(block))
    return m.group(1) if m else None


_TASK_NOTIFICATION = re.compile(r"<task-notification>(.*?)</task-notification>", re.S)


def _xml_field(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else None
