"""Real-time consumer of Claude Code JSONL output.

Two responsibilities, both driven from the same `LiveConsumer` instance:

1. **`ModelEventSink`** — installed on the agent bridge so the bridge hands
   us every `ModelEvent` for routing instead of emitting it to the transcript
   itself. We attribute each event to the correct agent span at `on_pending`
   time, then forward to the transcript with the attributed span_id.

   Claude Code 2.1.x makes sub-agents opaque in JSONL — sub-agent model
   calls have no `parent_tool_use_id`, no separate `session_id`, and no
   `isSidechain` marker. So spans are driven from the bridge:

   - **Launch.** When a call's output contains Task/Agent tool_calls,
     `on_complete` records each sub-agent and opens its agent span
     (`agent-{tool_use_id}`), synchronously, *before* the response is sent
     back to Claude Code. The sub-agent's first call reaches the bridge 1–2
     seconds before our stdout reader sees the parent's `assistant` line, so
     a JSONL-driven open would miss it.
   - **Attribution** (`attribution.resolve`): the call's history tells which
     sub-agent made it, and which of its runs.
   - **Resumed runs.** A finished background agent resumes when sent a
     message (SendMessage, or a message queued while it ran). Each run gets
     its own span, `agent-{tool_use_id}-run{N}`. It is opened in
     `on_complete` for a SendMessage, or lazily when a call for the agent
     shows a run boundary we have no span for yet, with JSONL
     `task_started` as a fallback.

2. **JSONL consumer** — `process_jsonl_line` reads each line printed by
   Claude Code's `--output-format stream-json` and closes agent spans (on
   `tool_result` for foreground Task/Agent, on `task_notification` for
   background ones), and emits `CompactionEvent` for `compact_boundary`
   system events. `task_started` binds task ids and run ids, and only opens
   a span for a resumed run that the bridge has not shown us yet.

With no stream-json (ACP), `BridgeOnlyLiveConsumer` closes spans from what
the bridge sees instead.

The sub-agents and runs we track, and their invariants, are in `subagents`;
what Claude Code's messages look like is in `shapes`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Any

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.model._model_output import StopReason
from inspect_ai.util._span import current_span_id

from .attribution import resolve
from .shapes import (
    ASYNC_LAUNCHED,
    FORK_STARTED,
    agent_id_in,
    async_launch_task_id,
    block_text,
    first_user_message,
    is_compaction_request,
    send_message_succeeded,
    summary_body,
    task_notifications,
)
from .subagents import Run, SubAgent, SubAgents
from .toolview import tool_view

logger = getLogger(__name__)


@contextmanager
def _tracking(what: str) -> Iterator[None]:
    """Log, not raise, a failure to track sub-agents.

    The bridge doesn't guard its sink calls: an exception here would fail
    the model call. A tracking bug should cost a wrong span, not the sample.
    """
    try:
        yield
    except Exception:
        logger.warning(
            "Claude Code sub-agent span tracking failed (%s)", what, exc_info=True
        )


class LiveConsumer(ModelEventSink):
    """Sink + JSONL consumer.

    The bridge calls `on_pending` / `on_complete` for every `ModelEvent`.
    The runner loop calls `process_jsonl_line` for every JSONL line printed
    by Claude Code.
    """

    def __init__(self) -> None:
        self._agents = SubAgents()

        # whether Claude Code reports background completions as `system`
        # task_notification events (2.1.303); older versions only show the
        # parent a `<task-notification>` in a `user` line
        self._system_notifications = False

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
        were written. Sub-agents are kept: a retry resumes the session.
        """
        for run in reversed(self._agents.open_runs()):
            self._close(run)
        for agent in self._agents:
            agent.pending_send = None
            agent.pending_send_parent = None
            # notifications of an attempt that is gone
            agent.owed.clear()
            agent.rolled_unnotified.clear()
        self._emitted_events.clear()
        self._last_stop_reason = None

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        event.span_id = self.outer_span_id
        with _tracking("attribution"):
            event.span_id = self._attribute(event.input)
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        # Record the terminal stop reason (guard against empty choices, where
        # ModelOutput.stop_reason raises). This fires for sub-agent calls too,
        # but the runner only consults it on a non-zero exit.
        if event.output and event.output.choices:
            self._last_stop_reason = event.output.stop_reason
        if (
            event.error is not None
            or not (event.output and event.output.choices)
            or event.output.stop_reason == "content_filter"
        ):
            # a failed or refused call: its retry is the same call again
            caller = self._agents.of_span(event.span_id)
            if caller is not None and caller.calls > 0:
                caller.calls -= 1

        msg = event.output.message if event.output and event.output.choices else None
        if msg is not None and msg.tool_calls:
            caller = self._agents.of_span(event.span_id)
            if caller is not None:
                for tc in msg.tool_calls:
                    self._agents.tool_call_owners[tc.id] = caller
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
            with _tracking("launches"):
                self._register_launches(event, msg)
        elif msg is not None and is_compaction_request(event.input):
            with _tracking("compaction summary"):
                self._record_summary(event.span_id, msg.text)

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    def _register_launches(self, event: ModelEvent, msg: ChatMessageAssistant) -> None:
        """Open spans for the sub-agents a call's output launches or resumes.

        Runs synchronously, before the bridge response is sent back to
        Claude Code, so the span is open before the sub-agent's first call.
        """
        parent_span_id = event.span_id or self.outer_span_id
        for tc in msg.tool_calls or []:
            if tc.function not in ("Task", "Agent"):
                continue
            args = tc.arguments or {}
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt or tc.id in self._agents:
                continue  # (already seen: defensive against retries)
            name = args.get("name")
            agent = SubAgent(
                origin=tc.id,
                prompt=prompt,
                span_name=str(args.get("subagent_type") or name or "agent"),
                description=str(args.get("description") or ""),
                parent_span_id=parent_span_id,
                launcher=self._agents.of_span(parent_span_id),
                name=name if isinstance(name, str) else None,
                fork=args.get("subagent_type") == "fork",
            )
            self._agents.add(agent)
            self._open(agent, tc.id, parent_span_id)

        # SendMessage to a finished background agent resumes it: open the
        # run's span now, before its first call arrives
        for tc in msg.tool_calls or []:
            if tc.function != "SendMessage" or self._agents.run(tc.id) is not None:
                continue
            self._agents.send_ids.add(tc.id)
            to = self._agents.find((tc.arguments or {}).get("to"), parent_span_id)
            if to is None:
                continue
            if to.open_run is not None or to.may_continue:
                # still running (the message is delivered in-run), or its
                # end hasn't reached us yet
                to.pending_send = tc.id
                to.pending_send_parent = parent_span_id
            else:
                self._open(to, tc.id, parent_span_id).speculative = True

    def _record_summary(self, span_id: str | None, text: str) -> None:
        """Remember a sub-agent's compaction summary, which starts its history next."""
        run = self._agents.run_of_span(span_id)
        summary = summary_body(text)
        if run is not None and summary is not None:
            run.agent.summary = summary
            run.agent.summary_run = run.number

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def _attribute(self, input_messages: list[ChatMessage]) -> str | None:
        """The span for a bridge call."""
        caller = resolve(self._agents, input_messages)
        if caller is None:
            return self.outer_span_id
        caller.agent.calls += 1
        return self._run_span(caller.agent, caller.finished_runs)

    def _run_span(self, agent: SubAgent, finished_runs: int) -> str | None:
        """Span for a call from a sub-agent whose history shows `finished_runs`.

        If that is as many runs as have started, this call starts the next
        run (a resume we have not seen yet, or one that raced its
        predecessor's close). If fewer, the call continues an earlier run.
        """
        new_run = finished_runs >= agent.started
        if new_run:
            # if this fires for what is one run, a turn Claude Code adds
            # within a run is missing from `shapes._IN_RUN_USER_PREFIXES`
            logger.debug(
                "sub-agent %s: history shows %d finished runs, opening run %d",
                agent.origin,
                finished_runs,
                agent.started + 1,
            )
        run = agent.open_run
        if run is None:
            if new_run:
                return self._open(agent, agent.pending_send or agent.origin).span_id
            # a call after its run closed: the run its history shows
            earlier = agent.run_number(finished_runs + 1) or agent.latest
            return earlier.span_id if earlier is not None else None
        if new_run:
            run = self._roll_over(run)
        elif finished_runs < run.number - 1:
            # an earlier run still going (e.g. through a blocking Stop hook)
            # after a SendMessage opened this one
            earlier = agent.run_number(finished_runs + 1)
            return earlier.span_id if earlier is not None else run.span_id
        run.speculative = False
        return run.span_id

    # ------------------------------------------------------------------
    # Spans
    # ------------------------------------------------------------------

    def _open(
        self,
        agent: SubAgent,
        key: str,
        parent_span_id: str | None = None,
        run_id: str | None = None,
        stale_run_ids: set[str] | None = None,
    ) -> Run:
        """Open the span of a new run of `agent`, keyed by `key`."""
        if key == agent.pending_send:
            parent_span_id = parent_span_id or agent.pending_send_parent
        agent.pending_send = None
        agent.pending_send_parent = None
        stray = agent.open_run
        if stray is not None:
            # should not happen: callers open a run only when none is open
            logger.warning("sub-agent %s: closing open run %s", agent.origin, stray.key)
            self._close(stray)
        resumed = bool(agent.runs)
        run = self._agents.new_run(agent, key)
        run.run_id = run_id
        run.stale_run_ids = stale_run_ids or set()
        metadata: dict[str, str] = {}
        if agent.description:
            metadata["description"] = agent.description
        if resumed:
            metadata["resumes"] = f"agent-{agent.origin}"
        transcript()._event(
            SpanBeginEvent(
                id=run.span_id,
                parent_id=parent_span_id or self._launcher_span(agent),
                type="agent",
                name=agent.span_name,
                metadata=metadata or None,
            )
        )
        return run

    def _launcher_span(self, agent: SubAgent) -> str | None:
        """The current span of the agent that launched `agent`.

        For a run we open without seeing who resumed it. Not the span current
        at launch, which may have ended since (a run of the launcher, or a
        checkpoint span of the main agent).
        """
        if agent.launcher is None:
            return self.outer_span_id
        latest = agent.launcher.latest
        return latest.span_id if latest is not None else agent.parent_span_id

    def _close(self, run: Run) -> None:
        if run.open:
            run.open = False
            transcript()._event(SpanEndEvent(id=run.span_id))

    def _roll_over(self, run: Run) -> Run:
        """Close a run's span and open the next run's."""
        agent = run.agent
        stale = set(run.stale_run_ids)
        if run.run_id:
            stale.add(run.run_id)
        self._close(run)
        # its notification is still to come
        if agent.background:
            agent.owed.append(run.key)
        else:
            agent.rolled_unnotified.append(run.key)
        return self._open(agent, agent.pending_send or run.key, stale_run_ids=stale)

    def _learned_background(self, agent: SubAgent) -> bool:
        """Record that a sub-agent runs in the background; False if known."""
        if agent.background:
            return False
        agent.background = True
        agent.owed.extend(agent.rolled_unnotified)
        agent.rolled_unnotified.clear()
        return True

    def _send_failed(self, key: str) -> None:
        """A SendMessage failed: it started no run."""
        if key not in self._agents.send_ids:
            return
        run = self._agents.run(key)
        if run is not None and run.open and run.via_send:
            if run.speculative:
                self._close(run)
                run.undone = True
            else:
                # a call showed the run started anyway (e.g. a child's
                # notification resumed the agent); only the key is the send's
                run.via_send = False
            return
        for agent in self._agents:
            if agent.pending_send == key:
                agent.pending_send = None
                agent.pending_send_parent = None

    def _notified(
        self,
        tool_use_id: str | None,
        task_id: str | None,
        run_id: str | None = None,
        closed_is_stale: bool = False,
    ) -> None:
        """A background agent's run finished: close its span.

        With no run ids to tell runs apart, `closed_is_stale` ignores a
        notification naming a span key we already closed.
        """
        named = self._agents.run(tool_use_id)
        agent = named.agent if named is not None else self._agents.by_agent_id(task_id)
        if agent is None:
            logger.debug(
                "task notification for unknown agent (tool_use_id=%s, task_id=%s)",
                tool_use_id,
                task_id,
            )
            return
        if self._was_owed(agent, tool_use_id, run_id):
            return  # for an earlier run we rolled past
        if named is not None and not named.open and closed_is_stale:
            return  # a run we already closed
        run = agent.open_run
        if run is None or run_id in run.stale_run_ids:
            return
        if named is not None and named is not run and task_id is None:
            return  # names a closed run, and nothing ties it to this one
        self._close(run)

    def _was_owed(
        self, agent: SubAgent, tool_use_id: str | None, run_id: str | None
    ) -> bool:
        """Whether a notification is the one owed by a run we rolled past.

        It names that run's key, or no key; a run id, when there is one,
        must not be the current run's.
        """
        if not agent.owed or (
            tool_use_id is not None and tool_use_id not in agent.owed
        ):
            return False
        current = agent.open_run
        if run_id is not None and current is not None and run_id == current.run_id:
            return False
        agent.owed.remove(tool_use_id if tool_use_id is not None else agent.owed[0])
        return True

    def _bind_agent_id(self, agent: SubAgent | None, agent_id: str) -> None:
        if agent is not None and not agent.agent_id:
            agent.agent_id = agent_id

    # ------------------------------------------------------------------
    # JSONL consumer
    # ------------------------------------------------------------------

    def process_jsonl_line(self, raw: dict[str, Any]) -> None:
        """Process one raw JSONL event from Claude Code.

        Called from the runner loop as each JSONL line arrives.

        Sub-agent spans open from the bridge (`on_complete`/`on_pending`,
        ahead of the race with stdout-buffered JSONL arrival). Here we
        consume `user` lines (tool_result → span close) and `system` lines
        (compaction, task lifecycle).
        """
        event_type = raw.get("type")
        with _tracking(f"stream-json {event_type} line"):
            if event_type == "user":
                self._handle_user(raw)
            elif event_type == "system":
                self._handle_system(raw)

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
            launched = self._agents.get(tool_use_id)
            if launched is not None:
                agent_id = agent_id_in(block_text(block))
                if agent_id:
                    self._bind_agent_id(launched, agent_id)
            run = self._agents.open_run(tool_use_id)
            if run is not None and run.via_send:
                if not send_message_succeeded(structured, block):
                    self._send_failed(tool_use_id)
                continue
            task_id = (
                async_launch_task_id(structured, block, launched.prompt)
                if launched is not None
                else None
            )
            if launched is not None and task_id is not None:
                # a background launch: still running, closed on its
                # notification (also when we already rolled past that run)
                self._learned_background(launched)
                if task_id:
                    self._bind_agent_id(launched, task_id)
            elif run is not None:
                self._close(run)
            elif not send_message_succeeded(structured, block):
                self._send_failed(tool_use_id)

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
        if isinstance(content, str):
            for tool_use_id, task_id in task_notifications(content):
                self._notified(tool_use_id, task_id)

    def _handle_system(self, raw: dict[str, Any]) -> None:
        """Handle compaction boundaries and background-agent completion.

        Spans open from the bridge (synchronous with the calls, ahead of any
        race). `task_started` binds the task and run ids, and opens a span
        only for a resumed run the bridge has not shown us yet.
        `task_notification` closes background agents, whose `tool_result`
        arrived at launch; foreground agents close in `_handle_user`. A late
        notification for a run we already rolled past is ignored.
        """
        subtype = raw.get("subtype")

        def text(field: str) -> str | None:
            value = raw.get(field)
            return value if isinstance(value, str) else None

        tool_use_id, task_id, run_id = (
            text("tool_use_id"),
            text("task_id"),
            text("run_id"),
        )
        if subtype == "compact_boundary":
            self._handle_compact_boundary(raw)
        elif subtype == "task_started" and task_id is not None:
            self._handle_task_started(tool_use_id, task_id, run_id)
        elif subtype == "task_notification":
            self._system_notifications = True
            self._notified(tool_use_id, task_id, run_id)

    def _handle_task_started(
        self, tool_use_id: str | None, task_id: str, run_id: str | None
    ) -> None:
        """Bind ids, and open or roll over to a run the bridge has not shown.

        Run ids arrive in run order, though stdout can lag the bridge: the
        n-th run id names run n, whichever span is open when it arrives.
        """
        self._bind_agent_id(self._agents.get(tool_use_id), task_id)
        agent = self._agents.by_agent_id(task_id)
        if agent is None:
            return
        number: int | None = None
        if run_id is not None:
            if run_id in agent.run_ids:
                return
            agent.run_ids.append(run_id)
            number = len(agent.run_ids)
        current = agent.open_run
        if current is None:
            if number is None or number > agent.started:
                key = tool_use_id or agent.pending_send or agent.origin
                self._open(agent, key, run_id=run_id)
            return
        if number is None or run_id is None:
            return
        if number < current.number:
            # a lagging start for an earlier run
            current.stale_run_ids.add(run_id)
        elif number == current.number:
            current.run_id = run_id
            current.speculative = False
        else:
            # a new run we have not seen from the bridge yet
            self._roll_over(current).run_id = run_id

    def _handle_compact_boundary(self, raw: dict[str, Any]) -> None:
        parent = raw.get("parent_tool_use_id")
        run = self._agents.open_run(parent if isinstance(parent, str) else None)
        compact_meta = raw.get("compactMetadata") or {}
        transcript()._event(
            CompactionEvent(
                source="claude_code",
                tokens_before=compact_meta.get("preTokens"),
                span_id=run.span_id if run is not None else self.outer_span_id,
                metadata={
                    "trigger": compact_meta.get("trigger", "auto"),
                    "content": raw.get("content") or "Conversation compacted",
                },
            )
        )


@dataclass
class _Notifications:
    """Notifications a `BridgeOnlyLiveConsumer` expects for a sub-agent."""

    # still to come for runs their own reply closed
    pending: int = 0
    # runs closed that way before we knew the agent was in the background
    # (its launch acknowledgement reached the parent late)
    early_reply_closes: int = 0


class BridgeOnlyLiveConsumer(LiveConsumer):
    """`LiveConsumer` for Claude Code with no stream-json output (ACP).

    Closes sub-agent spans from what the bridge sees: a run's own final
    reply, or the parent's next request showing the Task/Agent result
    (foreground) or a `<task-notification>` (background).
    """

    def __init__(self) -> None:
        super().__init__()
        self._notifications: dict[str, _Notifications] = {}
        # (conversation, index, role, tool call id, text) of parent messages
        # already acted on; a retried request (e.g. after a refusal) repeats
        # them
        self._handled: set[tuple[int, int, str, str, int]] = set()

    def reset(self) -> None:
        super().reset()
        self._notifications.clear()
        self._handled.clear()

    def _expected(self, agent: SubAgent) -> _Notifications:
        return self._notifications.setdefault(agent.origin, _Notifications())

    def on_pending(self, event: ModelEvent) -> None:
        with _tracking("parent input"):
            self._close_from_input(event.input)
        super().on_pending(event)

    def on_complete(self, event: ModelEvent) -> None:
        super().on_complete(event)
        with _tracking("final reply"):
            self._close_finished_run(event)

    def _close_finished_run(self, event: ModelEvent) -> None:
        """Close a sub-agent's span when its run ends.

        A run ends with a text-only `end_turn` reply. Without this, a
        background agent's span would wait for the parent's next request to
        show the notification, which in an idle interactive session can be
        much later. A resume later opens the next run lazily. A compaction
        summary is a text-only reply too, but not the end of the run.
        """
        msg = event.output.message if event.output and event.output.choices else None
        if msg is None or msg.tool_calls or event.output.stop_reason != "stop":
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
        """Close sub-agents whose completion the calling agent is now seeing.

        Only messages after the caller's last assistant turn are new; earlier
        ones were handled when they first arrived (and a resumed agent's old
        notification must not close its new run). A retried request repeats
        them, so each is handled once. A foreground agent finishes with its
        Task/Agent tool result; a background one with a `<task-notification>`.
        """
        # agent ids from Task/Agent results anywhere in history (SendMessage
        # names the agent by id, possibly after its span closed)
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
        elif not text.startswith((FORK_STARTED, ASYNC_LAUNCHED)):
            self._close(run)

    def _input_notification(self, tool_use_id: str | None, task_id: str | None) -> None:
        agent = self._agents.by_agent_id(task_id)
        if agent is not None and self._expected(agent).pending > 0:
            # for a run its own reply already closed
            self._expected(agent).pending -= 1
            return
        self._notified(tool_use_id, task_id, closed_is_stale=True)
