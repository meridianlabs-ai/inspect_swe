"""Real-time consumer of Claude Code's model calls and JSONL output.

As the bridge's `ModelEventSink`, `LiveConsumer` puts each model call in its
sub-agent's span: from the `x-claude-code-agent-id` request header when
inspect_ai exposes it, else from the call's history (`attribution`). Spans
open when a call's output launches a sub-agent, before its first call can
arrive. `process_jsonl_line` closes them from Claude Code's stream-json
output.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from inspect import signature
from logging import getLogger
from typing import Any

import inspect_ai.agent
from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.event import CompactionEvent, Event, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import ModelUsage
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.model._model_output import StopReason
from inspect_ai.util._span import current_span_id

from ..._util.model_event_sink import mark_started, started
from .attribution import resolve
from .shapes import (
    agent_id_in,
    async_launch_task_id,
    block_text,
    is_compaction_request,
    send_message_succeeded,
    summary_body,
    task_notifications,
)
from .subagents import Run, SubAgent, SubAgents
from .toolview import tool_view

logger = getLogger(__name__)

# the sub-agent making a request, and its parent if that is a sub-agent too
AGENT_ID_HEADER = "x-claude-code-agent-id"
PARENT_AGENT_ID_HEADER = "x-claude-code-parent-agent-id"


def bridge_header_args() -> dict[str, Any]:
    """`sandbox_agent_bridge` arguments that expose the agent id headers, if it can."""
    if "expose_client_headers" not in signature(sandbox_agent_bridge).parameters:
        return {}
    return {"expose_client_headers": [AGENT_ID_HEADER, PARENT_AGENT_ID_HEADER]}


def _bridge_request() -> Any:
    """The bridged request being handled (None on an inspect_ai without it)."""
    current = getattr(inspect_ai.agent, "current_bridge_request", None)
    return current() if current is not None else None


def _request_agent_ids() -> tuple[str | None, str | None] | None:
    """(agent id, parent agent id) of the request; None without its headers."""
    headers = getattr(_bridge_request(), "headers", None)
    if not isinstance(headers, Mapping):
        return None
    return (
        headers.get(AGENT_ID_HEADER) or None,
        headers.get(PARENT_AGENT_ID_HEADER) or None,
    )


# (sink, input fingerprint, span) of a sub-agent call that failed in this task,
# whose retry is the same call
_retrying: ContextVar[tuple[object, int, str | None] | None] = ContextVar(
    "inspect_swe_claude_code_retrying", default=None
)


def _fingerprint(input: list[ChatMessage]) -> int:
    return hash((len(input), input[-1].text if input else ""))


@contextmanager
def _tracking(what: str) -> Iterator[None]:
    """Log, not raise, a failure to track sub-agents (it would fail the model call)."""
    try:
        yield
    except Exception:
        logger.warning(
            "Claude Code sub-agent span tracking failed (%s)", what, exc_info=True
        )


class LiveConsumer(ModelEventSink):
    """Model event sink and stream-json consumer for one Claude Code session."""

    def __init__(self) -> None:
        self._agents = SubAgents()

        # whether background completions come as `system` task_notification
        # lines (else only as `<task-notification>`s in `user` lines)
        self._system_notifications = False

        # Track which event objects we've already _event()'d so on_complete
        # knows whether to emit _event_updated (yes if we emitted) vs swallow
        # (no if we didn't — shouldn't happen with current logic but defensive).
        self._emitted_events: set[int] = set()

        # whether requests name their sub-agent
        self._agent_ids_seen = False
        # events in the spans of sub-agents whose agent id is unconfirmed
        self._unconfirmed_events: dict[str, list[Event]] = {}

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
        """Close open spans and clear per-attempt state; sub-agents are kept."""
        for run in reversed(self._agents.open_runs()):
            self._close(run)
        for agent in self._agents:
            agent.pending_send = None
            agent.pending_send_parent = None
            agent.owed.clear()
            agent.rolled_unnotified.clear()
        self._emitted_events.clear()
        self._last_stop_reason = None

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_start(self, input: list[ChatMessage]) -> str | None:
        mark_started(self)
        span_id = self.outer_span_id
        with _tracking("attribution"):
            span_id = self._attribute(input)
        return span_id

    def on_pending(self, event: ModelEvent) -> None:
        # an inspect_ai without on_start leaves the placement to us here
        if not started(self):
            event.span_id = self.outer_span_id
            with _tracking("attribution"):
                event.span_id = self._attribute(event.input)
        else:
            with _tracking("attribution"):
                self._follow_swap(event)
        self._track(event.span_id, event)
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
            if self._agents.of_span(event.span_id) is not None:
                _retrying.set((self, _fingerprint(event.input), event.span_id))
        else:
            _retrying.set(None)

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
                if event.error is None:
                    self._record_summary(event.span_id, msg.text)
                    self._compacted(event)

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    def _register_launches(self, event: ModelEvent, msg: ChatMessageAssistant) -> None:
        """Open spans for the sub-agents a call's output launches or resumes."""
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

    def _compacted(self, event: ModelEvent) -> None:
        """Mark a sub-agent's compaction, which stream-json does not report."""
        run = self._agents.run_of_span(event.span_id)
        if run is None or summary_body(event.output.message.text) is None:
            return
        run.agent.compactions += 1
        self._compaction_event(run.span_id, event.output.usage)

    def _compaction_event(
        self,
        span_id: str | None,
        usage: ModelUsage | None = None,
        tokens_before: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tokens_before is None and usage is not None:
            tokens_before = usage.input_tokens + (usage.input_tokens_cache_read or 0)
        event = CompactionEvent(
            source="claude_code",
            tokens_before=tokens_before,
            span_id=span_id,
            metadata=metadata or {"content": "Conversation compacted"},
        )
        self._track(span_id, event)
        transcript()._event(event)

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def _attribute(self, input_messages: list[ChatMessage]) -> str | None:
        """The span for a bridge call."""
        ids = _request_agent_ids()
        if ids is not None and ids[0]:
            self._agent_ids_seen = True
            agent = self._agent_for_id(ids[0], ids[1], input_messages)
            if agent is None:
                return self._launcher_span_of(ids[1])
            agent.calls += 1
            # a history that isn't its own (e.g. a side request) is in its
            # latest run
            own = resolve(self._agents, input_messages, among=[agent])
            runs = own.finished_runs if own is not None else agent.started - 1
            return self._run_span(agent, runs)
        if ids is not None and self._agent_ids_seen:
            return self.outer_span_id  # sub-agents' requests are all named
        retrying = _retrying.get()
        if (
            retrying is not None
            and retrying[0] is self
            and retrying[1] == _fingerprint(input_messages)
        ):
            return retrying[2]
        caller = resolve(self._agents, input_messages)
        if caller is None:
            return self.outer_span_id
        caller.agent.calls += 1
        return self._run_span(caller.agent, caller.finished_runs)

    def _agent_for_id(
        self, agent_id: str, parent_id: str | None, input_messages: list[ChatMessage]
    ) -> SubAgent | None:
        """The sub-agent with an agent id (a new one is bound provisionally)."""
        agent = self._agents.by_agent_id(agent_id)
        if agent is not None:
            return agent
        launcher = self._agents.by_agent_id(parent_id)
        unbound = [
            a
            for a in self._agents
            if a.agent_id is None
            and (launcher is not None or parent_id is None)
            and a.launcher is launcher
        ]
        caller = resolve(self._agents, input_messages, among=unbound)
        if caller is not None:
            agent = caller.agent
        elif len(unbound) == 1:
            agent = unbound[0]
        else:
            logger.debug("no launch for Claude Code agent %s", agent_id)
            return None
        agent.agent_id = agent_id
        return agent

    def _follow_swap(self, event: ModelEvent) -> None:
        """Move a retry attempt to where its first call's agent was moved."""
        ids = _request_agent_ids()
        agent = self._agents.by_agent_id(ids[0] if ids is not None else None)
        span_agent = self._agents.of_span(event.span_id)
        if agent is not None and span_agent is not None and agent is not span_agent:
            run = agent.open_run or agent.latest
            if run is not None:
                event.span_id = run.span_id

    def _launcher_span_of(self, parent_id: str | None) -> str | None:
        """The current span of the agent with this id (None: the main agent)."""
        launcher = self._agents.by_agent_id(parent_id)
        run = launcher.latest if launcher is not None else None
        return run.span_id if run is not None else self.outer_span_id

    def _run_span(self, agent: SubAgent, finished_runs: int) -> str | None:
        """Span for a call from a sub-agent whose history shows `finished_runs`."""
        new_run = finished_runs >= agent.started
        if new_run:
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
        begin = SpanBeginEvent(
            id=run.span_id,
            parent_id=parent_span_id or self._launcher_span(agent),
            type="agent",
            name=agent.span_name,
            metadata=metadata or None,
        )
        self._track(begin.parent_id, begin)
        transcript()._event(begin)
        return run

    def _launcher_span(self, agent: SubAgent) -> str | None:
        """The current span of the agent that launched `agent`."""
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
        """A background agent's run finished: close its span."""
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
        """Whether a notification is the one owed by a run we rolled past."""
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
        """Bind a launch to the agent id its result or `task_started` names."""
        if agent is None or agent.id_confirmed:
            return
        holder = self._agents.by_agent_id(agent_id)
        if holder is not None and holder is not agent:
            if holder.id_confirmed or not self._swap_runtime(holder, agent):
                logger.debug("agent id %s is already %s's", agent_id, holder.origin)
                return
        agent.agent_id = agent_id
        agent.id_confirmed = True
        self._unconfirmed_events.pop(f"agent-{agent.origin}", None)

    def _swap_runtime(self, a: SubAgent, b: SubAgent) -> bool:
        """Swap what two sibling sub-agents did, while each is in its first run."""
        if a.launcher is not b.launcher or not all(
            len(x.runs) <= 1 and all(r.open for r in x.runs) and not x.run_ids
            for x in (a, b)
        ):
            return False
        span_a, span_b = f"agent-{a.origin}", f"agent-{b.origin}"
        relabel = {span_a: span_b, span_b: span_a}
        moved = (
            self._unconfirmed_events.pop(span_a, []),
            self._unconfirmed_events.pop(span_b, []),
        )
        for events in moved:
            for e in events:
                if isinstance(e, SpanBeginEvent):
                    e.parent_id = relabel.get(e.parent_id or "", e.parent_id)
                else:
                    e.span_id = relabel.get(e.span_id or "", e.span_id)
                transcript()._event_updated(e)
        self._unconfirmed_events[span_b] = moved[0]
        self._unconfirmed_events[span_a] = moved[1]
        for field in ("agent_id", "calls", "summary", "summary_run", "compactions"):
            value = getattr(a, field)
            setattr(a, field, getattr(b, field))
            setattr(b, field, value)
        swap = {a: b, b: a}
        for tool_call_id, owner in self._agents.tool_call_owners.items():
            self._agents.tool_call_owners[tool_call_id] = swap.get(owner, owner)
        for child in self._agents:
            if child.launcher in swap:
                child.launcher = swap[child.launcher]
                child.parent_span_id = relabel.get(
                    child.parent_span_id or "", child.parent_span_id
                )
        return True

    def _track(self, span_id: str | None, event: Event) -> None:
        """Keep an event in a sub-agent span until the agent id is confirmed."""
        agent = self._agents.of_span(span_id)
        if agent is not None and not agent.id_confirmed and span_id is not None:
            self._unconfirmed_events.setdefault(span_id, []).append(event)

    # ------------------------------------------------------------------
    # JSONL consumer
    # ------------------------------------------------------------------

    def process_jsonl_line(self, raw: dict[str, Any]) -> None:
        """Process one stream-json line from Claude Code."""
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
        """Close background agents from `<task-notification>`s in a user line."""
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
        """Handle compaction boundaries and background task lifecycle."""
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

        The n-th run id names run n, whichever span is open when it arrives.
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
        if run is not None and run.agent.compactions > 0:
            run.agent.compactions -= 1  # marked from the bridge already
            return
        compact_meta = raw.get("compactMetadata") or {}
        self._compaction_event(
            run.span_id if run is not None else self.outer_span_id,
            tokens_before=compact_meta.get("preTokens"),
            metadata={
                "trigger": compact_meta.get("trigger", "auto"),
                "content": raw.get("content") or "Conversation compacted",
            },
        )
