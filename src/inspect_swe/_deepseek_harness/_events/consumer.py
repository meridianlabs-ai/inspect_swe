"""Bridge `ModelEventSink` for DeepSeek Harness (dsh) sub-agent spans.

Installed on the agent bridge, which then hands us every `ModelEvent` instead
of writing it to the transcript. From the requests alone (dsh's `--json`
stdout is not used) we rebuild the agent-span tree:

  1. **Open** — when an agent's output calls `subagent` or `subagent_fork`,
     `on_complete` opens an agent `SpanBeginEvent` per call (`agent-<call id>`)
     before the response reaches dsh, so the span exists before the child's
     first request. An output the user's filter returned itself produces no
     `ModelEvent`; `on_filter_output` registers its calls instead.

  2. **Attribute** — every dsh agent is told its name in its session reminder
     (see `detection.py`), so `classify`, in the bridge filter, reads the
     requester's name off each request and decides the request's agent context
     and span together. `on_start` returns that span for inspect_ai to run the
     generate under. A name is tied to its call by the call's `name` argument,
     else by the delegated prompt the child's first request opens with, else
     by the call's `started subagent <name>` result.

  3. **Close** — dsh runs every child in the background: the call's result is
     only a launch acknowledgement. The span stays open until the parent's
     history carries a new settlement notice for the child. `reset()` closes
     what is left between attempts and at the end.

  4. **Result** — the notice carries the child's closing message. It is
     recorded as a `ToolEvent` beside the span, naming it (`agent_span_id`):
     the call that started the run, answered with what the run produced.
     That is where the timeline takes an agent span's result from; without
     it the viewer would show the launch acknowledgement. A run with no
     notice by `reset()` is answered with its own final answer, or as still
     running.

  5. **Resume** — a settled child that makes requests again (the parent sent
     it more with `send_message`) continues in a new span with the same name,
     which gets its own result when that run settles. A child can be
     restarted before its parent has seen the notice for the run that ended
     (the parent sent the message while that run was finishing): its own
     request shows the restart, so the run's span closes then and takes the
     next notice for the child, and the new run gets the one after.

Notices are matched to the ones already taken from the same parent's history
in order, not counted, so compaction cutting old notices from the history (or
a compaction request carrying only part of it) neither hides a new notice nor
repeats an old one.

Parallel children are told apart by name on every request, so interleaving
doesn't matter. A child we never saw start (a `workflow` script's children, a
child resumed in a later dsh process) gets a span when its first request
arrives.

Web search: dsh's `web_search` tool runs a model request of its own (the
query, with the provider's search tool). It is classified "utility" and
recorded in the root's span, since the request doesn't say which agent
searched.

Compaction: dsh's summarizer request is the span of an agent's conversation
being condensed plus a fixed instruction. It is classified "utility" and a
`CompactionEvent` is recorded in the span of the agent that compacted. That
span carries the agent's session reminder only when it reaches back to the
agent's first turn (or a later reminder), so a sub-agent's second and later
compactions name nobody and are recorded in the root's span.
"""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, NamedTuple
from uuid import uuid4

from inspect_ai.agent import AgentBridgeContext
from inspect_ai.event import (
    CompactionEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    Model,
    ModelOutput,
    StopReason,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.tool import ToolCall, ToolInfo
from inspect_ai.util._span import current_span_id

from ..._util.model_event_sink import (
    Decision,
    in_filter_generation,
    mark_started,
    remember_decision,
    remembered_decision,
    started,
)
from .detection import (
    ROOT_AGENT_NAME,
    SEND_MESSAGE_TOOL,
    Delegation,
    Identity,
    find_delegations,
    find_messages,
    initial_prompt,
    is_compaction_request,
    is_web_search_request,
    requester_identity,
    restarted,
    settlements,
    started_name,
)
from .toolview import tool_view

logger = getLogger(__name__)

COMPACTION_SOURCE = "deepseek_harness"


@dataclass(frozen=True)
class _Decision(Decision):
    """The filter-time decision for one request (`requester_path`: its agent's name)."""

    kind: str = "unknown"
    epoch: int = 0
    """The consumer's `reset()` count when the request was classified."""
    messages: list[ChatMessage] | None = field(default=None, compare=False, repr=False)
    """The request's conversation as it reached the bridge (the list the bridge
    tracks, which a user filter's rewrite of the model's input doesn't change)."""


class RootGeneration(NamedTuple):
    """The root agent's latest completed generation."""

    messages: list[ChatMessage]
    output: ModelOutput


@dataclass
class _Run:
    """The tool call a child's current run answers."""

    call_id: str
    function: str
    arguments: dict[str, Any]


@dataclass
class _Ended:
    """A run whose span closed before its settlement notice was seen."""

    run: _Run
    span_id: str
    span_parent_id: str | None
    closing: str
    """The child's own final answer, recorded if no notice comes."""


NO_RESULT = "No result: the sub-agent was still running when dsh exited."


@dataclass
class _Child:
    """A sub-agent and the span its requests currently go to."""

    span_id: str
    base_span_id: str
    display: str
    parent: str | None
    """Name of the agent that started it; `None` when unknown."""
    parent_span_id: str | None
    """Span its first span was opened under."""
    call_id: str | None = None
    """The delegation call that started it; `None` for a child we never saw start."""
    tool: str | None = None
    prompt: str = ""
    name: str | None = None
    """Its name in dsh's subagent tree, once known."""
    is_open: bool = True
    resumes: int = 0
    run: _Run | None = None
    """What started the run its open span holds."""
    span_parent_id: str | None = None
    """The span its open span sits in."""
    final: str | None = None
    """The open run's latest output, when it called no tools (its answer so far)."""
    answered_alone: bool = False
    """That output came with no children of its own running, so it ended the run."""
    ended: list[_Ended] = field(default_factory=list)
    """Runs closed by a restart, oldest first, still owed their notice."""
    restart_key: tuple[int, str] | None = None
    """Where its latest run began (`detection.Restart.key`)."""
    metadata: dict[str, str] = field(default_factory=dict)


class DshConsumer(ModelEventSink):
    def __init__(self) -> None:
        # delegation call id → child, for every start we saw (failed ones
        # are dropped when their error result arrives)
        self._by_call: dict[str, _Child] = {}

        # name → child, once a name is tied to it
        self._children: dict[str, _Child] = {}

        # children whose span is open, in open order (reset() closes
        # innermost first)
        self._open: list[_Child] = []

        # (parent name, child name) → the results of every settlement notice
        # taken from the parent's history so far, in order
        self._notices: dict[tuple[str, str], list[str]] = {}

        # recipient name → the latest `send_message` call to it that hasn't
        # been tied to a run yet (the one a settled child resumes on)
        self._sent: dict[str, ToolCall] = {}

        # ModelEvents we've _event()'d, so on_complete knows to _event_updated
        self._emitted_events: set[int] = set()

        self._root: RootGeneration | None = None

        # how many times reset() ran: a generation classified before a reset
        # belongs to a dsh process that is gone
        self._epoch = 0

    @property
    def outer_span_id(self) -> str | None:
        """Span for main-agent attribution, resolved at emission time.

        Must not be captured once at construction: with checkpointing active
        the enclosing span rotates and a frozen id would pin every event to
        the first one.
        """
        return current_span_id()

    @property
    def root_generation(self) -> RootGeneration | None:
        """The root agent's latest completed generation this attempt.

        dsh's own compaction request is not one. `deepseek_harness()` makes
        this the sample's conversation and answer, because the bridge can
        otherwise keep a sub-agent's (see `_apply_root_generation` there).
        """
        return self._root

    @property
    def last_stop_reason(self) -> StopReason | None:
        """Stop reason of the root agent's latest completed generation."""
        if self._root is None or not self._root.output.choices:
            return None
        return self._root.output.stop_reason

    def reset(self) -> None:
        """Close any open spans and clear per-attempt state.

        Called before each dsh launch and after the last one, so the span tree
        stays balanced even if dsh exited with children still running. A
        generation still in flight for the process that exited completes
        later: its event is updated, but what it would start or answer is
        ignored (`_epoch`), so it opens no span in the next attempt or after
        the last one.

        A run still open is answered with its final answer if its last output
        was one (it finished, but dsh exited before telling its parent), else
        as still running; a run closed by a restart, whose notice never came,
        with the answer it ended with.
        """
        for child in self._children.values():
            for ended in child.ended:
                self._record(
                    ended.run,
                    ended.span_id,
                    ended.span_parent_id,
                    ended.closing or NO_RESULT,
                )
            child.ended.clear()
        for child in reversed(self._open):
            if child.run is not None:
                self._record(
                    child.run,
                    child.span_id,
                    child.span_parent_id,
                    child.final or NO_RESULT,
                )
            transcript()._event(SpanEndEvent(id=child.span_id))
        self._open.clear()
        self._by_call.clear()
        self._children.clear()
        self._notices.clear()
        self._sent.clear()
        self._root = None
        self._epoch += 1

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_start(self, input: list[ChatMessage]) -> str | None:
        mark_started(self)
        return self._start(input).span_id

    def on_pending(self, event: ModelEvent) -> None:
        # an inspect_ai without on_start leaves the placement to us here
        if not started(self):
            event.span_id = self._start(event.input).span_id

        if is_compaction_request(event.input):
            transcript()._event(
                CompactionEvent(
                    source=COMPACTION_SOURCE,
                    span_id=event.span_id,
                    metadata={"trigger": "auto"},
                )
            )

        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        output = event.output if event.output and event.output.choices else None
        if output is not None:
            for call in output.message.tool_calls or []:
                if call.view is None:
                    call.view = tool_view(call.function, call.arguments or {})

            # a user filter's own generation reaches dsh only if the filter
            # returns it (`on_filter_output`)
            if not in_filter_generation(self):
                # the filter may have rewritten the input this generate saw:
                # go by what `classify` read off the request
                decision = self._start(event.input)
                if decision.epoch == self._epoch:
                    self._register_delegations(
                        output.message,
                        event.span_id or self.outer_span_id,
                        decision.requester_path,
                    )
                    self._note_generation(decision, event.input, output)

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    def on_filter_output(
        self, messages: list[ChatMessage], output: ModelOutput
    ) -> None:
        """Track an output the user's filter returned itself.

        Such an output bypasses the bridge's generation, so no `ModelEvent`
        reaches `on_complete`: without this its delegation calls would never
        open a span. Called by `classify_filter` before the output goes back
        to dsh.
        """
        if not output.choices:
            return
        decision = self._start(messages)
        if decision.epoch == self._epoch:
            self._register_delegations(
                output.message, decision.span_id, decision.requester_path
            )
            self._note_generation(decision, messages, output)

    # ------------------------------------------------------------------
    # filter-time classification (AgentContextClassifier)
    # ------------------------------------------------------------------

    def classify(
        self, _model: Model, messages: list[ChatMessage], _tools: list[ToolInfo]
    ) -> AgentBridgeContext:
        """Filter-time agent context, from the same decision as span placement.

        Runs in the bridge filter, before generation and before the user's
        filter can rewrite the request. The requester is the agent its latest
        session reminder names:

        - the root agent → "root";
        - a delegated agent → "subagent", with `subagent_id` its name in dsh's
          subagent tree and `subagent_type` the tool that started it
          (`subagent` or `subagent_fork`, `None` for a child we never saw
          start);
        - no reminder → "root" while no sub-agent has been seen this attempt
          (nothing else can be making requests), else "unknown".

        A request whose last block is dsh's compaction instruction is
        "utility" whoever sent it, and so is the model call behind dsh's
        `web_search` tool (which names nobody and goes to the root's span).
        """
        context, decision = self._decide(messages)
        # the generate this request leads to is placed by the same decision,
        # even if the user's filter rewrites its messages (see `_start`)
        remember_decision(self, decision)
        return context

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _start(self, messages: list[ChatMessage]) -> _Decision:
        """The decision for a generate: the one `classify` made, if it ran.

        `classify` decided over the request as it reached the bridge, and a
        filter may since have rewritten the messages, so only without that
        decision is the request read here (`_decide` is idempotent).
        """
        decided = remembered_decision(self)
        if isinstance(decided, _Decision):
            return decided
        return self._decide(messages)[1]

    def _decide(
        self, messages: list[ChatMessage]
    ) -> tuple[AgentBridgeContext, _Decision]:
        """Sync from a request, then decide its context and span together."""
        if is_web_search_request(messages):
            return AgentBridgeContext("utility"), _Decision(
                span_id=self.outer_span_id,
                requester_path=None,
                kind="utility",
                epoch=self._epoch,
                messages=messages,
            )

        # a compaction request holds only the span being condensed, which may
        # open with a reminder that isn't the agent's latest. It is placed,
        # never used to open or resume a span; a notice in it that is news
        # (it may never reach a later request) still settles its run.
        compaction = is_compaction_request(messages)
        identity = requester_identity(messages)
        self._sync(messages, identity)

        child: _Child | None = None
        if identity is None:
            span_id = self.outer_span_id
            context = AgentBridgeContext(
                "unknown" if self._by_call or self._children else "root"
            )
        elif identity.is_root:
            span_id = self.outer_span_id
            context = AgentBridgeContext("root")
        elif compaction:
            child = self._children.get(identity.name)
            span_id = (
                child.span_id
                if child is not None and child.is_open
                else self.outer_span_id
            )
            context = AgentBridgeContext("utility")
        else:
            child = self._child_for(identity, messages)
            span_id = child.span_id
            context = AgentBridgeContext(
                "subagent", subagent_id=identity.name, subagent_type=child.tool
            )

        if compaction:
            context = AgentBridgeContext("utility")

        return context, _Decision(
            span_id=span_id,
            requester_path=identity.name if identity is not None else None,
            kind=context.kind,
            epoch=self._epoch,
            messages=messages,
        )

    def _note_generation(
        self, decision: _Decision, messages: list[ChatMessage], output: ModelOutput
    ) -> None:
        """Keep the root's latest generation; forget it once we lose track.

        A generation we can't attribute may be the root's, and then the one we
        hold is no longer its latest: better to report none than a stale one.
        """
        if decision.kind == "root":
            # the conversation as it reached the bridge, which is what the
            # bridge itself reports; `messages` is what the model was sent,
            # after any rewrite by the user's filter
            conversation = (
                decision.messages if decision.messages is not None else messages
            )
            self._root = RootGeneration(list(conversation), output)
        elif decision.kind == "unknown":
            self._root = None
        elif decision.kind == "subagent" and decision.requester_path is not None:
            child = self._children.get(decision.requester_path)
            if (
                child is not None
                and child.is_open
                and child.span_id == decision.span_id
            ):
                answer = not output.message.tool_calls
                child.final = output.message.text if answer else None
                # waiting on children of its own, an answer doesn't end its run
                child.answered_alone = answer and not self._has_running_children(child)

    def _sync(self, messages: list[ChatMessage], identity: Identity | None) -> None:
        """Bind names from delegation results, then settle children's runs.

        Idempotent: results and notices stay in the requester's history, so a
        failed start is dropped once and each notice answers one run, the
        first time it is seen.
        """
        for message in messages:
            if not isinstance(message, ChatMessageTool) or message.tool_call_id is None:
                continue
            child = self._by_call.get(message.tool_call_id)
            if child is None:
                continue
            name = started_name(message)
            if name is not None:
                self._bind(child, name)
            elif message.error is not None:
                # dsh rejected the start: no agent will ever use the span
                del self._by_call[message.tool_call_id]
                if child.name is not None and self._children.get(child.name) is child:
                    del self._children[child.name]
                self._close(child)

        if identity is None:
            return
        visible: dict[str, list[str]] = {}
        for settlement in settlements(messages):
            visible.setdefault(settlement.name, []).append(settlement.result)
        for name, results in visible.items():
            child = self._children.get(name)
            # a notice goes to the child's direct parent; a forked sibling
            # inherits it with the parent's history and proves nothing new
            if child is not None and child.parent not in (None, identity.name):
                continue
            seen = self._notices.setdefault((identity.name, name), [])
            news = _new_notices(seen, results)
            seen.extend(news)
            if child is not None:
                for result in news:
                    self._answer(child, result)

    def _bind(self, child: _Child, name: str) -> None:
        """Tie a started child to its name, unless the name is another's."""
        if child.name is None and name not in self._children:
            child.name = name
            self._children[name] = child

    def _child_for(self, identity: Identity, messages: list[ChatMessage]) -> _Child:
        """The child a delegated agent's request belongs to, with an open span."""
        child = self._children.get(identity.name)
        if child is None:
            child = self._unnamed_start(identity, messages)
            if child is not None:
                self._bind(child, identity.name)
        restart = restarted(messages, identity.parent)
        if child is None or child.name != identity.name:
            child = self._open_unseen(identity)
        elif not child.is_open:
            self._resume(child)
        elif (
            restart is not None
            and restart.key != child.restart_key
            and child.answered_alone
        ):
            # it was sent more after a run whose notice its parent hasn't
            # seen yet: that run is over, and this is the next
            self._end_run(child, restart.closing)
            self._resume(child)
        if restart is not None:
            child.restart_key = restart.key
        return child

    def _has_running_children(self, child: _Child) -> bool:
        """Whether a child has children of its own whose span is open.

        Waiting on them, it can write an answer without tool calls that
        doesn't end its run.
        """
        return child.name is not None and any(
            other.parent == child.name for other in self._open
        )

    def _unnamed_start(
        self, identity: Identity, messages: list[ChatMessage]
    ) -> _Child | None:
        """The start a first request belongs to, when its call gave no name.

        dsh then names the child `agent-N`, which only its result reports. The
        child's first request opens with the delegated prompt, so that picks
        the call (calls with the same prompt are interchangeable); failing
        that, the caller's only unnamed start.
        """
        unnamed = [
            child
            for child in self._by_call.values()
            if child.name is None
            and child.is_open
            and child.parent in (None, identity.parent)
        ]
        prompt = initial_prompt(messages)
        for child in unnamed:
            if child.prompt and child.prompt == prompt:
                return child
        return unnamed[0] if len(unnamed) == 1 else None

    def _open_unseen(self, identity: Identity) -> _Child:
        """Open a span for a child whose start we never saw.

        Its span id can't come from a call id, and the same child can turn up
        again after `reset()` (a later turn of the sample resumes it in a new
        dsh process, with a new consumer), so the id is made unique.
        """
        parent = self._children.get(identity.parent) if identity.parent else None
        parent_span_id = (
            parent.span_id
            if parent is not None and parent.is_open
            else self.outer_span_id
        )
        span_id = f"agent-{identity.name}-{uuid4().hex[:8]}"
        child = _Child(
            span_id=span_id,
            base_span_id=span_id,
            display=identity.name,
            parent=identity.parent,
            parent_span_id=parent_span_id,
            name=identity.name,
        )
        self._children[identity.name] = child
        # a child that already exists is making requests because it was sent
        # a message (nothing else restarts one)
        child.run = self._resumed_run(child)
        self._begin(child, parent_span_id, dict(child.metadata))
        return child

    def _resume(self, child: _Child) -> None:
        """Open a continuation span for a settled child making requests again.

        `send_message` restarts a child after its settlement closed its span.
        The new span repeats the child's name and metadata and opens under the
        span its first one was opened under while that is still open (else the
        root's current span). It closes like the first, on the child's next
        settlement notice or at `reset()`.
        """
        child.resumes += 1
        child.span_id = f"{child.base_span_id}-resumed-{child.resumes}"
        child.run = self._resumed_run(child)
        open_spans = {open_child.span_id for open_child in self._open}
        parent_span_id = (
            child.parent_span_id
            if child.parent_span_id in open_spans
            else self.outer_span_id
        )
        self._begin(
            child, parent_span_id, {**child.metadata, "continues": child.base_span_id}
        )

    def _resumed_run(self, child: _Child) -> _Run:
        """The `send_message` call a child's new run answers.

        The call itself when we saw it; otherwise (it was sent by a dsh
        process that is gone, or by an agent whose output a filter replaced)
        one made up from the span's id, so the run can still carry a result.
        """
        sent = self._sent.pop(child.name, None) if child.name is not None else None
        if sent is not None:
            return _Run(sent.id, SEND_MESSAGE_TOOL, dict(sent.arguments or {}))
        return _Run(
            child.span_id.removeprefix("agent-"),
            SEND_MESSAGE_TOOL,
            {"agent": child.name} if child.name is not None else {},
        )

    def _answer(self, child: _Child, result: str) -> None:
        """Give a new settlement notice to the oldest run of a child still owed one."""
        if child.ended:
            ended = child.ended.pop(0)
            self._record(ended.run, ended.span_id, ended.span_parent_id, result)
        else:
            self._settle(child, result)

    def _settle(self, child: _Child, result: str) -> None:
        """Record what a finished run produced against its span, then close it."""
        if not child.is_open:
            return
        if child.name is not None:
            # a message sent to it while it ran was part of that run
            self._sent.pop(child.name, None)
        if child.run is not None:
            self._record(child.run, child.span_id, child.span_parent_id, result)
        self._close(child)

    def _end_run(self, child: _Child, closing: str) -> None:
        """Close the span of a run that ended unseen; the next notice answers it."""
        if child.run is not None:
            child.ended.append(
                _Ended(child.run, child.span_id, child.span_parent_id, closing)
            )
        self._close(child)

    def _record(
        self, run: _Run, span_id: str, span_parent_id: str | None, result: str
    ) -> None:
        """Record a run's call, answered with what the run produced.

        The timeline takes an agent span's result from a tool event beside the
        span that names it (`agent_span_id`), and only failing that, for a
        span named `agent-<call id>`, from that call's tool message, which
        for dsh is the launch acknowledgement (and nothing at all for a
        resumed run's span).
        """
        transcript()._event(
            ToolEvent(
                id=run.call_id,
                function=run.function,
                arguments=run.arguments,
                view=tool_view(run.function, run.arguments),
                result=result,
                agent_span_id=span_id,
                span_id=span_parent_id,
                # not a call Inspect ran: dsh ran the child, and this is
                # its answer as dsh reported it to the parent
                metadata={"source": "deepseek_harness", "from": "settlement notice"},
            )
        )

    def _begin(
        self, child: _Child, parent_span_id: str | None, metadata: dict[str, str]
    ) -> None:
        child.is_open = True
        child.final = None
        child.answered_alone = False
        child.span_parent_id = parent_span_id
        self._open.append(child)
        transcript()._event(
            SpanBeginEvent(
                id=child.span_id,
                parent_id=parent_span_id,
                type="agent",
                name=child.display,
                metadata=metadata or None,
            )
        )

    def _close(self, child: _Child) -> None:
        if not child.is_open:
            return
        child.is_open = False
        self._open.remove(child)
        transcript()._event(SpanEndEvent(id=child.span_id))

    def _register_delegations(
        self,
        message: ChatMessageAssistant,
        parent_span_id: str | None,
        parent: str | None,
    ) -> None:
        """Open a span for each delegation call in an agent's output.

        Runs synchronously before the response reaches dsh, so the span is
        open before the child's first request arrives. `send_message` calls
        are noted for the run they may start (`_resumed_run`).
        """
        for sent in find_messages(message.tool_calls):
            self._sent[str((sent.arguments or {})["agent"])] = sent

        for delegation in find_delegations(message.tool_calls):
            if delegation.call_id in self._by_call:
                continue  # idempotent (defensive against retries)
            span_id = f"agent-{delegation.call_id}"
            child = _Child(
                span_id=span_id,
                base_span_id=span_id,
                display=delegation.name or delegation.description or delegation.tool,
                parent=parent,
                parent_span_id=parent_span_id,
                call_id=delegation.call_id,
                tool=delegation.tool,
                prompt=delegation.prompt,
                metadata=_metadata(delegation),
                run=_Run(delegation.call_id, delegation.tool, delegation.arguments),
            )
            self._by_call[delegation.call_id] = child
            # a name some child already holds is one dsh will reject: such a
            # call stays unnamed until its (error) result drops it
            if delegation.name is not None:
                self._bind(child, _full_name(parent, delegation.name))
            self._begin(child, parent_span_id, dict(child.metadata))


def _new_notices(seen: list[str], visible: list[str]) -> list[str]:
    """The notices in a request that weren't taken before, in order.

    `seen` is every notice taken from one parent's history for one child;
    `visible` is those the latest request carries. That history only grows at
    the end, but compaction cuts it from the front, and a compaction request
    holds a stretch from its middle: what lies within what was seen is old,
    and past the longest overlap with its end is new. Notices are compared by
    their text, so a run that closes with exactly the text of one just cut
    away by compaction goes unseen.
    """
    width = len(visible)
    for start in range(len(seen) - width + 1):
        if seen[start : start + width] == visible:
            return []
    for overlap in range(min(len(seen), width), 0, -1):
        if seen[-overlap:] == visible[:overlap]:
            return visible[overlap:]
    return visible


def _full_name(parent: str | None, name: str) -> str:
    """A child's name in dsh's tree: its path from the root, without `lead/`."""
    return name if parent in (None, ROOT_AGENT_NAME) else f"{parent}/{name}"


def _metadata(delegation: Delegation) -> dict[str, str]:
    metadata = {"tool": delegation.tool}
    if delegation.description is not None:
        metadata["description"] = delegation.description
    return metadata
