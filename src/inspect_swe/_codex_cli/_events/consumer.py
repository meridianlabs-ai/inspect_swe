"""Bridge `ModelEventSink` for Codex CLI sub-agent spans (bridge-only).

Installed on the agent bridge so the bridge hands us every `ModelEvent` for
routing instead of emitting it to the transcript itself. From those events alone
(no Codex `--json` stdout parsing) we reconstruct the agent-span tree:

  1. **Open** (race-free) — when a parent's output contains `spawn_agent`
     tool-calls, `on_complete` opens an agent `SpanBeginEvent` for each, keyed by
     the spawn tool-call id, and registers the spawn prompt for attribution. This
     happens synchronously before the bridge response is returned, so the spans
     are open before any sub-agent can make its first call. An output the
     user's filter returned itself never produces a `ModelEvent`;
     `on_filter_output` registers its spawns instead.

  2. **Attribute** — `on_pending` resolves each call's span by substring-matching
     its user-message text against the open spawn prompts. Codex re-sends a
     sub-agent's spawn prompt as a user message on every request, so this works
     for every call (not just the first). Zero/multiple matches → outer span.

  3. **Bind thread id** — the `spawn_agent` tool *result* carries the sub-agent's
     `agent_id` (thread id), correlated to the spawn call by `tool_call_id`. We
     harvest it from `event.input` so spans can be closed by thread id.

  4. **Close** — on a `close_agent` tool-call (`target=thread_id`) and on any
     `status:completed` notification for a thread id (whichever comes first).
     `reset()` closes orphans between attempts and at the end.

Compaction: our custom bridge provider forces Codex's *local* compaction, a
normal `/v1/responses` call carrying `COMPACTION_MARKER`. We detect it in
`on_pending` and emit a `CompactionEvent` on the attributed span.

Concurrency: attribution is per-request (keyed on the call's own prompt, not
wall-clock state), so parallel sub-agents are handled correctly — each call
routes to its own span regardless of interleaving, and each thread binds/closes
independently via its unique spawn `tool_call_id`.
"""

from dataclasses import dataclass
from logging import getLogger

from inspect_ai.agent import AgentBridgeContext, current_bridge_request
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import Model, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.tool import ToolInfo
from inspect_ai.util._span import current_span_id

from ..config import GUARDIAN_MODEL_SLUG
from .detection import (
    agent_message_recipients,
    completed_thread_ids,
    final_answer_authors,
    find_close_targets,
    find_spawned_agents,
    is_compaction_request,
    spawn_failed,
    spawn_result,
)
from .toolview import tool_view

logger = getLogger(__name__)


# Minimum spawn-prompt length to consider for substring matching, guarding
# against short prompts accidentally matching unrelated content.
_MIN_PROMPT_LENGTH = 16


@dataclass
class _OpenAgent:
    """A sub-agent span currently open (spawn_agent seen, not yet closed)."""

    call_id: str
    span_id: str
    prompt: str
    name: str = "agent"
    thread_id: str | None = None
    agent_type: str | None = None
    """The spawn's `agent_type` argument, when given."""
    expected_path: str | None = None
    """V2 path Codex will give this agent (parent path + task_name), when
    the parent's own path is known."""


class CodexConsumer(ModelEventSink):
    def __init__(self) -> None:
        # spawn tool_call_id → open sub-agent span. Insertion order = open order
        # (used to close innermost-first in reset()).
        self._agents: dict[str, _OpenAgent] = {}

        # thread_id → spawn tool_call_id (bound when the spawn result is seen).
        self._thread_index: dict[str, str] = {}

        # thread_id → nickname (Codex's friendly per-agent name, for tool views).
        self._nicknames: dict[str, str] = {}

        # ModelEvents we've _event()'d, so on_complete knows to _event_updated.
        self._emitted_events: set[int] = set()

        # thread_id → spawn agent_type, kept after the thread closes so a V2
        # child resumed by followup_task still reports its type.
        self._closed_types: dict[str, str] = {}

        # spawn call_id → prompt of a closed V1 spawn; only vetoes a
        # sub-agent id (see `_subagent_context`), never attributes one.
        self._closed_prompts: dict[str, str] = {}

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

        Called between Codex attempts and after the attempt loop, so the span
        tree stays balanced even if Codex exited before closing a sub-agent.
        """
        for call_id in reversed(list(self._agents.keys())):
            agent = self._agents.pop(call_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))
        self._thread_index.clear()
        self._nicknames.clear()
        self._emitted_events.clear()
        self._closed_types.clear()
        self._closed_prompts.clear()

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        # idempotent if classify() already ran over this request
        self._sync_threads(event.input)

        # attribute this call to a span
        span_id = self._attribute(event.input)
        event.span_id = span_id

        # compaction summarization call → emit a marker on the same span
        if is_compaction_request(event.input):
            transcript()._event(
                CompactionEvent(
                    source="codex_cli",
                    span_id=span_id,
                    metadata={"trigger": "auto"},
                )
            )

        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            # custom rendering for Codex built-in tools (see toolview.py)
            for tc in msg.tool_calls:
                if tc.view is None:
                    custom = tool_view(tc.function, tc.arguments or {}, self._nicknames)
                    if custom is not None:
                        tc.view = custom

            self._register_delegation(
                msg,
                event.span_id or self.outer_span_id,
                self._requester_path(event.input),
            )

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    def on_filter_output(
        self, messages: list[ChatMessage], output: ModelOutput
    ) -> None:
        """Track delegation in an output the user's filter returned itself.

        A filter that generates on its own and returns a `ModelOutput` bypasses
        the bridge's generation, so no `ModelEvent` reaches `on_complete`.
        Without this, `spawn_agent` calls in that output would never open a
        span, and the spawned sub-agents' requests would classify `"root"`.
        Called by `classify_filter` before the output is returned to Codex.
        """
        msg = output.message if output.choices else None
        if msg is not None and msg.tool_calls:
            self._register_delegation(
                msg, self._attribute(messages), self._requester_path(messages)
            )

    # ------------------------------------------------------------------
    # filter-time classification (AgentContextClassifier)
    # ------------------------------------------------------------------

    def classify(
        self, _model: Model, messages: list[ChatMessage], _tools: list[ToolInfo]
    ) -> AgentBridgeContext:
        """Filter-time agent classification from the same maps as span attribution.

        Runs inside the bridge filter (before generation), so it must not
        assume `on_pending` has already seen this request — it does its own
        `_sync_threads` + `_attribute`, exactly as `on_pending` would. The
        sync matters for the root thread: under V1 the request that follows
        the last sub-agent's completion carries that completion in its own
        input (wait/close result or notification) and has no agent_message
        recipient, so it is only recognisably "root" once that thread is
        closed here rather than later in `on_pending`.

        Two positive claims are checked first, since neither is visible to
        `_attribute`: the guardian (`auto_review`) model slug and Codex's own
        local-compaction marker both mean "utility" regardless of any open
        sub-agent. Then the V2 `agent_message` recipients, which are inbound
        to the requester: a recipient other than "/root" is a positive
        sub-agent claim that needs no span state, so it holds for a child
        resumed by `followup_task` after its FINAL_ANSWER closed its span, and
        for one whose spawn was never seen; recipients of exactly "/root" are
        a positive root claim even while sub-agents are open (`_attribute`
        only ever falls back to the outer span there). Everything else
        defers to `_attribute`: a sub-agent span means "subagent"; the outer
        span means "root" only when no sub-agents are open (an unattributed
        call with open sub-agents is `_attribute`'s own ambiguity fallback,
        not a genuine root signal) — otherwise "unknown".
        """
        request = current_bridge_request()
        if request is not None and request.model == GUARDIAN_MODEL_SLUG:
            return AgentBridgeContext("utility")
        if is_compaction_request(messages):
            return AgentBridgeContext("utility")

        recipients = agent_message_recipients(messages)
        non_root_recipient = bool(recipients - {"/root"})

        # nothing open → nothing to bind, close, or attribute against
        if not self._agents:
            if non_root_recipient:
                return self._subagent_context(messages, recipients, None)
            if recipients == {"/root"} or not self._carries_closed_prompt(messages):
                return AgentBridgeContext("root")
            return AgentBridgeContext("unknown")

        self._sync_threads(messages)
        if recipients == {"/root"}:
            return AgentBridgeContext("root")

        span_id = self._attribute(messages)
        if span_id != self.outer_span_id or non_root_recipient:
            return self._subagent_context(messages, recipients, span_id)
        if self._agents or self._carries_closed_prompt(messages):
            return AgentBridgeContext("unknown")
        return AgentBridgeContext("root")

    def _carries_closed_prompt(self, messages: list[ChatMessage]) -> bool:
        """Whether a request carries the prompt of a V1 spawn that has closed.

        A completed V1 agent the parent resumes with send_input keeps its
        spawn prompt and has no agent_message to identify it, so such a
        request is never claimed as root. The spawner's own requests carry
        the spawn call or result and are excluded (as in `_attribute`).
        """
        if not self._closed_prompts:
            return False
        user_text = self._user_text(messages)
        user_messages = {
            msg.text.strip() for msg in messages if isinstance(msg, ChatMessageUser)
        }
        own_calls = self._history_call_ids(messages)
        # any length: a short prompt only counts as a whole user message,
        # since as a substring it could appear anywhere
        return any(
            call_id not in own_calls
            and (
                prompt.strip() in user_messages
                or (len(prompt) >= _MIN_PROMPT_LENGTH and prompt in user_text)
            )
            for call_id, prompt in self._closed_prompts.items()
        )

    def _subagent_context(
        self, messages: list[ChatMessage], recipients: set[str], span_id: str | None
    ) -> AgentBridgeContext:
        """The "subagent" context, with the sub-agent's identity when known.

        `subagent_id` is the agent's V2 path (its inbound agent_message
        recipient, e.g. "/root/worker"), present on every one of its requests
        including after a followup_task; under V1 (no agent_messages) it is
        the `spawn_agent` call id of the span the request was attributed to,
        stable from spawn to close (the thread id is only learned later, from
        the parent's next request). `None` when neither identifies one agent,
        or when a V1 request also carries a closed spawn's prompt: a
        completed agent the parent resumes with send_input would otherwise
        borrow the id of a later spawn with the same prompt. The V2 path is
        distinct between live agents; that a closed agent's path is never
        reused by a later spawn is assumed, not verified.
        `subagent_type` is the spawn's `agent_type` argument, when given.
        """
        agent = next((a for a in self._agents.values() if a.span_id == span_id), None)
        paths = recipients - {"/root"}
        if len(paths) == 1:
            path = next(iter(paths))
            agent_type = (
                agent.agent_type if agent is not None else self._closed_types.get(path)
            )
            return AgentBridgeContext(
                "subagent", subagent_id=path, subagent_type=agent_type
            )
        if (
            agent is not None
            and not paths
            and not self._carries_closed_prompt(messages)
        ):
            return AgentBridgeContext(
                "subagent", subagent_id=agent.call_id, subagent_type=agent.agent_type
            )
        return AgentBridgeContext("subagent")

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _register_delegation(
        self,
        msg: ChatMessageAssistant,
        parent_span_id: str | None,
        parent_path: str | None,
    ) -> None:
        """Open a span for each spawn_agent call and close close_agent targets.

        Runs synchronously before the response reaches Codex, so a span is
        open before the sub-agent's first call arrives.
        """
        for spawned in find_spawned_agents(msg.tool_calls):
            if spawned.call_id in self._agents:
                continue  # idempotent (defensive against retries)
            span_id = f"agent-{spawned.call_id}"
            self._agents[spawned.call_id] = _OpenAgent(
                call_id=spawned.call_id,
                span_id=span_id,
                prompt=spawned.message,
                name=spawned.name,
                agent_type=spawned.declared_type,
                expected_path=(
                    f"{parent_path}/{spawned.task_name}"
                    if parent_path is not None and spawned.task_name
                    else None
                ),
            )
            metadata: dict[str, str] = {"agent_type": spawned.agent_type}
            if spawned.task_name:
                metadata["task_name"] = spawned.task_name
            if spawned.reasoning_effort:
                metadata["reasoning_effort"] = spawned.reasoning_effort
            transcript()._event(
                SpanBeginEvent(
                    id=span_id,
                    parent_id=parent_span_id,
                    type="agent",
                    name=spawned.name,
                    metadata=metadata,
                )
            )

        # explicit close_agent calls
        for target in find_close_targets(msg.tool_calls):
            self._close_thread(target)

    @staticmethod
    def _requester_path(messages: list[ChatMessage]) -> str | None:
        """V2 path of the agent that sent a request, when the request proves it.

        Every agent_message in a request is inbound to its sender, so a
        single recipient names it ("/root" or a sub-agent path). Anything
        else proves nothing: no agent_message (root's early requests, but
        also any agent after Codex's local compaction, which drops them), or
        messages to several agents. A spawn from such a request binds to a
        path only from its spawn result, so its nesting and type appear from
        then on. The transcript span is never used: a resumed or untracked
        agent's request lands on the outer span.
        """
        recipients = agent_message_recipients(messages)
        return next(iter(recipients)) if len(recipients) == 1 else None

    def _sync_threads(self, input_messages: list[ChatMessage]) -> None:
        """Bind spawn-result thread ids, then close threads this request completes.

        Completion carriers: V1 wait/close status dicts and notifications; V2
        FINAL_ANSWER agent_messages. Idempotent: closing pops the thread, so a
        second pass over the same request emits no second SpanEndEvent.
        """
        self._harvest_bindings(input_messages)
        for thread_id in completed_thread_ids(input_messages):
            self._close_thread(thread_id)
        for author in final_answer_authors(input_messages):
            self._close_thread(author)

    def _harvest_bindings(self, input_messages: list[ChatMessage]) -> None:
        """Bind thread_id → span from spawn_agent tool results (by tool_call_id).

        A failed spawn (Codex returned an error instead of an agent) closes
        its span: no agent will ever make a request under it, and a retry
        with the same prompt would otherwise leave the real child matching
        two open spawns.
        """
        for msg in input_messages:
            if not isinstance(msg, ChatMessageTool) or msg.tool_call_id is None:
                continue
            result = spawn_result(msg)
            if result is None:
                failed = self._agents.get(msg.tool_call_id)
                if (
                    failed is not None
                    and failed.thread_id is None
                    and spawn_failed(msg)
                ):
                    del self._agents[msg.tool_call_id]
                    transcript()._event(SpanEndEvent(id=failed.span_id))
                continue
            if result.nickname is not None:
                self._nicknames[result.agent_id] = result.nickname
            agent = self._agents.get(msg.tool_call_id)
            if agent is not None and agent.thread_id is None:
                agent.thread_id = result.agent_id
                self._thread_index[result.agent_id] = msg.tool_call_id

    def _close_thread(self, thread_id: str) -> None:
        call_id = self._thread_index.pop(thread_id, None)
        if call_id is None:
            return
        agent = self._agents.pop(call_id, None)
        if agent is None:
            return
        if agent.agent_type is not None:
            self._closed_types[thread_id] = agent.agent_type
        self._closed_prompts[call_id] = agent.prompt
        transcript()._event(SpanEndEvent(id=agent.span_id))

    def _attribute(self, input_messages: list[ChatMessage]) -> str | None:
        """Resolve the span_id for an incoming bridge call.

        Multi-Agent V2 first: every agent_message in a request is inbound to
        the requester, so a single recipient path identifies the calling agent
        exactly (spawn prompts are encrypted under V2, so substring matching
        cannot work). Falls back to the V1 heuristic: substring-match the
        call's user-message text against open spawn prompts, skipping any
        spawn whose call or result is in the request's own history: the
        requester is then the spawner, never the spawned agent (a V1 child's
        thread never carries its own spawn call), and a spawner whose own
        prompt contains the delegated text would otherwise match it. Exactly
        one match → that sub-agent's span; zero/multiple → outer span
        (defensive default).
        """
        if not self._agents:
            return self.outer_span_id

        span_id = self._attribute_by_recipient(input_messages)
        if span_id is not None:
            return span_id

        user_text = self._user_text(input_messages)
        if not user_text:
            return self.outer_span_id

        own_calls = self._history_call_ids(input_messages)
        matches = [
            agent
            for agent in self._agents.values()
            if agent.call_id not in own_calls
            and len(agent.prompt) >= _MIN_PROMPT_LENGTH
            and agent.prompt in user_text
        ]
        if len(matches) == 1:
            return matches[0].span_id
        return self.outer_span_id

    def _attribute_by_recipient(self, input_messages: list[ChatMessage]) -> str | None:
        """Resolve a call's span from its agent_message recipient (V2).

        A bound thread id (from a spawn result or an earlier call) resolves
        directly. Otherwise the recipient is matched against the expected
        path of open spawns awaiting their first call (Codex joins the
        parent's path and the spawn's relative task_name, e.g. "/root" +
        "write_fizzbuzz") — a unique match binds the thread id so later calls
        and FINAL_ANSWER closes resolve directly. A spawn whose parent path
        is unknown binds only from its spawn result. Returns None (fall
        through) when no unambiguous match exists.
        """
        recipients = agent_message_recipients(input_messages)
        if len(recipients) != 1:
            return None
        recipient = next(iter(recipients))

        # already bound (spawn result or earlier call)
        call_id = self._thread_index.get(recipient)
        if call_id is not None:
            agent = self._agents.get(call_id)
            if agent is not None:
                return agent.span_id

        # first call from this thread: bind the open spawn Codex gives this
        # exact path (parent path + task_name). A basename alone could be a
        # same-named agent under another parent, or a finished agent resumed.
        candidates = [
            agent
            for agent in self._agents.values()
            if agent.thread_id is None and agent.expected_path == recipient
        ]
        if len(candidates) == 1:
            agent = candidates[0]
            agent.thread_id = recipient
            self._thread_index[recipient] = agent.call_id
            return agent.span_id

        return None

    @staticmethod
    def _history_call_ids(input_messages: list[ChatMessage]) -> set[str]:
        """Tool-call ids the requester made or received results for."""
        ids: set[str] = set()
        for msg in input_messages:
            if isinstance(msg, ChatMessageAssistant):
                ids.update(tc.id for tc in msg.tool_calls or [])
            elif isinstance(msg, ChatMessageTool) and msg.tool_call_id:
                ids.add(msg.tool_call_id)
        return ids

    @staticmethod
    def _user_text(input_messages: list[ChatMessage]) -> str:
        """Concatenated text of user messages used for attribution.

        Excludes `<subagent_notification>` messages: those appear in a *parent's*
        input and carry sub-agent *answers* (not spawn prompts), which could
        otherwise cause a parent call to false-match a sub-agent span.
        """
        return "\n".join(
            msg.text
            for msg in input_messages
            if isinstance(msg, ChatMessageUser)
            and msg.text
            and "<subagent_notification>" not in msg.text
        )
