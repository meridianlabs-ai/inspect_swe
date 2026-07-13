"""Bridge `ModelEventSink` for Codex CLI sub-agent spans (bridge-only).

Installed on the agent bridge so the bridge hands us every `ModelEvent` for
routing instead of emitting it to the transcript itself. From those events alone
(no Codex `--json` stdout parsing) we reconstruct the agent-span tree:

  1. **Open** (race-free) — when a parent's output contains `spawn_agent`
     tool-calls, `on_complete` opens an agent `SpanBeginEvent` for each, keyed by
     the spawn tool-call id, and registers the spawn prompt for attribution. This
     happens synchronously before the bridge response is returned, so the spans
     are open before any sub-agent can make its first call.

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

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.util._span import current_span_id

from .detection import (
    completed_thread_ids,
    find_close_targets,
    find_spawned_agents,
    is_compaction_request,
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
    thread_id: str | None = None


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

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        # bind thread ids from any spawn results, then close completed threads
        self._harvest_bindings(event.input)
        for thread_id in completed_thread_ids(event.input):
            self._close_thread(thread_id)

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

            # open a span for each spawned sub-agent — synchronously, before the
            # bridge response is returned, so the span is ready before the
            # sub-agent's first call arrives.
            parent_span_id = event.span_id or self.outer_span_id
            for spawned in find_spawned_agents(msg.tool_calls):
                if spawned.call_id in self._agents:
                    continue  # idempotent (defensive against retries)
                span_id = f"agent-{spawned.call_id}"
                self._agents[spawned.call_id] = _OpenAgent(
                    call_id=spawned.call_id,
                    span_id=span_id,
                    prompt=spawned.message,
                )
                metadata: dict[str, str] = {"agent_type": spawned.agent_type}
                if spawned.reasoning_effort:
                    metadata["reasoning_effort"] = spawned.reasoning_effort
                transcript()._event(
                    SpanBeginEvent(
                        id=span_id,
                        parent_id=parent_span_id,
                        type="agent",
                        name=spawned.agent_type,
                        metadata=metadata,
                    )
                )

            # explicit close_agent calls
            for target in find_close_targets(msg.tool_calls):
                self._close_thread(target)

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _harvest_bindings(self, input_messages: list[ChatMessage]) -> None:
        """Bind thread_id → span from spawn_agent tool results (by tool_call_id)."""
        for msg in input_messages:
            if not isinstance(msg, ChatMessageTool):
                continue
            result = spawn_result(msg)
            if result is None or msg.tool_call_id is None:
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
        transcript()._event(SpanEndEvent(id=agent.span_id))

    def _attribute(self, input_messages: list[ChatMessage]) -> str | None:
        """Resolve the span_id for an incoming bridge call.

        Substring-matches the call's user-message text against open spawn
        prompts. Exactly one match → that sub-agent's span; zero/multiple →
        outer span (defensive default).
        """
        if not self._agents:
            return self.outer_span_id

        user_text = self._user_text(input_messages)
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
