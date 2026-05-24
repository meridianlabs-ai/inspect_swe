"""Real-time consumer of Claude Code JSONL output.

Emits `SpanBeginEvent`/`SpanEndEvent` (for agent boundaries) and
`CompactionEvent` (for compact-boundary system events) to the transcript
as the JSONL stream arrives, and resolves per-request `span_id` for the
inspect bridge so each emitted `ModelEvent` is attributed to the correct
agent span.

The agent-tree is reconstructed from `Task`/`Agent` tool_use blocks in
assistant events and their matching `tool_result` blocks in user events.
The `parent_tool_use_id` field on each JSONL event tells us the enclosing
agent context (None for the main agent, the spawning Task's tool_use_id
for sub-agents). That's enough to compute correct `parent_id`s for span
emission and to handle nested sub-agents.

The resolver correlates bridge requests to agent spans via the first user
message's text content. Every sub-agent's `input[0]` is the prompt the
parent passed to its `Task` tool — sent verbatim by Claude Code, and
stable across all of the sub-agent's subsequent calls because Anthropic's
conversation history always begins with the initial user message.

Two refinements keep correlation correct and cheap:

* Open agents are indexed by `tool_use_id` (authoritative key) with a
  secondary `prompt → [tool_use_id, ...]` index. When an agent closes,
  both indexes are pruned, so a stale closed-span never gets returned.
  Duplicate concurrent prompts are detected (list length > 1) and
  treated as ambiguous — we fall back to outer span with a warning.

* The main agent's first user prompt is seeded via `set_main_prompt()`
  so its calls match immediately. As an additional safeguard, any
  request with `len(input_messages) > 1` (i.e. not a first call) is
  resolved immediately without the bounded wait, since the
  sub-agent-first-call race window only applies when `len(input) == 1`.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

import anyio
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
)

logger = getLogger(__name__)


# Bounded wait when a sub-agent's first call beats our JSONL processing.
# JSONL polling cycles at ~500ms, so we wait one cycle plus a small margin.
_RESOLVE_WAIT_SECONDS = 0.6


@dataclass
class _OpenAgent:
    """An agent span currently open (Task tool_use seen, no tool_result yet)."""

    span_id: str
    prompt: str


class LiveConsumer:
    """Drains Claude Code JSONL in real time and resolves bridge span_ids.

    Two callbacks form the contract:

    - `process_jsonl_line(raw)` is called from the main task as each JSONL
      line arrives. It updates internal state and emits agent SpanBegin/End
      and CompactionEvent directly to the transcript.

    - `resolve_span(input_messages)` is called from the bridge's per-request
      task. It returns the span_id this request belongs to (always
      non-None: a known sub-agent span, or the outer span as a fallback).
    """

    def __init__(self, outer_span_id: str | None) -> None:
        self.outer_span_id = outer_span_id

        # First user prompt of the main agent. Set via set_main_prompt()
        # after build_user_prompt() in claude_code.py. Used by resolve_span
        # to short-circuit main-agent calls without hitting the wait.
        self._main_prompt: str | None = None

        # tool_use_id → _OpenAgent for currently-open agent spans.
        # Authoritative — tool_use_id is the unique key Claude Code uses.
        self._open_agents: dict[str, _OpenAgent] = {}

        # prompt → list of currently-open agent tool_use_ids with that
        # prompt. Secondary index for resolver lookup. List (not single
        # value) so we can detect duplicate concurrent prompts as
        # ambiguous rather than silently returning a stale or wrong span.
        self._prompt_to_open_agents: dict[str, list[str]] = {}

        # Signaled after each JSONL line is processed. Consumed by
        # resolve_span() when it needs to wait for the parent's JSONL line
        # to arrive before a sub-agent's first bridge call can be
        # resolved. Replaced with a fresh Event on each notify
        # (anyio.Event is set-once).
        self._jsonl_progress = anyio.Event()

    def set_main_prompt(self, prompt: str) -> None:
        """Seed the main agent's first user prompt.

        Call this after building the prompt that's passed to Claude Code
        on the command line. The main agent's `input[0]` is always this
        prompt (stable across attempts because `--resume` preserves the
        original first user message). Seeding lets the resolver return
        the outer span_id immediately for main-agent calls instead of
        timing out on the bounded wait.
        """
        self._main_prompt = prompt

    def reset(self) -> None:
        """Close all open agent spans and clear state.

        Called between Claude Code subprocess restarts (retry attempts)
        and in a `finally` after the retry loop. Emits a `SpanEndEvent`
        for every still-open agent span (innermost first) so the
        transcript stays balanced even if Claude Code crashed mid-flight
        before its `tool_result`s were written. Without this, ACP depth
        tracking and other span-aware consumers would stay stuck inside
        a phantom sub-agent.

        `outer_span_id` and `_main_prompt` are preserved — they describe
        the enclosing `@agent` context, which spans all attempts.
        """
        for tool_use_id in reversed(list(self._open_agents.keys())):
            agent = self._open_agents.pop(tool_use_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))
        # _open_agents is now empty; clear the secondary index too.
        self._prompt_to_open_agents.clear()

    def process_jsonl_line(self, raw: dict[str, Any]) -> None:
        """Process one raw JSONL event from Claude Code.

        Called from the main task; no awaits — safe to call from sync code
        too.
        """
        event_type = raw.get("type")

        if event_type == "assistant":
            self._handle_assistant(raw)
        elif event_type == "user":
            self._handle_user(raw)
        elif event_type == "system":
            self._handle_system(raw)

        # Wake any resolver waiting for new agent boundaries.
        prev = self._jsonl_progress
        self._jsonl_progress = anyio.Event()
        prev.set()

    async def resolve_span(self, input_messages: list[ChatMessage]) -> str | None:
        """Return the span_id this bridge request belongs to.

        Resolution order:

        1. Empty input → outer span (defensive).
        2. Find the first ChatMessageUser, skipping any leading
           ChatMessageSystem entries (the bridge prepends one for any
           API request with a `system` field — Anthropic, OpenAI Chat,
           OpenAI Responses, Google all do this). Claude Code uses
           `--append-system-prompt`, so every request has at least one
           leading system message. If there's no user message at all →
           outer span (defensive).
        3. Seeded main prompt match → outer span (immediate).
        4. Open sub-agent match by prompt → that sub-agent's span
           (immediate; ambiguity falls back to outer with warning).
        5. Subsequent call (any messages after the first user) → outer
           span (immediate; the race window only applies to a sub-agent's
           very first bridge call).
        6. Bounded wait for a matching sub-agent prompt to appear in our
           map (sub-agent's first call before our JSONL caught up).
        7. Timeout → outer span (logged warning).
        """
        if not input_messages:
            return self.outer_span_id

        # Find the first user message past any leading system messages.
        first_user_idx: int | None = None
        for i, msg in enumerate(input_messages):
            if isinstance(msg, ChatMessageSystem):
                continue
            if isinstance(msg, ChatMessageUser):
                first_user_idx = i
            break
        if first_user_idx is None:
            return self.outer_span_id

        prompt = input_messages[first_user_idx].text

        # Main agent — known prompt, immediate.
        if self._main_prompt is not None and prompt == self._main_prompt:
            return self.outer_span_id

        # Known sub-agent — immediate.
        span_id = self._lookup_open_agent(prompt)
        if span_id is not None:
            return span_id

        # Not a first call: a sub-agent's first call has exactly one
        # message after the leading system prefix (the Task prompt).
        # Anything more means the conversation has progressed past the
        # first turn, so the race window for an unobserved sub-agent
        # first-call is past — if we still don't recognize the prompt,
        # it's not a sub-agent we know about. Fall back rather than
        # waste the wait timeout.
        if len(input_messages) > first_user_idx + 1:
            return self.outer_span_id

        # Bounded wait: rearm-aware loop that captures the current Event
        # before checking the dict, so we never miss a notification.
        # Also re-check main_prompt in case it gets seeded mid-wait.
        with anyio.move_on_after(_RESOLVE_WAIT_SECONDS):
            while True:
                event = self._jsonl_progress
                if self._main_prompt is not None and prompt == self._main_prompt:
                    return self.outer_span_id
                span_id = self._lookup_open_agent(prompt)
                if span_id is not None:
                    return span_id
                await event.wait()

        logger.warning(
            "span_id_resolver: timed out resolving first-call prompt "
            "(len=%d); falling back to outer span",
            len(prompt),
        )
        return self.outer_span_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_open_agent(self, prompt: str) -> str | None:
        """Look up the open agent span_id for a given prompt.

        Returns:
          - the span_id if exactly one open agent has this prompt.
          - `outer_span_id` (with a logged warning) if multiple open
            agents share this prompt — we can't disambiguate from
            input[0] alone, so we fall back rather than guess.
          - `None` if no open agent has this prompt.
        """
        ids = self._prompt_to_open_agents.get(prompt)
        if not ids:
            return None
        if len(ids) > 1:
            logger.warning(
                "span_id_resolver: %d open agents share the same first-user "
                "prompt — cannot disambiguate from request content alone; "
                "falling back to outer span",
                len(ids),
            )
            return self.outer_span_id
        return self._open_agents[ids[0]].span_id

    def _open_agent(
        self,
        tool_use_id: str,
        span_id: str,
        prompt: str,
    ) -> None:
        """Record a newly-opened agent in both indexes."""
        self._open_agents[tool_use_id] = _OpenAgent(span_id=span_id, prompt=prompt)
        self._prompt_to_open_agents.setdefault(prompt, []).append(tool_use_id)

    def _close_agent(self, tool_use_id: str) -> _OpenAgent | None:
        """Remove an agent from both indexes; return the _OpenAgent or None."""
        agent = self._open_agents.pop(tool_use_id, None)
        if agent is None:
            return None
        ids_list = self._prompt_to_open_agents.get(agent.prompt)
        if ids_list is not None:
            try:
                ids_list.remove(tool_use_id)
            except ValueError:
                pass
            if not ids_list:
                self._prompt_to_open_agents.pop(agent.prompt, None)
        return agent

    # ------------------------------------------------------------------
    # JSONL event handlers
    # ------------------------------------------------------------------

    def _handle_assistant(self, raw: dict[str, Any]) -> None:
        """Look for Task tool_use blocks and open agent spans."""
        message = raw.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            return

        enclosing_parent_tool_use_id = raw.get("parent_tool_use_id")

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") not in ("Task", "Agent"):
                continue

            tool_use_id = block.get("id")
            tool_input = block.get("input", {}) or {}
            prompt = str(tool_input.get("prompt", ""))
            if not tool_use_id or not prompt:
                continue

            agent_span_id = f"agent-{tool_use_id}"

            # Parent span: the enclosing agent context for the assistant
            # message that issued this Task call. None means main agent —
            # the outer @agent span.
            if enclosing_parent_tool_use_id:
                parent_agent = self._open_agents.get(enclosing_parent_tool_use_id)
                parent_id = (
                    parent_agent.span_id if parent_agent else self.outer_span_id
                )
            else:
                parent_id = self.outer_span_id

            # Record before emitting so any racing resolver finds the
            # mapping the moment _jsonl_progress is signaled.
            self._open_agent(
                tool_use_id=tool_use_id, span_id=agent_span_id, prompt=prompt
            )

            span_name = (
                tool_input.get("name") or tool_input.get("subagent_type") or "agent"
            )
            description = tool_input.get("description") or ""

            transcript()._event(
                SpanBeginEvent(
                    id=agent_span_id,
                    parent_id=parent_id,
                    type="agent",
                    name=str(span_name),
                    metadata={"description": description} if description else None,
                )
            )

    def _handle_user(self, raw: dict[str, Any]) -> None:
        """Look for tool_result blocks that close open agent spans."""
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
            agent = self._close_agent(tool_use_id)
            if agent is None:
                continue
            transcript()._event(SpanEndEvent(id=agent.span_id))

    def _handle_system(self, raw: dict[str, Any]) -> None:
        """Emit CompactionEvent for compact_boundary system events."""
        if raw.get("subtype") != "compact_boundary":
            return

        enclosing_parent_tool_use_id = raw.get("parent_tool_use_id")
        if enclosing_parent_tool_use_id:
            parent_agent = self._open_agents.get(enclosing_parent_tool_use_id)
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
