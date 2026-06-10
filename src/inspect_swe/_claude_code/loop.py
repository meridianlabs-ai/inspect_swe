"""The Claude Code agent loop as a policy-driven trampoline.

Claude Code runs headless (``--print``) and exits when a turn completes;
continuation is achieved by relaunching with ``--resume <session-id>`` and a
new prompt. Every iteration of the agent loop therefore asks one question:
given the outcome of a launch, do we relaunch, and with what prompt?

This module expresses each concern that answers that question as an
independent *policy* closure:

- operator interventions (interrupt, redirect, backstop, nudge) — only
  present in live (``--acp-server``) mode (`operator_policies`)
- retry of uncaught Claude Code errors (`error_retry_policy`)
- attempts/scoring (`attempts_policy`)

`run_agent_loop` drives them: launch → consult policies in order → first
non-``None`` step wins (``Relaunch`` loops, ``Done`` returns). Each policy
factory owns its state privately, so no loop state is shared across concerns.
`consume_outcome` folds Claude Code's JSONL event stream into a `RunOutcome`.
"""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Sequence,
)

import anyio
from inspect_ai.agent import AgentAttempts, AgentInterrupted, AgentState
from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.scorer import Score, score
from inspect_ai.util import StoreModel
from pydantic import Field

from inspect_swe._claude_code._events.stream import (
    ClaudeCodeStreamEvent,
    ExitEvent,
    JsonlEvent,
    JsonlParseError,
    StderrEvent,
)

from .._util._async import is_callable_coroutine


class ClaudeCodeDebug(StoreModel):
    stderr: list[str] = Field(default_factory=list)
    stdout: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Completed:
    """Claude Code ran to completion (exit code may be non-zero)."""

    exit_code: int
    stderr: str
    any_tool_uses: bool
    """Whether the run made any top-level tool call (sub-agents excluded)."""


@dataclass(frozen=True)
class Interrupted:
    """The operator interrupted the run (Esc); the process was killed."""


@dataclass(frozen=True)
class Redirected:
    """A queued operator message was delivered at a safe seam.

    The process was killed; `text` is the operator message to relaunch with.
    """

    text: str


RunOutcome = Completed | Interrupted | Redirected


@dataclass(frozen=True)
class Relaunch:
    """Relaunch Claude Code (via --resume) with `prompt`."""

    prompt: str


@dataclass(frozen=True)
class Done:
    """The agent loop is finished."""


Step = Relaunch | Done

Launch = Callable[[bool, str], Awaitable[RunOutcome]]
"""One Claude Code launch: (resume, prompt) -> outcome."""

Policy = Callable[[RunOutcome, str], Awaitable[Step | None]]
"""Decide what follows a launch: (outcome, launch_prompt) -> step or decline."""


async def run_agent_loop(
    launch: Launch,
    policies: Sequence[Policy],
    *,
    prompt: str,
    resume: bool,
) -> None:
    """Drive the agent loop: launch, then ask each policy in order.

    The first policy to return a non-``None`` step wins (so the order of
    `policies` encodes precedence). Every relaunch resumes the session —
    only the first launch's `resume` is caller-determined.
    """
    while True:
        outcome = await launch(resume, prompt)
        step: Step | None = None
        for policy in policies:
            step = await policy(outcome, prompt)
            if step is not None:
                break
        if step is None:
            raise RuntimeError(f"No policy handled agent outcome: {outcome}")
        if isinstance(step, Done):
            return
        resume, prompt = True, step.prompt


async def consume_outcome(
    events: AsyncIterator[ClaudeCodeStreamEvent],
    kill: Callable[[], Awaitable[None]],
    *,
    on_jsonl: Callable[[dict[str, Any]], None],
    turn_scope: Callable[[], AbstractContextManager[None]] | None = None,
    pending_message: Callable[[], Awaitable[str | None]] | None = None,
    cc_debug: ClaudeCodeDebug | None = None,
    debug_output: list[str] | None = None,
) -> RunOutcome:
    """Fold Claude Code's event stream into a `RunOutcome`.

    Accumulates stderr/exit-code/tool-use and feeds JSONL lines to `on_jsonl`.
    In live mode (`pending_message` provided), a queued operator message is
    delivered at the next safe seam (`_operator_delivery_gate`): the process
    is killed and `Redirected` returned. If `turn_scope` is provided (live),
    an operator interrupt raised within it kills the process (shielded) and
    returns `Interrupted`.
    """
    is_delivery_seam = (
        _operator_delivery_gate() if pending_message is not None else None
    )
    stderr_data = ""
    exit_code = 0
    any_tool_uses = False

    try:
        with turn_scope() if turn_scope is not None else nullcontext():
            async for cc_event in events:
                if isinstance(cc_event, JsonlEvent):
                    on_jsonl(cc_event.raw)
                    if cc_debug is not None:
                        cc_debug.stdout.append(cc_event.line)
                elif isinstance(cc_event, JsonlParseError):
                    if debug_output is not None:
                        debug_output.append(f"JSONL parse error: {cc_event.line}")
                elif isinstance(cc_event, StderrEvent):
                    stderr_data += cc_event.data
                    if cc_debug is not None:
                        cc_debug.stderr.append(cc_event.data)
                elif isinstance(cc_event, ExitEvent):
                    exit_code = cc_event.code

                if not any_tool_uses and _top_level_tool_use_ids(cc_event):
                    any_tool_uses = True

                if (
                    is_delivery_seam is not None
                    and pending_message is not None
                    and is_delivery_seam(cc_event)
                ):
                    pending = await pending_message()
                    if pending is not None:
                        await kill()
                        return Redirected(pending)
    except AgentInterrupted:
        # Esc. The scope cancel may land away from exec_remote's kill
        # handler, so kill explicitly (shielded + idempotent).
        with anyio.CancelScope(shield=True):
            await kill()
        return Interrupted()

    return Completed(
        exit_code=exit_code, stderr=stderr_data, any_tool_uses=any_tool_uses
    )


def operator_policies(
    *,
    pending: Callable[[], Awaitable[str | None]],
    interrupted: Callable[[], Awaitable[str | None]],
    continue_prompt: str,
) -> list[Policy]:
    """Policies for live operator interventions (in precedence order).

    - **interrupt**: operator pressed Esc — block for their redirect message
      (`interrupted`, i.e. the channel's ``after_cancel``, which runs here at
      the driver level and so is structurally outside ``turn_scope``); if
      they sent none, relaunch with the previous prompt.
    - **redirect**: a queued message was delivered mid-stream at a seam.
    - **backstop**: a message that landed during the final result — drained
      via `pending` regardless of exit code (an operator message beats error
      handling).
    - **nudge**: an operator interjection the model answered conversationally
      (no tool calls) would otherwise end the run; nudge it once back to work.

    The bundle privately owns `launched_for_operator`: set by the first three
    policies, read-AND-cleared by nudge on every completion it is consulted
    on — so the nudge fires at most once per interjection and never after a
    non-operator relaunch (e.g. an attempts re-prompt).
    """
    launched_for_operator = False

    async def on_interrupt(outcome: RunOutcome, prompt: str) -> Step | None:
        nonlocal launched_for_operator
        if not isinstance(outcome, Interrupted):
            return None
        follow = await interrupted()
        launched_for_operator = True
        return Relaunch(follow or prompt)

    async def on_redirect(outcome: RunOutcome, prompt: str) -> Step | None:
        nonlocal launched_for_operator
        if not isinstance(outcome, Redirected):
            return None
        launched_for_operator = True
        return Relaunch(outcome.text)

    async def backstop(outcome: RunOutcome, prompt: str) -> Step | None:
        nonlocal launched_for_operator
        if not isinstance(outcome, Completed):
            return None
        text = await pending()
        if text is None:
            return None
        launched_for_operator = True
        return Relaunch(text)

    async def nudge(outcome: RunOutcome, prompt: str) -> Step | None:
        nonlocal launched_for_operator
        if not isinstance(outcome, Completed):
            return None
        was_operator, launched_for_operator = launched_for_operator, False
        if was_operator and outcome.exit_code == 0 and not outcome.any_tool_uses:
            return Relaunch(continue_prompt.strip())
        return None

    return [on_interrupt, on_redirect, backstop, nudge]


def error_retry_policy(retry_uncaught_errors: int | None) -> Policy:
    """Retry uncaught Claude Code errors; raise on hard failures.

    Exit code 1 with no stderr means an uncaught exception reached the top of
    Claude Code's main loop — treated as a scaffold bug and retried (resumed,
    same prompt) up to `retry_uncaught_errors` times. Any other non-zero exit
    is a hard failure. The retry counter resets on clean completion.
    """
    uncaught_error_count = 0

    async def policy(outcome: RunOutcome, prompt: str) -> Step | None:
        nonlocal uncaught_error_count
        if not isinstance(outcome, Completed):
            return None
        if outcome.exit_code == 0:
            uncaught_error_count = 0
            return None
        if (
            outcome.exit_code == 1
            and len(outcome.stderr.strip()) == 0
            and retry_uncaught_errors is not None
            and uncaught_error_count < retry_uncaught_errors
        ):
            uncaught_error_count += 1
            return Relaunch(prompt)
        raise RuntimeError(
            f"Error executing claude code agent {outcome.exit_code}: {outcome.stderr}"
        )

    return policy


def attempts_policy(
    attempts: AgentAttempts,
    state: AgentState,
    score_fn: Callable[[AgentState], Awaitable[list[Score]]] = score,
) -> Policy:
    """Score completions and re-prompt for additional attempts.

    Counts an attempt per clean completion. At max attempts (note: with the
    default ``attempts=1`` scoring never runs) or on a correct score, the
    loop is done; otherwise relaunch with the configured incorrect message.
    """
    attempt_count = 0

    async def policy(outcome: RunOutcome, prompt: str) -> Step | None:
        nonlocal attempt_count
        if not isinstance(outcome, Completed) or outcome.exit_code != 0:
            return None
        attempt_count += 1
        if attempt_count >= attempts.attempts:
            return Done()
        answer_scores = await score_fn(state)
        if attempts.score_value(answer_scores[0].value) == 1.0:
            return Done()
        if callable(attempts.incorrect_message):
            if not is_callable_coroutine(attempts.incorrect_message):
                raise ValueError("The incorrect_message function must be async.")
            return Relaunch(await attempts.incorrect_message(state, answer_scores))
        return Relaunch(attempts.incorrect_message)

    return policy


def _user_text(msgs: Sequence[ChatMessage]) -> str | None:
    """Join operator user-message text drained from the agent channel.

    Filters out any synthetic ``ChatMessageTool`` repair messages (e.g. those
    ``after_cancel`` prepends) — Claude Code repairs its own conversation when
    we relaunch with ``--resume``, so appending ours would double-repair.
    Returns ``None`` when there are no user messages to forward.
    """
    texts = [m.text for m in msgs if isinstance(m, ChatMessageUser)]
    return "\n\n".join(texts) if texts else None


def _is_turn_boundary(cc_event: ClaudeCodeStreamEvent) -> bool:
    """True at the start of a new top-level Claude Code assistant turn.

    A top-level ``assistant`` event (``parent_tool_use_id is None``) only
    arrives once every prior top-level tool call — including any ``Task``
    sub-agents — has resolved (the top-level model can't start a new turn
    until its tool results are back). So it is a safe seam to abort at: no
    open sub-agent spans and no unresolved top-level tool calls. A sub-agent's
    own ``assistant`` events carry a ``parent_tool_use_id`` and are excluded.
    """
    return (
        isinstance(cc_event, JsonlEvent)
        and cc_event.raw.get("type") == "assistant"
        and cc_event.raw.get("parent_tool_use_id") is None
    )


def _operator_delivery_gate() -> Callable[[ClaudeCodeStreamEvent], bool]:
    """Build the per-run predicate for when a queued operator message may be delivered.

    A *plain* operator message is delivered at the next safe seam — one where
    no top-level tool call is unresolved and no sub-agent span is open, so the
    process can be killed and resumed cleanly. The returned predicate returns
    True at the first of two such seams:

    1. **Tools just completed** — every tool call from the CURRENT top-level
       turn has resolved. This lands BEFORE the next generation starts, so we
       don't wait for (and then discard on --resume) a full wasted turn. The
       fast path for tool-heavy turns.
    2. **Next top-level assistant turn** (:func:`_is_turn_boundary`) — the
       backstop for turns with no tool calls (the outstanding set never
       populates) and for a message that lands during the final result.

    Closes over the set of outstanding *top-level* tool calls: a ``Task``
    sub-agent's own tool calls carry a non-null ``parent_tool_use_id`` and are
    excluded, so the Task's top-level ``tool_result`` (which only arrives once
    the sub-agent has finished) is what clears it — preserving the "all
    top-level tools resolved, no open sub-agent spans" guarantee.
    """
    # Mutated in place (``update`` / ``difference_update``) so the closure never
    # rebinds it and no ``nonlocal`` is needed.
    outstanding: set[str] = set()

    def is_delivery_seam(cc_event: ClaudeCodeStreamEvent) -> bool:
        outstanding.update(_top_level_tool_use_ids(cc_event))
        resolved = _tool_result_ids(cc_event)
        tools_just_completed = False
        if resolved and outstanding:
            outstanding.difference_update(resolved)
            tools_just_completed = not outstanding
        return tools_just_completed or _is_turn_boundary(cc_event)

    return is_delivery_seam


def _top_level_tool_use_ids(cc_event: ClaudeCodeStreamEvent) -> set[str]:
    """IDs of the ``tool_use`` blocks in a *top-level* assistant event.

    Empty for sub-agent assistant events (non-null ``parent_tool_use_id``),
    non-assistant events, and assistant turns with no tool calls.
    """
    if not (
        isinstance(cc_event, JsonlEvent)
        and cc_event.raw.get("type") == "assistant"
        and cc_event.raw.get("parent_tool_use_id") is None
    ):
        return set()
    return _tool_block_ids(cc_event.raw, "tool_use", "id")


def _tool_result_ids(cc_event: ClaudeCodeStreamEvent) -> set[str]:
    """``tool_use_id``s of the ``tool_result`` blocks in a *top-level* user event.

    These mark top-level tool calls whose results have come back. Empty for
    sub-agent user events (non-null ``parent_tool_use_id``) and non-user
    events.
    """
    if not (
        isinstance(cc_event, JsonlEvent)
        and cc_event.raw.get("type") == "user"
        and cc_event.raw.get("parent_tool_use_id") is None
    ):
        return set()
    return _tool_block_ids(cc_event.raw, "tool_result", "tool_use_id")


def _tool_block_ids(raw: dict[str, Any], block_type: str, id_field: str) -> set[str]:
    """Collect ``id_field`` from each ``block_type`` block in ``raw.message.content``."""
    message = raw.get("message")
    if not isinstance(message, dict):
        return set()
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    ids: set[str] = set()
    for block in content:
        if isinstance(block, dict) and block.get("type") == block_type:
            block_id = block.get(id_field)
            if isinstance(block_id, str):
                ids.add(block_id)
    return ids
