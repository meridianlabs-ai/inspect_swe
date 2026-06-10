"""Unit tests for the claude_code policy-driven agent loop (loop.py).

Everything runs against fakes — scripted launches, scripted event streams,
and async-lambda channel drains — no sandbox. Covers the driver
(`run_agent_loop`), each policy factory, the stream fold (`consume_outcome`),
and chain-integration behaviors that pin the policy ORDER (operator
continuation beats error retry beats attempt scoring).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Sequence

import anyio
import pytest
from inspect_ai.agent import AgentAttempts, AgentInterrupted, AgentState
from inspect_ai.scorer import Score
from inspect_swe._claude_code._events.stream import (
    ClaudeCodeStreamEvent,
    ExitEvent,
    JsonlEvent,
    StderrEvent,
)
from inspect_swe._claude_code.loop import (
    Completed,
    Done,
    Interrupted,
    Launch,
    Policy,
    Redirected,
    Relaunch,
    RunOutcome,
    Step,
    attempts_policy,
    consume_outcome,
    error_retry_policy,
    operator_policies,
    run_agent_loop,
)

# --- fakes ----------------------------------------------------------------


def scripted_launch(
    outcomes: Sequence[RunOutcome], launches: list[tuple[bool, str]]
) -> Launch:
    """A Launch that replays `outcomes` and records (resume, prompt) calls."""
    remaining = list(outcomes)

    async def launch(resume: bool, prompt: str) -> RunOutcome:
        launches.append((resume, prompt))
        return remaining.pop(0)

    return launch


def returns(value: str | None) -> Callable[[], Awaitable[str | None]]:
    """An async channel drain returning a fixed value."""

    async def drain() -> str | None:
        return value

    return drain


async def consult(
    policies: Sequence[Policy], outcome: RunOutcome, prompt: str
) -> Step | None:
    """Mimic the driver: first non-None policy step wins."""
    for policy in policies:
        step = await policy(outcome, prompt)
        if step is not None:
            return step
    return None


def completed(
    exit_code: int = 0, stderr: str = "", any_tool_uses: bool = False
) -> Completed:
    return Completed(exit_code=exit_code, stderr=stderr, any_tool_uses=any_tool_uses)


# --- run_agent_loop -------------------------------------------------------


def test_loop_first_non_none_policy_wins() -> None:
    consulted: list[str] = []

    def recording_policy(name: str, step: Step | None) -> Policy:
        async def policy(outcome: RunOutcome, prompt: str) -> Step | None:
            consulted.append(name)
            return step

        return policy

    async def run() -> None:
        launches: list[tuple[bool, str]] = []
        await run_agent_loop(
            scripted_launch([completed()], launches),
            [
                recording_policy("declines", None),
                recording_policy("wins", Done()),
                recording_policy("never", Done()),
            ],
            prompt="p",
            resume=False,
        )

    anyio.run(run)
    assert consulted == ["declines", "wins"]


def test_loop_relaunch_then_done_and_resume_transitions() -> None:
    async def run() -> None:
        first = True

        async def relaunch_once(outcome: RunOutcome, prompt: str) -> Step | None:
            nonlocal first
            if first:
                first = False
                return Relaunch("second prompt")
            return Done()

        launches: list[tuple[bool, str]] = []
        await run_agent_loop(
            scripted_launch([completed(), completed()], launches),
            [relaunch_once],
            prompt="first prompt",
            resume=False,
        )
        # initial resume honored; every relaunch resumes
        assert launches == [(False, "first prompt"), (True, "second prompt")]

    anyio.run(run)


def test_loop_raises_when_no_policy_handles_outcome() -> None:
    async def run() -> None:
        launches: list[tuple[bool, str]] = []
        with pytest.raises(RuntimeError, match="No policy handled"):
            await run_agent_loop(
                scripted_launch([completed()], launches),
                [],
                prompt="p",
                resume=False,
            )

    anyio.run(run)


# --- operator_policies ----------------------------------------------------


def test_operator_interrupt_uses_follow_text() -> None:
    async def run() -> None:
        policies = operator_policies(
            pending=returns(None), interrupted=returns("go left"), continue_prompt="c"
        )
        step = await consult(policies, Interrupted(), "original")
        assert step == Relaunch("go left")

    anyio.run(run)


def test_operator_interrupt_without_follow_relaunches_previous_prompt() -> None:
    async def run() -> None:
        policies = operator_policies(
            pending=returns(None), interrupted=returns(None), continue_prompt="c"
        )
        step = await consult(policies, Interrupted(), "original")
        assert step == Relaunch("original")

    anyio.run(run)


def test_operator_redirect_relaunches_with_text() -> None:
    async def run() -> None:
        policies = operator_policies(
            pending=returns(None), interrupted=returns(None), continue_prompt="c"
        )
        step = await consult(policies, Redirected("new direction"), "original")
        assert step == Relaunch("new direction")

    anyio.run(run)


def test_operator_backstop_fires_and_declines() -> None:
    async def run() -> None:
        fires = operator_policies(
            pending=returns("queued"), interrupted=returns(None), continue_prompt="c"
        )
        assert await consult(fires, completed(), "p") == Relaunch("queued")

        declines = operator_policies(
            pending=returns(None), interrupted=returns(None), continue_prompt="c"
        )
        assert await consult(declines, completed(any_tool_uses=True), "p") is None

    anyio.run(run)


def test_operator_backstop_fires_even_on_nonzero_exit() -> None:
    # An operator message beats error handling: a crashed launch with a
    # queued message gets the operator relaunch, not an error retry.
    async def run() -> None:
        policies = operator_policies(
            pending=returns("queued"), interrupted=returns(None), continue_prompt="c"
        )
        step = await consult(policies, completed(exit_code=1), "p")
        assert step == Relaunch("queued")

    anyio.run(run)


def test_operator_nudge_fires_once_per_interjection() -> None:
    async def run() -> None:
        policies = operator_policies(
            pending=returns(None),
            interrupted=returns(None),
            continue_prompt=" keep going ",
        )
        # operator redirect arms the nudge
        await consult(policies, Redirected("note"), "p")
        # conversational completion (no tools) -> nudged, stripped prompt
        assert await consult(policies, completed(), "note") == Relaunch("keep going")
        # still no tools after the nudge -> NOT nudged again
        assert await consult(policies, completed(), "keep going") is None

    anyio.run(run)


def test_operator_nudge_declines_when_tools_used() -> None:
    async def run() -> None:
        policies = operator_policies(
            pending=returns(None), interrupted=returns(None), continue_prompt="c"
        )
        await consult(policies, Redirected("note"), "p")
        assert await consult(policies, completed(any_tool_uses=True), "note") is None

    anyio.run(run)


def test_operator_nudge_flag_dies_at_non_operator_relaunch() -> None:
    # Lifetime regression: operator redirect -> tool-using completion (nudge
    # declines AND clears the flag) -> a later no-tool completion (e.g. after
    # an attempts re-prompt) must NOT nudge.
    async def run() -> None:
        policies = operator_policies(
            pending=returns(None), interrupted=returns(None), continue_prompt="c"
        )
        await consult(policies, Redirected("note"), "p")
        assert await consult(policies, completed(any_tool_uses=True), "note") is None
        assert await consult(policies, completed(), "try again") is None

    anyio.run(run)


# --- error_retry_policy ---------------------------------------------------


def test_error_retry_retries_then_raises() -> None:
    async def run() -> None:
        policy = error_retry_policy(2)
        failure = completed(exit_code=1)
        assert await policy(failure, "p") == Relaunch("p")
        assert await policy(failure, "p") == Relaunch("p")
        with pytest.raises(RuntimeError, match="Error executing claude code agent 1"):
            await policy(failure, "p")

    anyio.run(run)


def test_error_retry_raises_immediately_with_stderr() -> None:
    async def run() -> None:
        policy = error_retry_policy(3)
        with pytest.raises(RuntimeError, match="boom"):
            await policy(completed(exit_code=1, stderr="boom"), "p")

    anyio.run(run)


def test_error_retry_raises_immediately_when_disabled() -> None:
    async def run() -> None:
        policy = error_retry_policy(None)
        with pytest.raises(RuntimeError):
            await policy(completed(exit_code=1), "p")

    anyio.run(run)


def test_error_retry_counter_resets_on_success() -> None:
    async def run() -> None:
        policy = error_retry_policy(1)
        failure = completed(exit_code=1)
        assert await policy(failure, "p") == Relaunch("p")
        assert await policy(completed(), "p") is None  # success resets
        assert await policy(failure, "p") == Relaunch("p")  # full budget again

    anyio.run(run)


def test_error_retry_declines_non_completed() -> None:
    async def run() -> None:
        policy = error_retry_policy(3)
        assert await policy(Interrupted(), "p") is None
        assert await policy(Redirected("x"), "p") is None

    anyio.run(run)


# --- attempts_policy ------------------------------------------------------


def scoring(
    values: Sequence[float], calls: list[int]
) -> Callable[[AgentState], Awaitable[list[Score]]]:
    async def score_fn(state: AgentState) -> list[Score]:
        calls.append(1)
        return [Score(value=values[len(calls) - 1])]

    return score_fn


def test_attempts_single_attempt_never_scores() -> None:
    async def run() -> None:
        calls: list[int] = []
        policy = attempts_policy(
            AgentAttempts(attempts=1), AgentState(messages=[]), scoring([1.0], calls)
        )
        assert await policy(completed(), "p") == Done()
        assert calls == []

    anyio.run(run)


def test_attempts_correct_score_is_done() -> None:
    async def run() -> None:
        calls: list[int] = []
        policy = attempts_policy(
            AgentAttempts(attempts=2), AgentState(messages=[]), scoring([1.0], calls)
        )
        assert await policy(completed(), "p") == Done()
        assert calls == [1]

    anyio.run(run)


def test_attempts_incorrect_relaunches_with_message_then_done_at_max() -> None:
    async def run() -> None:
        calls: list[int] = []
        policy = attempts_policy(
            AgentAttempts(attempts=2, incorrect_message="try again"),
            AgentState(messages=[]),
            scoring([0.0], calls),
        )
        assert await policy(completed(), "p") == Relaunch("try again")
        assert await policy(completed(), "try again") == Done()  # at max
        assert calls == [1]  # final attempt not scored

    anyio.run(run)


def test_attempts_async_incorrect_message() -> None:
    async def run() -> None:
        async def incorrect(state: AgentState, scores: list[Score]) -> str:
            return f"scored {scores[0].value}, try again"

        calls: list[int] = []
        policy = attempts_policy(
            AgentAttempts(attempts=3, incorrect_message=incorrect),
            AgentState(messages=[]),
            scoring([0.0], calls),
        )
        assert await policy(completed(), "p") == Relaunch("scored 0.0, try again")

    anyio.run(run)


def test_attempts_sync_incorrect_message_raises() -> None:
    async def run() -> None:
        def incorrect(state: AgentState, scores: list[Score]) -> str:
            return "nope"

        calls: list[int] = []
        policy = attempts_policy(
            AgentAttempts(attempts=2, incorrect_message=incorrect),  # type: ignore[arg-type]
            AgentState(messages=[]),
            scoring([0.0], calls),
        )
        with pytest.raises(ValueError, match="must be async"):
            await policy(completed(), "p")

    anyio.run(run)


def test_attempts_declines_failures_and_non_completed() -> None:
    async def run() -> None:
        calls: list[int] = []
        policy = attempts_policy(
            AgentAttempts(attempts=2), AgentState(messages=[]), scoring([1.0], calls)
        )
        assert await policy(completed(exit_code=1), "p") is None
        assert await policy(Interrupted(), "p") is None
        assert calls == []

    anyio.run(run)


# --- consume_outcome ------------------------------------------------------


def _jsonl(raw: dict[str, Any]) -> JsonlEvent:
    return JsonlEvent(raw=raw, line="{}")


def _assistant(*tool_ids: str, parent: str | None = None) -> JsonlEvent:
    raw: dict[str, Any] = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "thinking"},
                *(
                    {"type": "tool_use", "id": tid, "name": "Bash", "input": {}}
                    for tid in tool_ids
                ),
            ]
        },
    }
    if parent is not None:
        raw["parent_tool_use_id"] = parent
    return _jsonl(raw)


async def _events(
    events: Sequence[ClaudeCodeStreamEvent],
    raise_interrupt_after: int | None = None,
) -> AsyncIterator[ClaudeCodeStreamEvent]:
    for i, event in enumerate(events):
        if raise_interrupt_after is not None and i >= raise_interrupt_after:
            raise AgentInterrupted()
        yield event


class _Kill:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


def test_consume_accumulates_completed() -> None:
    async def run() -> None:
        jsonl_lines: list[dict[str, Any]] = []
        kill = _Kill()
        outcome = await consume_outcome(
            _events(
                [
                    _assistant("t1"),
                    StderrEvent(data="warn: "),
                    StderrEvent(data="thing"),
                    ExitEvent(code=3),
                ]
            ),
            kill,
            on_jsonl=jsonl_lines.append,
        )
        assert outcome == Completed(
            exit_code=3, stderr="warn: thing", any_tool_uses=True
        )
        assert len(jsonl_lines) == 1
        assert kill.calls == 0

    anyio.run(run)


def test_consume_subagent_tools_do_not_count_as_tool_uses() -> None:
    async def run() -> None:
        outcome = await consume_outcome(
            _events([_assistant("s1", parent="toolu_task"), ExitEvent(code=0)]),
            _Kill(),
            on_jsonl=lambda raw: None,
        )
        assert outcome == Completed(exit_code=0, stderr="", any_tool_uses=False)

    anyio.run(run)


def test_consume_delivers_pending_message_at_seam() -> None:
    async def run() -> None:
        kill = _Kill()
        outcome = await consume_outcome(
            # a top-level assistant event is a delivery seam
            _events([_assistant(), _assistant()]),
            kill,
            on_jsonl=lambda raw: None,
            pending_message=returns("steer this way"),
        )
        assert outcome == Redirected("steer this way")
        assert kill.calls == 1

    anyio.run(run)


def test_consume_continues_when_no_pending_message() -> None:
    async def run() -> None:
        kill = _Kill()
        outcome = await consume_outcome(
            _events([_assistant(), ExitEvent(code=0)]),
            kill,
            on_jsonl=lambda raw: None,
            pending_message=returns(None),
        )
        assert outcome == Completed(exit_code=0, stderr="", any_tool_uses=False)
        assert kill.calls == 0

    anyio.run(run)


def test_consume_non_live_never_drains() -> None:
    async def run() -> None:
        # pending_message=None: no gate is built, nothing is drained
        outcome = await consume_outcome(
            _events([_assistant("t1"), ExitEvent(code=0)]),
            _Kill(),
            on_jsonl=lambda raw: None,
            pending_message=None,
        )
        assert isinstance(outcome, Completed)

    anyio.run(run)


def test_consume_interrupt_kills_shielded_and_returns_interrupted() -> None:
    @contextmanager
    def fake_turn_scope() -> Iterator[None]:
        yield

    async def run() -> None:
        kill = _Kill()
        outcome = await consume_outcome(
            _events([_assistant(), _assistant()], raise_interrupt_after=1),
            kill,
            on_jsonl=lambda raw: None,
            turn_scope=fake_turn_scope,
        )
        assert outcome == Interrupted()
        assert kill.calls == 1

    anyio.run(run)


def test_consume_debug_capture() -> None:
    from inspect_swe._claude_code._events.stream import JsonlParseError

    async def run() -> None:
        debug_output: list[str] = []
        outcome = await consume_outcome(
            _events(
                [
                    JsonlParseError(line="not json"),
                    StderrEvent(data="err"),
                    ExitEvent(code=0),
                ]
            ),
            _Kill(),
            on_jsonl=lambda raw: None,
            debug_output=debug_output,
        )
        assert outcome == Completed(exit_code=0, stderr="err", any_tool_uses=False)
        assert debug_output == ["JSONL parse error: not json"]

    anyio.run(run)


# --- chain integration (pins policy ORDER) --------------------------------


def full_chain(
    *,
    pending: str | None = None,
    interrupted: str | None = None,
    continue_prompt: str = "continue",
    retry_uncaught_errors: int | None = 3,
    attempts: AgentAttempts | None = None,
    score_calls: list[int] | None = None,
    score_values: Sequence[float] = (1.0,),
) -> list[Policy]:
    attempts = attempts if attempts is not None else AgentAttempts(attempts=1)
    return [
        *operator_policies(
            pending=returns(pending),
            interrupted=returns(interrupted),
            continue_prompt=continue_prompt,
        ),
        error_retry_policy(retry_uncaught_errors),
        attempts_policy(
            attempts,
            AgentState(messages=[]),
            scoring(score_values, score_calls if score_calls is not None else []),
        ),
    ]


def test_chain_score_never_runs_after_operator_relaunch() -> None:
    async def run() -> None:
        score_calls: list[int] = []
        launches: list[tuple[bool, str]] = []
        await run_agent_loop(
            scripted_launch(
                [Redirected("steer"), completed(any_tool_uses=True)], launches
            ),
            full_chain(score_calls=score_calls),
            prompt="task",
            resume=False,
        )
        # redirect relaunched (resumed) with the operator text; attempts=1
        # finished the loop without scoring
        assert launches == [(False, "task"), (True, "steer")]
        assert score_calls == []

    anyio.run(run)


def test_chain_error_retry_does_not_consume_an_attempt() -> None:
    async def run() -> None:
        score_calls: list[int] = []
        launches: list[tuple[bool, str]] = []
        await run_agent_loop(
            scripted_launch(
                [completed(exit_code=1), completed(any_tool_uses=True)], launches
            ),
            full_chain(
                attempts=AgentAttempts(attempts=2),
                score_calls=score_calls,
                score_values=(1.0,),
            ),
            prompt="task",
            resume=False,
        )
        # the failed launch was retried with the SAME prompt and did not
        # count as an attempt: the clean completion was attempt #1 and scored
        assert launches == [(False, "task"), (True, "task")]
        assert score_calls == [1]

    anyio.run(run)


def test_chain_nudge_relaunches_once_then_loop_ends() -> None:
    async def run() -> None:
        launches: list[tuple[bool, str]] = []
        await run_agent_loop(
            scripted_launch(
                [
                    Redirected("just checking in"),
                    completed(),  # conversational reply, no tools -> nudge
                    completed(),  # still no tools -> nudge spent, loop ends
                ],
                launches,
            ),
            full_chain(continue_prompt=" continue working "),
            prompt="task",
            resume=False,
        )
        assert launches == [
            (False, "task"),
            (True, "just checking in"),
            (True, "continue working"),
        ]

    anyio.run(run)
