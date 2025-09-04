from typing import Literal

from inspect_ai.log import ScoreEvent

from tests.conftest import (
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_attempts() -> None:
    check_attempts("claude_code", "anthropic/claude-sonnet-4-0")


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_attempts() -> None:
    check_attempts("codex_cli", "openai/gpt-5")


def check_attempts(agent: Literal["claude_code", "codex_cli"], model: str) -> None:
    log = run_example("multiple_attempts", agent, model)[0]
    assert log.samples
    score_events = [
        event for event in log.samples[0].events if isinstance(event, ScoreEvent)
    ]
    assert len(score_events) == 2
