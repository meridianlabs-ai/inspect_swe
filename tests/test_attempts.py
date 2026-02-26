from typing import Literal

import pytest
from inspect_ai.event import ScoreEvent

from tests.conftest import (
    get_available_sandboxes,
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_google,
    skip_if_no_openai,
)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_claude_code_attempts(sandbox: str) -> None:
    check_attempts("claude_code", "anthropic/claude-sonnet-4-0", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_attempts(sandbox: str) -> None:
    check_attempts("codex_cli", "openai/gpt-5", sandbox)


@skip_if_no_google
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_gemini_cli_attempts(sandbox: str) -> None:
    check_attempts("gemini_cli", "google/gemini-2.5-pro", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_mini_swe_attempts(sandbox: str) -> None:
    check_attempts("mini_swe_agent", "openai/gpt-5-mini", sandbox)


def check_attempts(
    agent: Literal["claude_code", "codex_cli", "gemini_cli", "mini_swe_agent"],
    model: str,
    sandbox: str,
) -> None:
    log = run_example("multiple_attempts", agent, model, sandbox=sandbox)[0]
    assert log.samples
    score_events = [
        event for event in log.samples[0].events if isinstance(event, ScoreEvent)
    ]
    assert len(score_events) == 2
