from typing import Literal

import pytest

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
def test_claude_code_system_explorer(sandbox: str) -> None:
    check_system_explorer_example("claude_code", "anthropic/claude-sonnet-4-5", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_system_explorer(sandbox: str) -> None:
    check_system_explorer_example("codex_cli", "openai/gpt-5.4", sandbox)


@skip_if_no_google
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_gemini_cli_system_explorer(sandbox: str) -> None:
    check_system_explorer_example(
        "gemini_cli", "google/gemini-3.1-pro-preview", sandbox
    )


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_mini_swe_agent_system_explorer(sandbox: str) -> None:
    check_system_explorer_example("mini_swe_agent", "openai/gpt-5-mini", sandbox)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_opencode_system_explorer(sandbox: str) -> None:
    check_system_explorer_example("opencode", "anthropic/claude-sonnet-4-5", sandbox)


def check_system_explorer_example(
    agent: Literal[
        "claude_code", "codex_cli", "gemini_cli", "mini_swe_agent", "opencode"
    ],
    model: str,
    sandbox: str | None = None,
) -> None:
    log = run_example("system_explorer", agent, model, sandbox=sandbox)[0]
    # run_example already fails the test on a task-level failure, an errored
    # sample, a sample cut short by a limit, or a sample with no agent turn --
    # which is everything the old `assert log.status == "success"` here covered
    # and more. What it cannot know is that this example, unlike the others,
    # defines a scorer (model_graded_qa), so a run that got all the way through
    # must carry a score. Assert that, and not the grade itself: the examples
    # exist to exercise the agents, not to measure how well they explore.
    assert log.samples and log.samples[0].scores, (
        "sample completed but was never scored"
    )
