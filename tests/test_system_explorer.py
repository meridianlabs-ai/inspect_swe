from typing import Literal

import pytest

from tests.conftest import (
    get_available_sandboxes,
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_claude_code_system_explorer(sandbox: str) -> None:
    check_system_explorer_example("claude_code", "anthropic/claude-sonnet-4-0", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_system_explorer(sandbox: str) -> None:
    check_system_explorer_example("codex_cli", "openai/gpt-5", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_mini_swe_agent_system_explorer(sandbox: str) -> None:
    check_system_explorer_example("mini_swe_agent", "openai/gpt-5-mini", sandbox)


def check_system_explorer_example(
    agent: Literal["claude_code", "codex_cli", "mini_swe_agent"],
    model: str,
    sandbox: str | None = None,
) -> None:
    log = run_example("system_explorer", agent, model, sandbox=sandbox)[0]
    assert log.status == "success"
