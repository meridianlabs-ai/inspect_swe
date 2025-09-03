from typing import Literal

from tests.conftest import run_example, skip_if_no_anthropic, skip_if_no_docker


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_system_explorer() -> None:
    check_system_explorer_example("claude_code", "anthropic/claude-sonnet-4-0")


def check_system_explorer_example(
    agent: Literal["claude_code", "codex_cli"], model: str
) -> None:
    log = run_example("system_explorer", agent, model)[0]
    assert log.status == "success"
