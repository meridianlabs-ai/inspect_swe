from tests.conftest import run_example, skip_if_no_anthropic, skip_if_no_docker


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_anthropic() -> None:
    log = run_example("system_explorer", "anthropic/claude-sonnet-4-0")[0]
    assert log.status == "success"
