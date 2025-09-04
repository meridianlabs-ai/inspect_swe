from inspect_ai.model import ChatMessageAssistant, ContentToolUse

from tests.conftest import (
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_web_search() -> None:
    log = run_example("web_search", "claude_code", "anthropic/claude-sonnet-4-0")[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    assert next((tc for tc in tool_calls if tc.function == "WebSearch"), None)


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_web_search() -> None:
    log = run_example("web_search", "codex_cli", "openai/gpt-5")[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    assert next(
        (
            c
            for c in assistant_messages[0].content
            if isinstance(c, ContentToolUse) and c.tool_type == "web_search"
        ),
        None,
    )
