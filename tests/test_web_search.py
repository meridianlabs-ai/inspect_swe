import pytest
from inspect_ai.model import ChatMessageAssistant, ContentToolUse

from tests.conftest import (
    get_available_sandboxes,
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_google,
    skip_if_no_openai,
)


@skip_if_no_docker
@skip_if_no_anthropic
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_claude_code_web_search(sandbox: str) -> None:
    log = run_example(
        "web_search", "claude_code", "anthropic/claude-sonnet-4-0", sandbox=sandbox
    )[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    assert next((tc for tc in tool_calls if tc.function == "WebSearch"), None)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_web_search(sandbox: str) -> None:
    log = run_example("web_search", "codex_cli", "openai/gpt-5", sandbox=sandbox)[0]
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


@skip_if_no_google
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_gemini_cli_web_search(sandbox: str) -> None:
    log = run_example(
        "web_search", "gemini_cli", "google/gemini-2.5-pro", sandbox=sandbox
    )[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    # Check for web search tool usage (adjust based on Gemini CLI's tool naming)
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    assert next((tc for tc in tool_calls if "search" in tc.function.lower()), None)
