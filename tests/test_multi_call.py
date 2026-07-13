from typing import Any, Literal

import pytest
from inspect_ai.log import EvalSample, resolve_sample_attachments
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

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
def test_claude_code_multi_call(sandbox: str) -> None:
    check_multi_call("claude_code", "anthropic/claude-sonnet-4-0", sandbox)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_claude_code_system_prompt_not_duplicated(sandbox: str) -> None:
    """Regression test: the system prompt must be sent exactly once per turn.

    The bridge captures Claude Code's own system prompt back into
    ``state.messages`` as a ``ChatMessageSystem``. A previous bug re-passed that
    captured prompt via ``--append-system-prompt`` on every resumed turn, so the
    entire system prompt was concatenated onto itself (2x on turn 2, 3x on turn
    3, ...). We verify against the *raw* Anthropic request payload
    (``ModelEvent.call.request``).
    """
    log = run_example(
        "multi_call", "claude_code", "anthropic/claude-sonnet-4-0", sandbox=sandbox
    )[0]
    assert log.samples
    sample = log.samples[0]

    counts = _env_block_counts(sample)
    assert counts, "expected at least one model call carrying the system prompt"
    # No single request may contain the Claude Code system prompt more than once.
    assert max(counts) == 1, (
        f"system prompt duplicated in a resumed-turn request: per-call "
        f"'# Environment' counts were {counts}"
    )


# Marker that appears exactly once in the Claude Code system prompt.
_ENV_MARKER = "# Environment"


def _flatten_system(value: Any) -> str:
    """Flatten an Anthropic ``system`` field (str, or list of text blocks) to text."""
    if isinstance(value, list):
        return " ".join(_flatten_system(block) for block in value)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def _env_block_counts(sample: EvalSample) -> list[int]:
    """Per-model-call occurrences of the system-prompt env block in the raw request."""
    sample = resolve_sample_attachments(sample, "full")

    counts: list[int] = []
    for event in sample.events:
        if getattr(event, "event", None) != "model":
            continue
        call = getattr(event, "call", None)
        if call is None:
            continue
        system = call.request.get("system")
        if system is None:
            continue
        n = _flatten_system(system).count(_ENV_MARKER)
        if n:
            counts.append(n)
    return counts


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_multi_call(sandbox: str) -> None:
    check_multi_call("codex_cli", "openai/gpt-5", sandbox)


@skip_if_no_google
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_gemini_cli_multi_call(sandbox: str) -> None:
    check_multi_call("gemini_cli", "google/gemini-3.1-pro-preview", sandbox)


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_mini_swe_agent_multi_call_openai(sandbox: str) -> None:
    check_multi_call("mini_swe_agent", "openai/gpt-5-mini", sandbox)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_mini_swe_agent_multi_call_anthropic(sandbox: str) -> None:
    check_multi_call("mini_swe_agent", "anthropic/claude-haiku-4-5", sandbox)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_opencode_multi_call(sandbox: str) -> None:
    check_multi_call("opencode", "anthropic/claude-sonnet-4-0", sandbox)


def check_multi_call(
    agent: Literal[
        "claude_code", "codex_cli", "gemini_cli", "mini_swe_agent", "opencode"
    ],
    model: str,
    sandbox: str,
) -> None:
    log = run_example("multi_call", agent, model, sandbox=sandbox)[0]
    assert log.samples
    sample = log.samples[0]

    user_messages = [m for m in sample.messages if isinstance(m, ChatMessageUser)]
    assistant_messages = [
        m for m in sample.messages if isinstance(m, ChatMessageAssistant)
    ]

    # Codex CLI includes extra scaffold messages in conversation history
    # Gemini CLI may use built-in tools (e.g. google_web_search), causing variable counts
    match agent:
        case "claude_code":
            assert len(user_messages) == 4
            assert len(assistant_messages) == 4
        case "codex_cli":
            assert len(user_messages) == 5
            assert len(assistant_messages) == 4
        case "gemini_cli":
            assert len(user_messages) >= 4
            assert len(assistant_messages) >= 4
        case "mini_swe_agent":
            # model may have more messages due to bash tool use(user/assistant share tool call and tool response)
            assert len(user_messages) >= 4
            assert len(assistant_messages) >= 4
        case "opencode":
            # opencode may emit extra scaffolding messages and tool turns
            assert len(user_messages) >= 4
            assert len(assistant_messages) >= 4
