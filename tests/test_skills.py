from typing import Literal
from tests.conftest import (
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_google,
    skip_if_no_openai,
)


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_skills() -> None:
    check_skills("claude_code", "anthropic/claude-sonnet-4-5")


@skip_if_no_openai
@skip_if_no_docker
def test_codex_cli_skills() -> None:
    check_skills("codex_cli", "openai/gpt-5.1-codex")


@skip_if_no_google
@skip_if_no_docker
def test_gemini_cli_skills() -> None:
    check_skills("gemini_cli", "google/gemini-2.5-pro")


def check_skills(
    agent: Literal["claude_code", "codex_cli", "gemini_cli"], model: str
) -> None:
    log = run_example("skills", agent, model)[0]
    assert log.status == "success"
    assert log.samples

    # Find all assistant messages and tool outputs to check content
    all_content = []
    sample = log.samples[0]
    messages = sample.messages
    for msg in messages:
        if hasattr(msg, "content"):
            if isinstance(msg.content, str):
                all_content.append(msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if hasattr(item, "text"):
                        all_content.append(item.text)

    combined_output = " ".join(all_content)

    # Verify the model read the asset (ALPHA-BRAVO-CHARLIE should appear)
    assert "ALPHA-BRAVO-CHARLIE" in combined_output, (
        "Agent did not read the asset file. "
        f"Expected 'ALPHA-BRAVO-CHARLIE' in output but got: {combined_output[:500]}..."
    )

    # Verify the model ran the script (DELTA-ECHO-FOXTROT should appear)
    assert "DELTA-ECHO-FOXTROT" in combined_output, (
        "Agent did not run the script. "
        f"Expected 'DELTA-ECHO-FOXTROT' in output but got: {combined_output[:500]}..."
    )
