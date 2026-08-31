from typing import Any

import anyio
import pytest
from inspect_ai import eval
from inspect_ai.agent import AgentState
from inspect_ai.log import EvalSample, resolve_sample_attachments
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_swe._codex_cli import codex_cli as codex_cli_module
from inspect_swe._util.centaur import CentaurOptions

from tests.conftest import (
    get_available_sandboxes,
    skip_if_no_docker,
    skip_if_no_openai,
)

# color names a model might reasonably use for the two solid-color images in
# examples/image_input (magenta = rgb(255,0,255), green = rgb(0,200,0))
MAGENTA_NAMES = ["magenta", "fuchsia", "pink", "purple", "violet"]
GREEN_NAMES = ["green"]


@skip_if_no_openai
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_codex_cli_image_input(sandbox: str) -> None:
    """Images in the input reach codex (via `codex exec --image`).

    Regression test: image content in sample input was silently dropped
    (`build_user_prompt` keeps only text). Verifies both the initial prompt
    and a follow-up message after an assistant response (`exec resume`).
    """
    log = eval(
        "examples/image_input",
        model="openai/gpt-5",
        limit=1,
        task_args={"sandbox": sandbox},
        time_limit=600,
        token_limit=500_000,
    )[0]
    assert log.status == "success", f"eval failed: {log.error}"
    assert log.samples
    sample = log.samples[0]

    # mechanism: the raw model requests actually carried image content
    assert _image_content_counts(sample), (
        "no model request contained image content: images did not reach codex"
    )

    # behavior: the model identified the color of each image
    turn_1_answer, turn_2_answer = _turn_answers(sample)
    assert any(color in turn_1_answer.lower() for color in MAGENTA_NAMES), (
        f"first answer did not identify the magenta image: {turn_1_answer!r}"
    )
    assert any(color in turn_2_answer.lower() for color in GREEN_NAMES), (
        f"second answer did not identify the green image: {turn_2_answer!r}"
    )


def test_codex_cli_centaur_alias_attaches_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The centaur `codex` alias carries the staged `--image` args.

    Regression test: image args were only consumed by the headless exec
    branch, so in centaur mode images were staged as files but the alias
    never attached them -- a silent drop (the image-content warning is
    suppressed because the content is declared handled).
    """
    captured: dict[str, str] = {}

    async def fake_run_centaur(
        options: CentaurOptions, instructions: str, bashrc: str, state: AgentState
    ) -> None:
        captured["instructions"] = instructions
        captured["bashrc"] = bashrc

    monkeypatch.setattr(codex_cli_module, "run_centaur", fake_run_centaur)

    image_file = "/home/user/.codex/images/image-0.png"
    anyio.run(
        lambda: codex_cli_module._run_codex_cli_centaur(
            options=CentaurOptions(),
            codex_cmd=["/usr/bin/codex", "--model", "gpt-5", "-c", "key=value"],
            image_args=["--image", image_file],
            agent_env={},
            state=AgentState(messages=[]),
        )
    )

    alias = next(
        line
        for line in captured["bashrc"].splitlines()
        if line.startswith("alias codex=")
    )
    assert f"--image {image_file}" in alias

    # image args must be spliced after the binary, not appended: `--image` is
    # multi-value, so an alias ending with it would swallow whatever the human
    # types next (`codex resume`, or a prompt) as another image path
    assert alias.index("--image") < alias.index("--model")

    # the human is told about the attached images
    assert image_file in captured["instructions"]


def test_codex_cli_centaur_alias_without_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No image args: the alias and instructions are unchanged."""
    captured: dict[str, str] = {}

    async def fake_run_centaur(
        options: CentaurOptions, instructions: str, bashrc: str, state: AgentState
    ) -> None:
        captured["instructions"] = instructions
        captured["bashrc"] = bashrc

    monkeypatch.setattr(codex_cli_module, "run_centaur", fake_run_centaur)

    anyio.run(
        lambda: codex_cli_module._run_codex_cli_centaur(
            options=CentaurOptions(),
            codex_cmd=["/usr/bin/codex", "--model", "gpt-5"],
            image_args=[],
            agent_env={},
            state=AgentState(messages=[]),
        )
    )

    assert "--image" not in captured["bashrc"]
    assert "image" not in captured["instructions"].lower()


def _turn_answers(sample: EvalSample) -> tuple[str, str]:
    """Assistant text for turn 1 (before the second user image) and turn 2."""
    second_user_idx = [
        i
        for i, m in enumerate(sample.messages)
        if isinstance(m, ChatMessageUser) and "second image" in m.text
    ][0]
    turn_1 = " ".join(
        m.text
        for m in sample.messages[:second_user_idx]
        if isinstance(m, ChatMessageAssistant)
    )
    turn_2 = " ".join(
        m.text
        for m in sample.messages[second_user_idx:]
        if isinstance(m, ChatMessageAssistant)
    )
    return turn_1, turn_2


def _image_content_counts(sample: EvalSample) -> list[int]:
    """Per-model-call count of image content blocks in the raw request."""
    sample = resolve_sample_attachments(sample, "full")

    def count_images(value: Any) -> int:
        if isinstance(value, dict):
            n = 1 if value.get("type") == "input_image" else 0
            return n + sum(count_images(v) for v in value.values())
        if isinstance(value, list):
            return sum(count_images(v) for v in value)
        return 0

    counts: list[int] = []
    for event in sample.events:
        if getattr(event, "event", None) != "model":
            continue
        call = getattr(event, "call", None)
        if call is None:
            continue
        n = count_images(call.request)
        if n:
            counts.append(n)
    return counts
