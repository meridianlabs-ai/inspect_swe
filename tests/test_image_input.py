import subprocess
from pathlib import Path
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
    assert _any_request_has_images(sample), (
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


def test_codex_cli_centaur_attaches_images_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The centaur `codex` command attaches staged images, on first use only.

    Regression tests, in branch-history order: image files were staged but
    never attached in centaur mode (silent drop, with the image-content
    warning suppressed because the content is declared handled); then a
    permanent alias re-attached the images on every invocation, including
    `codex resume`, duplicating images codex had already embedded in its
    rollout. Now a self-disarming shell function attaches them exactly once.
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
            image_files=[image_file],
            agent_env={},
            state=AgentState(messages=[]),
        )
    )

    bashrc = captured["bashrc"]

    # a self-disarming function, not a permanent alias
    assert "codex()" in bashrc
    assert "alias codex=" not in bashrc
    assert "/home/user/.codex/images/.attached" in bashrc

    # exactly one invocation attaches the images (the marker-guarded branch)
    assert bashrc.count(f"--image {image_file}") == 1

    # image args must be spliced after the binary, not appended: `--image` is
    # multi-value, so a command ending with it would swallow whatever the
    # human types next (`codex resume`, or a prompt) as another image path
    attach_line = next(line for line in bashrc.splitlines() if "--image" in line)
    assert attach_line.index("--image") < attach_line.index("--model")

    # the generated bashrc is valid bash
    rc_file = tmp_path / "bashrc"
    rc_file.write_text(bashrc)
    subprocess.run(["bash", "-n", str(rc_file)], check=True)

    # the human is told about the attached images
    assert image_file in captured["instructions"]
    assert "first invocation" in captured["instructions"]


def test_codex_cli_centaur_alias_without_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No images: the plain alias and instructions are unchanged."""
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
            image_files=[],
            agent_env={},
            state=AgentState(messages=[]),
        )
    )

    assert "alias codex=" in captured["bashrc"]
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


def _any_request_has_images(sample: EvalSample) -> bool:
    """Whether any raw model request contains an image content block."""
    sample = resolve_sample_attachments(sample, "full")

    def has_image(value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("type") == "input_image" or any(
                has_image(v) for v in value.values()
            )
        if isinstance(value, list):
            return any(has_image(v) for v in value)
        return False

    for event in sample.events:
        if getattr(event, "event", None) != "model":
            continue
        call = getattr(event, "call", None)
        if call is not None and has_image(call.request):
            return True
    return False
