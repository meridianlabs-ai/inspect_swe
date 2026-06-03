"""End-to-end tests that the inner agent presents the real served model.

Runs Claude Code in a real sandbox and captures its first bridged request via a
``GenerateFilter``. The bridge hands the filter Claude Code's system prompt
(which contains the "# Environment ... You are powered by the model X" line) and
the resolved Inspect ``Model``, so we can assert:

- the displayed model is the *real* served model, NOT the bridge sentinel
  ``inspect`` (the regression this change fixes);
- bridged routing still resolves to the served model;
- it works even cross-provider (Claude Code accepts an arbitrary model string,
  so serving an OpenAI model presents ``openai/gpt-5`` and routes there).

Slow: requires Docker + a live model API (mirrors ``tests/test_codex_align.py``).
"""

from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    Model,
    ModelOutput,
)
from inspect_swe import claude_code

from tests.conftest import (
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)

_DOCKERFILE = str(Path(__file__).parent.parent / "examples" / "mcp" / "Dockerfile")


class _CaptureDisplay:
    """Record the first bridged request's prompt + resolved model, then stop."""

    def __init__(self) -> None:
        self.system_prompt: str | None = None
        self.resolved_model: str | None = None

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        *_: object,  # bridge also passes tools, tool_choice, config
    ) -> ModelOutput:
        if self.system_prompt is None:
            self.system_prompt = next(
                (m.text for m in messages if isinstance(m, ChatMessageSystem)), ""
            )
            self.resolved_model = model.canonical_name()
        return ModelOutput.from_content(
            model=str(model), content="DONE", stop_reason="stop"
        )


def _run_claude(model: str) -> _CaptureDisplay:
    capture = _CaptureDisplay()
    task = Task(
        dataset=[Sample(input="Print the word DONE and then stop.")],
        solver=claude_code(model=model, filter=capture),
        sandbox=("docker", _DOCKERFILE),
    )
    eval(task, model=model, limit=1)
    assert capture.system_prompt is not None, "Claude Code made no bridged request"
    return capture


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_presents_real_model_native() -> None:
    capture = _run_claude("anthropic/claude-sonnet-4-5")
    sp = capture.system_prompt or ""
    assert "powered by the model claude-sonnet-4-5" in sp, (
        f"expected the real model name in the system prompt, got:\n{sp}"
    )
    # bridged routing still resolves to the served model
    assert capture.resolved_model == "anthropic/claude-sonnet-4-5"


@skip_if_no_openai
@skip_if_no_docker
def test_claude_code_presents_real_model_cross_provider() -> None:
    # Claude Code accepts an arbitrary --model string, so serving an OpenAI model
    # presents the real served model AND routes to it (the bridge handles the
    # protocol). Proves the default is robust cross-provider without a fallback.
    capture = _run_claude("openai/gpt-5")
    sp = capture.system_prompt or ""
    assert "gpt-5" in sp, (
        f"expected the real served model in the system prompt, got:\n{sp}"
    )
    assert capture.resolved_model == "openai/gpt-5"
