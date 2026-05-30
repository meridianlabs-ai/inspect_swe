"""End-to-end tests that Codex's prompt/tooling is aligned to the real model.

These run Codex CLI in a real sandbox and capture its first bridged request via
a ``GenerateFilter``. The bridge hands the filter Codex's chosen instructions as
the leading system message and the tool set Codex offered, so we can assert that
the catalog-driven mapping took effect:

- an OpenAI model maps to a native catalog entry -> ``apply_patch`` is offered;
- a non-OpenAI model falls back to Codex's generic prompt -> no ``apply_patch``.

Slow: requires Docker + a live model API (mirrors ``tests/test_mcp.py``).
"""

from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe import codex_cli

from tests.conftest import (
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)

# Dockerfile with the prerequisites for running an agent in a sandbox.
_DOCKERFILE = str(Path(__file__).parent.parent / "examples" / "mcp" / "Dockerfile")


class _CaptureFirstRequest:
    """A bridge ``GenerateFilter`` that records the first request and passes through."""

    def __init__(self) -> None:
        self.system_prompt: str | None = None
        self.tool_names: list[str] | None = None

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        if self.tool_names is None:
            self.tool_names = [t.name for t in tools]
            self.system_prompt = next(
                (m.text for m in messages if isinstance(m, ChatMessageSystem)), None
            )
        # passthrough: let the real model handle generation
        return None


def _run_codex(model: str) -> _CaptureFirstRequest:
    capture = _CaptureFirstRequest()
    task = Task(
        dataset=[Sample(input="Print the word DONE and then stop.")],
        solver=codex_cli(model=model, filter=capture),
        sandbox=("docker", _DOCKERFILE),
    )
    eval(task, model=model, limit=1)
    assert capture.tool_names is not None, "Codex made no bridged request"
    return capture


def _offers_apply_patch(tool_names: list[str]) -> bool:
    return any("apply_patch" in name for name in tool_names)


@skip_if_no_openai
@skip_if_no_docker
def test_codex_align_openai_gets_native_tools() -> None:
    capture = _run_codex("openai/gpt-5")
    assert capture.system_prompt, "expected a non-empty system prompt"
    assert _offers_apply_patch(capture.tool_names or []), (
        f"expected apply_patch for an OpenAI model, got tools: {capture.tool_names}"
    )


@skip_if_no_anthropic
@skip_if_no_docker
def test_codex_align_non_openai_uses_generic_fallback() -> None:
    capture = _run_codex("anthropic/claude-sonnet-4-0")
    assert capture.system_prompt, "expected a non-empty system prompt"
    assert not _offers_apply_patch(capture.tool_names or []), (
        f"expected no apply_patch for a non-OpenAI model, got tools: {capture.tool_names}"
    )
