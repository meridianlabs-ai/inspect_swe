"""End-to-end test that auto_review engages Codex's guardian reviewer.

Runs Codex CLI with ``auto_review=True`` in a real sandbox and captures all
bridged requests via a ``GenerateFilter``. The task requires network access,
which Codex's own sandbox does not permit without escalation (and in
containers without a working sandbox launcher the command fails outright),
so Codex requests escalated permissions — forcing an approval that the
guardian must review. We expect at least one bridged request whose system
prompt is the guardian review prompt (identified by marker phrases).

Slow: requires Docker + a live model API (mirrors ``tests/test_codex_align.py``).
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

from tests.conftest import skip_if_no_docker, skip_if_no_openai

# Dockerfile with the prerequisites for running an agent in a sandbox.
_DOCKERFILE = str(Path(__file__).parent.parent / "examples" / "mcp" / "Dockerfile")

# Phrases expected in the guardian review prompt and absent from Codex's main
# system prompt. These are the field names of the JSON verdict schema the
# guardian is asked to emit ("judging one planned coding-agent action" ...
# derive `outcome` from tenant policy, `risk_level`, and `user_authorization`),
# so they should survive upstream prompt rewording; update from captured
# output if they don't.
_GUARDIAN_MARKERS = ["risk_level", "user_authorization"]


class _CaptureSystemPrompts:
    """Bridge ``GenerateFilter`` that records every request's system prompt."""

    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        self.system_prompts.append(
            "\n".join(m.text for m in messages if isinstance(m, ChatMessageSystem))
        )
        # passthrough: let the real model handle generation
        return None


def _is_guardian_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return all(marker in lowered for marker in _GUARDIAN_MARKERS)


@skip_if_no_docker
@skip_if_no_openai
def test_auto_review_triggers_guardian_review() -> None:
    capture = _CaptureSystemPrompts()
    task = Task(
        dataset=[
            Sample(
                input="Fetch https://example.com with curl and print the HTTP "
                "status code. If you need approval for network access, request it."
            )
        ],
        solver=codex_cli(auto_review=True, version="latest", filter=capture),
        sandbox=("docker", _DOCKERFILE),
    )
    eval(task, model="openai/gpt-5.5", limit=1)
    assert capture.system_prompts, "Codex made no bridged requests"
    assert any(_is_guardian_prompt(p) for p in capture.system_prompts), (
        "no guardian review request observed; captured system prompts:\n"
        + "\n---\n".join(p[:500] for p in capture.system_prompts)
    )
