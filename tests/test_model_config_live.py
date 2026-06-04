"""End-to-end tests that the inner agent presents the genuine served-model identity.

Runs Claude Code in a real sandbox and captures its first bridged request via a
``GenerateFilter``. The bridge hands the filter Claude Code's system prompt (which
contains the "# Environment ... You are powered by the model X" block) plus the
resolved Inspect ``Model`` — so we can assert what the *underlying model* sees,
the eval-awareness surface this change targets.

Claude Code builds the identity block from an internal catalog: a recognized
Anthropic id renders the genuine "You are powered by the model named <Friendly>.
The exact model ID is <id>." (plus a knowledge-cutoff line), canonicalizing
dated / ``[1m]`` / provider-prefixed variants via substring match. An
unrecognized id (cross-provider, or a Claude model newer than the installed
Claude Code) honestly degrades to "You are powered by the model <id>." with no
fabricated Claude identity. So we assert:

- a native Anthropic model yields the *genuine* block — NOT the bridge sentinel
  ``inspect`` (the regression this change fixes) and NOT a degraded raw echo;
- a dated Anthropic id still yields the genuine friendly name — Claude Code
  normalizes internally, so no normalization is needed on our side;
- a cross-provider model honestly degrades (no fabricated Claude identity);
- ``model_config`` overrides the presented identity without changing routing;
- bridged routing still resolves to the served model in every case.

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


def _run_claude(model: str, model_config: str | None = None) -> _CaptureDisplay:
    capture = _CaptureDisplay()
    task = Task(
        dataset=[Sample(input="Print the word DONE and then stop.")],
        solver=claude_code(model=model, model_config=model_config, filter=capture),
        sandbox=("docker", _DOCKERFILE),
    )
    eval(task, model=model, limit=1)
    assert capture.system_prompt is not None, "Claude Code made no bridged request"
    return capture


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_presents_genuine_model_native() -> None:
    capture = _run_claude("anthropic/claude-sonnet-4-5")
    sp = capture.system_prompt or ""
    # Recognized model → genuine catalog identity (friendly name + exact id),
    # NOT the bridge sentinel and NOT a degraded raw echo.
    assert "powered by the model named Sonnet 4.5" in sp, sp
    assert "exact model ID is claude-sonnet-4-5" in sp, sp
    assert "powered by the model inspect" not in sp
    # bridged routing still resolves to the served model
    assert capture.resolved_model == "anthropic/claude-sonnet-4-5"


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_normalizes_dated_anthropic_id() -> None:
    # A dated Anthropic id is canonicalized by Claude Code internally (substring
    # match), so the genuine friendly name still appears with no normalization on
    # our side; the exact-id line echoes the real dated snapshot.
    capture = _run_claude("anthropic/claude-sonnet-4-5-20250929")
    sp = capture.system_prompt or ""
    assert "powered by the model named Sonnet 4.5" in sp, sp
    assert "exact model ID is claude-sonnet-4-5-20250929" in sp, sp
    assert capture.resolved_model == "anthropic/claude-sonnet-4-5-20250929"


@skip_if_no_openai
@skip_if_no_docker
def test_claude_code_cross_provider_honest_degrade() -> None:
    # A non-Anthropic served model isn't in Claude Code's catalog, so it honestly
    # degrades to the raw-id block (no fabricated Claude identity) and still routes
    # to the served model via the bridge.
    capture = _run_claude("openai/gpt-5")
    sp = capture.system_prompt or ""
    assert "powered by the model gpt-5" in sp, sp
    assert capture.resolved_model == "openai/gpt-5"


@skip_if_no_openai
@skip_if_no_docker
def test_claude_code_model_config_override() -> None:
    # model_config overrides the presented identity without changing routing:
    # serve gpt-5 but present (genuinely) as Claude Sonnet 4.5.
    capture = _run_claude("openai/gpt-5", model_config="claude-sonnet-4-5")
    sp = capture.system_prompt or ""
    assert "powered by the model named Sonnet 4.5" in sp, sp
    # routing still goes to the real served model, not the presented identity
    assert capture.resolved_model == "openai/gpt-5"
