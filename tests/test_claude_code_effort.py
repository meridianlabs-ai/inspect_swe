"""``effort`` is applied host-side, on the served model's ``GenerateConfig``.

Passing it as a Claude Code CLI flag has no effect on the model actually
serving the request: the bridge drops the inner agent's request-level
generation config by default (``sandbox_agent_bridge``'s
``forward_generation_config`` defaults to ``False``), and only the resolved
Inspect model's own config governs. See ``resolve_claude_code_models``.
"""

from inspect_ai.model import Model
from inspect_swe._claude_code.model import resolve_claude_code_models


def test_effort_sets_reasoning_effort_on_served_model() -> None:
    models = resolve_claude_code_models("mockllm/model", None, effort="max")
    served_model = models.aliases[models.presented]
    assert isinstance(served_model, Model)
    assert served_model.config.reasoning_effort == "max"


def test_unconfigured_effort_leaves_served_model_config_untouched() -> None:
    models = resolve_claude_code_models("mockllm/model", None, effort=None)
    served_model = models.aliases[models.presented]
    assert isinstance(served_model, Model)
    assert served_model.config.reasoning_effort is None


def test_effort_applies_to_every_role_this_function_resolves() -> None:
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        effort="low",
        opus_model="mockllm/opus",
    )
    presented_model = models.aliases[models.presented]
    opus_model = models.aliases[models.opus]
    assert isinstance(presented_model, Model)
    assert isinstance(opus_model, Model)
    assert presented_model.config.reasoning_effort == "low"
    assert opus_model.config.reasoning_effort == "low"


def test_effort_does_not_override_caller_supplied_model_aliases() -> None:
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        effort="high",
        model_aliases={"model": "mockllm/override"},
    )
    # the caller-supplied alias is a plain model spec string, untouched by effort
    assert models.aliases["model"] == "mockllm/override"


def test_effort_applies_to_an_explicit_subagent_model() -> None:
    """The subagent role gets `effort` like every other role it resolves."""
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        effort="high",
        subagent_model="mockllm/sub",
    )
    subagent_model = models.aliases[models.subagent]
    assert isinstance(subagent_model, Model)
    assert subagent_model.config.reasoning_effort == "high"


def test_unset_subagent_follows_caller_alias_untouched_by_effort() -> None:
    """The default subagent follows the presented name's caller alias, effort and all.

    The presented alias carries the effort-merged served model, but a caller
    override replaces it and effort is not applied to caller-supplied
    aliases -- the synthetic subagent slug follows that same (un-efforted)
    override rather than keeping the effort-merged served model.
    """
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        effort="high",
        model_aliases={"model": "mockllm/override"},
    )
    assert models.aliases[models.subagent] == "mockllm/override"
    assert models.aliases[models.subagent] is models.aliases[models.presented]
