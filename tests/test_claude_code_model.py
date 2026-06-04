"""Fast unit tests for Claude Code model identity + bridge-alias resolution.

Uses the keyless ``mockllm`` provider so these run without Docker or API keys
(unlike ``tests/test_model_config_live.py``). Covers the per-role alias routing
and ``model_config`` override logic in ``resolve_claude_code_models``.
"""

from inspect_ai.model import Model
from inspect_swe._claude_code.model import resolve_claude_code_models


def test_defaults_present_served_model_and_share_one_alias() -> None:
    models = resolve_claude_code_models("mockllm/model", None)
    # presented defaults to the served model's name
    assert models.presented == "model"
    # every unset role inherits the primary presented name
    assert models.opus == models.sonnet == models.haiku == models.subagent == "model"
    # a single alias maps the presented name to the served Model
    assert set(models.aliases) == {"model"}
    assert isinstance(models.aliases["model"], Model)
    # bridge sentinel preserves the inspect/<model> routing form
    assert models.bridge_model == "inspect/mockllm/model"


def test_model_config_overrides_presented_identity() -> None:
    models = resolve_claude_code_models("mockllm/model", "claude-sonnet-4-5")
    assert models.presented == "claude-sonnet-4-5"
    # the override name is what routes to the served model
    assert "claude-sonnet-4-5" in models.aliases
    # routing target (the real served model) is unchanged
    assert models.bridge_model == "inspect/mockllm/model"


def test_set_role_gets_own_name_and_alias_unset_roles_inherit() -> None:
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        opus_model="mockllm/opus",
    )
    # the set role routes to its own model via its own name + alias...
    assert models.opus == "opus"
    assert "opus" in models.aliases
    # ...while unset roles still inherit the primary presented name
    assert models.sonnet == models.haiku == models.subagent == "model"


def test_caller_model_aliases_take_precedence() -> None:
    # a caller alias on the same key as a derived name wins
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        model_aliases={"model": "mockllm/override"},
    )
    assert models.aliases["model"] == "mockllm/override"
