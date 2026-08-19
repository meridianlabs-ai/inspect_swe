"""Fast unit tests for Claude Code model identity + bridge-alias resolution.

Uses the keyless ``mockllm`` provider so these run without Docker or API keys
(unlike ``tests/test_model_config_live.py``). Covers the per-role alias routing
and ``model_config`` override logic in ``resolve_claude_code_models``.
"""

from inspect_ai.model import Model, get_model
from inspect_swe._claude_code.model import resolve_claude_code_models


def test_defaults_present_served_model_and_share_one_alias() -> None:
    models = resolve_claude_code_models("mockllm/model", None)
    # presented defaults to the served model's name
    assert models.presented == "model"
    # every unset opus/sonnet/haiku role inherits the primary presented name
    assert models.opus == models.sonnet == models.haiku == "model"
    # subagent is the one exception: it never presents the same slug as the
    # primary, even when unset -- see test_subagent_default_gets_synthetic_name
    assert models.subagent == "model-subagent"
    # aliases: the presented name and the synthetic subagent name, both
    # routing to the served Model
    assert set(models.aliases) == {"model", "model-subagent"}
    assert isinstance(models.aliases["model"], Model)
    assert isinstance(models.aliases["model-subagent"], Model)
    # bridge sentinel preserves the inspect/<model> routing form
    assert models.bridge_model == "inspect/mockllm/model"


def test_subagent_default_gets_synthetic_name_aliased_to_served_model() -> None:
    """Default (unset subagent_model) gets a synthetic '<presented>-subagent' name.

    Routed to the exact same served model as the primary -- zero behavior
    change, only the presented label differs.
    """
    models = resolve_claude_code_models("mockllm/model", None)
    assert models.subagent == f"{models.presented}-subagent"
    assert models.subagent != models.presented
    assert models.aliases[models.subagent] is models.aliases[models.presented]


def test_subagent_explicit_distinct_model_unchanged() -> None:
    """Explicit subagent_model resolving to a distinct name is unchanged.

    Pre-existing behavior -- its own name, its own alias.
    """
    models = resolve_claude_code_models(
        "mockllm/model", None, subagent_model="mockllm/subagent"
    )
    assert models.subagent == "subagent"
    assert "subagent" in models.aliases
    assert models.aliases["subagent"] is not models.aliases["model"]


def test_subagent_explicit_same_as_main_gets_synthetic_name() -> None:
    """Degenerate case: caller explicitly points subagent_model at the main model.

    Distinctness is still enforced (synthetic suffix applied), routed to the
    caller's own resolved subagent model, and the primary's alias is left
    untouched.
    """
    # get_model is not cached (verified: two calls with the same spec return
    # distinct instances), so passing the already-resolved Model through as
    # `model` (get_model passes a Model argument through unchanged) is the
    # only way to assert the primary alias is untouched by *identity* rather
    # than merely "some non-None value" -- a stray `aliases[role.name] = role`
    # clobber (the bug this test guards against) would silently replace it
    # with a different, but equally non-None, Model instance.
    served = get_model("mockllm/model")
    models = resolve_claude_code_models(
        served,  # type: ignore[arg-type]
        None,
        subagent_model="mockllm/model",
    )
    assert models.subagent == f"{models.presented}-subagent"
    assert models.subagent != models.presented
    # primary alias untouched -- still the exact served Model instance, not
    # clobbered by the subagent block's own resolution of the same
    # underlying model string
    assert models.aliases[models.presented] is served
    # and the synthetic subagent alias routes to the caller's own resolved
    # subagent model (a distinct instance, since get_model isn't cached),
    # not to the primary's served model
    assert models.aliases[models.subagent] is not served


def test_subagent_explicit_same_as_haiku_gets_synthetic_name() -> None:
    """Degenerate case: caller points subagent_model at their haiku_model.

    The natural "cheap model for background AND subagents" config. Without
    the fix, `models.subagent == models.haiku != models.presented` would
    make classify()'s subagent branch shadow its utility branch for all
    small-fast traffic (utility becomes unreachable). Distinctness must be
    enforced against every other role name, not just `presented`.
    """
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        haiku_model="mockllm/haiku",
        subagent_model="mockllm/haiku",
    )
    assert models.haiku == "haiku"
    assert models.subagent == f"{models.presented}-subagent"
    assert models.subagent != models.haiku
    assert models.subagent != models.presented
    # routes to the caller's chosen (haiku) model, not the served primary
    assert models.aliases[models.subagent] is not models.aliases[models.presented]
    # and the haiku alias itself is undisturbed by the subagent collision
    assert models.aliases[models.haiku] is not None


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
    # ...while unset opus/sonnet/haiku peers still inherit the primary
    # presented name (subagent is the exception -- see the dedicated
    # subagent tests -- it never inherits, even when unset)
    assert models.sonnet == models.haiku == "model"
    assert models.subagent == "model-subagent"


def test_caller_model_aliases_take_precedence() -> None:
    # a caller alias on the same key as a derived name wins
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        model_aliases={"model": "mockllm/override"},
    )
    assert models.aliases["model"] == "mockllm/override"
