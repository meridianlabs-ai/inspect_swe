"""Fast unit tests for Claude Code model identity + bridge-alias resolution.

Uses the keyless ``mockllm`` provider so these run without Docker or API keys
(unlike ``tests/test_model_config_live.py``). Covers the per-role alias routing
and ``model_config`` override logic in ``resolve_claude_code_models``.
"""

from typing import Any

import pytest
from inspect_ai.agent._bridge.util import resolve_inspect_model
from inspect_ai.model import Model, get_model
from inspect_swe._claude_code.model import (
    distinct_subagent_name,
    resolve_claude_code_models,
)


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


@pytest.mark.parametrize("role", ["opus_model", "sonnet_model", "haiku_model"])
def test_synthetic_subagent_name_avoids_role_collision(role: str) -> None:
    """The synthetic '<presented>-subagent' name is itself checked for collisions.

    A caller whose opus/sonnet/haiku model happens to resolve to
    '<presented>-subagent' must keep that role's name and alias intact; the
    subagent role takes the next free suffix instead of clobbering it.
    """
    kwargs: dict[str, Any] = {role: "mockllm/model-subagent"}
    models = resolve_claude_code_models("mockllm/model", None, **kwargs)
    role_name = getattr(models, role.removesuffix("_model"))
    assert role_name == "model-subagent"
    assert models.subagent == "model-subagent-2"
    assert models.subagent not in {
        models.presented,
        models.opus,
        models.sonnet,
        models.haiku,
    }
    # the colliding role's alias still routes to the caller's model, not the
    # served primary that the default subagent route would have installed
    assert models.aliases[role_name] is not models.aliases[models.presented]
    assert models.aliases[models.subagent] is models.aliases[models.presented]


def test_synthetic_subagent_name_skips_every_taken_suffix() -> None:
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        opus_model="mockllm/model-subagent",
        sonnet_model="mockllm/model-subagent-2",
    )
    assert models.subagent == "model-subagent-3"
    assert models.aliases[models.opus] is not models.aliases[models.presented]
    assert models.aliases[models.sonnet] is not models.aliases[models.presented]


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


def test_transparent_proxy_presented_identity_resolves_via_alias() -> None:
    # reproduces the review finding on #100: claude_code's presented identity
    # is a bare, unprefixed name (e.g. "claude-sonnet-4-5", or a real model's
    # own bare name), which can't resolve as a raw model name -- get_model()
    # requires "<api_name>/<model_name>". claude_code()'s execute() keeps the
    # presented-identity alias table under transparent_proxy=True precisely
    # so this, the main-line request, still resolves to the real served
    # model instead of raising.
    models = resolve_claude_code_models("mockllm/model", "claude-sonnet-4-5")
    resolved = resolve_inspect_model(models.presented, models.aliases, None)
    assert resolved.name == "model"


def test_unset_subagent_follows_caller_alias_for_presented_name() -> None:
    """A caller alias on the presented name reroutes the default subagent too.

    With ``subagent_model`` unset the synthetic ``"<presented>-subagent"``
    slug is only a *label* for the same served model as the primary, so it
    must route wherever the presented name routes -- including after a
    caller's ``model_aliases`` override of that name. Otherwise main-thread
    traffic would go to the override while Task sub-agent traffic silently
    stayed on the un-overridden served model.
    """
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        model_aliases={"model": "mockllm/override"},
    )
    assert models.subagent == "model-subagent"
    assert models.aliases["model"] == "mockllm/override"
    assert models.aliases[models.subagent] is models.aliases[models.presented]


def test_explicit_subagent_model_unaffected_by_presented_alias() -> None:
    """An explicit subagent_model is the caller's choice; a presented override leaves it alone."""
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        subagent_model="mockllm/sub",
        model_aliases={"model": "mockllm/override"},
    )
    assert models.aliases["model"] == "mockllm/override"
    sub = models.aliases[models.subagent]
    assert isinstance(sub, Model)
    assert sub.name == "sub"


def test_explicit_colliding_subagent_model_unaffected_by_presented_alias() -> None:
    """Explicit subagent_model that collides with the primary keeps its own route.

    The synthetic slug is applied for distinctness, but its target is the
    caller's resolved subagent model -- not the primary's (overridden) route.
    """
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        subagent_model="mockllm/model",
        model_aliases={"model": "mockllm/override"},
    )
    assert models.subagent == "model-subagent"
    assert models.aliases["model"] == "mockllm/override"
    sub = models.aliases[models.subagent]
    assert isinstance(sub, Model)
    assert sub.name == "model"


def test_caller_alias_for_synthetic_subagent_slug_wins() -> None:
    """A caller who aliases the synthetic slug themselves gets exactly that."""
    models = resolve_claude_code_models(
        "mockllm/model",
        None,
        model_aliases={
            "model": "mockllm/override",
            "model-subagent": "mockllm/sub-override",
        },
    )
    assert models.aliases["model"] == "mockllm/override"
    assert models.aliases["model-subagent"] == "mockllm/sub-override"


def test_distinct_subagent_name_collision_free() -> None:
    assert distinct_subagent_name("model", {"model"}) == "model-subagent"
    assert distinct_subagent_name("model", set()) == "model-subagent"


def test_distinct_subagent_name_advances_past_taken_suffixes() -> None:
    assert (
        distinct_subagent_name("model", {"model", "model-subagent"})
        == "model-subagent-2"
    )
    assert (
        distinct_subagent_name("model", {"model", "model-subagent", "model-subagent-2"})
        == "model-subagent-3"
    )
    # a gap in the taken suffixes is filled, not skipped past
    assert (
        distinct_subagent_name("model", {"model-subagent", "model-subagent-3"})
        == "model-subagent-2"
    )
