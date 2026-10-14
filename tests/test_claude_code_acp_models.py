"""Unit tests for Claude Code (ACP) model-name + bridge-alias resolution.

Keyless ``mockllm`` provider, no sandbox. `resolve_claude_code_acp_models`
mirrors the native `resolve_claude_code_models`; the classification-facing
subagent-distinctness cases live in ``tests/test_agent_context_gemini.py``.
These cover how a caller's ``model_map`` override interacts with the
synthetic ``"<presented>-subagent"`` slug.
"""

import pytest
from inspect_ai.model import get_model
from inspect_swe.acp._agents.claude_code.claude_code import (
    ClaudeCode,
    resolve_claude_code_acp_models,
)


def test_unset_subagent_follows_model_map_override_of_presented_name() -> None:
    """A ``model_map`` override of the presented name reroutes the default subagent too.

    `ACPAgent` layers the caller's ``model_map`` over the resolved aliases;
    with ``subagent_model`` unset the synthetic slug is only a label for the
    primary's route, so it must follow the presented name's override rather
    than silently staying on the un-overridden served model.
    """
    models = resolve_claude_code_acp_models(
        "mockllm/model", model_map={"model": "mockllm/override"}
    )
    assert models.subagent == "model-subagent"
    assert models.aliases[models.subagent].canonical_name() == "override"


def test_unset_subagent_follows_model_map_model_instance_as_is() -> None:
    override = get_model("mockllm/override")
    models = resolve_claude_code_acp_models(
        "mockllm/model", model_map={"model": override}
    )
    assert models.aliases[models.subagent] is override


def test_unset_subagent_without_override_routes_to_served_model() -> None:
    models = resolve_claude_code_acp_models(
        "mockllm/model", model_map={"unrelated": "mockllm/other"}
    )
    assert models.aliases[models.subagent] is models.aliases[models.presented]


def test_explicit_subagent_model_unaffected_by_presented_override() -> None:
    models = resolve_claude_code_acp_models(
        "mockllm/model",
        subagent_model="mockllm/sub",
        model_map={"model": "mockllm/override"},
    )
    assert models.subagent == "sub"
    assert models.aliases[models.subagent].canonical_name() == "sub"


def test_explicit_colliding_subagent_model_unaffected_by_presented_override() -> None:
    sub = get_model("mockllm/model")
    models = resolve_claude_code_acp_models(
        "mockllm/model",
        subagent_model=sub,
        model_map={"model": "mockllm/override"},
    )
    assert models.subagent == "model-subagent"
    assert models.aliases[models.subagent] is sub


def test_caller_override_of_synthetic_slug_is_left_to_the_caller() -> None:
    """A caller who maps the synthetic slug themselves wins: the resolver doesn't redirect it.

    The resolver leaves the derived alias in place; `ACPAgent` then applies
    the caller's mapping for that slug on top (see the agent-level test).
    """
    models = resolve_claude_code_acp_models(
        "mockllm/model",
        model_map={
            "model": "mockllm/override",
            "model-subagent": "mockllm/sub-override",
        },
    )
    assert models.aliases[models.subagent] is models.aliases[models.presented]


def test_agent_model_map_reroutes_unset_subagent_with_presented_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through `ClaudeCode` (what `interactive_claude_code` builds): the final ``model_map`` the bridge sees."""
    import inspect_swe.acp.agent as acp_agent_mod

    # ACPAgent requires an active sample to construct; only the model_map
    # wiring is under test, so stub the unrelated guard
    monkeypatch.setattr(acp_agent_mod, "sample_active", lambda: object())

    agent = ClaudeCode(model="mockllm/model", model_map={"model": "mockllm/override"})
    assert agent.model_map["model"] == "mockllm/override"
    sub = agent.model_map["model-subagent"]
    assert not isinstance(sub, str)
    assert sub.canonical_name() == "override"

    agent = ClaudeCode(
        model="mockllm/model",
        model_map={
            "model": "mockllm/override",
            "model-subagent": "mockllm/sub-override",
        },
    )
    assert agent.model_map["model-subagent"] == "mockllm/sub-override"

    agent = ClaudeCode(
        model="mockllm/model",
        subagent_model="mockllm/sub",
        model_map={"model": "mockllm/override"},
    )
    sub = agent.model_map["sub"]
    assert not isinstance(sub, str)
    assert sub.canonical_name() == "sub"
