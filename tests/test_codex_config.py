from typing import Any

import pytest
from inspect_ai.model import Model
from inspect_swe import codex_cli, interactive_codex_cli
from inspect_swe._codex_cli.config import (
    GUARDIAN_MODEL_SLUG,
    CodexAutoReview,
    check_codex_auto_review_version,
    codex_cli_config_overrides,
    codex_config_options,
    resolve_codex_auto_review,
    resolve_codex_auto_review_model_aliases,
    resolve_codex_deprecated_args,
    resolve_codex_web_search,
)
from inspect_swe._util.toml import to_toml


def test_codex_config_defaults() -> None:
    config = codex_config_options("live", True)

    assert config["web_search"] == "live"
    assert config["features.goals"] is True
    toml = to_toml(config)
    assert 'web_search = "live"' in toml
    assert "features.goals = true" in toml


@pytest.mark.parametrize("web_search", ["live", "cached", "disabled"])
def test_resolve_codex_web_search_modes(web_search: str) -> None:
    assert resolve_codex_web_search(web_search) == web_search


def test_resolve_codex_web_search_invalid_mode() -> None:
    with pytest.raises(ValueError, match="web_search must be one of"):
        resolve_codex_web_search("offline")


def test_deprecated_disallowed_tools_disable_web_search() -> None:
    disallowed_tools = resolve_codex_deprecated_args(
        {"disallowed_tools": ["web_search"]}
    )

    assert resolve_codex_web_search("live", disallowed_tools) == "disabled"


def test_deprecated_disallowed_tools_reject_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unsupported Codex disallowed_tools"):
        resolve_codex_deprecated_args({"disallowed_tools": ["bash"]})


def test_deprecated_args_reject_unexpected_keyword() -> None:
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        resolve_codex_deprecated_args({"unexpected": True})


def test_codex_cli_config_overrides_format_values_for_cli() -> None:
    assert codex_cli_config_overrides("cached", False) == {
        "web_search": '"cached"',
        "features.goals": "false",
    }


def test_to_toml_escapes_control_characters() -> None:
    toml = to_toml({"policy": 'line one\nline "two"\ttabbed'})
    assert toml == 'policy = "line one\\nline \\"two\\"\\ttabbed"'


def test_resolve_codex_auto_review_false_is_none() -> None:
    assert resolve_codex_auto_review(False) is None


def test_resolve_codex_auto_review_true_is_defaults() -> None:
    resolved = resolve_codex_auto_review(True)
    assert resolved == CodexAutoReview()
    assert resolved is not None
    assert resolved.policy is None
    assert resolved.model is None


def test_resolve_codex_auto_review_passes_through_options() -> None:
    options = CodexAutoReview(policy="Deny all network access.")
    assert resolve_codex_auto_review(options) is options


def test_codex_config_options_auto_review_off_by_default() -> None:
    config = codex_config_options("live", True)
    assert "approvals_reviewer" not in config
    assert "approval_policy" not in config
    assert "sandbox_mode" not in config


def test_codex_config_options_auto_review_enabled() -> None:
    config = codex_config_options("live", True, auto_review=CodexAutoReview())
    assert config["approval_policy"] == "on-request"
    assert config["sandbox_mode"] == "workspace-write"
    assert config["approvals_reviewer"] == "auto_review"
    assert config["features.guardian_approval"] is True
    assert "auto_review" not in config  # no [auto_review] table without a policy
    toml = to_toml(config)
    assert 'approvals_reviewer = "auto_review"' in toml
    assert 'approval_policy = "on-request"' in toml


def test_codex_config_options_auto_review_policy_table() -> None:
    config = codex_config_options(
        "live",
        True,
        auto_review=CodexAutoReview(policy="Never allow curl.\nAllow pip."),
    )
    assert config["auto_review"] == {"policy": "Never allow curl.\nAllow pip."}
    toml = to_toml(config)
    assert "[auto_review]" in toml
    assert 'policy = "Never allow curl.\\nAllow pip."' in toml


def test_codex_cli_config_overrides_auto_review() -> None:
    overrides = codex_cli_config_overrides(
        "live", True, auto_review=CodexAutoReview(policy="Never allow curl.")
    )
    assert overrides["approval_policy"] == '"on-request"'
    assert overrides["sandbox_mode"] == '"workspace-write"'
    assert overrides["approvals_reviewer"] == '"auto_review"'
    assert overrides["features.guardian_approval"] == "true"
    # policy goes only into config.toml (multiline-safe), never -c
    assert not any(key.startswith("auto_review") for key in overrides)


def test_codex_cli_config_overrides_auto_review_off_by_default() -> None:
    overrides = codex_cli_config_overrides("live", True)
    assert "approvals_reviewer" not in overrides
    assert "approval_policy" not in overrides


def test_auto_review_model_aliases_none_passthrough() -> None:
    existing: dict[str, str | Model] = {"alias": "openai/gpt-4o"}
    assert (
        resolve_codex_auto_review_model_aliases(CodexAutoReview(), existing) is existing
    )
    assert resolve_codex_auto_review_model_aliases(None, existing) is existing


def test_auto_review_model_aliases_adds_guardian_string() -> None:
    # outside a task, model_roles() is {}, so plain strings pass through
    aliases = resolve_codex_auto_review_model_aliases(
        CodexAutoReview(model="openai/gpt-4o"), {"alias": "x"}
    )
    assert aliases == {"alias": "x", "codex-auto-review": "openai/gpt-4o"}
    assert GUARDIAN_MODEL_SLUG == "codex-auto-review"


def test_auto_review_model_aliases_binds_role(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect_swe._codex_cli.config as config_mod

    guardian_model = object()

    def fake_model_roles() -> dict[str, Any]:
        return {"guardian": object()}

    def fake_get_model(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("role") == "guardian"
        return guardian_model

    monkeypatch.setattr(config_mod, "model_roles", fake_model_roles)
    monkeypatch.setattr(config_mod, "get_model", fake_get_model)

    aliases = resolve_codex_auto_review_model_aliases(
        CodexAutoReview(model="guardian"), None
    )
    assert aliases == {"codex-auto-review": guardian_model}


def test_check_codex_auto_review_version() -> None:
    check_codex_auto_review_version("0.137.0")
    check_codex_auto_review_version("0.145.0")
    check_codex_auto_review_version(None)  # undetectable: proceed
    with pytest.raises(RuntimeError, match="0.137.0"):
        check_codex_auto_review_version("0.136.0")
    with pytest.raises(RuntimeError, match="0.137.0"):
        check_codex_auto_review_version("0.99.0")


def test_codex_cli_accepts_auto_review() -> None:
    codex_cli(auto_review=True)
    codex_cli(auto_review=False)
    codex_cli(
        auto_review=CodexAutoReview(policy="Deny package installs.", model="guardian")
    )


def test_interactive_codex_cli_accepts_auto_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CodexCli (ACPAgent) requires an active sample to construct; this smoke
    # test only cares that `auto_review` is accepted and threaded through, so
    # stub the unrelated guard rather than standing up a full sample.
    import inspect_swe.acp.agent as acp_agent_mod

    monkeypatch.setattr(acp_agent_mod, "sample_active", lambda: object())

    interactive_codex_cli(model="mockllm/model", auto_review=True)
    interactive_codex_cli(
        model="mockllm/model",
        auto_review=CodexAutoReview(policy="Deny network."),
    )


def test_codex_auto_review_exported_from_package_root() -> None:
    import inspect_swe

    assert inspect_swe.CodexAutoReview is CodexAutoReview
    assert "CodexAutoReview" in inspect_swe.__all__
