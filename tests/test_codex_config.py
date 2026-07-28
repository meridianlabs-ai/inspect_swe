import pytest
from inspect_swe._codex_cli.config import (
    CodexAutoReview,
    codex_cli_config_overrides,
    codex_config_options,
    resolve_codex_auto_review,
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
