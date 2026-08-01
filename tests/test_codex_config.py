import pytest
from inspect_swe._codex_cli.config import (
    CodexApprovalPolicy,
    CodexSandboxMode,
    codex_cli_config_overrides,
    codex_config_options,
    codex_mcp_server_toml,
    codex_sandbox_args,
    resolve_codex_approval_policy,
    resolve_codex_deprecated_args,
    resolve_codex_sandbox_mode,
    resolve_codex_web_search,
    validate_codex_network_access,
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


def test_codex_mcp_server_toml_sets_approve_when_never() -> None:
    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    result = codex_mcp_server_toml(dump, "never")
    assert result == {
        "type": "http",
        "url": "http://localhost:8901/mcp/taiga-mcp",
        "default_tools_approval_mode": "approve",
    }


def test_codex_mcp_server_toml_leaves_other_policies_untouched() -> None:
    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    for policy in ("untrusted", "on-request"):
        assert codex_mcp_server_toml(dump, policy) == dump


def test_codex_mcp_server_toml_does_not_mutate_input() -> None:
    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    codex_mcp_server_toml(dump, "never")
    assert "default_tools_approval_mode" not in dump


@pytest.mark.parametrize(
    ("sandbox_mode", "approval_policy", "network_access", "expected"),
    [
        (
            "danger-full-access",
            "never",
            True,
            ["--dangerously-bypass-approvals-and-sandbox"],
        ),
        (
            "read-only",
            "never",
            True,
            ["--sandbox", "read-only", "-c", "approval_policy=never"],
        ),
        (
            "workspace-write",
            "on-request",
            False,
            [
                "--sandbox",
                "workspace-write",
                "-c",
                "approval_policy=on-request",
                "-c",
                "sandbox_workspace_write.network_access=false",
            ],
        ),
    ],
)
def test_codex_sandbox_args(
    sandbox_mode: CodexSandboxMode,
    approval_policy: CodexApprovalPolicy,
    network_access: bool,
    expected: list[str],
) -> None:
    assert codex_sandbox_args(sandbox_mode, approval_policy, network_access) == expected


def test_config_override_resolves_effective_approval_policy() -> None:
    assert (
        resolve_codex_approval_policy("on-request", {"approval_policy": "never"})
        == "never"
    )


def test_config_override_rejects_unknown_approval_policy() -> None:
    with pytest.raises(ValueError, match="approval_policy"):
        resolve_codex_approval_policy("never", {"approval_policy": "always"})


def test_resolve_codex_sandbox_mode_prefers_override() -> None:
    assert (
        resolve_codex_sandbox_mode("danger-full-access", {"sandbox_mode": "read-only"})
        == "read-only"
    )
    assert resolve_codex_sandbox_mode("workspace-write", None) == "workspace-write"


def test_resolve_codex_sandbox_mode_validates_both_paths() -> None:
    with pytest.raises(ValueError):
        resolve_codex_sandbox_mode("readonly", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        resolve_codex_sandbox_mode("danger-full-access", {"sandbox_mode": "nope"})


def test_config_override_sandbox_mode_never_emits_bypass() -> None:
    """A caller who asked for a restricted sandbox must never get the bypass flag.

    Previously `config_overrides={"sandbox_mode": "read-only"}` with the default
    argument emitted `--dangerously-bypass-approvals-and-sandbox` alongside
    `-c sandbox_mode=read-only`, silently granting no sandbox at all.
    """
    effective = resolve_codex_sandbox_mode(
        "danger-full-access", {"sandbox_mode": "read-only"}
    )
    args = codex_sandbox_args(effective, "never", True)
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert args[:2] == ["--sandbox", "read-only"]


def test_resolve_codex_approval_policy_validates_argument() -> None:
    with pytest.raises(ValueError):
        resolve_codex_approval_policy("on_request", None)  # type: ignore[arg-type]


def test_validate_codex_network_access() -> None:
    assert validate_codex_network_access(True) is True
    assert validate_codex_network_access(False) is False
    with pytest.raises(ValueError):
        validate_codex_network_access("nope")  # type: ignore[arg-type]


def test_headless_non_never_policy_raises_without_reviewer() -> None:
    """Headless prompting policies fail fast.

    `codex exec` hard-overrides the runtime policy to `never`; a prompting
    policy without an approvals reviewer would silently cancel every bridged
    tool call.
    """
    from inspect_swe import codex_cli

    with pytest.raises(ValueError, match="headless"):
        codex_cli(approval_policy="on-request")
    # supported paths do not raise
    codex_cli(approval_policy="on-request", centaur=True)
    codex_cli(
        approval_policy="on-request",
        config_overrides={"approvals_reviewer": '"auto_review"'},
    )
    codex_cli()  # default never is fine


def test_static_mcp_server_toml_opt_in_path() -> None:
    """The static-server opt-in reuses the bridged helper.

    Approve under effective `never`, untouched otherwise.
    """
    dump = {"type": "http", "url": "http://localhost:9/mcp/x"}
    assert codex_mcp_server_toml(dump, "never")["default_tools_approval_mode"] == (
        "approve"
    )
    assert codex_mcp_server_toml(dump, "on-request") == dump
