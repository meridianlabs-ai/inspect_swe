from typing import Any

import pytest
from inspect_ai.model import Model
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_swe import codex_cli, interactive_codex_cli
from inspect_swe._codex_cli.config import (
    GUARDIAN_MODEL_SLUG,
    MCP_STARTUP_TIMEOUT_SEC,
    CodexApprovalPolicy,
    CodexAutoReview,
    CodexSandboxMode,
    check_codex_auto_review_version,
    codex_cli_config_overrides,
    codex_config_options,
    codex_mcp_server_config,
    codex_mcp_server_toml,
    codex_mcp_servers_toml,
    codex_network_access_args,
    codex_sandbox_args,
    codex_sandbox_uses_bwrap,
    resolve_codex_approval_policy,
    resolve_codex_auto_review,
    resolve_codex_auto_review_model_aliases,
    resolve_codex_deprecated_args,
    resolve_codex_sandbox_mode,
    resolve_codex_web_search,
    validate_codex_bool_arg,
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


def test_to_toml_escapes_remaining_control_characters() -> None:
    value = "esc \x1b bell \x07 del \x7f"
    toml = to_toml({"policy": value})
    assert toml == 'policy = "esc \\u001B bell \\u0007 del \\u007F"'
    # round-trip via the stdlib parser (tomllib requires Python >= 3.11)
    tomllib = pytest.importorskip("tomllib")
    assert tomllib.loads(toml) == {"policy": value}


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


def test_auto_review_model_aliases_default_binds_guardian() -> None:
    # bridges without a fallback model (e.g. ACP) pass a default so the
    # guardian slug always resolves
    aliases = resolve_codex_auto_review_model_aliases(
        CodexAutoReview(), {"alias": "x"}, default="openai/gpt-4o"
    )
    assert aliases == {"alias": "x", GUARDIAN_MODEL_SLUG: "openai/gpt-4o"}


def test_auto_review_model_aliases_explicit_model_beats_default() -> None:
    aliases = resolve_codex_auto_review_model_aliases(
        CodexAutoReview(model="openai/gpt-4o"), None, default="openai/other"
    )
    assert aliases == {GUARDIAN_MODEL_SLUG: "openai/gpt-4o"}


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


def test_check_codex_auto_review_version_names_the_feature() -> None:
    # the reviewer escape hatch rides the same codex relent, so it reports the
    # same floor under its own name rather than mentioning auto_review
    with pytest.raises(RuntimeError, match=r"approvals_reviewer"):
        check_codex_auto_review_version(
            "0.136.0", feature="config_overrides['approvals_reviewer']"
        )
    check_codex_auto_review_version("0.137.0", feature="anything")


def test_codex_sandbox_uses_bwrap_only_on_releases_that_launch_it() -> None:
    # <= 0.77 sandboxes via Landlock+seccomp and 0.98 makes bwrap opt-in, so a
    # bwrap preflight on those releases fails over a binary that never runs
    assert not codex_sandbox_uses_bwrap("0.77.0")
    assert not codex_sandbox_uses_bwrap("0.98.0")
    assert codex_sandbox_uses_bwrap("0.99.0")
    assert codex_sandbox_uses_bwrap("0.145.0")
    # undetectable or non-numeric: keep the preflight rather than assume
    assert codex_sandbox_uses_bwrap(None)
    assert codex_sandbox_uses_bwrap("sandbox")


def test_codex_cli_accepts_auto_review() -> None:
    codex_cli(auto_review=True)
    codex_cli(auto_review=False)
    codex_cli(
        auto_review=CodexAutoReview(policy="Deny package installs.", model="guardian")
    )


def test_codex_cli_accepts_auto_review_with_transparent_proxy() -> None:
    codex_cli(auto_review=True, transparent_proxy=True)


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


def test_config_override_resolves_effective_approval_policy() -> None:
    assert (
        resolve_codex_approval_policy("on-request", {"approval_policy": "never"})
        == "never"
    )


def test_config_override_rejects_unknown_approval_policy() -> None:
    with pytest.raises(ValueError, match="approval_policy"):
        resolve_codex_approval_policy("never", {"approval_policy": "always"})


def test_config_override_accepts_quoted_approval_policy() -> None:
    """A quoted `approval_policy` override must not regress.

    `codex_cli_config_overrides` itself emits the TOML-quoted form (e.g.
    `'"on-request"'`), and `main` accepted a quoted `approval_policy`
    override.
    """
    assert (
        resolve_codex_approval_policy("never", {"approval_policy": '"never"'})
        == "never"
    )


def test_config_override_accepts_on_failure_approval_policy() -> None:
    """`"on-failure"` is a real upstream `AskForApproval` spelling.

    It is a serde alias of the same `OnRequest` variant as `"on-request"`
    (`#[serde(alias = "on-failure")] OnRequest`, codex-rs `protocol.rs`) --
    not a typo, so it must resolve rather than raise, quoted or not.
    """
    assert (
        resolve_codex_approval_policy("never", {"approval_policy": "on-failure"})
        == "on-request"
    )
    assert (
        resolve_codex_approval_policy("never", {"approval_policy": '"on-failure"'})
        == "on-request"
    )


def test_resolve_codex_sandbox_mode_prefers_override() -> None:
    assert (
        resolve_codex_sandbox_mode("danger-full-access", {"sandbox_mode": "read-only"})
        == "read-only"
    )
    assert resolve_codex_sandbox_mode("workspace-write", None) == "workspace-write"


def test_config_override_accepts_quoted_sandbox_mode() -> None:
    """A quoted `sandbox_mode` override must not regress, same as `approval_policy`.

    `codex_cli_config_overrides` uses the TOML-quoted convention throughout
    (e.g. `'"workspace-write"'` for `auto_review`), so a caller who copies it
    for `sandbox_mode` must not have the value rejected.
    """
    assert (
        resolve_codex_sandbox_mode(
            "danger-full-access", {"sandbox_mode": '"read-only"'}
        )
        == "read-only"
    )


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


def test_validate_codex_bool_arg() -> None:
    assert validate_codex_bool_arg("network_access", True) is True
    assert validate_codex_bool_arg("network_access", False) is False
    with pytest.raises(ValueError, match="network_access must be a bool"):
        validate_codex_bool_arg("network_access", "nope")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, None])
def test_validate_codex_bool_arg_rejects_non_bools(value: object) -> None:
    """Truthy and falsy non-bools alike must be rejected, not coerced.

    Agent kwargs arrive from task configs and `-S` args as arbitrary values, so
    the string `"false"` is truthy and would otherwise select the opposite of
    what the caller wrote.
    """
    with pytest.raises(ValueError, match="approve_static_mcp_tools must be a bool"):
        validate_codex_bool_arg("approve_static_mcp_tools", value)  # type: ignore[arg-type]


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
        (
            "workspace-write",
            "never",
            True,
            [
                "--sandbox",
                "workspace-write",
                "-c",
                "approval_policy=never",
                "-c",
                "sandbox_workspace_write.network_access=true",
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


def test_codex_network_access_args_spellings() -> None:
    """`-c` values are parsed as TOML: Python's `True` capitalisation is invalid."""
    assert codex_network_access_args(True) == [
        "-c",
        "sandbox_workspace_write.network_access=true",
    ]
    assert codex_network_access_args(False) == [
        "-c",
        "sandbox_workspace_write.network_access=false",
    ]


def test_codex_cli_rejects_non_bool_approve_static_mcp_tools() -> None:
    """`approve_static_mcp_tools="false"` is truthy and must not be coerced.

    Agent kwargs arrive from task configs and `-S` args as arbitrary strings, so
    an unvalidated `"false"` would pre-approve every static MCP tool call under
    the default `never` policy -- the opposite of the caller's opt-out.
    """
    with pytest.raises(ValueError, match="approve_static_mcp_tools must be a bool"):
        codex_cli(approve_static_mcp_tools="false")  # type: ignore[arg-type]
    # real booleans are accepted
    codex_cli(approve_static_mcp_tools=True)
    codex_cli(approve_static_mcp_tools=False)


def test_codex_cli_validates_network_access_and_allows_it_with_auto_review() -> None:
    with pytest.raises(ValueError, match="network_access must be a bool"):
        codex_cli(network_access="false")  # type: ignore[arg-type]
    codex_cli(auto_review=True, network_access=False)
    codex_cli(auto_review=True, network_access=True)


def test_bridged_mcp_server_is_marked_required() -> None:
    server = MCPServerConfigHTTP(
        type="http", name="bridged-tools", url="http://localhost:9000/mcp"
    )

    config = codex_mcp_server_config(server, {"bridged-tools"}, None)

    assert config["required"] is True
    assert "name" not in config
    assert "tools" not in config
    assert "required = true" in to_toml({"mcp_servers.bridged-tools": config})


def test_static_mcp_server_is_not_marked_required() -> None:
    server = MCPServerConfigHTTP(
        type="http", name="caller-tools", url="http://localhost:9001/mcp"
    )

    config = codex_mcp_server_config(server, {"bridged-tools"}, None)

    assert "required" not in config


def test_bridged_mcp_server_gets_the_startup_timeout() -> None:
    server = MCPServerConfigHTTP(
        type="http", name="bridged-tools", url="http://localhost:9000/mcp"
    )

    config = codex_mcp_server_config(server, {"bridged-tools"}, MCP_STARTUP_TIMEOUT_SEC)

    assert config["startup_timeout_sec"] == 300


def test_static_mcp_server_keeps_codex_default_startup_timeout() -> None:
    server = MCPServerConfigHTTP(
        type="http", name="caller-tools", url="http://localhost:9001/mcp"
    )

    config = codex_mcp_server_config(server, {"bridged-tools"}, MCP_STARTUP_TIMEOUT_SEC)

    assert "startup_timeout_sec" not in config


def test_bridged_mcp_server_accepts_an_explicit_startup_timeout() -> None:
    server = MCPServerConfigHTTP(
        type="http", name="bridged-tools", url="http://localhost:9000/mcp"
    )

    config = codex_mcp_server_config(server, {"bridged-tools"}, 60)

    assert config["startup_timeout_sec"] == 60


def test_codex_cli_rejects_non_bool_centaur() -> None:
    """`centaur` must not silently bypass the headless guard.

    A non-bool, non-`CentaurOptions` value must raise rather than pass
    through: that guard keys off `centaur is False` identity, and agent
    kwargs arrive from task configs and `-S` args as arbitrary values (bare
    `-S centaur=false` coerces via YAML, but quoted values, config files,
    and programmatic callers don't), so an unvalidated
    `centaur=0`/`None`/`"false"` would skip the guard and produce a
    malformed headless run.
    """
    with pytest.raises(ValueError, match="centaur must be a bool"):
        codex_cli(centaur="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="centaur must be a bool"):
        codex_cli(centaur=0)  # type: ignore[arg-type]
    # real values are accepted
    codex_cli(centaur=True)
    codex_cli(centaur=False)


def test_codex_cli_rejects_approve_static_mcp_tools_with_auto_review() -> None:
    """`approve_static_mcp_tools=True` is inert under `auto_review`.

    The effective policy is `"on-request"`, never `"never"`, so combining
    them must fail loudly like the sibling `sandbox_mode`/`approval_policy`
    controls rather than silently doing nothing.
    """
    with pytest.raises(ValueError, match="approve_static_mcp_tools"):
        codex_cli(auto_review=True, approve_static_mcp_tools=True)
    # the default (False) does not conflict
    codex_cli(auto_review=True, approve_static_mcp_tools=False)


def test_codex_mcp_servers_toml_gates_on_force_approve() -> None:
    """Regression coverage for the static-vs-bridged wiring in `codex_cli`.

    Static servers only get the headless approval override when the caller
    opts in (`force_approve=False` by default); bridged servers always do
    (`force_approve=True`, unconditionally). A regression that gates on the
    wrong flag or mixes the two server classes fails here.
    """
    static = MCPServerConfigHTTP(
        type="http", name="caller-tools", url="http://localhost:9001/mcp"
    )
    bridged = MCPServerConfigHTTP(
        type="http", name="bridged-tools", url="http://localhost:9000/mcp"
    )
    bridged_server_names = {"bridged-tools"}

    static_toml = codex_mcp_servers_toml(
        [static],
        bridged_server_names,
        "never",
        force_approve=False,
        bridged_startup_timeout=MCP_STARTUP_TIMEOUT_SEC,
    )
    assert "default_tools_approval_mode" not in static_toml["mcp_servers.caller-tools"]
    assert "required" not in static_toml["mcp_servers.caller-tools"]

    static_opted_in = codex_mcp_servers_toml(
        [static],
        bridged_server_names,
        "never",
        force_approve=True,
        bridged_startup_timeout=MCP_STARTUP_TIMEOUT_SEC,
    )
    assert (
        static_opted_in["mcp_servers.caller-tools"]["default_tools_approval_mode"]
        == "approve"
    )

    bridged_toml = codex_mcp_servers_toml(
        [bridged],
        bridged_server_names,
        "never",
        force_approve=True,
        bridged_startup_timeout=MCP_STARTUP_TIMEOUT_SEC,
    )
    assert (
        bridged_toml["mcp_servers.bridged-tools"]["default_tools_approval_mode"]
        == "approve"
    )
    assert bridged_toml["mcp_servers.bridged-tools"]["required"] is True
