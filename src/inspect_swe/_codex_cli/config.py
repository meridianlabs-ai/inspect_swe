from typing import Any, Literal, Mapping, cast

from typing_extensions import TypedDict

CodexWebSearch = Literal["live", "cached", "disabled"]
CodexSandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
CodexApprovalPolicy = Literal["untrusted", "on-request", "never"]


class CodexDeprecatedArgs(TypedDict, total=False):
    disallowed_tools: list[Literal["web_search"]] | None


def resolve_codex_deprecated_args(
    deprecated_args: Mapping[str, Any],
) -> list[Literal["web_search"]]:
    unexpected_args = set(deprecated_args) - {"disallowed_tools"}
    if unexpected_args:
        unexpected = ", ".join(sorted(unexpected_args))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    disallowed_tools = deprecated_args.get("disallowed_tools") or []
    unsupported_tools = set(disallowed_tools) - {"web_search"}
    if unsupported_tools:
        unsupported = ", ".join(sorted(unsupported_tools))
        raise ValueError(f"Unsupported Codex disallowed_tools value(s): {unsupported}")

    return list(disallowed_tools)


def resolve_codex_web_search(
    web_search: str,
    disallowed_tools: list[Literal["web_search"]] | None = None,
) -> CodexWebSearch:
    if web_search not in ("live", "cached", "disabled"):
        raise ValueError("web_search must be one of 'live', 'cached', or 'disabled'.")
    if disallowed_tools and "web_search" in disallowed_tools:
        return "disabled"
    return cast(CodexWebSearch, web_search)


def codex_config_options(web_search: CodexWebSearch, goals: bool) -> dict[str, Any]:
    return {
        "web_search": web_search,
        "features.goals": goals,
    }


def codex_cli_config_overrides(
    web_search: CodexWebSearch, goals: bool
) -> dict[str, str]:
    return {
        "web_search": f'"{web_search}"',
        "features.goals": "true" if goals else "false",
    }


def codex_mcp_server_toml(
    mcp_server_dump: dict[str, Any], approval_policy: CodexApprovalPolicy
) -> dict[str, Any]:
    """Build one `[mcp_servers.<name>]` TOML table for a bridged MCP server.

    MCP tool calls have their OWN approval gate
    (`default_tools_approval_mode`, one of "prompt"/"writes"/"auto"/"approve" --
    `AppToolApproval` in `codex-rs/config/src/mcp_types.rs`, defaulting to
    "auto"), separate from the top-level `approval_policy`: with
    `approval_policy="never"` alone, write-type MCP tool calls (e.g. an
    `edit_file` call) are auto-denied ("user cancelled MCP tool call") rather
    than run, because headless `codex exec` has no way to answer the resulting
    approval prompt. "auto" is NOT sufficient either (confirmed empirically:
    same auto-denial) -- only "approve" actually skips the gate. The override is
    applied only when `approval_policy` is `"never"`, so callers who choose a
    prompting policy keep the per-server default.

    Version caveat: the MCP approval gate applies to all servers from codex
    0.117, but `default_tools_approval_mode` is only honoured from 0.122 --
    on 0.117-0.121 this key is parsed and silently ignored, so restricted-mode
    runs with bridged tools on those versions still hit the auto-denial. The
    key is silently ignored (not a parse error) back to at least 0.50, so it is
    safe to emit for old pins.
    """
    mcp_server_toml = dict(mcp_server_dump)
    if approval_policy == "never":
        mcp_server_toml["default_tools_approval_mode"] = "approve"
    return mcp_server_toml


def codex_sandbox_args(
    sandbox_mode: CodexSandboxMode,
    approval_policy: CodexApprovalPolicy,
    network_access: bool,
) -> list[str]:
    if sandbox_mode == "danger-full-access" and approval_policy == "never":
        return ["--dangerously-bypass-approvals-and-sandbox"]

    sandbox_args = [
        "--sandbox",
        sandbox_mode,
        "-c",
        f"approval_policy={approval_policy}",
    ]
    if sandbox_mode == "workspace-write":
        sandbox_args.extend(
            [
                "-c",
                f"sandbox_workspace_write.network_access={str(network_access).lower()}",
            ]
        )
    return sandbox_args


def validate_codex_sandbox_mode(value: str) -> CodexSandboxMode:
    """Validate a sandbox mode at runtime (agent kwargs arrive as arbitrary strings)."""
    if value not in ("read-only", "workspace-write", "danger-full-access"):
        raise ValueError(
            "sandbox_mode must be one of 'read-only', 'workspace-write', or "
            f"'danger-full-access', got {value!r}."
        )
    return cast(CodexSandboxMode, value)


def validate_codex_approval_policy(value: str) -> CodexApprovalPolicy:
    """Validate an approval policy at runtime (agent kwargs arrive as arbitrary strings)."""
    if value not in ("untrusted", "on-request", "never"):
        raise ValueError(
            "approval_policy must be one of 'untrusted', 'on-request', or "
            f"'never', got {value!r}."
        )
    return cast(CodexApprovalPolicy, value)


def validate_codex_network_access(value: bool) -> bool:
    """Validate network_access at runtime.

    Agent kwargs arrive from task configs and `-S` args as arbitrary values; an
    unvalidated value would be emitted as
    `-c sandbox_workspace_write.network_access=<value>` and only fail when Codex
    parses it mid-evaluation.
    """
    if not isinstance(value, bool):
        raise ValueError(f"network_access must be a bool, got {value!r}.")
    return value


def resolve_codex_approval_policy(
    approval_policy: CodexApprovalPolicy,
    config_overrides: Mapping[str, str] | None,
) -> CodexApprovalPolicy:
    """Resolve one effective approval policy from the argument and overrides.

    `config_overrides["approval_policy"]` is intercepted (not passed through as
    a raw `-c` pair) so command construction and the bridged-MCP TOML are
    generated from a single effective value; a raw pass-through would let the
    two disagree.
    """
    configured_policy = (
        config_overrides.get("approval_policy")
        if config_overrides is not None
        else None
    )
    if configured_policy is None:
        return validate_codex_approval_policy(approval_policy)
    return validate_codex_approval_policy(configured_policy)


def resolve_codex_sandbox_mode(
    sandbox_mode: CodexSandboxMode,
    config_overrides: Mapping[str, str] | None,
) -> CodexSandboxMode:
    """Resolve one effective sandbox mode from the argument and overrides.

    `config_overrides["sandbox_mode"]` is intercepted for the same reason as
    `approval_policy`: the explicit `--sandbox`/bypass arguments are derived
    from the effective mode, and a raw pass-through would leave them
    contradicting the caller's requested mode -- e.g.
    `config_overrides={"sandbox_mode": "read-only"}` with the default argument
    previously emitted `--dangerously-bypass-approvals-and-sandbox` alongside
    `-c sandbox_mode=read-only`, silently granting no sandbox at all.
    """
    configured_mode = (
        config_overrides.get("sandbox_mode") if config_overrides is not None else None
    )
    if configured_mode is None:
        return validate_codex_sandbox_mode(sandbox_mode)
    return validate_codex_sandbox_mode(configured_mode)
