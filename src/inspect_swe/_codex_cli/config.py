from typing import Any, Literal, Mapping, cast

from inspect_ai.model import Model, get_model, model_roles
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

CodexWebSearch = Literal["live", "cached", "disabled"]


class CodexAutoReview(BaseModel):
    """Options for Codex automated approval review (`auto_review`).

    When enabled, Codex runs with its own sandbox active (`workspace-write`)
    and `approval_policy` set to `on-request`; escalation requests are
    adjudicated by a guardian model rather than a human.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: str | None = Field(default=None)
    """Additional policy instructions inserted into the guardian review prompt."""

    model: str | Model | None = Field(default=None)
    """Model that serves guardian review requests.

    A `str` naming an Inspect model role binds that role; any other string is
    treated as a model name. Defaults to `None`, which serves guardian
    requests with the model the agent is running with (the task's main model
    unless the agent's `model` option is set).
    """


def resolve_codex_auto_review(
    auto_review: bool | CodexAutoReview,
) -> CodexAutoReview | None:
    if auto_review is False:
        return None
    if auto_review is True:
        return CodexAutoReview()
    return auto_review


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


def codex_config_options(
    web_search: CodexWebSearch,
    goals: bool,
    auto_review: CodexAutoReview | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "web_search": web_search,
        "features.goals": goals,
    }
    if auto_review is not None:
        # auto_review only functions with on-request approvals and Codex's own
        # sandbox engaged (mirrors Codex's "Approve for me" preset)
        options["approval_policy"] = "on-request"
        options["sandbox_mode"] = "workspace-write"
        options["approvals_reviewer"] = "auto_review"
        options["features.guardian_approval"] = True
        if auto_review.policy is not None:
            options["auto_review"] = {"policy": auto_review.policy}
    return options


def codex_cli_config_overrides(
    web_search: CodexWebSearch,
    goals: bool,
    auto_review: CodexAutoReview | None = None,
) -> dict[str, str]:
    overrides = {
        "web_search": f'"{web_search}"',
        "features.goals": "true" if goals else "false",
    }
    if auto_review is not None:
        overrides["approval_policy"] = '"on-request"'
        overrides["sandbox_mode"] = '"workspace-write"'
        overrides["approvals_reviewer"] = '"auto_review"'
        overrides["features.guardian_approval"] = "true"
        # auto_review.policy is emitted only via config.toml: -c values are
        # parsed as TOML and multiline policies don't survive shell quoting
    return overrides


GUARDIAN_MODEL_SLUG = "codex-auto-review"
"""Model slug Codex uses for guardian (auto_review) requests."""

CODEX_AUTO_REVIEW_MIN_VERSION = "0.137.0"
"""First Codex CLI release where `codex exec` preserves auto_review approvals."""


def resolve_codex_auto_review_model_aliases(
    auto_review: CodexAutoReview | None,
    model_aliases: dict[str, str | Model] | None,
    default: str | Model | None = None,
) -> dict[str, str | Model] | None:
    """Bind the guardian model slug to the configured auto_review model.

    Call within a running task (role resolution reads task context). A `str`
    naming a defined Inspect model role binds via `get_model(role=...)`;
    other values pass through for the bridge to resolve. `default` is bound
    when auto_review is enabled but no model is configured — pass it for
    bridges without a fallback model (e.g. the ACP bridge), where the
    guardian slug would otherwise fail to resolve.
    """
    if auto_review is None:
        return model_aliases
    guardian: str | Model | None = (
        auto_review.model if auto_review.model is not None else default
    )
    if guardian is None:
        return model_aliases
    if isinstance(guardian, str) and guardian in model_roles():
        guardian = get_model(role=guardian)
    return {**(model_aliases or {}), GUARDIAN_MODEL_SLUG: guardian}


def check_codex_auto_review_version(version: str | None) -> None:
    """Raise if the installed Codex CLI can't run auto_review headlessly."""
    if version is None:
        return
    installed = tuple(int(part) for part in version.split(".")[:3])
    required = tuple(int(part) for part in CODEX_AUTO_REVIEW_MIN_VERSION.split("."))
    if installed < required:
        raise RuntimeError(
            f"auto_review requires Codex CLI >= {CODEX_AUTO_REVIEW_MIN_VERSION} "
            f"(found {version}). Pass version='latest' (or an explicit newer "
            "version) to codex_cli()."
        )


CodexSandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
CodexApprovalPolicy = Literal["untrusted", "on-request", "never"]


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


def codex_network_access_args(network_access: bool) -> list[str]:
    """`-c` args granting or denying network to model-generated commands.

    Only `workspace-write` reads this setting, but that mode is reached two ways:
    explicitly via `--sandbox`, and implicitly by `auto_review`, which selects it
    through its own `-c` overrides. Codex's own default is `false`, so a path
    that omits these args silently discards the caller's setting.
    """
    return [
        "-c",
        f"sandbox_workspace_write.network_access={str(network_access).lower()}",
    ]


def codex_sandbox_args(
    sandbox_mode: CodexSandboxMode,
    approval_policy: CodexApprovalPolicy,
    network_access: bool,
    auto_review: bool = False,
) -> list[str]:
    """CLI args selecting Codex's own sandbox and approval policy.

    With `auto_review` the mode and policy come from its own final `-c`
    overrides (`workspace-write` + `on-request`) instead: an explicit
    `--sandbox` or bypass flag would pin them at a precedence `-c` cannot beat.
    Its effective mode is still `workspace-write`, so network access is
    configured on that path too.
    """
    if auto_review:
        return codex_network_access_args(network_access)

    if sandbox_mode == "danger-full-access" and approval_policy == "never":
        return ["--dangerously-bypass-approvals-and-sandbox"]

    sandbox_args = [
        "--sandbox",
        sandbox_mode,
        "-c",
        f"approval_policy={approval_policy}",
    ]
    if sandbox_mode == "workspace-write":
        sandbox_args.extend(codex_network_access_args(network_access))
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


def validate_codex_bool_arg(name: str, value: bool) -> bool:
    """Validate a boolean agent argument at runtime.

    Agent kwargs arrive from task configs and `-S` args as arbitrary values, so
    the `bool` annotation is no protection: the string `"false"` is truthy and
    would silently select the opposite of what the caller asked for.

    `network_access` would otherwise be emitted as
    `-c sandbox_workspace_write.network_access=<value>` and fail only when Codex
    parses it mid-evaluation; `approve_static_mcp_tools` would pre-approve every
    static MCP tool call, reversing an explicit opt-out with no signal at all.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}.")
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
