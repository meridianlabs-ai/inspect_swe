from collections.abc import Set as AbstractSet
from typing import Any, Literal, Mapping, cast

from inspect_ai.tool import MCPServerConfig
from typing_extensions import TypedDict

CodexWebSearch = Literal["live", "cached", "disabled"]


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


def codex_mcp_server_config(
    mcp_server: MCPServerConfig, bridged_server_names: AbstractSet[str]
) -> dict[str, Any]:
    """TOML table for one `mcp_servers.<name>` entry in codex config.

    Bridged servers are marked `required = true`: codex >= 0.119.0 then blocks
    session init on their initialize+tools/list and `codex exec` exits non-zero
    if one fails to come up, closing the client-connect half of the first-turn
    race (`wait_for_mcp_endpoints` covers the endpoint half; Claude Code's
    equivalent is `BLOCKING_MCP_ENV`). Older codex versions ignore the key
    (serde tolerates unknown fields in the `mcp_servers` table). Static
    caller-provided servers stay optional: their availability is the caller's
    contract, mirroring the readiness-gate scoping.
    """
    server_config = mcp_server.model_dump(exclude={"name", "tools"}, exclude_none=True)
    if mcp_server.name in bridged_server_names:
        server_config["required"] = True
    return server_config
