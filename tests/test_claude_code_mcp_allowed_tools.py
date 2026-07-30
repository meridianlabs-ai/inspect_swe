"""Regression tests for Claude Code MCP allowed-tool spellings.

Claude Code recognizes wildcard rules only after the literal
``mcp__<server>__`` prefix, so ``mcp__<server>__*`` uses a double underscore
before the glob. The older, more portable spelling ``mcp__<server>`` also
allows every tool from a server, which matters when ``version="auto"`` or
``"sandbox"`` selects an older installed Claude Code version.
"""

import pytest
from inspect_ai.tool import MCPServerConfig
from inspect_swe._claude_code.claude_code import (
    claude_code,
    resolve_allowed_mcp_tools,
    resolve_claude_code_deprecated_args,
    resolve_mcp_servers,
)


def test_default_registers_and_allowlists_explicit_mcp_tools() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )

    mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])

    assert mcp_config_cmds[0] == "--mcp-config"
    assert allowed_tools == [
        "mcp__taiga-mcp__list_files",
        "mcp__taiga-mcp__read_file",
    ]


def test_disabling_static_mcp_allowlist_keeps_servers_registered() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )

    mcp_config_cmds, _ = resolve_mcp_servers([server])
    allowed_tools = resolve_allowed_mcp_tools([server], [], allowlist_mcp_tools=False)

    assert mcp_config_cmds[0] == "--mcp-config"
    assert allowed_tools == []


def test_disabling_static_mcp_allowlist_preserves_bridged_allowlist() -> None:
    static_server = MCPServerConfig(type="http", name="caller-tools", tools=["lookup"])
    bridged_server = MCPServerConfig(
        type="http", name="inspect-tools", tools=["submit"]
    )

    allowed_tools = resolve_allowed_mcp_tools(
        [static_server], [bridged_server], allowlist_mcp_tools=False
    )

    assert allowed_tools == ["mcp__inspect-tools__submit"]


def test_disabling_bridged_allowlist_suppresses_bridged_allow_rules() -> None:
    """``allowlist_bridged_tools=False`` is what sends bridged calls to the classifier.

    An allow rule resolves at step 1 of the permission flow, before the
    ``permission_mode="auto"`` classifier, so a bridged tool left on
    ``--allowed-tools`` is never adjudicated. An evaluation whose whole tool
    surface is bridged therefore records zero adjudications while appearing to
    run under a reviewer.
    """
    bridged_server = MCPServerConfig(type="http", name="inspect-tools", tools="all")

    allowed_tools = resolve_allowed_mcp_tools(
        [], [bridged_server], allowlist_mcp_tools=False, allowlist_bridged_tools=False
    )

    assert allowed_tools == []


def test_bridged_allowlist_defaults_to_enabled() -> None:
    """Omitting the argument keeps every existing caller's behavior unchanged."""
    bridged_server = MCPServerConfig(type="http", name="inspect-tools", tools="all")

    assert resolve_allowed_mcp_tools(
        [], [bridged_server], allowlist_mcp_tools=True
    ) == ["mcp__inspect-tools__*"]


def test_static_and_bridged_allowlists_are_independent() -> None:
    static_server = MCPServerConfig(type="http", name="caller-tools", tools=["lookup"])
    bridged_server = MCPServerConfig(
        type="http", name="inspect-tools", tools=["submit"]
    )

    assert resolve_allowed_mcp_tools(
        [static_server],
        [bridged_server],
        allowlist_mcp_tools=True,
        allowlist_bridged_tools=False,
    ) == ["mcp__caller-tools__lookup"]


def test_claude_code_accepts_allowlist_bridged_tools() -> None:
    claude_code(permission_mode="auto", allowlist_bridged_tools=False)


def test_all_tools_wildcard_uses_double_underscore_before_glob() -> None:
    server = MCPServerConfig(type="http", name="taiga-mcp", tools="all")

    mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])

    assert mcp_config_cmds[0] == "--mcp-config"
    assert allowed_tools == ["mcp__taiga-mcp__*"]


def test_deprecated_auto_mode_resolves_to_auto_permission_mode() -> None:
    assert resolve_claude_code_deprecated_args({"auto_mode": True}, None) == "auto"


def test_truthy_deprecated_auto_mode_resolves_to_auto_permission_mode() -> None:
    assert resolve_claude_code_deprecated_args({"auto_mode": 1}, None) == "auto"


def test_claude_code_accepts_deprecated_auto_mode() -> None:
    claude_code(auto_mode=True)


def test_deprecated_auto_mode_rejects_conflicting_permission_mode() -> None:
    with pytest.raises(ValueError, match="auto_mode"):
        resolve_claude_code_deprecated_args({"auto_mode": True}, "acceptEdits")


def test_invalid_permission_mode_raises_construction_error() -> None:
    with pytest.raises(ValueError, match="permission_mode"):
        resolve_claude_code_deprecated_args({}, "unrestricted")
