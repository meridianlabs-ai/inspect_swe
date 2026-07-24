"""Regression test for the ``resolve_mcp_servers`` allowed-tools glob.

Claude Code's ``--allowed-tools`` parser only recognizes a wildcard rule in
the exact form ``mcp__<server>__*`` (double underscore before the glob). A
single-underscore variant (``mcp__<server>_*``) is silently ignored (logged
as a warning, not an error), so under a permission mode that actually
consults the allow-list (e.g. ``--permission-mode auto`` -- as opposed to
``--dangerously-skip-permissions``, which bypasses the allow-list check
entirely) every bridged MCP tool call is denied. Confirmed empirically
against the real ``claude`` CLI (2.1.205 and 2.1.212) in a fresh sandbox
home: the single-underscore rule is ignored and tool calls report "you
haven't granted it yet"; the double-underscore rule is honored and the same
tool call succeeds.
"""

from inspect_ai.tool import MCPServerConfig

from inspect_swe._claude_code.claude_code import resolve_mcp_servers


def test_all_tools_wildcard_uses_double_underscore_before_glob() -> None:
    server = MCPServerConfig(type="http", name="taiga-mcp", tools="all")
    _mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])
    assert allowed_tools == ["mcp__taiga-mcp__*"]


def test_explicit_tool_list_is_unaffected() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )
    _mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])
    assert allowed_tools == [
        "mcp__taiga-mcp__list_files",
        "mcp__taiga-mcp__read_file",
    ]
