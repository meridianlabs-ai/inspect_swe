"""Tests for the `tools` and `strict_mcp_config` options of `claude_code()`.

Claude Code's `--tools` restricts the built-in tool set itself, which
`--allowed-tools` / `--disallowed-tools` cannot do: those only allow or deny
calls to tools that are still offered to the model.
"""

from inspect_ai.tool import MCPServerConfig
from inspect_swe._claude_code.claude_code import (
    claude_code,
    resolve_allowed_mcp_tools,
    resolve_mcp_servers,
    resolve_tool_set_args,
    resolve_web_search_enabled,
)


def test_tools_reaches_command_as_comma_separated_list() -> None:
    assert resolve_tool_set_args(["Bash", "Read"], strict_mcp_config=False) == [
        "--tools",
        "Bash,Read",
    ]


def test_empty_tools_disables_every_builtin_tool() -> None:
    """`--tools ""` is how the CLI removes every built-in tool.

    The empty value must be passed explicitly; a bare `--tools` is not accepted.
    """
    assert resolve_tool_set_args([], strict_mcp_config=False) == ["--tools", ""]


def test_strict_mcp_config_reaches_command() -> None:
    assert resolve_tool_set_args(None, strict_mcp_config=True) == [
        "--strict-mcp-config"
    ]


def test_tools_and_strict_mcp_config_combine() -> None:
    assert resolve_tool_set_args([], strict_mcp_config=True) == [
        "--tools",
        "",
        "--strict-mcp-config",
    ]


def test_unset_options_leave_command_unchanged() -> None:
    assert resolve_tool_set_args(None, strict_mcp_config=False) == []


def test_strict_mcp_config_keeps_static_and_bridged_servers() -> None:
    """Static and bridged servers both stay available under strict MCP config.

    Both kinds of server travel via `--mcp-config`, which is the one source
    `--strict-mcp-config` keeps.
    """
    static_server = MCPServerConfig(type="http", name="caller-tools", tools=["lookup"])
    bridged_server = MCPServerConfig(
        type="http", name="inspect-tools", tools=["submit"]
    )

    mcp_config_cmds, _ = resolve_mcp_servers([static_server, bridged_server])
    allowed_tools = resolve_allowed_mcp_tools(
        [static_server], [bridged_server], allowlist_mcp_tools=True
    )
    args = resolve_tool_set_args([], strict_mcp_config=True)

    assert mcp_config_cmds[0] == "--mcp-config"
    assert '"caller-tools"' in mcp_config_cmds[1]
    assert '"inspect-tools"' in mcp_config_cmds[1]
    assert allowed_tools == ["mcp__caller-tools__lookup", "mcp__inspect-tools__submit"]
    assert "--strict-mcp-config" in args


def test_web_search_enabled_by_default() -> None:
    assert resolve_web_search_enabled(None, None)
    assert resolve_web_search_enabled([], None)


def test_web_search_disabled_when_disallowed() -> None:
    assert not resolve_web_search_enabled(["WebSearch"], None)
    assert not resolve_web_search_enabled(["WebSearch"], ["WebSearch"])


def test_web_search_disabled_when_left_out_of_tools() -> None:
    assert not resolve_web_search_enabled(None, [])
    assert not resolve_web_search_enabled(None, ["Bash", "Read"])


def test_web_search_enabled_when_included_in_tools() -> None:
    assert resolve_web_search_enabled(None, ["Bash", "WebSearch"])
    assert resolve_web_search_enabled(None, ["default"])


def test_claude_code_accepts_tools_and_strict_mcp_config() -> None:
    claude_code(tools=[], strict_mcp_config=True)
    claude_code(tools=["Bash", "Read"])
