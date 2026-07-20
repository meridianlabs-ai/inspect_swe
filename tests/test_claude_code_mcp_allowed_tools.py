from inspect_ai.tool import MCPServerConfig
from inspect_swe._claude_code.claude_code import resolve_mcp_servers


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


def test_disabling_mcp_allowlist_keeps_servers_registered() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )

    mcp_config_cmds, allowed_tools = resolve_mcp_servers(
        [server], allowlist_mcp_tools=False
    )

    assert mcp_config_cmds[0] == "--mcp-config"
    assert allowed_tools == []
