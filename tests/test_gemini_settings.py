"""Unit tests for the Gemini CLI settings.json builder."""

import json

from inspect_ai.tool import MCPServerConfigHTTP, MCPServerConfigStdio
from inspect_swe._gemini_cli.gemini_cli import build_gemini_settings


def _mcp_servers(*servers: MCPServerConfigHTTP | MCPServerConfigStdio) -> object:
    return json.loads(build_gemini_settings(list(servers)))["mcpServers"]


def test_http_server_uses_http_url() -> None:
    servers = _mcp_servers(
        MCPServerConfigHTTP(name="remote", type="http", url="http://host:9000/mcp")
    )
    assert servers == {"remote": {"httpUrl": "http://host:9000/mcp"}}


def test_sse_server_keeps_url_with_sse_type() -> None:
    servers = _mcp_servers(
        MCPServerConfigHTTP(name="legacy", type="sse", url="http://host:8080/sse")
    )
    assert servers == {"legacy": {"url": "http://host:8080/sse", "type": "sse"}}


def test_tools_list_becomes_include_tools() -> None:
    servers = _mcp_servers(
        MCPServerConfigHTTP(
            name="remote", type="http", url="http://host:9000/mcp", tools=["search"]
        ),
        MCPServerConfigStdio(name="local", command="serve", tools=["query", "status"]),
    )
    assert servers == {
        "remote": {"httpUrl": "http://host:9000/mcp", "includeTools": ["search"]},
        "local": {
            "command": "serve",
            "args": [],
            "includeTools": ["query", "status"],
        },
    }
