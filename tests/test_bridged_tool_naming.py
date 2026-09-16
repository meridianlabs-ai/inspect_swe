"""Each wrapper tells the sandbox agent bridge how its scaffold names bridged tools.

Vectors come from the scaffolds' own code (Kimi Code's names were produced by
its ``qualifyMcpToolName``; Codex's by reproducing ``normalize_tools_for_model``
on the review's examples).
"""

import pytest
from inspect_ai.agent import BridgedToolCall, BridgedToolName
from inspect_ai.tool import ToolCall
from inspect_swe._antigravity.naming import AntigravityToolNaming
from inspect_swe._claude_code.naming import ClaudeCodeToolNaming
from inspect_swe._codex_cli.naming import CodexCliToolNaming
from inspect_swe._gemini_cli.naming import GeminiCliToolNaming
from inspect_swe._kimi_code.naming import KimiCodeToolNaming
from inspect_swe._opencode.naming import OpenCodeToolNaming


@pytest.mark.parametrize(
    ("server", "tool", "name"),
    [
        ("secrets", "secret_lookup", "mcp__secrets__secret_lookup"),
        ("host.tools", "read_file", "mcp__host_tools__read_file"),
        ("host-tools", "read.file", "mcp__host-tools__read_file"),
    ],
)
def test_claude_code_names(server: str, tool: str, name: str) -> None:
    assert ClaudeCodeToolNaming().declared_names(server, tool) == [
        BridgedToolName(name)
    ]


@pytest.mark.parametrize(
    ("server", "tool", "name"),
    [
        ("secrets", "secret_lookup", "secrets_secret_lookup"),
        ("host.tools", "read_file", "host_tools_read_file"),
        ("a.b", "c", "a_b_c"),
    ],
)
def test_opencode_names(server: str, tool: str, name: str) -> None:
    assert OpenCodeToolNaming().declared_names(server, tool) == [BridgedToolName(name)]


@pytest.mark.parametrize(
    ("server", "tool", "name"),
    [
        ("secrets", "secret_lookup", "mcp_secrets_secret_lookup"),
        ("host.tools", "read_file", "mcp_host.tools_read_file"),
        ("mcp_host", "read_file", "mcp_host_read_file"),
        # mcp_ + 60 s + _read_file is 74 characters: the first 30 and the last 30 survive
        ("s" * 60, "read_file", "mcp_" + "s" * 26 + "..." + "s" * 20 + "_read_file"),
    ],
)
def test_gemini_cli_names(server: str, tool: str, name: str) -> None:
    assert GeminiCliToolNaming().declared_names(server, tool) == [BridgedToolName(name)]


@pytest.mark.parametrize(
    ("server", "tool", "name"),
    [
        ("secrets", "secret_lookup", "mcp__secrets__secret_lookup"),
        ("my-server.v2", "read.file", "mcp__my-server_v2__read_file"),
        ("a__b", "c__d", "mcp__a_b__c_d"),
        ("srv", "x" * 80, "mcp__srv__" + "x" * 45 + "_245dc304"),
        (
            "inspect-bridge-tools-server",
            "very_long_tool_name_for_testing_hash_truncation",
            "mcp__inspect-bridge-tools-server__very_long_tool_name__-532a7d1b",
        ),
    ],
)
def test_kimi_code_names(server: str, tool: str, name: str) -> None:
    assert KimiCodeToolNaming().declared_names(server, tool) == [BridgedToolName(name)]


def test_codex_cli_declares_the_bare_name_in_a_namespace() -> None:
    names = CodexCliToolNaming(caps=(128,)).declared_names("secrets", "secret_lookup")
    assert BridgedToolName("secret_lookup", "mcp__secrets") in names
    # releases before rust-v0.150 sent the flat form instead of a namespace
    assert BridgedToolName("mcp__secrets__secret_lookup") in names


def test_codex_cli_sanitizes_both_parts() -> None:
    names = CodexCliToolNaming(caps=(128,)).declared_names("host-tools", "read.file")
    assert BridgedToolName("read_file", "mcp__host_tools") in names


@pytest.mark.parametrize(
    ("server", "tool", "cap", "name"),
    [
        # mcp__ + 60 s (65) + __ + 61 t is exactly 128 bytes: the name is untouched
        ("s" * 60, "t" * 61, 128, "t" * 61),
        # one byte over: the name is cut and given the identity hash suffix
        ("s" * 60, "t" * 62, 128, "t" * 48 + "_9f2b038d5e15"),
        ("s" * 80, "t" * 60, 128, "t" * 28 + "_ad31d8f69652"),
        # a namespace that leaves no room for the suffix is cut instead
        ("s" * 120, "t" * 5, 128, "_797a814e6ef0"),
        # 77 bytes fits the 128-byte cap but not the 64-byte cap of rust-v0.149
        ("s" * 20, "t" * 50, 64, "t" * 24 + "_01f03f38fbf3"),
        ("s" * 20, "t" * 50, 128, "t" * 50),
    ],
)
def test_codex_cli_cuts_and_hashes_long_names(
    server: str, tool: str, cap: int, name: str
) -> None:
    names = CodexCliToolNaming(caps=(cap,)).declared_names(server, tool)
    assert any(n.name == name and n.namespace for n in names)


def test_codex_cli_declares_every_known_cap_by_default() -> None:
    names = CodexCliToolNaming().declared_names("s" * 20, "t" * 50)
    assert {n.name for n in names if n.namespace} == {
        "t" * 50,
        "t" * 24 + "_01f03f38fbf3",
    }


def test_antigravity_declares_no_per_tool_functions() -> None:
    assert AntigravityToolNaming().declared_names("secrets", "secret_lookup") == []


def test_antigravity_dispatches_call_mcp_tool() -> None:
    call = ToolCall(
        id="1",
        function="call_mcp_tool",
        arguments={
            "ServerName": "secrets",
            "ToolName": "secret_lookup",
            "Arguments": {"key": "alpha"},
        },
    )
    assert AntigravityToolNaming().dispatched_call(call) == BridgedToolCall(
        "secrets", "secret_lookup", {"key": "alpha"}
    )


@pytest.mark.parametrize(
    "call",
    [
        ToolCall(id="1", function="view_file", arguments={"AbsolutePath": "/x"}),
        ToolCall(id="2", function="call_mcp_tool", arguments={"ServerName": "s"}),
        ToolCall(
            id="3",
            function="call_mcp_tool",
            arguments={"ServerName": "s", "ToolName": "t", "Arguments": "{}"},
        ),
    ],
    ids=["other-tool", "incomplete", "arguments-not-an-object"],
)
def test_antigravity_ignores_calls_that_are_not_dispatches(call: ToolCall) -> None:
    assert AntigravityToolNaming().dispatched_call(call) is None
