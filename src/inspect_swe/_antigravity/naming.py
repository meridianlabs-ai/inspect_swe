"""How Antigravity names bridged MCP tools to its model."""

from inspect_ai.agent import BridgedToolCall, BridgedToolName, BridgedToolNaming
from inspect_ai.tool import ToolCall

DISPATCHER = "call_mcp_tool"
"""The one function through which Antigravity's harness exposes MCP tools."""


class AntigravityToolNaming(BridgedToolNaming):
    """Antigravity exposes MCP tools through a single dispatcher, not per-tool functions.

    The model calls ``call_mcp_tool(ServerName, ToolName, Arguments)`` (the
    ``google-antigravity`` harness's ``CallMcpToolConverter``); the bridged tool
    and its arguments are taken from that call. No bridged tool is declared under
    its own name, so a call to any other name denotes nothing.
    """

    def declared_names(self, server: str, tool: str) -> list[BridgedToolName]:
        return []

    def dispatched_call(self, call: ToolCall) -> BridgedToolCall | None:
        if call.function != DISPATCHER:
            return None
        server = call.arguments.get("ServerName")
        tool = call.arguments.get("ToolName")
        arguments = call.arguments.get("Arguments")
        if not (
            isinstance(server, str)
            and isinstance(tool, str)
            and isinstance(arguments, dict)
        ):
            return None
        return BridgedToolCall(server, tool, arguments)
