"""How Claude Code names bridged MCP tools to its model."""

import re

from inspect_ai.agent import BridgedToolName, BridgedToolNaming

_INVALID = re.compile(r"[^a-zA-Z0-9_-]")


class ClaudeCodeToolNaming(BridgedToolNaming):
    """Claude Code declares an MCP tool as ``mcp__<server>__<tool>``.

    Characters outside ``[A-Za-z0-9_-]`` in either part become ``_`` (Claude
    Code 2.1.x; read from the CLI's own name normalizer).
    """

    def declared_names(self, server: str, tool: str) -> list[BridgedToolName]:
        return [
            BridgedToolName(
                f"mcp__{_INVALID.sub('_', server)}__{_INVALID.sub('_', tool)}"
            )
        ]
