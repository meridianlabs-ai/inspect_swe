"""How OpenCode names bridged MCP tools to its model."""

import re

from inspect_ai.agent import BridgedToolName, BridgedToolNaming

_INVALID = re.compile(r"[^a-zA-Z0-9_-]")


class OpenCodeToolNaming(BridgedToolNaming):
    """OpenCode declares an MCP tool as ``<server>_<tool>``.

    Reproduces ``McpCatalog.toolName``: characters outside ``[A-Za-z0-9_-]`` in
    either part become ``_``.
    """

    def declared_names(self, server: str, tool: str) -> list[BridgedToolName]:
        return [
            BridgedToolName(f"{_INVALID.sub('_', server)}_{_INVALID.sub('_', tool)}")
        ]
