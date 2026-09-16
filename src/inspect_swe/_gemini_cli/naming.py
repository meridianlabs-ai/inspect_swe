"""How Gemini CLI names bridged MCP tools to its model."""

import re

from inspect_ai.agent import BridgedToolName, BridgedToolNaming

_INVALID = re.compile(r"[^a-zA-Z0-9_.:-]")
_MAX_LENGTH = 63


class GeminiCliToolNaming(BridgedToolNaming):
    """Gemini CLI declares an MCP tool as ``mcp_<server>_<tool>``.

    Reproduces ``generateValidName``: the ``mcp_`` prefix is not doubled when the
    server name already starts with it, characters outside ``[A-Za-z0-9_.:-]``
    become ``_``, and a name over 63 characters is collapsed to its first and last
    30 around ``...``.
    """

    def declared_names(self, server: str, tool: str) -> list[BridgedToolName]:
        name = f"{server}_{tool}"
        if not name.startswith("mcp_"):
            name = f"mcp_{name}"
        name = _INVALID.sub("_", name)
        if len(name) > _MAX_LENGTH:
            name = f"{name[:30]}...{name[-30:]}"
        return [BridgedToolName(name)]
