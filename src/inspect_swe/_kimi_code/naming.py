"""How Kimi Code names bridged MCP tools to its model."""

import re

from inspect_ai.agent import BridgedToolName, BridgedToolNaming

_INVALID = re.compile(r"[^a-zA-Z0-9_-]")
_UNDERSCORES = re.compile(r"_+")
_MAX_LENGTH = 64


class KimiCodeToolNaming(BridgedToolNaming):
    """Kimi Code declares an MCP tool as ``mcp__<server>__<tool>``.

    Reproduces ``qualifyMcpToolName`` in ``@moonshot-ai/kimi-code``: characters
    outside ``[A-Za-z0-9_-]`` in either part become ``_`` and runs of ``_``
    collapse; a name over 64 characters is cut and given ``_`` plus the FNV-1a
    hash of the full name (``stableHash8``).
    """

    def declared_names(self, server: str, tool: str) -> list[BridgedToolName]:
        name = f"mcp__{_part(server)}__{_part(tool)}"
        if len(name) > _MAX_LENGTH:
            digest = _stable_hash8(name)
            name = f"{name[: _MAX_LENGTH - len(digest) - 1]}_{digest}"
        return [BridgedToolName(name)]


def _part(value: str) -> str:
    return _UNDERSCORES.sub("_", _INVALID.sub("_", value))


def _stable_hash8(value: str) -> str:
    """Kimi Code's ``stableHash8``: 32-bit FNV-1a in JavaScript integer arithmetic.

    ``Math.imul`` yields a signed 32-bit product and ``toString(16)`` renders a
    negative one with a leading ``-``, so the digest is 8 hex digits, or ``-`` and
    8; the name is ASCII by then, so code points are single code units.
    """
    digest = 0x811C9DC5
    for char in value:
        digest = ((digest ^ ord(char)) * 0x01000193) & 0xFFFFFFFF
    if digest & 0x80000000:
        return f"-{(1 << 32) - digest:x}".rjust(8, "0")
    return f"{digest:x}".rjust(8, "0")
