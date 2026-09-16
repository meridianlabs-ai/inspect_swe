"""How Codex CLI names bridged MCP tools to its model."""

import hashlib
import re
from collections.abc import Sequence

from inspect_ai.agent import BridgedToolName, BridgedToolNaming

_INVALID = re.compile(r"[^a-zA-Z0-9_]")
_SEPARATOR = "__"
_HASH_LENGTH = 12

CODEX_NAME_CAPS: tuple[int, ...] = (64, 128)
"""Byte caps on namespace + separator + name across Codex releases.

64 up to rust-v0.149 (still embedded in codex-acp), 128 from rust-v0.150.
"""


class CodexCliToolNaming(BridgedToolNaming):
    """Codex CLI declares the sanitized tool name inside an ``mcp__<server>`` namespace.

    Reproduces ``normalize_tools_for_model`` in ``codex-rs/codex-mcp``: characters
    outside ``[A-Za-z0-9_]`` become ``_``; when namespace, separator and name
    exceed the cap, the name is cut to fit and given a ``_<12 hex>`` suffix, the
    SHA-1 of the tool's identity (server, namespace, connector id, name, name;
    for an MCP server the raw server name, empty, and the raw tool name), and a
    namespace that leaves no room for the suffix is cut instead. Releases before
    rust-v0.150 sent the flat ``mcp__<server>__<tool>`` instead of a namespace,
    so that form is declared too. The installed Codex version is only known once
    the sandbox is up, so by default the names under every known cap are given;
    a call matching any of them denotes the tool.

    Args:
        caps: Byte caps to produce names for.
    """

    def __init__(self, caps: Sequence[int] = CODEX_NAME_CAPS) -> None:
        self._caps = tuple(caps)

    def declared_names(self, server: str, tool: str) -> list[BridgedToolName]:
        names: list[BridgedToolName] = []
        for cap in self._caps:
            namespace, name = _callable_parts(server, tool, cap)
            names.append(BridgedToolName(name, namespace))
            flat = f"{namespace.rstrip('_')}{_SEPARATOR}{name.lstrip('_')}"
            names.append(BridgedToolName(flat))
        return names


def _callable_parts(server: str, tool: str, cap: int) -> tuple[str, str]:
    namespace = _INVALID.sub("_", server)
    if not namespace.startswith("mcp__"):
        namespace = f"mcp__{namespace}"
    name = _INVALID.sub("_", tool)
    reserved = len(_SEPARATOR)
    if len(namespace) + len(name) + reserved > cap:
        identity = f"{server}\0{server}\0\0{tool}\0{tool}".encode()
        digest = hashlib.sha1(identity, usedforsecurity=False).hexdigest()
        suffix = f"_{digest[:_HASH_LENGTH]}"
        max_name = max(cap - len(namespace) - reserved, 0)
        if max_name >= len(suffix):
            name = name[: max_name - len(suffix)] + suffix
        else:
            namespace = namespace[: cap - len(suffix) - reserved]
            name = suffix
    return namespace, name
