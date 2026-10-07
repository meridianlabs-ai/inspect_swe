"""Fail fast when Codex code mode would deny every bridged host tool call.

In code mode (`tool_mode = "code_mode_only"`) the model proposes only an
`exec` call, and the JavaScript it writes calls MCP tools itself. The bridge
runs a host tool only for a call the model proposed, so it denies every call
to a server whose `BridgedToolsSpec` keeps `require_proposal=True`, and the
sample runs on without its tools.

The check mirrors the parts of Codex's configuration that decide this
(`openai/codex` at `rust-v0.160.1`): the model catalog entry, the
`features.code_mode_only` flag, `features.code_mode.direct_only_tool_namespaces`
and how Codex names MCP tool namespaces. Where the effective setting cannot be
established (an unknown catalog, a selected config profile, a namespace Codex
would rewrite) it does not raise, so the run behaves as it did before rather
than failing on a guess.
"""

import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import ToolDef
from inspect_ai.util import OutputLimitExceededError, SandboxEnvironment

from .model_catalog import codex_catalog_tool_mode

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Codex's `MAX_TOOL_NAME_LENGTH` and `MCP_TOOL_NAME_DELIMITER` (codex-mcp
# `tools.rs`): a namespace plus tool name longer than this is hashed.
_MAX_TOOL_NAME_LENGTH = 128
_MCP_TOOL_NAME_DELIMITER = "__"

# Feature tables whose boolean and table forms Codex merges (config `merge.rs`).
_STRUCTURED_FEATURES = ("code_mode", "multi_agent_v2", "network_proxy", "sleep_tool")


def check_codex_code_mode_bridged_tools(
    codex_model: str,
    catalog: dict[str, Any] | None,
    config_overrides: Mapping[str, str] | None,
    bridged_tools: Sequence[BridgedToolsSpec] | None,
    other_mcp_servers: Sequence[str] = (),
) -> None:
    """Raise when Codex code mode would make every bridged tool call fail.

    The opt-out is never set on the author's behalf.

    Args:
        codex_model: The `--model` slug Codex runs with.
        catalog: The model catalog the installed Codex reads (the
            `model_catalog_json` file when set, else its release's), or `None`
            when it is unknown.
        config_overrides: The agent's `config_overrides`, which Codex
            receives as `-c key=value` pairs.
        bridged_tools: The agent's bridged tool specs.
        other_mcp_servers: Names of the static MCP servers Codex is given,
            whose namespaces can collide with a bridged server's.
    """
    specs = codex_specs_requiring_proposal(bridged_tools)
    if not specs or catalog is None:
        return

    config = codex_config_overrides_tree(config_overrides)
    if "profile" in config:
        # a selected profile can change any of the settings read below
        return
    features = _table(config.get("features"))

    # Codex reads the feature flag only when the catalog entry sets no mode
    tool_mode = codex_catalog_tool_mode(codex_model, catalog)
    if tool_mode is not None:
        if tool_mode != "code_mode_only":
            return
        cause = 'its catalog entry sets tool_mode = "code_mode_only"'
    elif features.get("code_mode_only") is True:
        cause = "features.code_mode_only is enabled"
    else:
        return

    direct_only = _table(features.get("code_mode")).get("direct_only_tool_namespaces")
    server_names = [
        *(s.name for s in bridged_tools or []),
        *other_mcp_servers,
        *_table(config.get("mcp_servers")),
    ]
    servers: list[str] = []
    for spec in specs:
        if isinstance(direct_only, list) and direct_only:
            namespace = _codex_mcp_namespace(spec, features, server_names)
            if namespace is None or namespace in direct_only:
                # a direct model tool, or a namespace we cannot establish
                continue
        servers.append(spec.name)
    if not servers:
        return

    names = ", ".join(f"'{name}'" for name in servers)
    raise ValueError(
        f"Codex runs model '{codex_model}' in code mode ({cause}): the model "
        "calls MCP tools from the code it writes instead of proposing each "
        "call, so the bridge denies every call to bridged server(s) "
        f"{names}. Set require_proposal=False on the existing "
        "BridgedToolsSpec for each of them. Approval policies then review "
        "only the exec call that runs the code, not the tool calls made "
        "from it."
    )


def codex_specs_requiring_proposal(
    bridged_tools: Sequence[BridgedToolsSpec] | None,
) -> list[BridgedToolsSpec]:
    """The bridged specs whose tools run only for a proposed call."""
    # inspect_ai releases before require_proposal have no proposal check
    return [s for s in bridged_tools or [] if getattr(s, "require_proposal", False)]


def codex_model_catalog_json(config_overrides: Mapping[str, str] | None) -> Any:
    """The `model_catalog_json` Codex is given, or `None` when it is unset.

    When it is set, Codex replaces its own catalog with that file.
    """
    return codex_config_overrides_tree(config_overrides).get("model_catalog_json")


async def read_codex_model_catalog(
    sandbox: SandboxEnvironment, path: Any
) -> dict[str, Any] | None:
    """The catalog file at `path` in the sandbox, or `None` if it is unknown.

    Only an absolute path is read, since Codex resolves a relative one itself.
    A missing file or one that is not a JSON object is also unknown.
    """
    if not isinstance(path, str) or not PurePosixPath(path).is_absolute():
        return None
    try:
        catalog = json.loads(await sandbox.read_file(path))
    except (OSError, UnicodeDecodeError, ValueError, OutputLimitExceededError):
        return None
    return catalog if isinstance(catalog, dict) else None


def codex_config_overrides_tree(
    config_overrides: Mapping[str, str] | None,
) -> dict[str, Any]:
    """The configuration Codex builds from the `-c key=value` pairs.

    Mirrors Codex: the pair splits at its first `=`, the value parses as a TOML
    value (else a string with surrounding quotes trimmed), and each dotted key
    is applied in order (config `overrides.rs`).
    """
    root: dict[str, Any] = {}
    for key, value in (config_overrides or {}).items():
        path, _, raw = f"{key}={value}".partition("=")
        _apply_override(root, path.strip(), _parse_override_value(raw.strip()))
    return root


def _parse_override_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"_x_ = {raw}")["_x_"]
    except tomllib.TOMLDecodeError:
        return raw.strip().strip("\"'")


def _is_structured_feature_path(path: Sequence[str]) -> bool:
    if len(path) == 4 and path[0] == "profiles":
        path = path[2:]
    return len(path) == 2 and path[0] == "features" and path[1] in _STRUCTURED_FEATURES


def _apply_override(root: dict[str, Any], key: str, value: Any) -> None:
    segments = key.split(".")
    current = root
    for index, segment in enumerate(segments[:-1]):
        child = current.setdefault(segment, {})
        if isinstance(child, bool) and _is_structured_feature_path(
            segments[: index + 1]
        ):
            child = {"enabled": child}
        elif not isinstance(child, dict):
            child = {}
        current[segment] = child
        current = child

    last = segments[-1]
    existing = current.get(last)
    if existing is not None and _is_structured_feature_path(segments):
        if isinstance(existing, dict) and isinstance(value, bool):
            existing["enabled"] = value
            return
        if isinstance(existing, bool) and isinstance(value, dict):
            current[last] = {"enabled": existing}
            _merge(current[last], value, segments)
            return
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge(existing, value, segments)
            return
    current[last] = value


def _merge(base: dict[str, Any], overlay: dict[str, Any], path: list[str]) -> None:
    """Deep-merge `overlay` into `base` the way Codex merges config tables."""
    for key, value in overlay.items():
        child_path = [*path, key]
        existing = base.get(key)
        if _is_structured_feature_path(child_path):
            if isinstance(existing, bool) and isinstance(value, dict):
                existing = base[key] = {"enabled": existing}
            elif isinstance(existing, dict) and isinstance(value, bool):
                existing["enabled"] = value
                continue
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge(existing, value, child_path)
        else:
            base[key] = value


def _codex_mcp_namespace(
    spec: BridgedToolsSpec, features: dict[str, Any], server_names: Sequence[str]
) -> str | None:
    """The namespace Codex gives `spec`'s tools, or `None` if it rewrites it.

    Codex replaces characters outside `[A-Za-z0-9_]` with `_` and adds an
    `mcp__` prefix, which `features.non_prefixed_mcp_tool_names` drops for
    every server or, with `server_names`, for those servers. It hashes a
    namespace that collides with another server's or that makes a tool name
    too long; those return `None`.
    """
    non_prefixed = features.get("non_prefixed_mcp_tool_names")
    enabled = non_prefixed is True or _table(non_prefixed).get("enabled") is True
    unprefixed_servers = _table(non_prefixed).get("server_names") if enabled else None
    prefix = not enabled or unprefixed_servers is not None

    def namespace(server: str) -> str:
        sanitized = _sanitize(server)
        unprefixed = isinstance(unprefixed_servers, list) and (
            server in unprefixed_servers
        )
        if not prefix or unprefixed or sanitized.startswith("mcp__"):
            return sanitized
        return f"mcp__{sanitized}"

    result = namespace(spec.name)
    if any(namespace(s) == result for s in server_names if s != spec.name):
        return None
    for tool in spec.tools:
        tool_name = _sanitize(ToolDef(tool).name)
        if len(result) + len(tool_name) + len(_MCP_TOOL_NAME_DELIMITER) > (
            _MAX_TOOL_NAME_LENGTH
        ):
            return None
    return result


def _sanitize(name: str) -> str:
    """Codex's `sanitize_responses_api_tool_name`."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name) or "_"


def _table(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
