"""Fail fast when Codex code mode would deny every bridged host tool call.

In code mode (`tool_mode = "code_mode_only"`) the model proposes only an
`exec` call, and the JavaScript it writes calls MCP tools itself. The bridge
runs a host tool only for a call the model proposed, so it denies every call
to a server whose `BridgedToolsSpec` keeps `require_proposal=True`, and the
sample runs on without its tools.

Only the plain case is checked: the installed release's own catalog puts the
`--model` slug in code mode and the caller has not configured Codex's tool or
catalog settings. Anything else runs as it would without the check.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from inspect_ai.agent import BridgedToolsSpec

from .model_catalog import codex_catalog_tool_mode

# config_overrides keys that can change the tool mode or which tools are
# direct: a caller who sets any of them has configured Codex deliberately
_SKIP_KEYS = ("features", "model_catalog_json", "profile")


def check_codex_code_mode_bridged_tools(
    codex_model: str,
    release_catalog: dict[str, Any] | None,
    config_overrides: Mapping[str, str] | None,
    bridged_tools: Sequence[BridgedToolsSpec] | None,
) -> None:
    """Raise when Codex code mode would make every bridged tool call fail.

    The opt-out is never set on the author's behalf.

    Args:
        codex_model: The `--model` slug Codex runs with.
        release_catalog: The `models.json` of the installed Codex release, or
            `None` when it is unknown (the check is then skipped).
        config_overrides: The agent's `config_overrides`.
        bridged_tools: The agent's bridged tool specs.
    """
    if release_catalog is None or any(
        key.strip() == skip or key.strip().startswith(f"{skip}.")
        for key in config_overrides or {}
        for skip in _SKIP_KEYS
    ):
        return
    if codex_catalog_tool_mode(codex_model, release_catalog) != "code_mode_only":
        return

    # inspect_ai releases before require_proposal have no proposal check
    servers = [
        spec.name
        for spec in bridged_tools or []
        if getattr(spec, "require_proposal", False)
    ]
    if not servers:
        return

    names = ", ".join(f"'{name}'" for name in servers)
    raise ValueError(
        f"Codex runs model '{codex_model}' in code mode (its catalog entry sets "
        'tool_mode = "code_mode_only"): the model calls MCP tools from the code '
        "it writes instead of proposing each call, so the bridge denies every "
        f"call to bridged server(s) {names}. Set require_proposal=False on the "
        "existing BridgedToolsSpec for each of them. Approval policies then "
        "review only the exec call that runs the code, not the tool calls made "
        "from it."
    )
