"""Fallback Codex model catalog snapshot.

Used by ``codex_models_catalog`` when the version-matched ``models.json`` can't be
fetched (offline, rate-limited ``raw.githubusercontent.com``, or a pre-
``models-manager`` Codex). We only consult the catalog to *decide* the ``--model``
slug — Codex supplies the actual prompt/tools from its own bundled catalog — so a
trimmed snapshot of the fields the resolver reads
(``slug``/``priority``/``apply_patch_tool_type``/``supports_search_tool``) is
sufficient.

Snapshot source: ``openai/codex`` ``codex-rs/models-manager/models.json``
(``rust-v0.145.0``, July 2026). Refresh when bumping the default Codex version;
the live fetch keeps this exact when ``raw.githubusercontent.com`` is reachable.
"""

from typing import Any

BUNDLED_CODEX_CATALOG: dict[str, Any] = {
    "models": [
        {
            "slug": "gpt-5.6-sol",
            "priority": 1,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
        {
            "slug": "gpt-5.6-terra",
            "priority": 2,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
        {
            "slug": "gpt-5.6-luna",
            "priority": 3,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
        {
            "slug": "gpt-5.5",
            "priority": 7,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
        {
            "slug": "gpt-5.4",
            "priority": 16,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
        {
            "slug": "gpt-5.4-mini",
            "priority": 23,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
        {
            "slug": "gpt-5.2",
            "priority": 29,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
        {
            "slug": "codex-auto-review",
            "priority": 43,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },
    ]
}
