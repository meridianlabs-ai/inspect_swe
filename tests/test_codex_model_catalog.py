"""Unit tests for the Codex real-model -> --model slug mapping.

These are pure and fast (no Docker / network); they exercise the mapping policy
in `inspect_swe._codex_cli.model_catalog`.
"""

from typing import Any

from inspect_swe._codex_cli.model_catalog import (
    latest_openai_slug,
    resolve_codex_model_slug,
)

# A catalog shaped like Codex's models.json (only the fields we use).
CATALOG: dict[str, Any] = {
    "models": [
        {"slug": "gpt-5.5", "priority": 0, "apply_patch_tool_type": "freeform"},
        {"slug": "gpt-5.4", "priority": 2, "apply_patch_tool_type": "freeform"},
        {"slug": "gpt-5.4-mini", "priority": 4, "apply_patch_tool_type": "freeform"},
    ]
}


def test_latest_openai_slug_prefers_highest_priority_non_mini() -> None:
    assert latest_openai_slug(CATALOG) == "gpt-5.5"


def test_latest_openai_slug_none_when_empty() -> None:
    assert latest_openai_slug(None) is None
    assert latest_openai_slug({"models": []}) is None
    assert latest_openai_slug({}) is None


def test_latest_openai_slug_falls_back_to_mini_when_only_option() -> None:
    catalog = {"models": [{"slug": "gpt-5.4-mini", "priority": 4}]}
    assert latest_openai_slug(catalog) == "gpt-5.4-mini"


def test_explicit_override_is_verbatim() -> None:
    # an explicit model_config wins regardless of catalog/provider
    assert (
        resolve_codex_model_slug(
            "claude-sonnet-4-0", api="anthropic", catalog=CATALOG, override="gpt-5.4"
        )
        == "gpt-5.4"
    )


def test_openai_model_present_in_catalog_uses_real_name() -> None:
    # longest-prefix match: "gpt-5.5-preview" resolves natively via "gpt-5.5"
    assert (
        resolve_codex_model_slug(
            "gpt-5.5-preview", api="openai", catalog=CATALOG, override=None
        )
        == "gpt-5.5-preview"
    )


def test_openai_model_absent_from_catalog_aliases_to_latest() -> None:
    # gpt-5.1 is not in the catalog -> alias to latest (gpt-5.5)
    assert (
        resolve_codex_model_slug(
            "gpt-5.1", api="openai", catalog=CATALOG, override=None
        )
        == "gpt-5.5"
    )


def test_non_openai_model_passes_through_for_generic_fallback() -> None:
    # non-OpenAI -> real (unrecognized) name -> Codex's generic fallback
    assert (
        resolve_codex_model_slug(
            "claude-sonnet-4-0", api="anthropic", catalog=CATALOG, override=None
        )
        == "claude-sonnet-4-0"
    )


def test_openai_model_with_no_catalog_passes_through() -> None:
    # catalog unavailable -> defer to Codex's own bundled catalog (pass the name)
    assert (
        resolve_codex_model_slug("gpt-5.1", api="openai", catalog=None, override=None)
        == "gpt-5.1"
    )
