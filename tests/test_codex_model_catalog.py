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


def _slug(
    model_name: str,
    *,
    api: str | None = "openai",
    catalog: dict[str, Any] | None = CATALOG,
    override: str | None = None,
    known_to_inspect: bool = True,
) -> str:
    return resolve_codex_model_slug(
        model_name,
        api=api,
        catalog=catalog,
        override=override,
        known_to_inspect=known_to_inspect,
    ).slug


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
    result = resolve_codex_model_slug(
        "claude-sonnet-4-0",
        api="anthropic",
        catalog=CATALOG,
        override="gpt-5.4",
        known_to_inspect=True,
    )
    assert result.slug == "gpt-5.4"
    assert "override" in result.reason


def test_openai_model_present_in_catalog_uses_real_name() -> None:
    # longest-prefix match: "gpt-5.5-preview" resolves natively via "gpt-5.5"
    result = resolve_codex_model_slug(
        "gpt-5.5-preview",
        api="openai",
        catalog=CATALOG,
        override=None,
        known_to_inspect=True,
    )
    assert result.slug == "gpt-5.5-preview"
    assert "matches Codex catalog" in result.reason


def test_pre_tool_search_models_pass_through_to_generic_fallback() -> None:
    # tool_search is supported only on gpt-5.4+. Earlier models -- including
    # gpt-5, base gpt-5.1, and sub-5.4 -codex variants (e.g. gpt-5.1-codex) --
    # must pass through to Codex's generic fallback rather than alias up to
    # gpt-5.5 (which would 400 on tool_search).
    result = resolve_codex_model_slug(
        "gpt-5", api="openai", catalog=CATALOG, override=None, known_to_inspect=True
    )
    assert result.slug == "gpt-5"
    assert "predates tool_search" in result.reason

    assert _slug("gpt-5.1") == "gpt-5.1"
    assert _slug("gpt-5.1-codex") == "gpt-5.1-codex"
    assert _slug("gpt-5-codex") == "gpt-5-codex"
    # older families and non-gpt models likewise
    assert _slug("gpt-4.1") == "gpt-4.1"
    assert _slug("o3") == "o3"


def test_tool_search_capable_absent_from_catalog_aliases_to_latest() -> None:
    # gpt-5.4+ supports tool_search -> alias to latest for full tooling. (5.4/5.5
    # are in CATALOG and match natively; any 5.4+ name absent from it is 5.6+.)
    for known in (True, False):
        assert _slug("gpt-5.6", known_to_inspect=known) == "gpt-5.5"
        assert _slug("gpt-6", known_to_inspect=known) == "gpt-5.5"


def test_unrecognized_openai_codename_aliases_to_latest() -> None:
    # a name Inspect doesn't recognize (and isn't version-comparable) is likely a
    # pre-deployment codename -> alias to latest for full tooling.
    assert _slug("frontier-x", known_to_inspect=False) == "gpt-5.5"


def test_non_openai_model_passes_through_for_generic_fallback() -> None:
    # non-OpenAI -> real (unrecognized) name -> Codex's generic fallback
    assert _slug("claude-sonnet-4-0", api="anthropic") == "claude-sonnet-4-0"


def test_openai_model_with_no_catalog_passes_through() -> None:
    # catalog unavailable -> defer to Codex's own bundled catalog (pass the name).
    # With no catalog there's no "latest" to alias to, so even a tool-capable or
    # unrecognized model passes through.
    assert _slug("gpt-5.1-codex", catalog=None) == "gpt-5.1-codex"
    assert _slug("frontier-x", catalog=None, known_to_inspect=False) == "frontier-x"
