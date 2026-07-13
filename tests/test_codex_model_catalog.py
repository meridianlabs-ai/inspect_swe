"""Unit tests for the Codex real-model -> --model slug mapping.

These are pure and fast (no Docker / network); they exercise the mapping policy
in `inspect_swe._codex_cli.model_catalog`.
"""

from typing import Any

from inspect_swe._codex_cli.model_catalog import (
    _GENERIC_FALLBACK_SLUG,
    is_latest_openai_model,
    is_openai_derived_api,
    latest_openai_slug,
    openai_service_model_name,
    resolve_codex_model_slug,
)

# A catalog shaped like Codex's models.json (only the fields we use). Note that
# real catalogs mark sub-5.4 slugs (e.g. gpt-5.2, gpt-5.3-codex) supports_search_
# tool=True — that reflects Codex's backend, not the public Responses API the
# bridge uses (which 400s on tool_search below gpt-5.4).
CATALOG: dict[str, Any] = {
    "models": [
        {
            "slug": "gpt-5.5",
            "priority": 0,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },  # noqa: E501
        {
            "slug": "gpt-5.4",
            "priority": 2,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },  # noqa: E501
        {
            "slug": "gpt-5.4-mini",
            "priority": 4,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },  # noqa: E501
        {
            "slug": "gpt-5.3-codex",
            "priority": 6,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },  # noqa: E501
        {
            "slug": "gpt-5.2",
            "priority": 10,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": True,
        },  # noqa: E501
        # hypothetical sub-5.4 entry that does NOT bind tool_search (native is safe)
        {
            "slug": "gpt-5.0-lite",
            "priority": 20,
            "apply_patch_tool_type": "freeform",
            "supports_search_tool": False,
        },  # noqa: E501
    ]
}


def _slug(
    model_name: str,
    *,
    api: str | None = "openai",
    catalog: dict[str, Any] | None = CATALOG,
    override: str | None = None,
    is_latest: bool = False,
) -> str:
    return resolve_codex_model_slug(
        model_name,
        api=api,
        catalog=catalog,
        override=override,
        is_latest=is_latest,
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
        is_latest=False,
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
        is_latest=False,
    )
    assert result.slug == "gpt-5.5-preview"
    assert "matches Codex catalog" in result.reason


def test_pre_tool_search_models_pass_through_to_generic_fallback() -> None:
    # tool_search is supported only on gpt-5.4+. Earlier models -- including
    # gpt-5, base gpt-5.1, and sub-5.4 -codex variants (e.g. gpt-5.1-codex) --
    # must pass through to Codex's generic fallback rather than alias up to
    # gpt-5.5 (which would 400 on tool_search).
    result = resolve_codex_model_slug(
        "gpt-5", api="openai", catalog=CATALOG, override=None, is_latest=False
    )
    assert result.slug == "gpt-5"
    assert "predates tool_search" in result.reason

    assert _slug("gpt-5.1") == "gpt-5.1"
    assert _slug("gpt-5.1-codex") == "gpt-5.1-codex"
    assert _slug("gpt-5-codex") == "gpt-5-codex"
    # older families and non-gpt models likewise
    assert _slug("gpt-4.1") == "gpt-4.1"
    assert _slug("o3") == "o3"


def test_catalog_present_sub_boundary_model_forces_generic_fallback() -> None:
    # gpt-5.2 / gpt-5.3-codex ARE in the catalog (marked supports_search_tool=True
    # by Codex's backend) but predate the public-API tool_search boundary (5.4+).
    # Returning the real slug would make Codex emit tool_search -> 400, so we force
    # a generic-fallback slug instead (no apply_patch, no tool_search).
    for name in ("gpt-5.2", "gpt-5.3-codex"):
        result = resolve_codex_model_slug(
            name, api="openai", catalog=CATALOG, override=None, is_latest=False
        )
        assert result.slug == _GENERIC_FALLBACK_SLUG, name
        assert "tool_search" in result.reason
        # the sentinel must not prefix-match any catalog slug (-> Codex generic)
        assert not any(
            _GENERIC_FALLBACK_SLUG.startswith(m["slug"]) for m in CATALOG["models"]
        )


def test_catalog_present_entry_without_search_stays_native() -> None:
    # a sub-5.4 catalog entry that does NOT bind tool_search is safe to use natively
    assert _slug("gpt-5.0-lite") == "gpt-5.0-lite"


def test_catalog_boundary_models_use_native_tools() -> None:
    # gpt-5.4+ are at/above the boundary -> native (tool_search is accepted)
    assert _slug("gpt-5.4") == "gpt-5.4"
    assert _slug("gpt-5.5") == "gpt-5.5"


def test_tool_search_capable_absent_from_catalog_aliases_to_latest() -> None:
    # gpt-5.4+ supports tool_search -> alias to latest for full tooling. (5.4/5.5
    # are in CATALOG and match natively; any 5.4+ name absent from it is 5.6+.)
    # The version signal aliases regardless of the is_latest flag.
    for latest in (True, False):
        assert _slug("gpt-5.6", is_latest=latest) == "gpt-5.5"
        assert _slug("gpt-6", is_latest=latest) == "gpt-5.5"


def test_latest_codename_aliases_to_latest() -> None:
    # a "latest"/codename model (per the provider's is_latest()) isn't version-
    # comparable -> alias to latest for full tooling.
    assert _slug("frontier-x", is_latest=True) == "gpt-5.5"


def test_non_openai_model_passes_through_for_generic_fallback() -> None:
    # non-OpenAI -> real (unrecognized) name -> Codex's generic fallback
    assert _slug("claude-sonnet-4-0", api="anthropic") == "claude-sonnet-4-0"


def test_openai_model_with_no_catalog_passes_through() -> None:
    # catalog unavailable -> defer to Codex's own bundled catalog (pass the name).
    # With no catalog there's no "latest" to alias to, so even a tool-capable or
    # unrecognized model passes through.
    assert _slug("gpt-5.1-codex", catalog=None) == "gpt-5.1-codex"
    assert _slug("frontier-x", catalog=None, is_latest=True) == "frontier-x"


# --- is_openai_derived_api: stand-ins mirror Inspect's provider class names ---


class OpenAIAPI:  # mirrors inspect_ai's concrete OpenAI provider class name
    pass


class _PreDeploymentProvider(OpenAIAPI):  # custom provider subclassing OpenAIAPI
    pass


class OpenAICompatibleAPI:  # sibling base for protocol-compatible providers
    pass


class _OllamaLikeAPI(OpenAICompatibleAPI):
    pass


class _BespokeAPI:  # unrelated custom ModelAPI
    pass


def test_is_openai_derived_api_matches_openai_and_subclasses() -> None:
    assert is_openai_derived_api(OpenAIAPI())
    assert is_openai_derived_api(_PreDeploymentProvider())


def test_is_openai_derived_api_rejects_compatible_and_bespoke() -> None:
    # OpenAI-protocol-compatible (e.g. Ollama/OpenRouter) is NOT "really OpenAI"
    assert not is_openai_derived_api(OpenAICompatibleAPI())
    assert not is_openai_derived_api(_OllamaLikeAPI())
    assert not is_openai_derived_api(_BespokeAPI())


# --- is_latest_openai_model: delegates to the provider's is_latest() ---


class _LatestOpenAIProvider(OpenAIAPI):  # provider flags a codename/frontier model
    def is_latest(self) -> bool:
        return True


class _EstablishedOpenAIProvider(OpenAIAPI):  # recognized model -> not "latest"
    def is_latest(self) -> bool:
        return False


class _CompatibleWithIsLatest(OpenAICompatibleAPI):  # is_latest() but NOT OpenAI
    def is_latest(self) -> bool:
        return True


def test_is_latest_openai_model_reflects_provider() -> None:
    assert is_latest_openai_model(_LatestOpenAIProvider())
    assert not is_latest_openai_model(_EstablishedOpenAIProvider())


def test_is_latest_openai_model_false_when_not_openai_derived() -> None:
    # is_latest() on a non-OpenAI provider must not count
    assert not is_latest_openai_model(_CompatibleWithIsLatest())
    assert not is_latest_openai_model(_BespokeAPI())


def test_is_latest_openai_model_robust_when_method_absent() -> None:
    # OpenAI-derived but without is_latest() (defensive) -> False, not an error
    assert not is_latest_openai_model(OpenAIAPI())


# --- openai_service_model_name: declared identity of OpenAI-derived providers ---


class _OtterProvider(OpenAIAPI):  # custom provider reporting a real model identity
    def service_model_name(self) -> str:
        return "gpt-5.5"


def test_openai_service_model_name_uses_declared_identity() -> None:
    # the otter case: registry name is 'otter' but the provider declares 'gpt-5.5'
    assert openai_service_model_name(_OtterProvider(), "otter") == "gpt-5.5"


def test_openai_service_model_name_falls_back_when_not_openai() -> None:
    # non-OpenAI providers (and bespoke APIs) keep the registry name
    assert openai_service_model_name(_CompatibleWithIsLatest(), "qwen") == "qwen"
    assert openai_service_model_name(_BespokeAPI(), "mystery") == "mystery"


def test_openai_service_model_name_falls_back_when_method_absent() -> None:
    # OpenAI-derived but without service_model_name() (defensive) -> fallback
    assert openai_service_model_name(OpenAIAPI(), "gpt-5") == "gpt-5"
