"""Unit tests for the version-matched Codex catalog fetch/cache helper."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_swe._codex_cli import agentbinary
from inspect_swe._codex_cli._bundled_catalog import BUNDLED_CODEX_CATALOG
from inspect_swe._codex_cli.model_catalog import latest_openai_slug
from inspect_swe._util.download import download_text_file

from tests.conftest import skip_if_github_action

_CATALOG_JSON = json.dumps({"models": [{"slug": "gpt-5.5", "priority": 0}]})


def test_codex_models_catalog_none_version_uses_bundled() -> None:
    # no version to fetch against -> fall back to the bundled snapshot
    assert anyio.run(agentbinary.codex_models_catalog, None) is BUNDLED_CODEX_CATALOG


def test_codex_models_catalog_fetches_and_caches(tmp_path: Path) -> None:
    cache_file = tmp_path / "codex-0.50.0-models.json"
    with (
        patch.object(agentbinary, "_cached_catalog_path", return_value=cache_file),
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(return_value=_CATALOG_JSON),
        ) as mock_download,
    ):
        catalog = anyio.run(agentbinary.codex_models_catalog, "0.50.0")
        assert catalog == {"models": [{"slug": "gpt-5.5", "priority": 0}]}
        assert cache_file.exists()
        mock_download.assert_awaited_once()

        # second call hits the cache (no further download)
        with patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(side_effect=AssertionError("should not download")),
        ):
            cached = anyio.run(agentbinary.codex_models_catalog, "0.50.0")
            assert cached == catalog


def test_codex_models_catalog_falls_back_to_bundled_on_fetch_error(
    tmp_path: Path,
) -> None:
    # a failed fetch (offline / rate-limited / pre-models-manager) degrades to the
    # bundled snapshot rather than None, so alignment stays deterministic.
    cache_file = tmp_path / "codex-9.9.9-models.json"
    with (
        patch.object(agentbinary, "_cached_catalog_path", return_value=cache_file),
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(side_effect=RuntimeError("404")),
        ),
    ):
        catalog = anyio.run(agentbinary.codex_models_catalog, "9.9.9")
        assert catalog is BUNDLED_CODEX_CATALOG
        assert not cache_file.exists()


@skip_if_github_action
def test_bundled_catalog_tracks_live_latest() -> None:
    """Drift check for the bundled fallback (``_bundled_catalog.py``).

    The fallback is only consulted when the live catalog can't be fetched, and it
    is used to *decide* the ``--model`` slug — so its "latest" entry must still
    exist in (and ideally match) the live latest Codex catalog. This fetches the
    live catalog and fails when OpenAI ships a new frontier slug, signalling that
    the snapshot needs refreshing. Skips when the catalog can't be fetched
    (offline / rate-limited); github-action-gated so a freshly-shipped model can't
    block unrelated CI.
    """
    try:
        version = anyio.run(agentbinary._fetch_latest_stable_version)
        url = (
            "https://raw.githubusercontent.com/openai/codex/"
            f"rust-v{version}/codex-rs/models-manager/models.json"
        )
        text = anyio.run(download_text_file, url)
    except Exception as ex:  # offline / rate-limited / tag without models.json
        pytest.skip(f"live Codex catalog unavailable: {ex}")

    live = agentbinary.cast_catalog(json.loads(text))
    assert live is not None, "live Codex catalog is not a dict"
    live_slugs = {
        m["slug"]
        for m in live.get("models", [])
        if isinstance(m, dict) and isinstance(m.get("slug"), str)
    }

    bundled_latest = latest_openai_slug(BUNDLED_CODEX_CATALOG)
    live_latest = latest_openai_slug(live)

    assert bundled_latest in live_slugs, (
        f"bundled fallback aliases unknown models to '{bundled_latest}', which is "
        f"absent from the live Codex catalog (rust-v{version}); refresh "
        f"src/inspect_swe/_codex_cli/_bundled_catalog.py"
    )
    assert bundled_latest == live_latest, (
        f"live Codex catalog (rust-v{version}) latest is '{live_latest}' but the "
        f"bundled fallback's latest is '{bundled_latest}'; refresh "
        f"src/inspect_swe/_codex_cli/_bundled_catalog.py"
    )
