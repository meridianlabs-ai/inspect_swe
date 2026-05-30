"""Unit tests for the version-matched Codex catalog fetch/cache helper."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
from inspect_swe._codex_cli import agentbinary

_CATALOG_JSON = json.dumps({"models": [{"slug": "gpt-5.5", "priority": 0}]})


def test_codex_models_catalog_none_version_returns_none() -> None:
    assert anyio.run(agentbinary.codex_models_catalog, None) is None


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


def test_codex_models_catalog_returns_none_on_fetch_error(tmp_path: Path) -> None:
    cache_file = tmp_path / "codex-9.9.9-models.json"
    with (
        patch.object(agentbinary, "_cached_catalog_path", return_value=cache_file),
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(side_effect=RuntimeError("404")),
        ),
    ):
        assert anyio.run(agentbinary.codex_models_catalog, "9.9.9") is None
        assert not cache_file.exists()
