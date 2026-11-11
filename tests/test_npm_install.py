"""Tests for the release cooldown and script policy of host npm installs."""

import json
import shutil
import subprocess
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import pytest
from inspect_swe._gemini_cli import agentbinary as gemini_agentbinary
from inspect_swe._util import node, versioncache

from tests.conftest import skip_if_github_action

_CUTOFF = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def fake_npm(tmp_path: Path) -> Iterator[list[list[str]]]:
    """Stub npm (`install` writes an empty node_modules) and the cache dir."""
    calls: list[list[str]] = []

    def run(cmd: list[str], cwd: str | None = None, **kwargs: Any) -> Any:
        calls.append(cmd)
        if cmd[1] == "pack":
            stdout = json.dumps([{"name": "pkg", "version": "1.2.0"}])
        else:
            (Path(str(cwd)) / "node_modules").mkdir()
            stdout = ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout)

    with (
        patch("subprocess.run", run),
        patch("shutil.which", return_value="/usr/bin/npm"),
        patch.object(node, "package_cache_dir", return_value=tmp_path),
        patch.object(node, "npm_cooldown_cutoff", return_value=_CUTOFF),
    ):
        yield calls


def test_resolution_asks_npm_for_latest_before_cutoff(
    fake_npm: list[list[str]],
) -> None:
    assert node.resolve_npm_package_version("pkg") == "1.2.0"
    [cmd] = fake_npm
    assert cmd[:3] == ["npm", "pack", "pkg@latest"]
    assert "--dry-run" in cmd
    assert cmd[cmd.index("--before") + 1] == "2026-01-01T12:00:00+00:00"


def test_resolution_failure_names_the_cooldown() -> None:
    failed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="npm error code ETARGET"
    )
    with (
        patch("subprocess.run", MagicMock(return_value=failed)),
        patch("shutil.which", return_value="/usr/bin/npm"),
    ):
        with pytest.raises(RuntimeError, match="7-day cooldown.*ETARGET"):
            node.resolve_npm_package_version("pkg")


@skip_if_github_action
@pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")
def test_resolution_skips_unpublished_releases() -> None:
    """Live registry check: @ctrl/tinycolor 4.1.1 and 4.1.2 were unpublished.

    Their publish times stay in the registry's `time` map, so a resolver that
    reads `time` picks 4.1.2, which then fails to install.
    """
    cutoff = datetime(2025, 9, 16, tzinfo=timezone.utc)
    with patch.object(node, "npm_cooldown_cutoff", return_value=cutoff):
        assert node.resolve_npm_package_version("@ctrl/tinycolor") == "4.1.0"


def test_bundle_install_applies_cooldown_and_skips_scripts(
    fake_npm: list[list[str]],
) -> None:
    node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")

    [cmd] = fake_npm
    assert cmd[:2] == ["npm", "install"]
    assert "--ignore-scripts" in cmd
    assert cmd[cmd.index("--before") + 1] == "2026-01-01T12:00:00+00:00"


def test_bundle_cached_before_cooldown_is_rebuilt(
    fake_npm: list[list[str]], tmp_path: Path
) -> None:
    # the cache name used before installs had a cooldown and skipped scripts
    (tmp_path / "pkg-bundles-1.0.0-linux-x64.tar.gz").write_bytes(b"old")

    bundle = node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")

    assert bundle != b"old"
    assert len(fake_npm) == 1


@pytest.fixture
def clear_version_cache() -> Iterator[None]:
    versioncache._resolved_versions.clear()
    versioncache._failed_resolutions.clear()
    yield
    versioncache._resolved_versions.clear()
    versioncache._failed_resolutions.clear()


@pytest.mark.usefixtures("clear_version_cache")
def test_gemini_latest_resolves_through_npm_cooldown() -> None:
    resolve = MagicMock(return_value="1.2.0")
    with patch.object(gemini_agentbinary, "resolve_npm_package_version", resolve):
        version = anyio.run(gemini_agentbinary.resolve_gemini_version, "latest")
    assert version == "1.2.0"
    resolve.assert_called_once_with("@google/gemini-cli")
