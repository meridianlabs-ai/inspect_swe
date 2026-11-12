"""Tests for the release cooldown and script policy of host npm installs."""

import json
import shutil
import subprocess
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import pytest
from inspect_swe._gemini_cli import agentbinary as gemini_agentbinary
from inspect_swe._util import node, versioncache

from tests.conftest import skip_if_github_action

_CUTOFF = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _days_before_cutoff(days: float) -> str:
    published = _CUTOFF - timedelta(days=days)
    return published.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _resolve(packument: dict[str, Any]) -> tuple[str, list[str]]:
    """Run resolve_npm_package_version against a stubbed `npm view`."""
    run = MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(packument), stderr=""
        )
    )
    with (
        patch("subprocess.run", run),
        patch("shutil.which", return_value="/usr/bin/npm"),
        patch.object(node, "npm_cooldown_cutoff", return_value=_CUTOFF),
    ):
        version = node.resolve_npm_package_version("pkg")
    return version, run.call_args.args[0]


def test_resolution_uses_latest_when_past_cooldown() -> None:
    version, cmd = _resolve(
        {
            "dist-tags": {"latest": "1.2.0"},
            "versions": ["1.1.0", "1.2.0", "1.3.0-rc.0"],
            "time": {
                "created": _days_before_cutoff(30),
                "1.1.0": _days_before_cutoff(30),
                # exactly at the cutoff counts as past it, as with npm's --before
                "1.2.0": _days_before_cutoff(0),
                "1.3.0-rc.0": _days_before_cutoff(0.5),
            },
        }
    )
    assert version == "1.2.0"
    assert cmd == ["npm", "view", "pkg", "time", "versions", "dist-tags", "--json"]


def test_resolution_falls_back_below_latest_inside_cooldown() -> None:
    version, _ = _resolve(
        {
            "dist-tags": {"latest": "1.12.0"},
            "versions": [
                "1.0.0",
                "1.9.0",
                "1.9.5",
                "1.10.0",
                "1.11.0-rc.0",
                "1.12.0",
                "2.0.0",
            ],
            "time": {
                "created": _days_before_cutoff(30),
                "modified": _days_before_cutoff(-1),
                "1.0.0": _days_before_cutoff(30),
                "1.9.0": _days_before_cutoff(12),
                "1.10.0": _days_before_cutoff(9),
                # a patch to an older line, published after 1.10.0
                "1.9.5": _days_before_cutoff(2),
                # unpublished: still in `time`, gone from `versions`
                "1.11.0": _days_before_cutoff(3),
                "1.11.0-rc.0": _days_before_cutoff(4),
                # above the latest dist-tag
                "2.0.0": _days_before_cutoff(20),
                # inside the cooldown
                "1.12.0": _days_before_cutoff(-1),
            },
        }
    )
    assert version == "1.10.0"


def test_resolution_fails_when_no_release_is_past_cooldown() -> None:
    with pytest.raises(RuntimeError, match="7-day cooldown"):
        _resolve(
            {
                "dist-tags": {"latest": "1.0.0"},
                "versions": ["1.0.0"],
                "time": {"1.0.0": _days_before_cutoff(-1)},
            }
        )


@skip_if_github_action
@pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")
def test_resolution_skips_unpublished_releases() -> None:
    """Live registry check: @ctrl/tinycolor 4.1.1 and 4.1.2 were unpublished.

    Their publish times stay in the registry's `time` map, so a resolver that
    reads only `time` picks 4.1.2, which then fails to install.
    """
    cutoff = datetime(2025, 9, 16, tzinfo=timezone.utc)
    with patch.object(node, "npm_cooldown_cutoff", return_value=cutoff):
        assert node.resolve_npm_package_version("@ctrl/tinycolor") == "4.1.0"


class FakeNpm:
    def __init__(self) -> None:
        self.installs: list[tuple[list[str], dict[str, str]]] = []

    def run(self, cmd: list[str], cwd: str, **kwargs: Any) -> Any:
        dependencies = json.loads((Path(cwd) / "package.json").read_text())
        self.installs.append((cmd, dependencies["dependencies"]))
        (Path(cwd) / "node_modules").mkdir()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")


@pytest.fixture
def fake_npm(tmp_path: Path) -> Iterator[FakeNpm]:
    """Stub `npm install` (it writes an empty node_modules) and the cache dir."""
    npm = FakeNpm()
    with (
        patch("subprocess.run", npm.run),
        patch("shutil.which", return_value="/usr/bin/npm"),
        patch.object(node, "package_cache_dir", return_value=tmp_path),
        patch.object(node, "npm_cooldown_cutoff", return_value=_CUTOFF),
    ):
        yield npm


def test_bundle_installs_exact_version_with_cooldown_and_no_scripts(
    fake_npm: FakeNpm,
) -> None:
    node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")

    [(cmd, dependencies)] = fake_npm.installs
    assert cmd[:2] == ["npm", "install"]
    assert "--ignore-scripts" in cmd
    assert cmd[cmd.index("--before") + 1] == "2026-01-01T12:00:00+00:00"
    # an exact version, not a tag or range: npm then skips its preference for
    # versions whose `engines` accept the host's Node
    assert dependencies == {"pkg": "1.0.0"}


def test_bundle_cached_before_cooldown_is_rebuilt(
    fake_npm: FakeNpm, tmp_path: Path
) -> None:
    # the cache name used before installs had a cooldown and skipped scripts
    (tmp_path / "pkg-bundles-1.0.0-linux-x64.tar.gz").write_bytes(b"old")

    bundle = node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")

    assert bundle != b"old"
    assert len(fake_npm.installs) == 1


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
