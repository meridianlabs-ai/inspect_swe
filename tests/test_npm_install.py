"""Unit tests for the release cooldown and script policy of host npm installs."""

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
import pytest
from inspect_swe._gemini_cli import agentbinary as gemini_agentbinary
from inspect_swe._util import node, versioncache


def _iso(days_ago: float) -> str:
    published = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return published.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _npm_view(packument: dict[str, Any]) -> MagicMock:
    return MagicMock(
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(packument), stderr=""
        )
    )


def test_resolves_newest_stable_release_older_than_cooldown() -> None:
    packument = {
        "dist-tags": {"latest": "1.3.0"},
        "time": {
            "created": _iso(30),
            "modified": _iso(1),
            "1.0.0": _iso(30),
            "1.2.0": _iso(9),
            # a patch to an older line, published after 1.2.0
            "1.1.5": _iso(8),
            "1.3.0-preview.0": _iso(8),
            # above the latest dist-tag
            "2.0.0": _iso(20),
            # inside the cooldown
            "1.3.0": _iso(6),
        },
    }
    run = _npm_view(packument)
    with (
        patch("subprocess.run", run),
        patch("shutil.which", return_value="/usr/bin/npm"),
    ):
        assert node.resolve_npm_package_version("pkg") == "1.2.0"
    assert run.call_args.args[0] == [
        "npm",
        "view",
        "pkg",
        "time",
        "dist-tags",
        "--json",
    ]


def test_resolution_fails_when_no_release_is_past_cooldown() -> None:
    packument = {
        "dist-tags": {"latest": "1.0.0"},
        "time": {"created": _iso(2), "modified": _iso(2), "1.0.0": _iso(2)},
    }
    with (
        patch("subprocess.run", _npm_view(packument)),
        patch("shutil.which", return_value="/usr/bin/npm"),
    ):
        with pytest.raises(RuntimeError, match="7-day cooldown"):
            node.resolve_npm_package_version("pkg")


@pytest.fixture
def fake_npm_install(tmp_path: Path) -> Any:
    """Stub `npm install` (it writes an empty node_modules) and the cache dir."""
    calls: list[list[str]] = []

    def run(cmd: list[str], cwd: str, **kwargs: Any) -> Any:
        calls.append(cmd)
        (Path(cwd) / "node_modules").mkdir()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")

    with (
        patch("subprocess.run", run),
        patch("shutil.which", return_value="/usr/bin/npm"),
        patch.object(node, "package_cache_dir", return_value=tmp_path),
    ):
        yield calls


def test_bundle_install_applies_cooldown_and_skips_scripts(
    fake_npm_install: list[list[str]],
) -> None:
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    with patch.object(node, "npm_cooldown_cutoff", return_value=cutoff):
        node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")

    [cmd] = fake_npm_install
    assert cmd[:2] == ["npm", "install"]
    assert "--ignore-scripts" in cmd
    assert cmd[cmd.index("--before") + 1] == "2026-01-01T12:00:00+00:00"


def test_bundle_cached_before_cooldown_is_rebuilt(
    fake_npm_install: list[list[str]], tmp_path: Path
) -> None:
    # the cache name used before installs had a cooldown and skipped scripts
    (tmp_path / "pkg-bundles-1.0.0-linux-x64.tar.gz").write_bytes(b"old")

    bundle = node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")

    assert bundle != b"old"
    assert len(fake_npm_install) == 1


def test_gemini_latest_resolves_through_npm_cooldown() -> None:
    versioncache._resolved_versions.clear()
    resolve = MagicMock(return_value="1.2.0")
    try:
        with patch.object(gemini_agentbinary, "resolve_npm_package_version", resolve):
            version = anyio.run(gemini_agentbinary.resolve_gemini_version, "latest")
    finally:
        versioncache._resolved_versions.clear()
    assert version == "1.2.0"
    resolve.assert_called_once_with("@google/gemini-cli")
