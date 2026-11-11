"""Tests for the release cooldown and script policy of host npm installs."""

import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from inspect_swe._gemini_cli.agentbinary import resolve_gemini_version
from inspect_swe._util import node

from tests.conftest import skip_if_github_action

_CUTOFF = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


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


def test_bundle_install_applies_cooldown_and_skips_scripts(
    fake_npm: FakeNpm,
) -> None:
    node.create_npm_bundle("pkg", "latest", "linux-x64", "pkg-bundles")

    [(cmd, dependencies)] = fake_npm.installs
    assert cmd[:2] == ["npm", "install"]
    assert "--ignore-scripts" in cmd
    assert cmd[cmd.index("--before") + 1] == "2026-01-01T12:00:00+00:00"
    assert dependencies == {"pkg": "latest"}


def test_bundle_cached_before_cooldown_is_rebuilt(
    fake_npm: FakeNpm, tmp_path: Path
) -> None:
    # the cache name used before installs had a cooldown and skipped scripts
    (tmp_path / "pkg-bundles-1.0.0-linux-x64.tar.gz").write_bytes(b"old")

    bundle = node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")

    assert bundle != b"old"
    assert len(fake_npm.installs) == 1


@pytest.mark.parametrize(
    "version,age_hours,rebuilt",
    [("latest", 1, False), ("latest", 25, True), ("1.0.0", 25 * 30, False)],
)
def test_only_day_old_latest_bundles_are_rebuilt(
    fake_npm: FakeNpm, tmp_path: Path, version: str, age_hours: int, rebuilt: bool
) -> None:
    cached = tmp_path / f"pkg-bundles-{version}-linux-x64-noscripts.tar.gz"
    cached.write_bytes(b"cached")
    mtime = time.time() - age_hours * 3600
    os.utime(cached, (mtime, mtime))

    bundle = node.create_npm_bundle("pkg", version, "linux-x64", "pkg-bundles")

    assert (bundle != b"cached") is rebuilt
    assert len(fake_npm.installs) == int(rebuilt)


@skip_if_github_action
@pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")
def test_latest_skips_releases_after_cutoff_and_unpublished(tmp_path: Path) -> None:
    """Live registry check with @ctrl/tinycolor.

    4.1.1 and 4.1.2 were published and then unpublished in September 2025;
    as of 2025-09-16 `latest` must resolve to 4.1.0.
    """
    cutoff = datetime(2025, 9, 16, tzinfo=timezone.utc)
    with (
        patch.object(node, "package_cache_dir", return_value=tmp_path),
        patch.object(node, "npm_cooldown_cutoff", return_value=cutoff),
    ):
        bundle = node.create_npm_bundle(
            "@ctrl/tinycolor", "latest", "linux-x64", "tinycolor-bundles"
        )

    with tarfile.open(fileobj=io.BytesIO(bundle)) as tar:
        member = tar.extractfile("node_modules/@ctrl/tinycolor/package.json")
        assert member is not None
        assert json.load(member)["version"] == "4.1.0"


@pytest.mark.parametrize("alias", ["auto", "sandbox", "stable", "latest"])
def test_gemini_version_aliases_install_npm_latest(alias: str) -> None:
    assert resolve_gemini_version(alias) == "latest"


def test_gemini_explicit_version_is_kept() -> None:
    assert resolve_gemini_version("0.60.0") == "0.60.0"
