"""Tests for host npm installs of agent CLIs bundled for the sandbox."""

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest
from inspect_swe._util import node
from inspect_swe._util.sandbox import SandboxPlatform


def _write_npm_tarball(path: Path, package_json: dict[str, object]) -> None:
    """Write an npm package tarball holding only `package.json`."""
    data = json.dumps(package_json).encode()
    info = tarfile.TarInfo("package/package.json")
    info.size = len(data)
    with tarfile.open(path, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))


@pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")
@pytest.mark.parametrize(
    ("platform", "installed", "skipped"),
    [
        ("linux-x64", "native-glibc", "native-musl"),
        ("linux-x64-musl", "native-musl", "native-glibc"),
    ],
)
def test_bundle_installs_optional_packages_for_sandbox_libc(
    platform: SandboxPlatform,
    installed: str,
    skipped: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real npm, offline: optional native packages are picked by the sandbox libc.

    Mirrors `@anthropic-ai/claude-agent-sdk`, whose native binary ships in
    optional `-linux-<cpu>` (glibc) and `-linux-<cpu>-musl` packages. Without
    `--libc`, npm uses the host's libc: on macOS it installs neither, and on
    a glibc Linux host it installs the glibc one for a musl sandbox.
    """
    pkgs = tmp_path / "pkgs"
    pkgs.mkdir()
    for libc in ("glibc", "musl"):
        _write_npm_tarball(
            pkgs / f"native-{libc}.tgz",
            {
                "name": f"native-{libc}",
                "version": "1.0.0",
                "os": ["linux"],
                "cpu": ["x64"],
                "libc": [libc],
            },
        )
    _write_npm_tarball(
        pkgs / "cli.tgz",
        {
            "name": "cli",
            "version": "1.0.0",
            "optionalDependencies": {
                name: f"file:{pkgs / name}.tgz"
                for name in ("native-glibc", "native-musl")
            },
        },
    )
    real_run = subprocess.run

    def run_npm_with_local_cli(cmd: list[str], cwd: str, **kwargs: Any) -> Any:
        """Point the bundle's `cli` dependency at the local tarball."""
        if cmd[:2] == ["npm", "--version"]:
            return real_run(cmd, cwd=cwd, **kwargs)
        package_json = Path(cwd) / "package.json"
        manifest = json.loads(package_json.read_text())
        manifest["dependencies"] = {"cli": f"file:{pkgs / 'cli.tgz'}"}
        package_json.write_text(json.dumps(manifest))
        return real_run(cmd, cwd=cwd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_npm_with_local_cli)
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setattr(node, "package_cache_dir", lambda _: tmp_path)

    bundle = node.create_npm_bundle("cli", "1.0.0", platform, "cli-bundles")

    with tarfile.open(fileobj=io.BytesIO(bundle)) as tar:
        names = tar.getnames()
    assert f"node_modules/{installed}/package.json" in names
    assert f"node_modules/{skipped}/package.json" not in names


def test_bundle_cached_without_libc_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the cache name used before installs passed --libc
    (tmp_path / "pkg-bundles-1.0.0-linux-x64.tar.gz").write_bytes(b"old")
    monkeypatch.setattr(node, "package_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _: None)

    # with the old bundle ignored, the build gets as far as looking for npm
    with pytest.raises(RuntimeError, match="npm is required"):
        node.create_npm_bundle("pkg", "1.0.0", "linux-x64", "pkg-bundles")


def test_parse_npm_version() -> None:
    assert node._parse_npm_version("10.2.0\n") == (10, 2, 0)
    assert node._parse_npm_version("10.3.0") == (10, 3, 0)
    assert node._parse_npm_version("11.14.1\n") == (11, 14, 1)
    assert node._parse_npm_version("") is None
    assert node._parse_npm_version("not-a-version") is None


@pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")
def test_bundle_not_cached_when_npm_lacks_libc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On npm before 10.3, --libc is ignored; do not cache that bundle under the -libc name."""
    pkgs = tmp_path / "pkgs"
    pkgs.mkdir()
    _write_npm_tarball(
        pkgs / "native-glibc.tgz",
        {
            "name": "native-glibc",
            "version": "1.0.0",
            "os": ["linux"],
            "cpu": ["x64"],
            "libc": ["glibc"],
        },
    )
    _write_npm_tarball(
        pkgs / "cli.tgz",
        {
            "name": "cli",
            "version": "1.0.0",
            "optionalDependencies": {
                "native-glibc": f"file:{pkgs / 'native-glibc.tgz'}"
            },
        },
    )
    real_run = subprocess.run

    def run_npm_with_local_cli(cmd: list[str], cwd: str, **kwargs: Any) -> Any:
        if cmd[:2] == ["npm", "--version"]:
            return real_run(cmd, cwd=cwd, **kwargs)
        package_json = Path(cwd) / "package.json"
        manifest = json.loads(package_json.read_text())
        manifest["dependencies"] = {"cli": f"file:{pkgs / 'cli.tgz'}"}
        package_json.write_text(json.dumps(manifest))
        return real_run(cmd, cwd=cwd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_npm_with_local_cli)
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setattr(node, "package_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(node, "_npm_version", lambda: (10, 2, 0))

    bundle = node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles")
    cache_path = tmp_path / "cli-bundles-1.0.0-linux-x64-libc.tar.gz"
    assert not cache_path.exists()
    assert len(bundle) > 0

    monkeypatch.setattr(node, "_npm_version", lambda: (10, 3, 0))
    bundle2 = node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles")
    assert cache_path.exists()
    assert cache_path.read_bytes() == bundle2
