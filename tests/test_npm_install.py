"""Tests for host npm installs of agent CLIs bundled for the sandbox."""

import io
import json
import os
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


def _local_cli_packages(pkgs: Path) -> None:
    """Write a `cli` package tarball with one glibc-only optional dependency."""
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


@pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")
def test_bundle_cached_under_pre_libc_name_when_npm_lacks_libc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On npm before 10.3 (no --libc), cache under the pre-libc name, not "-libc"."""
    pkgs = tmp_path / "pkgs"
    _local_cli_packages(pkgs)
    cache = tmp_path / "cache"
    cache.mkdir()
    real_npm = shutil.which("npm")
    assert real_npm is not None

    # an npm 10.2 on PATH: reports 10.2.0, otherwise runs the real npm
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_npm = fake_bin / "npm"
    fake_npm.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo 10.2.0; exit 0; fi\n'
        f'exec "{real_npm}" "$@"\n'
    )
    fake_npm.chmod(0o755)
    real_path = os.environ["PATH"]

    real_run = subprocess.run
    installs = 0

    def run_npm_with_local_cli(cmd: list[str], cwd: str, **kwargs: Any) -> Any:
        """Point the bundle's `cli` dependency at the local tarball."""
        nonlocal installs
        if cmd[:2] == ["npm", "install"]:
            installs += 1
            package_json = Path(cwd) / "package.json"
            manifest = json.loads(package_json.read_text())
            manifest["dependencies"] = {"cli": f"file:{pkgs / 'cli.tgz'}"}
            package_json.write_text(json.dumps(manifest))
        return real_run(cmd, cwd=cwd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_npm_with_local_cli)
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setattr(node, "package_cache_dir", lambda _: cache)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{real_path}")

    libc_cache = cache / "cli-bundles-1.0.0-linux-x64-libc.tar.gz"
    old_cache = cache / "cli-bundles-1.0.0-linux-x64.tar.gz"

    bundle = node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles")
    assert len(bundle) > 0
    assert not libc_cache.exists()
    assert old_cache.exists()
    assert old_cache.read_bytes() == bundle
    assert installs == 1

    # a second build on the old npm reuses the pre-libc cache
    assert node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles") == bundle
    assert installs == 1

    # the real npm: a libc-aware npm builds and caches the "-libc" bundle
    monkeypatch.setenv("PATH", real_path)
    real_version = node._parse_npm_version(
        real_run([real_npm, "--version"], capture_output=True, text=True).stdout
    )
    if real_version is None or real_version < (10, 3, 0):
        pytest.skip(f"real npm {real_version} is older than 10.3")
    bundle2 = node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles")
    assert installs == 2
    assert libc_cache.read_bytes() == bundle2


@pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")
def test_bundle_returned_when_npm_version_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed `npm --version` keeps the bundle but does not cache it as "-libc"."""
    pkgs = tmp_path / "pkgs"
    _local_cli_packages(pkgs)
    cache = tmp_path / "cache"
    cache.mkdir()
    real_run = subprocess.run

    def run_npm_with_failing_probe(cmd: list[str], cwd: str, **kwargs: Any) -> Any:
        if cmd[:2] == ["npm", "--version"]:
            raise subprocess.TimeoutExpired(cmd, 30)
        package_json = Path(cwd) / "package.json"
        manifest = json.loads(package_json.read_text())
        manifest["dependencies"] = {"cli": f"file:{pkgs / 'cli.tgz'}"}
        package_json.write_text(json.dumps(manifest))
        return real_run(cmd, cwd=cwd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_npm_with_failing_probe)
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setattr(node, "package_cache_dir", lambda _: cache)

    bundle = node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles")
    with tarfile.open(fileobj=io.BytesIO(bundle)) as tar:
        assert "node_modules/cli/package.json" in tar.getnames()
    assert not (cache / "cli-bundles-1.0.0-linux-x64-libc.tar.gz").exists()
