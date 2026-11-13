"""Tests for host npm installs of agent CLIs bundled for the sandbox."""

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from inspect_ai.util import SandboxEnvironment
from inspect_swe._gemini_cli import agentbinary as gemini_agentbinary
from inspect_swe._util import node, versioncache
from inspect_swe._util.sandbox import SandboxPlatform
from inspect_swe.acp._agents.claude_code import agentbinary as claude_acp_agentbinary
from inspect_swe.acp._agents.codex_cli import agentbinary as codex_acp_agentbinary

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
        if cmd[:2] == ["npm", "--version"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="11.0.0")
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
    # the cache names used before installs had a cooldown and skipped scripts
    (tmp_path / "pkg-bundles-1.0.0-linux-x64.tar.gz").write_bytes(b"old")
    (tmp_path / "pkg-bundles-1.0.0-linux-x64-libc.tar.gz").write_bytes(b"old")

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


# ---------------------------------------------------------------------------
# Real npm against a local registry
# ---------------------------------------------------------------------------

needs_npm = pytest.mark.skipif(shutil.which("npm") is None, reason="needs npm")


def _ago(days: float) -> str:
    published = datetime.now(timezone.utc) - timedelta(days=days)
    return published.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _tarball(manifest: dict[str, Any]) -> bytes:
    data = json.dumps(manifest).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("package/package.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class Registry:
    """A minimal npm registry serving full packuments and tarballs over HTTP.

    `releases` maps a package name to `(version, days ago, extra manifest
    fields)` tuples; the last one listed is the `latest` dist-tag.
    """

    def __init__(
        self, releases: dict[str, list[tuple[str, float, dict[str, Any]]]]
    ) -> None:
        self.files: dict[str, bytes] = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        for name, versions in releases.items():
            packument: dict[str, Any] = {
                "name": name,
                "dist-tags": {"latest": versions[-1][0]},
                "versions": {},
                "time": {"created": _ago(365), "modified": _ago(0)},
            }
            for version, days_ago, extra in versions:
                manifest = {"name": name, "version": version, **extra}
                tgz = _tarball(manifest)
                path = f"{name}/-/{name}-{version}.tgz"
                self.files[path] = tgz
                digest = base64.b64encode(hashlib.sha512(tgz).digest()).decode()
                packument["versions"][version] = {
                    **manifest,
                    "dist": {
                        "tarball": self.url + path,
                        "integrity": f"sha512-{digest}",
                    },
                }
                packument["time"][version] = _ago(days_ago)
            self.files[name] = json.dumps(packument).encode()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        files = self.files

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = files.get(self.path.lstrip("/"))
                self.send_response(200 if body is not None else 404)
                self.end_headers()
                self.wfile.write(body or b"{}")

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return Handler

    def __enter__(self) -> "Registry":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def npm_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate npm from the developer's config and cache; return the userconfig."""
    userconfig = tmp_path / "npmrc"
    userconfig.write_text("")
    (tmp_path / "global-npmrc").write_text("")
    monkeypatch.setenv("npm_config_userconfig", str(userconfig))
    monkeypatch.setenv("npm_config_globalconfig", str(tmp_path / "global-npmrc"))
    monkeypatch.setenv("npm_config_cache", str(tmp_path / "npm-cache"))
    monkeypatch.setenv("npm_config_update_notifier", "false")
    monkeypatch.setattr(node, "package_cache_dir", lambda _: tmp_path / "bundles")
    return userconfig


def _installed_version(bundle: bytes, package: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(bundle)) as tar:
        member = tar.extractfile(f"node_modules/{package}/package.json")
        assert member is not None
        return str(json.load(member)["version"])


# The agent's newest release is inside the cooldown. Below it, 1.1.0 needs a
# newer Node than any host has and 1.0.0 doesn't. `agent@latest --before`
# lets npm prefer the release whose `engines` accept the host's Node (1.0.0),
# so the result would depend on the host, whose Node is not the sandbox's.
_AGENT_RELEASES: dict[str, list[tuple[str, float, dict[str, Any]]]] = {
    "agent": [
        ("1.0.0", 30, {}),
        ("1.1.0", 20, {"engines": {"node": ">=99"}}),
        ("1.2.0", 1, {}),
    ]
}


async def _install_gemini(sandbox: SandboxEnvironment) -> None:
    version = await gemini_agentbinary.resolve_gemini_version("latest")
    await gemini_agentbinary.ensure_gemini_cli_installed(
        sandbox, "node", version, "linux-x64"
    )


async def _install_claude_acp(sandbox: SandboxEnvironment) -> None:
    await claude_acp_agentbinary._ensure_acp_installed(sandbox, "linux-x64")


async def _install_codex_acp(sandbox: SandboxEnvironment) -> None:
    await codex_acp_agentbinary._ensure_acp_installed(sandbox, "linux-x64")


@needs_npm
@pytest.mark.usefixtures("clear_version_cache")
@pytest.mark.parametrize(
    "module,package_attr,install",
    [
        (gemini_agentbinary, "_GEMINI_CLI_PACKAGE", _install_gemini),
        (claude_acp_agentbinary, "_ACP_ADAPTER_PACKAGE", _install_claude_acp),
        (codex_acp_agentbinary, "_ACP_ADAPTER_PACKAGE", _install_codex_acp),
    ],
)
def test_agent_install_picks_newest_cooled_down_release_whatever_host_node(
    npm_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    package_attr: str,
    install: Callable[[SandboxEnvironment], Awaitable[None]],
) -> None:
    # a user's engine-strict would otherwise refuse 1.1.0 on this host
    npm_env.write_text("engine-strict=true\n")
    sandbox = MagicMock()
    # nothing is installed in the sandbox yet
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(success=False))
    install_bundle = AsyncMock(return_value="binary")
    monkeypatch.setattr(module, package_attr, "agent")
    monkeypatch.setattr(module, "install_npm_bundle", install_bundle)

    with Registry(_AGENT_RELEASES) as registry:
        monkeypatch.setenv("npm_config_registry", registry.url)
        anyio.run(install, sandbox)

    bundle = install_bundle.call_args.kwargs["bundle_data"]
    assert _installed_version(bundle, "agent") == "1.1.0"


def _npm_version() -> tuple[int, ...]:
    version = subprocess.run(
        ["npm", "--version"], capture_output=True, text=True, check=True
    ).stdout
    return tuple(int(part) for part in version.strip().split(".")[:2])


@needs_npm
def test_user_exclude_does_not_lift_cooldown(
    npm_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user's min-release-age-exclude must not lift `--before` (npm 11.17+)."""
    if _npm_version() < (11, 17):
        pytest.skip("npm before 11.17 has no min-release-age-exclude")
    npm_env.write_text("min-release-age-exclude=dep\n")
    releases: dict[str, list[tuple[str, float, dict[str, Any]]]] = {
        "app": [("1.0.0", 30, {"dependencies": {"dep": "^1.0.0"}})],
        "dep": [("1.0.0", 30, {}), ("1.1.0", 1, {})],
    }
    with Registry(releases) as registry:
        monkeypatch.setenv("npm_config_registry", registry.url)
        bundle = node.create_npm_bundle("app", "1.0.0", "linux-x64", "app")

    assert _installed_version(bundle, "dep") == "1.0.0"


# ---------------------------------------------------------------------------
# Sandbox libc
# ---------------------------------------------------------------------------


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

    libc_cache = cache / "cli-bundles-1.0.0-linux-x64-libc-noscripts.tar.gz"
    old_cache = cache / "cli-bundles-1.0.0-linux-x64-noscripts.tar.gz"

    bundle = node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles")
    assert len(bundle) > 0
    assert not libc_cache.exists()
    assert old_cache.exists()
    assert old_cache.read_bytes() == bundle
    assert installs == 1

    # a second build on the old npm reuses the pre-libc cache
    assert node.create_npm_bundle("cli", "1.0.0", "linux-x64", "cli-bundles") == bundle
    assert installs == 1

    # after an upgrade to npm 10.3, the bundle is built again and cached as "-libc"
    fake_npm.write_text(fake_npm.read_text().replace("echo 10.2.0", "echo 10.3.0"))
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
    assert not (cache / "cli-bundles-1.0.0-linux-x64-libc-noscripts.tar.gz").exists()
