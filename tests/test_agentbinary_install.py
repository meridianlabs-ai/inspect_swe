"""Unit tests for package-archive installs in the agent binary machinery."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.util import SandboxEnvironment
from inspect_swe._util import agentbinary
from inspect_swe._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
    download_agent_binary_async,
    ensure_agent_binary_installed,
)
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR


def _package_source(
    tmp_path: Path, resolved: AgentBinaryVersion | None = None
) -> AgentBinarySource:
    async def resolve_version(version: str, platform: str) -> AgentBinaryVersion:
        assert resolved is not None
        return resolved

    return AgentBinarySource(
        agent="codex cli",
        binary="codex",
        resolve_version=resolve_version,
        cached_binary_path=lambda v, p: tmp_path / f"codex-{v}-{p}",
        list_cached_binaries=lambda: [],
        post_download=None,
        post_install=None,
        package_entrypoint="bin/codex",
        cached_package_path=lambda v, p: tmp_path / f"codex-package-{v}-{p}.tar.gz",
    )


class _FakeSandbox:
    """Records write_file calls; exec returns canned results by command."""

    def __init__(self, installed: bool = False) -> None:
        self.installed = installed
        self.written: list[str] = []

    async def exec(self, cmd: list[str], **kwargs: object) -> SimpleNamespace:
        script = cmd[-1]
        if script.startswith("test -x"):
            return SimpleNamespace(success=self.installed, stdout="", stderr="")
        return SimpleNamespace(success=True, stdout="", stderr="")

    async def write_file(self, path: str, data: bytes) -> None:
        self.written.append(path)


def test_package_install_extracts_in_sandbox(tmp_path: Path) -> None:
    source = _package_source(tmp_path)
    assert source.cached_package_path is not None
    cache = source.cached_package_path("9.9.9", "linux-arm64")
    cache.write_bytes(b"tarball-bytes")

    sandbox = _FakeSandbox(installed=False)
    execs: list[str] = []

    async def record_exec(sb: object, cmd: str, user: str | None = None) -> str:
        execs.append(cmd)
        return ""

    with (
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-arm64"),
        ),
        patch.object(agentbinary, "trace", lambda msg: None),
        patch.object(agentbinary, "sandbox_exec", record_exec),
    ):
        binary_path = anyio.run(
            ensure_agent_binary_installed,
            source,
            "9.9.9",
            None,
            cast(SandboxEnvironment, sandbox),
        )

    install_dir = f"{SANDBOX_INSTALL_DIR}/codex-9.9.9-linux-arm64"
    assert binary_path == f"{install_dir}/bin/codex"
    assert sandbox.written == [f"{install_dir}.tar.gz"]
    assert any("tar -xzf" in cmd and install_dir in cmd for cmd in execs)


def test_package_install_skips_when_already_installed(tmp_path: Path) -> None:
    source = _package_source(tmp_path)
    assert source.cached_package_path is not None
    source.cached_package_path("9.9.9", "linux-arm64").write_bytes(b"tarball-bytes")

    sandbox = _FakeSandbox(installed=True)
    with (
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-arm64"),
        ),
        patch.object(agentbinary, "trace", lambda msg: None),
        patch.object(agentbinary, "sandbox_exec", AsyncMock(return_value="")),
    ):
        binary_path = anyio.run(
            ensure_agent_binary_installed,
            source,
            "9.9.9",
            None,
            cast(SandboxEnvironment, sandbox),
        )

    assert binary_path.endswith("/bin/codex")
    assert sandbox.written == []


def test_download_caches_package_archive_verbatim(tmp_path: Path) -> None:
    data = b"package-tarball-bytes"
    resolved = AgentBinaryVersion(
        "9.9.8",
        hashlib.sha256(data).hexdigest(),
        "https://example.com/pkg.tar.gz",
        True,
    )
    source = _package_source(tmp_path, resolved)

    with patch.object(agentbinary, "download_file", AsyncMock(return_value=data)):
        downloaded, out = anyio.run(
            download_agent_binary_async, source, "9.9.8", "linux-arm64"
        )

    assert downloaded == data
    assert out.package is True
    assert source.cached_package_path is not None
    cache = source.cached_package_path("9.9.8", "linux-arm64")
    assert cache.read_bytes() == data

    # second call verifies the checksum against the verbatim cache (no download)
    with patch.object(
        agentbinary,
        "download_file",
        AsyncMock(side_effect=AssertionError("should not download")),
    ):
        cached, out = anyio.run(
            download_agent_binary_async, source, "9.9.8", "linux-arm64"
        )
    assert cached == data


def test_package_without_entrypoint_raises(tmp_path: Path) -> None:
    source = _package_source(tmp_path)
    source.package_entrypoint = None
    assert source.cached_package_path is not None
    source.cached_package_path("9.9.9", "linux-arm64").write_bytes(b"tarball-bytes")

    sandbox = _FakeSandbox(installed=False)
    with (
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-arm64"),
        ),
        patch.object(agentbinary, "trace", lambda msg: None),
        patch.object(agentbinary, "sandbox_exec", AsyncMock(return_value="")),
        pytest.raises(RuntimeError, match="package_entrypoint"),
    ):
        anyio.run(
            ensure_agent_binary_installed,
            source,
            "9.9.9",
            None,
            cast(SandboxEnvironment, sandbox),
        )


def test_stale_single_binary_cache_still_installs_package(tmp_path: Path) -> None:
    # a pinned version cached as a single binary by an older inspect_swe must
    # not short-circuit resolution: the package is fetched, installed, and the
    # superseded cache entry removed
    data = b"package-tarball-bytes"
    resolved = AgentBinaryVersion(
        "9.9.7",
        hashlib.sha256(data).hexdigest(),
        "https://example.com/pkg.tar.gz",
        True,
    )
    source = _package_source(tmp_path, resolved)
    stale = source.cached_binary_path("9.9.7", "linux-arm64")
    stale.write_bytes(b"stale-single-binary")

    sandbox = _FakeSandbox(installed=False)
    with (
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-arm64"),
        ),
        patch.object(agentbinary, "trace", lambda msg: None),
        patch.object(agentbinary, "sandbox_exec", AsyncMock(return_value="")),
        patch.object(agentbinary, "download_file", AsyncMock(return_value=data)),
    ):
        binary_path = anyio.run(
            ensure_agent_binary_installed,
            source,
            "9.9.7",
            None,
            cast(SandboxEnvironment, sandbox),
        )

    assert binary_path.endswith("/bin/codex")
    assert sandbox.written == [f"{SANDBOX_INSTALL_DIR}/codex-9.9.7-linux-arm64.tar.gz"]
    assert source.cached_package_path is not None
    assert source.cached_package_path("9.9.7", "linux-arm64").read_bytes() == data
    assert not stale.exists()


def test_offline_pinned_version_falls_back_to_cached_binary(tmp_path: Path) -> None:
    # when resolution fails (offline) a pinned version with a cached single
    # binary still installs, without the package
    source = _package_source(tmp_path)  # resolve_version raises (resolved=None)
    source.cached_binary_path("9.9.6", "linux-arm64").write_bytes(b"single-binary")

    sandbox = _FakeSandbox(installed=False)
    with (
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-arm64"),
        ),
        patch.object(agentbinary, "trace", lambda msg: None),
        patch.object(agentbinary, "sandbox_exec", AsyncMock(return_value="")),
    ):
        binary_path = anyio.run(
            ensure_agent_binary_installed,
            source,
            "9.9.6",
            None,
            cast(SandboxEnvironment, sandbox),
        )

    assert binary_path == f"{SANDBOX_INSTALL_DIR}/codex-9.9.6-linux-arm64"
    assert sandbox.written == [binary_path]


def test_concurrent_version_resolution_shares_one_request(tmp_path: Path) -> None:
    # a cold-start burst of concurrent installs for the same (binary,
    # version, platform) key shares one resolve_version call, not one each
    calls = 0

    async def resolve_version(version: str, platform: str) -> AgentBinaryVersion:
        nonlocal calls
        calls += 1
        # suspend so the other tasks reach the cache miss while this one is
        # still in flight
        await anyio.sleep(0.05)
        return AgentBinaryVersion("9.5.1", "checksum", "https://example.com/pkg.tar.gz")

    source = AgentBinarySource(
        agent="test agent",
        binary="test-agent-concurrent-ok",
        resolve_version=resolve_version,
        cached_binary_path=lambda v, p: tmp_path / f"{v}-{p}",
        list_cached_binaries=lambda: [],
        post_download=None,
        post_install=None,
    )
    results: list[AgentBinaryVersion] = []

    async def run() -> None:
        async def one() -> None:
            results.append(
                await agentbinary._resolve_agent_binary_version(
                    source, "9.5.1", "linux-x64"
                )
            )

        async with anyio.create_task_group() as tg:
            for _ in range(10):
                tg.start_soon(one)

    anyio.run(run)
    assert calls == 1
    assert results == [results[0]] * 10
    assert results[0].version == "9.5.1"


def test_concurrent_version_resolution_failure_is_shared(tmp_path: Path) -> None:
    # callers queued behind a failing resolve_version share its exception —
    # a rate-limit cascade across queued samples produces one failed API
    # call, not one retry per queued sample
    calls = 0
    errors: list[Exception] = []

    async def failing_resolve_version(
        version: str, platform: str
    ) -> AgentBinaryVersion:
        nonlocal calls
        calls += 1
        await anyio.sleep(0.05)
        raise RuntimeError("403 rate limited")

    source = AgentBinarySource(
        agent="test agent",
        binary="test-agent-concurrent-fail",
        resolve_version=failing_resolve_version,
        cached_binary_path=lambda v, p: tmp_path / f"{v}-{p}",
        list_cached_binaries=lambda: [],
        post_download=None,
        post_install=None,
    )

    async def run() -> None:
        async def one() -> None:
            try:
                await agentbinary._resolve_agent_binary_version(
                    source, "9.5.2", "linux-x64"
                )
            except RuntimeError as ex:
                errors.append(ex)

        async with anyio.create_task_group() as tg:
            for _ in range(10):
                tg.start_soon(one)

    anyio.run(run)
    assert calls == 1
    assert len(errors) == 10
    assert all(error is errors[0] for error in errors)

    # a genuinely later call (after the failure completed) retries rather
    # than replaying the stale failure forever — same (binary, version,
    # platform) key, a fresh source whose resolve_version now succeeds
    async def recovered_resolve_version(
        version: str, platform: str
    ) -> AgentBinaryVersion:
        return AgentBinaryVersion("9.5.2", "checksum", "https://example.com/pkg.tar.gz")

    retry_source = AgentBinarySource(
        agent="test agent",
        binary="test-agent-concurrent-fail",
        resolve_version=recovered_resolve_version,
        cached_binary_path=lambda v, p: tmp_path / f"{v}-{p}",
        list_cached_binaries=lambda: [],
        post_download=None,
        post_install=None,
    )
    resolved = anyio.run(
        agentbinary._resolve_agent_binary_version, retry_source, "9.5.2", "linux-x64"
    )
    assert resolved.version == "9.5.2"
