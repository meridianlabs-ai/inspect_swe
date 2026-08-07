"""Unit tests for package-archive installs in the agent binary machinery."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_swe._util import agentbinary
from inspect_swe._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
    download_agent_binary_async,
    ensure_agent_binary_installed,
)
from inspect_ai.util import SandboxEnvironment
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
