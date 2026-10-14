"""Unit tests for the per-sample agent binary provenance event."""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.event import InfoEvent
from inspect_ai.log import transcript
from inspect_ai.util import SandboxEnvironment
from inspect_swe._util import agentbinary
from inspect_swe._util.agentbinary import (
    AgentBinaryInstall,
    AgentBinarySource,
    AgentBinaryVersion,
    ensure_agent_binary_installed,
)
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR


@pytest.fixture(autouse=True)
def _clear_resolution_caches() -> Iterator[None]:
    """Isolate the module-level resolution caches, which are never evicted.

    These tests otherwise rely on every one of them choosing a unique `binary`
    name, since the cache keys on it. Clearing on the way in as well as out
    means reusing a name is merely redundant rather than silently coupling two
    tests together.
    """
    agentbinary._resolved_versions.clear()
    agentbinary._failed_resolutions.clear()
    yield
    agentbinary._resolved_versions.clear()
    agentbinary._failed_resolutions.clear()


def _source(
    tmp_path: Path,
    binary: str,
    resolved: AgentBinaryVersion | None = None,
    post_download: Any = None,
    package_capable: bool = False,
) -> AgentBinarySource:
    async def resolve_version(version: str, platform: str) -> AgentBinaryVersion:
        if resolved is None:
            raise RuntimeError("offline")
        return resolved

    # a package-capable source checks only its package cache on the pinned fast
    # path, so the single-binary cache is reachable only as the offline fallback
    return AgentBinarySource(
        agent="codex cli",
        binary=binary,
        resolve_version=resolve_version,
        cached_binary_path=lambda v, p: tmp_path / f"{binary}-{v}-{p}",
        list_cached_binaries=lambda: [],
        post_download=post_download,
        post_install=None,
        package_entrypoint="bin/codex" if package_capable else None,
        cached_package_path=(
            (lambda v, p: tmp_path / f"{binary}-package-{v}-{p}.tar.gz")
            if package_capable
            else None
        ),
    )


class _FakeSandbox:
    """exec() answers `which` per `which_success`; everything else succeeds."""

    def __init__(
        self,
        which_success: bool = False,
        which_path: str = "",
        already_installed: bool = False,
    ) -> None:
        self.which_success = which_success
        self.which_path = which_path
        self.already_installed = already_installed
        self.written: list[str] = []

    async def exec(self, cmd: list[str], **kwargs: object) -> SimpleNamespace:
        script = cmd[-1]
        if script.startswith("which "):
            return SimpleNamespace(
                success=self.which_success, stdout=self.which_path, stderr=""
            )
        if script.startswith("test -x"):
            # whether this version's package is already extracted in the sandbox
            return SimpleNamespace(success=self.already_installed, stdout="", stderr="")
        return SimpleNamespace(success=True, stdout="", stderr="")

    async def write_file(self, path: str, data: bytes) -> None:
        self.written.append(path)


def _install(
    source: AgentBinarySource, version: str, sandbox: _FakeSandbox
) -> tuple[str, list[AgentBinaryInstall]]:
    """Run an install and return its path plus the provenance events it wrote."""
    before = len(transcript().events)
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
            version,
            None,
            cast(SandboxEnvironment, sandbox),
        )
    events = [
        AgentBinaryInstall.model_validate(event.data)
        for event in transcript().events[before:]
        if isinstance(event, InfoEvent) and event.source == "inspect_swe"
    ]
    return binary_path, events


def test_sandbox_preinstalled_binary_is_recorded(tmp_path: Path) -> None:
    # the default path: the image already has the binary, so there is no
    # version, checksum or platform to report -- but the sample still records
    # that this is what it ran
    source = _source(tmp_path, "codex-prov-sandbox")
    sandbox = _FakeSandbox(which_success=True, which_path="/usr/local/bin/codex\n")

    binary_path, events = _install(source, "auto", sandbox)

    assert binary_path == "/usr/local/bin/codex"
    assert len(events) == 1
    install = events[0]
    assert install.agent == "codex_cli"
    assert install.origin == "sandbox"
    assert install.requested == "auto"
    assert install.version is None
    assert install.platform is None
    assert install.checksum is None


def test_pinned_cache_hit_is_recorded(tmp_path: Path) -> None:
    source = _source(tmp_path, "codex-prov-cache")
    data = b"cached-single-binary"
    source.cached_binary_path("9.9.1", "linux-arm64").write_bytes(data)
    sandbox = _FakeSandbox()

    binary_path, events = _install(source, "9.9.1", sandbox)

    assert binary_path == f"{SANDBOX_INSTALL_DIR}/codex-prov-cache-9.9.1-linux-arm64"
    assert len(events) == 1
    install = events[0]
    assert install.origin == "cache"
    assert install.requested == "9.9.1"
    assert install.version == "9.9.1"
    assert install.platform == "linux-arm64"
    assert install.checksum == hashlib.sha256(data).hexdigest()


def test_download_records_resolved_version_and_requested_alias(
    tmp_path: Path,
) -> None:
    # "auto" that finds nothing in the sandbox is rewritten to "stable"
    # internally; `requested` must still report what the caller passed, since
    # the point of the event is to compare the task's intent against reality
    data = b"downloaded-binary"
    resolved = AgentBinaryVersion(
        "2.1.236",
        hashlib.sha256(data).hexdigest(),
        "https://example.com/codex",
    )
    source = _source(tmp_path, "codex-prov-download", resolved)
    sandbox = _FakeSandbox(which_success=False)

    with patch.object(agentbinary, "download_file", AsyncMock(return_value=data)):
        _, events = _install(source, "auto", sandbox)

    assert len(events) == 1
    install = events[0]
    assert install.origin == "download"
    assert install.requested == "auto"
    assert install.version == "2.1.236"
    assert install.checksum == hashlib.sha256(data).hexdigest()


def test_offline_fallback_is_recorded_as_unverified(tmp_path: Path) -> None:
    # resolution fails and a cached binary installs without any digest to
    # check it against; the event has to say so, because the checksum it
    # reports is of bytes nobody verified
    source = _source(
        tmp_path, "codex-prov-offline", package_capable=True
    )  # resolve_version raises
    data = b"unverified-cached-binary"
    source.cached_binary_path("9.9.6", "linux-arm64").write_bytes(data)
    sandbox = _FakeSandbox()

    _, events = _install(source, "9.9.6", sandbox)

    assert len(events) == 1
    install = events[0]
    assert install.origin == "cache_unverified"
    assert install.version == "9.9.6"
    assert install.checksum == hashlib.sha256(data).hexdigest()


def test_checksum_is_of_installed_bytes_not_the_manifest_digest(
    tmp_path: Path,
) -> None:
    # codex_cli sets post_download=extract_tarball, so what it installs is not
    # what it downloaded. the manifest digest describes the archive; the event
    # has to describe the artifact that actually reached the sandbox.
    archive = b"archive-bytes"
    extracted = b"extracted-binary-bytes"
    resolved = AgentBinaryVersion(
        "9.9.7",
        hashlib.sha256(archive).hexdigest(),
        "https://example.com/codex.tar.gz",
    )
    source = _source(
        tmp_path,
        "codex-prov-transform",
        resolved,
        post_download=lambda _: extracted,
    )
    sandbox = _FakeSandbox()

    with patch.object(agentbinary, "download_file", AsyncMock(return_value=archive)):
        _, events = _install(source, "9.9.7", sandbox)

    assert len(events) == 1
    install = events[0]
    assert install.checksum == hashlib.sha256(extracted).hexdigest()
    assert install.checksum != resolved.expected_checksum


def test_failed_install_records_nothing(tmp_path: Path) -> None:
    # nothing was installed, so there is no provenance to claim
    source = _source(tmp_path, "codex-prov-failure")  # resolve_version raises
    sandbox = _FakeSandbox()

    before = len(transcript().events)
    with (
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-arm64"),
        ),
        patch.object(agentbinary, "trace", lambda msg: None),
        patch.object(agentbinary, "sandbox_exec", AsyncMock(return_value="")),
        pytest.raises(RuntimeError, match="offline"),
    ):
        anyio.run(
            ensure_agent_binary_installed,
            source,
            "9.9.9",
            None,
            cast(SandboxEnvironment, sandbox),
        )

    assert [
        event
        for event in transcript().events[before:]
        if isinstance(event, InfoEvent) and event.source == "inspect_swe"
    ] == []


def test_resolved_version_served_from_cache_is_not_reported_as_download(
    tmp_path: Path,
) -> None:
    # "stable" resolves to a concrete version that is usually already cached
    # from an earlier eval on the same host, so nothing is fetched. reporting
    # that as origin="download" would make the field useless in the common
    # case, and the caller cannot tell: the cache path is only known after
    # resolution, which happens inside download_agent_binary_async.
    data = b"already-on-disk-from-a-previous-eval"
    resolved = AgentBinaryVersion(
        "2.1.236",
        hashlib.sha256(data).hexdigest(),
        "https://example.com/claude",
    )
    source = _source(tmp_path, "codex-prov-warm", resolved)
    source.cached_binary_path("2.1.236", "linux-arm64").write_bytes(data)
    sandbox = _FakeSandbox(which_success=False)

    # any network access is a test failure, not a fallback
    with patch.object(
        agentbinary,
        "download_file",
        AsyncMock(side_effect=AssertionError("should not download")),
    ):
        _, events = _install(source, "stable", sandbox)

    assert len(events) == 1
    install = events[0]
    assert install.origin == "cache"
    assert install.requested == "stable"
    assert install.version == "2.1.236"


def test_package_archive_checksum_is_the_installed_archive(tmp_path: Path) -> None:
    # post_download is skipped for package archives, so the archive is what is
    # cached and installed. its digest legitimately matches the manifest -- the
    # opposite of the transformed single-binary case above, and the shape
    # codex_cli and opencode actually take today.
    archive = b"package-archive-bytes"
    resolved = AgentBinaryVersion(
        "9.9.5",
        hashlib.sha256(archive).hexdigest(),
        "https://example.com/codex-package.tar.gz",
        True,
    )
    source = _source(
        tmp_path,
        "codex-prov-package",
        resolved,
        post_download=lambda _: b"never-applied-to-a-package",
        package_capable=True,
    )
    sandbox = _FakeSandbox()

    with patch.object(agentbinary, "download_file", AsyncMock(return_value=archive)):
        _, events = _install(source, "9.9.5", sandbox)

    assert len(events) == 1
    install = events[0]
    assert install.checksum == hashlib.sha256(archive).hexdigest()
    assert install.checksum == resolved.expected_checksum


def test_already_extracted_package_still_records_what_the_sample_runs(
    tmp_path: Path,
) -> None:
    # when this version's package is already extracted in the sandbox the write
    # and extract are skipped. the sample still runs that binary, so a record is
    # still written -- but the checksum is of the cached archive the install came
    # from, not of bytes this call placed in the sandbox.
    archive = b"package-archive-already-extracted"
    source = _source(tmp_path, "codex-prov-extracted", package_capable=True)
    assert source.cached_package_path is not None
    source.cached_package_path("9.9.4", "linux-arm64").write_bytes(archive)
    sandbox = _FakeSandbox(already_installed=True)

    _, events = _install(source, "9.9.4", sandbox)

    assert sandbox.written == []
    assert len(events) == 1
    install = events[0]
    assert install.origin == "cache"
    assert install.version == "9.9.4"
    assert install.checksum == hashlib.sha256(archive).hexdigest()
