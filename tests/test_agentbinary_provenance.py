"""Unit tests for the per-sample agent binary provenance event."""

import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import ExitStack, asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState, agent, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.event import InfoEvent, SpanBeginEvent
from inspect_ai.log import read_eval_log, transcript
from inspect_ai.solver import chain
from inspect_ai.util import SandboxEnvironment
from inspect_swe._util import agentbinary
from inspect_swe._util.agentbinary import (
    AgentBinaryInstall,
    AgentBinarySource,
    AgentBinaryVersion,
    ensure_agent_binary_installed,
)
from inspect_swe._util.checksum import sha256_checksum
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR


@pytest.fixture(autouse=True)
def _clear_resolution_caches() -> Iterator[None]:
    """Isolate the module-level resolution and checksum caches (never evicted).

    These tests otherwise rely on every one of them choosing a unique `binary`
    name, since the cache keys on it. Clearing on the way in as well as out
    means reusing a name is merely redundant rather than silently coupling two
    tests together.
    """
    agentbinary._resolved_versions.clear()
    agentbinary._failed_resolutions.clear()
    agentbinary._artifact_checksums.clear()
    yield
    agentbinary._resolved_versions.clear()
    agentbinary._failed_resolutions.clear()
    agentbinary._artifact_checksums.clear()


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


def _patched_installer() -> ExitStack:
    """Stub the installer's sandbox platform probe, trace and exec helpers."""
    stack = ExitStack()
    stack.enter_context(
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-arm64"),
        )
    )
    stack.enter_context(patch.object(agentbinary, "trace", lambda msg: None))
    stack.enter_context(
        patch.object(agentbinary, "sandbox_exec", AsyncMock(return_value=""))
    )
    return stack


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
    source: AgentBinarySource,
    version: str,
    sandbox: _FakeSandbox,
    sandbox_version: Callable[[str], Awaitable[str | None]] | None = None,
) -> tuple[str, list[AgentBinaryInstall]]:
    """Run an install and return its path plus the provenance events it wrote."""
    before = len(transcript().events)
    with _patched_installer():
        binary_path = anyio.run(
            partial(
                ensure_agent_binary_installed,
                source,
                version,
                None,
                cast(SandboxEnvironment, sandbox),
                sandbox_version=sandbox_version,
            )
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


def test_sandbox_version_probe_is_recorded(tmp_path: Path) -> None:
    # codex_cli probes `codex --version` anyway, so it hands that probe to the
    # installer and a binary found in the image is recorded with its version
    source = _source(tmp_path, "codex-prov-sandbox-probe")
    sandbox = _FakeSandbox(which_success=True, which_path="/usr/local/bin/codex\n")
    probe = AsyncMock(return_value="0.140.0")

    _, events = _install(source, "auto", sandbox, sandbox_version=probe)

    probe.assert_awaited_once_with("/usr/local/bin/codex")
    assert len(events) == 1
    assert events[0].origin == "sandbox"
    assert events[0].version == "0.140.0"
    assert events[0].checksum is None


def test_sandbox_version_probe_is_not_run_for_an_installed_binary(
    tmp_path: Path,
) -> None:
    # an installed binary's version is already known from resolution
    source = _source(tmp_path, "codex-prov-cache-probe")
    source.cached_binary_path("9.9.2", "linux-arm64").write_bytes(b"cached")
    probe = AsyncMock(return_value="0.140.0")

    _, events = _install(source, "9.9.2", _FakeSandbox(), sandbox_version=probe)

    probe.assert_not_awaited()
    assert len(events) == 1
    assert events[0].version == "9.9.2"


def test_saved_log_has_one_record_per_agent_span_per_sample(tmp_path: Path) -> None:
    # the per-sample contract end to end: concurrent samples each pin their own
    # version and run the agent twice. after a save and reload, every record
    # sits in its own agent span and carries only its own sample's version.
    source = _source(tmp_path, "codex-prov-eval")
    versions = ["9.8.1", "9.8.2", "9.8.3", "9.8.4"]
    for version in versions:
        source.cached_binary_path(version, "linux-arm64").write_bytes(
            f"binary-{version}".encode()
        )

    @agent
    def installing_agent() -> Agent:
        async def execute(state: AgentState) -> AgentState:
            await ensure_agent_binary_installed(
                source,
                state.messages[0].text,
                None,
                cast(SandboxEnvironment, _FakeSandbox()),
            )
            return state

        return execute

    task = Task(
        dataset=[Sample(input=version) for version in versions],
        solver=chain(as_solver(installing_agent()), as_solver(installing_agent())),
    )
    with _patched_installer():
        [log] = eval(
            task,
            model="mockllm/model",
            max_samples=len(versions),
            log_dir=str(tmp_path / "logs"),
            display="none",
        )
    assert log.status == "success"

    saved = read_eval_log(log.location)
    assert saved.samples is not None
    assert len(saved.samples) == len(versions)
    for sample in saved.samples:
        spans = {
            event.id: event
            for event in sample.events
            if isinstance(event, SpanBeginEvent)
        }
        records = [
            event
            for event in sample.events
            if isinstance(event, InfoEvent) and event.source == "inspect_swe"
        ]
        assert len(records) == 2
        assert len({event.span_id for event in records}) == 2
        for event in records:
            assert event.span_id is not None
            assert spans[event.span_id].type == "agent"
            install = AgentBinaryInstall.model_validate(event.data)
            assert install.version == sample.input
            assert install.checksum == (
                hashlib.sha256(f"binary-{sample.input}".encode()).hexdigest()
            )

    # the events_df recipe from the docs extracts the same records
    pytest.importorskip("pandas")
    from inspect_ai.analysis import EventColumn, EventInfo, events_df

    events = events_df(
        str(tmp_path / "logs"),
        columns=EventInfo
        + [
            EventColumn("source", path="source"),
            EventColumn("version", path="data.version"),
            EventColumn("origin", path="data.origin"),
        ],
    )
    installs = events[events["source"] == "inspect_swe"]
    assert sorted(installs["version"]) == sorted(versions * 2)
    assert set(installs["origin"]) == {"cache"}


def _hash_spy() -> MagicMock:
    """Spy on the provenance hash, separate from download/cache verification."""
    return MagicMock(wraps=sha256_checksum)


def test_verified_bytes_record_the_resolved_digest_without_hashing(
    tmp_path: Path,
) -> None:
    # a download, and later a warm-cache read, are both verified against the
    # resolved digest already, so the record reuses it instead of hashing again
    data = b"verified-single-binary"
    resolved = AgentBinaryVersion(
        "9.7.1", hashlib.sha256(data).hexdigest(), "https://example.com/bin"
    )
    source = _source(tmp_path, "codex-prov-verified", resolved)
    spy = _hash_spy()

    with (
        patch.object(agentbinary, "sha256_checksum", spy),
        patch.object(agentbinary, "download_file", AsyncMock(return_value=data)),
    ):
        _, first = _install(source, "stable", _FakeSandbox())
        _, second = _install(source, "stable", _FakeSandbox())

    assert [e.origin for e in first + second] == ["download", "cache"]
    assert {e.checksum for e in first + second} == {resolved.expected_checksum}
    spy.assert_not_called()


def test_unverified_artifact_is_hashed_once_per_process(tmp_path: Path) -> None:
    # a pinned cache read has no resolved digest, so it is hashed -- once, by
    # the first install; later samples reuse the stored value
    data = b"pinned-cached-binary"
    source = _source(tmp_path, "codex-prov-memo")
    source.cached_binary_path("9.7.2", "linux-arm64").write_bytes(data)
    spy = _hash_spy()

    with patch.object(agentbinary, "sha256_checksum", spy):
        events = [
            event
            for _ in range(3)
            for event in _install(source, "9.7.2", _FakeSandbox())[1]
        ]

    assert [e.checksum for e in events] == [hashlib.sha256(data).hexdigest()] * 3
    spy.assert_called_once()


def test_transformed_download_is_hashed_once_per_process(tmp_path: Path) -> None:
    # post_download output has no verified digest either: the first install
    # (download) hashes it and the next (served from cache) reuses the value
    archive = b"archive-to-transform"
    extracted = b"transformed-binary"
    resolved = AgentBinaryVersion(
        "9.7.3", hashlib.sha256(archive).hexdigest(), "https://example.com/c.tgz"
    )
    source = _source(
        tmp_path,
        "codex-prov-memo-transform",
        resolved,
        post_download=lambda _: extracted,
    )
    spy = _hash_spy()

    with (
        patch.object(agentbinary, "sha256_checksum", spy),
        patch.object(agentbinary, "download_file", AsyncMock(return_value=archive)),
    ):
        _, first = _install(source, "stable", _FakeSandbox())
        _, second = _install(source, "stable", _FakeSandbox())

    assert [e.origin for e in first + second] == ["download", "cache"]
    assert {e.checksum for e in first + second} == {
        hashlib.sha256(extracted).hexdigest()
    }
    spy.assert_called_once()


@pytest.mark.parametrize("version", ["auto", "9.6.1"])
def test_codex_cli_probes_its_version_once(tmp_path: Path, version: str) -> None:
    # codex_cli hands its `codex --version` probe to the installer for a binary
    # found in the sandbox, and reuses that one result for its own version
    # checks: on both the sandbox and the installed path, exactly one probe
    # runs, the record carries the sandbox version, and auto_review's version
    # gate gets the probed value
    from inspect_swe import codex_cli
    from inspect_swe._codex_cli import agentbinary as codex_agentbinary
    from inspect_swe._codex_cli import codex_cli as codex_cli_module

    source = _source(tmp_path, "codex")
    source.cached_binary_path("9.6.1", "linux-arm64").write_bytes(b"codex-binary")
    sandbox = _FakeSandbox(
        which_success=version == "auto", which_path="/usr/local/bin/codex\n"
    )
    probe = AsyncMock(return_value="codex-cli 0.150.0")
    gate = MagicMock(side_effect=_StopAfterVersionGate)

    @asynccontextmanager
    async def fake_checkpointer() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(attempt="first")

    @asynccontextmanager
    async def fake_bridge(
        state: AgentState, **kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(state=state)

    task = Task(
        dataset=[Sample(input="hello")],
        solver=codex_cli(version=version, auto_review=True),
    )
    with (
        _patched_installer(),
        patch.object(codex_cli_module, "codex_cli_binary_source", lambda: source),
        patch.object(codex_cli_module, "sandbox_env", lambda *_: sandbox),
        patch.object(codex_cli_module, "checkpointer", fake_checkpointer),
        patch.object(codex_cli_module, "sandbox_agent_bridge", fake_bridge),
        patch.object(codex_cli_module, "check_codex_auto_review_version", gate),
        patch.object(codex_agentbinary, "sandbox_exec", probe),
    ):
        [log] = eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))

    assert log.samples is not None
    assert "_StopAfterVersionGate" in str(log.samples[0].error)
    expected_binary = (
        "/usr/local/bin/codex"
        if version == "auto"
        else f"{SANDBOX_INSTALL_DIR}/codex-9.6.1-linux-arm64"
    )
    probe.assert_awaited_once()
    assert probe.await_args is not None
    assert probe.await_args.args[1] == f"{expected_binary} --version"
    gate.assert_called_once_with("0.150.0")
    [install] = [
        AgentBinaryInstall.model_validate(event.data)
        for event in log.samples[0].events
        if isinstance(event, InfoEvent) and event.source == "inspect_swe"
    ]
    assert install.version == ("0.150.0" if version == "auto" else "9.6.1")


class _StopAfterVersionGate(Exception):
    """Ends a codex_cli run once its version gate has been reached."""
