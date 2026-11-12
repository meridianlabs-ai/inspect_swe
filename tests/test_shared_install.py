"""Agent binaries installed inside a sandbox identity scope.

Every member of a scope runs the same binary at a fixed path, so it is
provisioned as the sandbox's default user inside Inspect's framework-operation
marker, under the verified-directory contract, and an existing artifact is
verified before a member uses it. A stub sandbox stands in for the
verified-directory script and records the commands it wraps; the two shell
scripts the installer adds are run for real against a temporary directory.
"""

import io
import os
import stat
import subprocess
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, NamedTuple, overload
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.util import ExecResult, SandboxEnvironment
from inspect_ai.util._sandbox._framework_directory import _SCRIPT, _VERIFIED_MARKER
from inspect_ai.util._sandbox._framework_operation import (
    in_framework_sandbox_operation,
)
from inspect_ai.util._sandbox.context import (
    sandbox_default_context_var,
    sandbox_environments_context_var,
)
from inspect_ai.util._sandbox.environment import SandboxEnvironmentConfigType
from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy
from inspect_ai.util._sandbox.identity import SandboxIdentityError, sandbox_identity
from inspect_swe._util import agentbinary
from inspect_swe._util.agentbinary import (
    AgentBinaryDownload,
    AgentBinarySource,
    AgentBinaryVersion,
    ensure_agent_binary_installed,
)
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR, SandboxPlatform
from inspect_swe._util.shared_install import _CHECK, _EXTRACT, PACKAGE_COMPLETE

MEMBER = "swarm-w-1"
VERSION = "9.9.9"
PLATFORM: SandboxPlatform = "linux-arm64"


class Call(NamedTuple):
    directory: str | None
    """Verified directory, for a framework-directory call."""
    command: list[str]
    """Command wrapped in the verification (or the plain command)."""
    user: str | None
    framework: bool
    input: str | bytes | None


class InstallSandbox(SandboxEnvironment):
    """Answers the verified-directory script, and records what it wraps.

    `artifacts` maps an artifact's path to "ok", "partial" (a package with its
    entrypoint but no completion marker) or "tampered"; a path absent from it
    is missing until it is written or extracted. With `fail_extract`, an
    extraction writes the entrypoint and then fails.
    """

    def __init__(
        self, artifacts: dict[str, str] | None = None, fail_extract: bool = False
    ) -> None:
        super().__init__()
        self.artifacts = dict(artifacts or {})
        self.fail_extract = fail_extract
        self.calls: list[Call] = []

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        framework = in_framework_sandbox_operation()
        if len(cmd) < 3 or cmd[2] != _SCRIPT:
            self.calls.append(Call(None, cmd, user, framework, input))
            return ExecResult(True, 0, "", "")
        parent, leaf, wrapped = cmd[8], cmd[9], cmd[10:]
        directory = f"{parent}/{leaf}"
        self.calls.append(Call(directory, wrapped, user, framework, input))
        stdout, returncode, stderr = "", 0, ""
        if wrapped and wrapped[2] == _CHECK:
            tree = os.path.normpath(f"{directory}/{wrapped[4]}")
            executable, complete = wrapped[5], wrapped[6]
            state = self.artifacts.get(tree, "missing")
            if state == "partial":
                state = "ok" if complete == executable else "missing"
            if state == "tampered":
                returncode, stderr = 3, "not owned by uid 0: ./x"
            else:
                stdout = f"{state}\n"
        elif wrapped and wrapped[2] == _EXTRACT:
            self.artifacts[directory] = "partial"
            if self.fail_extract:
                returncode, stderr = 2, "tar: unexpected end of archive"
        elif wrapped and wrapped[4] == PACKAGE_COMPLETE:
            self.artifacts[directory] = "ok"
        elif wrapped:  # the publish of a single file
            self.artifacts[f"{directory}/{wrapped[4]}"] = "ok"
        return ExecResult(
            returncode == 0, returncode, stdout, f"{_VERIFIED_MARKER}\n{stderr}"
        )

    async def write_file(
        self, file: str, contents: str | bytes, *, user: str | None = None
    ) -> None:
        raise AssertionError(f"unexpected write_file({file!r})")

    @overload
    async def read_file(
        self, file: str, text: Literal[True] = True, *, user: str | None = None
    ) -> str: ...

    @overload
    async def read_file(
        self, file: str, text: Literal[False], *, user: str | None = None
    ) -> bytes: ...

    async def read_file(
        self, file: str, text: bool = True, *, user: str | None = None
    ) -> str | bytes:
        raise FileNotFoundError(file)

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        pass

    def wrapped(self) -> list[tuple[str | None, str]]:
        """(directory, kind) of each call, with the script named by its role."""
        kinds = {_CHECK: "check", _EXTRACT: "extract"}
        result: list[tuple[str | None, str]] = []
        for call in self.calls:
            if not call.command:
                kind = "ensure"
            elif len(call.command) > 2 and call.command[2] in kinds:
                kind = kinds[call.command[2]]
            elif call.directory is not None:
                kind = "write"
            else:
                kind = " ".join(call.command)
            result.append((call.directory, kind))
        return result


@pytest.fixture(autouse=True)
def transcript() -> None:
    init_transcript(Transcript())


@contextmanager
def scope(stub: InstallSandbox) -> Iterator[SandboxEnvironment]:
    proxy = SandboxEnvironmentProxy(stub)
    envs = sandbox_environments_context_var.set({"default": proxy})
    default = sandbox_default_context_var.set("default")
    try:
        with sandbox_identity(MEMBER):
            yield proxy
    finally:
        sandbox_default_context_var.reset(default)
        sandbox_environments_context_var.reset(envs)


def _source(tmp_path: Path, package: bool) -> AgentBinarySource:
    async def resolve_version(version: str, platform: str) -> AgentBinaryVersion:
        raise AssertionError("a pinned cached version is not resolved")

    return AgentBinarySource(
        agent="codex cli" if package else "claude code",
        binary="codex" if package else "claude",
        resolve_version=resolve_version,
        cached_binary_path=lambda v, p: tmp_path / f"binary-{v}-{p}",
        list_cached_binaries=lambda: [],
        post_download=None,
        post_install=None,
        package_entrypoint="bin/codex" if package else None,
        cached_package_path=(
            (lambda v, p: tmp_path / f"package-{v}-{p}.tar.gz") if package else None
        ),
    )


async def _install(
    tmp_path: Path, stub: InstallSandbox, package: bool
) -> tuple[str, bytes]:
    source = _source(tmp_path, package)
    data = b"package-archive" if package else b"binary"
    cache = (
        source.cached_package_path(VERSION, PLATFORM)
        if source.cached_package_path is not None
        else source.cached_binary_path(VERSION, PLATFORM)
    )
    cache.write_bytes(data)
    with (
        patch.object(
            agentbinary, "detect_sandbox_platform", AsyncMock(return_value=PLATFORM)
        ),
        patch.object(agentbinary, "trace", lambda message: None),
        scope(stub) as sandbox,
    ):
        path = await ensure_agent_binary_installed(source, VERSION, MEMBER, sandbox)
    return path, data


def _assert_framework_operations(stub: InstallSandbox) -> None:
    # Inspect's own operations, as the default user: inside the scope an
    # unmarked call without a user would have run as the member
    assert stub.calls
    assert all(c.framework and c.user is None for c in stub.calls)


async def test_single_binary_is_published_and_verified(tmp_path: Path) -> None:
    stub = InstallSandbox()
    path, data = await _install(tmp_path, stub, package=False)

    binary = f"{SANDBOX_INSTALL_DIR}/claude-{VERSION}-{PLATFORM}"
    assert path == binary
    assert stub.wrapped() == [
        (SANDBOX_INSTALL_DIR, "ensure"),
        (SANDBOX_INSTALL_DIR, "check"),
        (SANDBOX_INSTALL_DIR, "write"),
        (SANDBOX_INSTALL_DIR, "check"),
    ]
    [write] = [c for c in stub.calls if c.input is not None]
    assert write.input == data
    _assert_framework_operations(stub)


async def test_package_is_extracted_and_verified(tmp_path: Path) -> None:
    stub = InstallSandbox()
    path, data = await _install(tmp_path, stub, package=True)

    install_dir = f"{SANDBOX_INSTALL_DIR}/codex-{VERSION}-{PLATFORM}"
    assert path == f"{install_dir}/bin/codex"
    assert stub.wrapped() == [
        (SANDBOX_INSTALL_DIR, "ensure"),
        (install_dir, "ensure"),
        (install_dir, "check"),
        (install_dir, "extract"),
        (install_dir, "check"),
        (install_dir, "write"),
    ]
    [extract, marker] = [c for c in stub.calls if c.input is not None]
    assert extract.input == data
    assert marker.command[4] == PACKAGE_COMPLETE
    assert stub.artifacts[install_dir] == "ok"
    _assert_framework_operations(stub)


async def test_interrupted_package_install_is_extracted_again(
    tmp_path: Path,
) -> None:
    # an extraction that fails after writing the entrypoint leaves no
    # completion marker, so the next install extracts again rather than
    # running a partial package
    install_dir = f"{SANDBOX_INSTALL_DIR}/codex-{VERSION}-{PLATFORM}"
    stub = InstallSandbox(fail_extract=True)
    with pytest.raises(RuntimeError, match="Unable to extract"):
        await _install(tmp_path, stub, package=True)
    assert stub.artifacts[install_dir] == "partial"

    stub.fail_extract = False
    stub.calls.clear()
    await _install(tmp_path, stub, package=True)

    assert [kind for _, kind in stub.wrapped()][-4:] == [
        "check",
        "extract",
        "check",
        "write",
    ]
    assert stub.artifacts[install_dir] == "ok"


@pytest.mark.parametrize("package", [False, True])
async def test_verified_artifact_is_reused(tmp_path: Path, package: bool) -> None:
    name = f"codex-{VERSION}-{PLATFORM}" if package else f"claude-{VERSION}-{PLATFORM}"
    stub = InstallSandbox({f"{SANDBOX_INSTALL_DIR}/{name}": "ok"})
    await _install(tmp_path, stub, package)

    assert [kind for _, kind in stub.wrapped()][-1] == "check"
    assert all(c.input is None for c in stub.calls)
    _assert_framework_operations(stub)


@pytest.mark.parametrize("package", [False, True])
async def test_tampered_artifact_is_refused_before_use(
    tmp_path: Path, package: bool
) -> None:
    name = f"codex-{VERSION}-{PLATFORM}" if package else f"claude-{VERSION}-{PLATFORM}"
    stub = InstallSandbox({f"{SANDBOX_INSTALL_DIR}/{name}": "tampered"})
    with pytest.raises(RuntimeError, match="cannot be used inside a sandbox identity"):
        await _install(tmp_path, stub, package)

    # refused, not rewritten, and nothing ran it
    assert [kind for _, kind in stub.wrapped()][-1] == "check"
    assert all(c.input is None for c in stub.calls)
    _assert_framework_operations(stub)


@pytest.mark.parametrize("package", [False, True])
async def test_auto_installs_a_verified_copy_in_a_scope(
    tmp_path: Path, package: bool
) -> None:
    # a binary on the member's PATH may have been planted by a peer, so
    # inside a scope "auto" never looks for one and never runs it
    source = _source(tmp_path, package)
    data = b"package-archive" if package else b"binary"
    resolved = AgentBinaryVersion(VERSION, "digest", "https://example.com/x", package)
    stub = InstallSandbox()
    with (
        patch.object(
            agentbinary, "detect_sandbox_platform", AsyncMock(return_value=PLATFORM)
        ),
        patch.object(agentbinary, "trace", lambda message: None),
        patch.object(
            agentbinary,
            "download_agent_binary_async",
            AsyncMock(return_value=AgentBinaryDownload(data, resolved, True, None)),
        ),
        scope(stub) as sandbox,
    ):
        probe = AsyncMock(return_value="9.9.9")
        path = await ensure_agent_binary_installed(
            source, "auto", MEMBER, sandbox, sandbox_version=probe
        )

    assert path.startswith(f"{SANDBOX_INSTALL_DIR}/")
    assert not any("which" in " ".join(c.command) for c in stub.calls)
    probe.assert_not_awaited()
    _assert_framework_operations(stub)


async def test_sandbox_version_is_refused_in_a_scope(tmp_path: Path) -> None:
    stub = InstallSandbox()
    with scope(stub) as sandbox:
        with pytest.raises(SandboxIdentityError, match="version='sandbox'"):
            await ensure_agent_binary_installed(
                _source(tmp_path, False), "sandbox", MEMBER, sandbox
            )
    assert stub.calls == []


async def test_auto_outside_a_scope_still_uses_the_sandbox_binary(
    tmp_path: Path,
) -> None:
    stub = InstallSandbox()
    with patch.object(agentbinary, "trace", lambda message: None):
        await ensure_agent_binary_installed(
            _source(tmp_path, False), "auto", None, SandboxEnvironmentProxy(stub)
        )
    assert [c.command[-1] for c in stub.calls] == ["which claude"]


# The two scripts, run for real. They use only POSIX sh, find, tar and chmod.


def _check(
    directory: Path,
    tree: str,
    executable: str,
    complete: str | None = None,
    uid: int | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if uid is not None:
        # an `id` that reports another uid: every entry is then owned by
        # someone other than the user running the check
        fake = directory.parent / "fake-bin"
        fake.mkdir(exist_ok=True)
        (fake / "id").write_text(f"#!/bin/sh\necho {uid}\n")
        (fake / "id").chmod(0o755)
        env["PATH"] = f"{fake}:{env['PATH']}"
    return subprocess.run(
        ["sh", "-c", _CHECK, "sh", tree, executable, complete or executable],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
    )


MARKER = f"./{PACKAGE_COMPLETE}"


@pytest.fixture
def install_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "install"
    directory.mkdir(mode=0o755)
    return directory


def test_check_reports_a_missing_executable(install_dir: Path) -> None:
    result = _check(install_dir, "./claude", "./claude")
    assert (result.returncode, result.stdout) == (0, "missing\n")


def test_check_accepts_an_owned_unwritable_binary(install_dir: Path) -> None:
    binary = install_dir / "claude"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    result = _check(install_dir, "./claude", "./claude")
    assert (result.returncode, result.stdout) == (0, "ok\n")


@pytest.mark.parametrize("mode", [0o775, 0o757])
def test_check_refuses_a_writable_binary(install_dir: Path, mode: int) -> None:
    binary = install_dir / "claude"
    binary.write_bytes(b"binary")
    binary.chmod(mode)
    result = _check(install_dir, "./claude", "./claude")
    assert result.returncode == 3
    assert "writable by group or others" in result.stderr


def test_check_refuses_a_binary_owned_by_another_user(install_dir: Path) -> None:
    binary = install_dir / "claude"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    result = _check(install_dir, "./claude", "./claude", uid=os.getuid() + 1)
    assert result.returncode == 3
    assert "./claude" in result.stderr


def test_check_refuses_a_symbolic_link(install_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.write_bytes(b"binary")
    target.chmod(0o755)
    (install_dir / "claude").symlink_to(target)
    result = _check(install_dir, "./claude", "./claude")
    assert result.returncode == 3
    assert "not an executable regular file" in result.stderr


def test_check_refuses_a_writable_entry_anywhere_in_a_package(
    install_dir: Path,
) -> None:
    (install_dir / "bin").mkdir(mode=0o755)
    (install_dir / "bin" / "codex").write_bytes(b"codex")
    (install_dir / "bin" / "codex").chmod(0o755)
    (install_dir / "lib").mkdir()
    (install_dir / "lib").chmod(0o777)
    # refused whether or not the package was marked complete
    assert _check(install_dir, ".", "./bin/codex", MARKER).returncode == 3
    (install_dir / MARKER).write_bytes(b"")
    assert _check(install_dir, ".", "./bin/codex", MARKER).returncode == 3
    (install_dir / "lib").chmod(0o755)
    assert _check(install_dir, ".", "./bin/codex", MARKER).stdout == "ok\n"


def test_check_reports_an_unmarked_package_as_missing(install_dir: Path) -> None:
    # the entrypoint alone (an interrupted extraction) is not an install
    (install_dir / "bin").mkdir(mode=0o755)
    (install_dir / "bin" / "codex").write_bytes(b"codex")
    (install_dir / "bin" / "codex").chmod(0o755)
    assert _check(install_dir, ".", "./bin/codex", MARKER).stdout == "missing\n"
    (install_dir / MARKER).write_bytes(b"")
    assert _check(install_dir, ".", "./bin/codex", MARKER).stdout == "ok\n"


def _archive(*members: tuple[str, int, bytes | None]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, mode, content in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.uid = 4242
            if content is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_extract_gives_others_read_and_run_but_never_write(install_dir: Path) -> None:
    archive = _archive(
        ("bin", 0o777, None),
        ("bin/codex", 0o777, b"codex"),
        ("bin/notes.txt", 0o666, b"notes"),
    )
    # the verified-directory script runs the extraction under umask 077
    result = subprocess.run(
        ["sh", "-c", f"umask 077; {_EXTRACT}"],
        cwd=install_dir,
        input=archive,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

    def mode(path: str) -> int:
        return stat.S_IMODE((install_dir / path).stat().st_mode)

    assert (mode("bin"), mode("bin/codex"), mode("bin/notes.txt")) == (
        0o755,
        0o755,
        0o644,
    )
    assert (install_dir / "bin" / "codex").stat().st_uid == os.getuid()
    assert _check(install_dir, ".", "./bin/codex").stdout == "ok\n"
