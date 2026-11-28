import io
import logging
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anyio
from inspect_ai.util import SandboxEnvironment
from inspect_swe._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
    ensure_agent_binary_installed,
)
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR, SandboxPlatform


class LocalSandbox:
    """Run sandbox commands locally, with install paths isolated in a temporary directory."""

    def __init__(self, directory: Path, gate: anyio.Event | None = None) -> None:
        self.directory = directory
        self.gate = gate
        self.entered = anyio.Event()
        self.writes = 0
        self.writing = False

    async def exec(self, cmd: list[str], **kwargs: object) -> SimpleNamespace:
        command = [
            part.replace(SANDBOX_INSTALL_DIR, str(self.directory)) for part in cmd
        ]
        result = await anyio.run_process(command, check=False)
        return SimpleNamespace(
            success=result.returncode == 0,
            stdout=result.stdout.decode(),
            stderr=result.stderr.decode(),
        )

    async def write_file(self, path: str, data: bytes) -> None:
        assert not self.writing
        self.writing = True
        self.writes += 1
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        target = Path(path.replace(SANDBOX_INSTALL_DIR, str(self.directory)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.writing = False


def test_sandbox_installs_are_isolated(tmp_path: Path) -> None:
    logging.addLevelName(5, "TRACE")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        data = b"#!/bin/sh\necho test\n"
        entry = tarfile.TarInfo("bin/agent")
        entry.mode = 0o755
        entry.size = len(data)
        tar.addfile(entry, io.BytesIO(data))
    package = archive.getvalue()
    cache = tmp_path / "cache.tar.gz"
    cache.write_bytes(package)

    async def resolve(version: str, platform: SandboxPlatform) -> AgentBinaryVersion:
        raise AssertionError("pinned package should be read from cache")

    source = AgentBinarySource(
        agent="concurrency test",
        binary="concurrency-test-agent",
        resolve_version=resolve,
        cached_binary_path=lambda version, platform: tmp_path / "binary",
        list_cached_binaries=lambda: [],
        post_download=None,
        post_install=None,
        package_entrypoint="bin/agent",
        cached_package_path=lambda version, platform: cache,
    )

    async def run() -> None:
        gate = anyio.Event()
        first = LocalSandbox(tmp_path / "first", gate)
        second = LocalSandbox(tmp_path / "second", gate)

        async def install(sandbox: LocalSandbox) -> None:
            await ensure_agent_binary_installed(
                source=source, version="1.0", sandbox=cast(SandboxEnvironment, sandbox)
            )

        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(install, first)
                group.start_soon(install, first)
                group.start_soon(install, second)
                await first.entered.wait()
                await second.entered.wait()
                gate.set()

        assert first.writes == 1
        assert second.writes == 1
        for sandbox in (first, second):
            # The platform can differ across hosts; find the installed binary instead.
            installed = list(sandbox.directory.glob("*/bin/agent"))
            assert len(installed) == 1
            assert installed[0].read_bytes() == data

    anyio.run(run)
