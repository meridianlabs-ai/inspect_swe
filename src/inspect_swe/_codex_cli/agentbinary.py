from pathlib import Path

from typing_extensions import Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.sandbox import SandboxPlatform


def codex_cli_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("codex-cli-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        version = "TODO"
        expected_checksum = "TODO"
        download_url = "TODO"

        return AgentBinaryVersion(version, expected_checksum, download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"codex-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("codex-*"))

    return AgentBinarySource(
        agent="codex cli",
        binary="codex",
        post_install=None,
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
    )
