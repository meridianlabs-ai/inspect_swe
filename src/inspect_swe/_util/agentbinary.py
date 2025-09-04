from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal, NamedTuple

from inspect_ai.util import SandboxEnvironment, concurrency
from inspect_ai.util import sandbox as sandbox_env

from inspect_swe._util.trace import trace

from .checksum import verify_checksum
from .download import download_file
from .sandbox import (
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
    sandbox_exec,
)


class AgentBinaryVersion(NamedTuple):
    version: str
    expected_checksum: str
    download_url: str


@dataclass
class AgentBinarySource:
    agent: str
    binary: str
    resolve_version: Callable[
        [Literal["stable", "latest"] | str, SandboxPlatform],
        Awaitable[AgentBinaryVersion],
    ]
    cached_binary_path: Callable[[str, SandboxPlatform], Path]
    list_cached_binaries: Callable[[], list[Path]]
    post_download: Callable[[bytes], bytes] | None
    post_install: str | None


async def ensure_agent_binary_installed(
    source: AgentBinarySource,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    user: str | None = None,
    sandbox: SandboxEnvironment | None = None,
) -> str:
    # resolve sandbox
    sandbox = sandbox or sandbox_env()

    # look in the sandbox first if we need to
    if version == "auto" or version == "sandbox":
        result = await sandbox.exec(bash_command(f"which {source.binary}"), user=user)
        if result.success:
            binary_path = result.stdout.strip()
            trace(f"Using {source.agent} installed in sandbox: {binary_path}")
            return binary_path

        # if version == "sandbox" and we don't find it that's an error
        if version == "sandbox":
            raise RuntimeError(f"unable to locate {source.agent} in sandbox")

        # otherwise set to "stable"
        version = "stable"

    # detect the sandbox target platform
    platform = await detect_sandbox_platform(sandbox)

    # use concurrency so multiple samples don't attempt the same download all at once
    async with concurrency(f"{source.binary}-install", 1, visible=False):
        # if a specific version is requested, first try to read it directly from the cache
        if version not in ["stable", "latest"]:
            binary_bytes: bytes | None = read_cached_binary(
                source, version, platform, None
            )
            if binary_bytes is not None:
                trace(f"Used claude code binary from cache: {version} ({platform})")
        else:
            binary_bytes = None

        # download the binary
        if binary_bytes is None:
            binary_bytes, resolved_version = await download_agent_binary_async(
                source, version, platform, trace
            )
        else:
            # If we got it from cache, version is already the resolved version
            resolved_version = version

        # write it into the container and return it
        binary_path = f"/opt/{source.binary}-{resolved_version}-{platform}"
        await sandbox.write_file(binary_path, binary_bytes)
        await sandbox_exec(sandbox, f"chmod +x {binary_path}")
        if source.post_install:
            await sandbox_exec(
                sandbox, f"{binary_path} {source.post_install}", user=user
            )
        return binary_path


async def download_agent_binary_async(
    source: AgentBinarySource,
    version: Literal["stable", "latest"] | str,
    platform: SandboxPlatform,
    logger: Callable[[str], None] | None = None,
) -> tuple[bytes, str]:
    # resovle logger
    logger = logger or print

    # determine version and checksum
    version, expected_checksum, download_url = await source.resolve_version(
        version, platform
    )

    # check the cache (if post_download is used, don't verify checksum since cached is processed)
    cache_checksum = None if source.post_download else expected_checksum
    binary_data = read_cached_binary(source, version, platform, cache_checksum)
    if binary_data is None:
        # not in cache, download and verify checksum
        binary_data = await download_file(download_url)
        if not verify_checksum(binary_data, expected_checksum):
            raise ValueError("Checksum verification failed")

        # apply post-download processing if provided (e.g., extract from tar.gz)
        if source.post_download is not None:
            binary_data = source.post_download(binary_data)

        # save to cache
        write_cached_binary(source, binary_data, version, platform)

        # trace
        logger(f"Downloaded {source.agent} binary: {version} ({platform})")
    else:
        logger(f"Used {source.agent} binary from cache: {version} ({platform})")

    # return data and resolved version
    return binary_data, version


def read_cached_binary(
    source: AgentBinarySource,
    version: str,
    platform: SandboxPlatform,
    expected_checksum: str | None,
) -> bytes | None:
    # no cached binary
    cache_path = source.cached_binary_path(version, platform)
    if not cache_path.exists():
        return None

    # read binary
    with open(cache_path, "rb") as f:
        binary_data = f.read()

    if expected_checksum is None or verify_checksum(binary_data, expected_checksum):
        cache_path.touch()
        return binary_data
    else:
        cache_path.unlink()
        return None


def write_cached_binary(
    source: AgentBinarySource,
    binary_data: bytes,
    version: str,
    platform: SandboxPlatform,
) -> None:
    binary_path = source.cached_binary_path(version, platform)

    with open(binary_path, "wb") as f:
        f.write(binary_data)

    _cleanup_binary_cache(source, keep_count=3)


def _cleanup_binary_cache(source: AgentBinarySource, keep_count: int = 5) -> None:
    # get all cached binaries
    cache_files = source.list_cached_binaries()
    if len(cache_files) <= keep_count:
        return

    # remove oldest
    cache_files.sort(key=lambda f: f.stat().st_atime)
    files_to_remove = cache_files[:-keep_count]
    for file_path in files_to_remove:
        file_path.unlink()
