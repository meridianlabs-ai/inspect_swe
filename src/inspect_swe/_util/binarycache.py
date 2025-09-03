from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .checksum import verify_checksum
from .sandbox import SandboxPlatform


@dataclass
class AgentBinaryCache:
    cached_binary_path: Callable[[str, SandboxPlatform], Path]
    list_cached_binaries: Callable[[], list[Path]]


def read_cached_binary(
    cache: AgentBinaryCache,
    version: str,
    platform: SandboxPlatform,
    expected_checksum: str | None,
) -> bytes | None:
    # no cached binary
    cache_path = cache.cached_binary_path(version, platform)
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
    cache: AgentBinaryCache, binary_data: bytes, version: str, platform: SandboxPlatform
) -> None:
    binary_path = cache.cached_binary_path(version, platform)

    with open(binary_path, "wb") as f:
        f.write(binary_data)

    _cleanup_binary_cache(cache, keep_count=3)


def _cleanup_binary_cache(cache: AgentBinaryCache, keep_count: int = 5) -> None:
    # get all cached binaries
    cache_files = cache.list_cached_binaries()
    if len(cache_files) <= keep_count:
        return

    # remove oldest
    cache_files.sort(key=lambda f: f.stat().st_atime)
    files_to_remove = cache_files[:-keep_count]
    for file_path in files_to_remove:
        file_path.unlink()
