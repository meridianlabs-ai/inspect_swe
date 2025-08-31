from pathlib import Path

from inspect_swe._util.sandbox import SandboxPlatform

from ..._util.appdirs import package_cache_dir
from ..._util.checksum import verify_checksum


def read_cached_claude_code_binary(
    version: str, platform: SandboxPlatform, expected_checksum: str | None
) -> bytes | None:
    # no cached binary
    cache_path = _claude_code_cached_binary(version, platform)
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


def write_cached_claude_code_binary(
    binary_data: bytes, version: str, platform: SandboxPlatform
) -> None:
    binary_path = _claude_code_cached_binary(version, platform)

    with open(binary_path, "wb") as f:
        f.write(binary_data)

    _cleanup_claude_code_binary_cache(keep_count=3)


def _cleanup_claude_code_binary_cache(keep_count: int = 5) -> None:
    # get all cached binaries
    cache_files = list(_claude_code_cached_binary_dir().glob("claude-*"))
    if len(cache_files) <= keep_count:
        return

    # remove oldest
    cache_files.sort(key=lambda f: f.stat().st_atime)
    files_to_remove = cache_files[:-keep_count]
    for file_path in files_to_remove:
        file_path.unlink()


def _claude_code_cached_binary_dir() -> Path:
    return package_cache_dir("claude-code-downloads")


def _claude_code_cached_binary(version: str, platform: SandboxPlatform) -> Path:
    return _claude_code_cached_binary_dir() / f"claude-{version}-{platform}"
