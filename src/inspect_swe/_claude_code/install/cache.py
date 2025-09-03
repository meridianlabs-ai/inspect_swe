from pathlib import Path

from inspect_swe._util.sandbox import SandboxPlatform

from ..._util.appdirs import package_cache_dir
from ..._util.binarycache import AgentBinaryCache


def claude_code_binary_cache() -> AgentBinaryCache:
    cached_binary_dir = package_cache_dir("claude-code-downloads")

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"claude-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("claude-*"))

    return AgentBinaryCache(
        cached_binary_path=cached_binary_path, list_cached_binaries=list_cached_binaries
    )
