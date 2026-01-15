from typing import Literal

import pytest
from inspect_swe import (
    _list_cached_wheels,
    _wheels_cache_dir,
    cached_agent_binaries,
    download_agent_binary,
    download_wheels_tarball,
)


@pytest.mark.slow
def test_claude_code_binary_download() -> None:
    """Test Claude Code binary download and checksum verification.

    Downloads the stable Claude Code binary for linux-x64 and verifies:
    - Download completes successfully
    - Checksum verification passes (implicit - raises on failure)
    - Binary is cached locally
    """
    download_agent_binary("claude_code", "stable", "linux-x64")

    cached = cached_agent_binaries("claude_code")
    assert len(cached) >= 1
    assert cached[0].agent == "claude_code"
    assert cached[0].path.exists()
    assert cached[0].path.stat().st_size > 0


@pytest.mark.slow
def test_codex_cli_binary_download() -> None:
    """Test Codex CLI binary download and checksum verification.

    Downloads the stable Codex CLI binary for linux-x64 and verifies:
    - Download completes successfully
    - Checksum verification passes (implicit - raises on failure)
    - Binary is cached locally
    """
    download_agent_binary("codex_cli", "stable", "linux-x64")

    cached = cached_agent_binaries("codex_cli")
    assert len(cached) >= 1
    assert cached[0].agent == "codex_cli"
    assert cached[0].path.exists()
    assert cached[0].path.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "version,platform",
    [
        ("1.16.0", "linux-x64"),
        (None, "linux-x64"),  # Latest
        ("1.17.4", "linux-arm64"),
        ("1.17.4", "linux-x64"),
    ],
)
def test_mini_swe_agent_wheels_download(
    version: str | None,
    platform: Literal["linux-x64", "linux-arm64", "linux-x64-musl", "linux-arm64-musl"],
) -> None:
    """Test mini-swe-agent wheels download and caching.

    Downloads wheels for the specified version/platform and verifies:
    - Download completes successfully
    - Version is resolved correctly
    - Tarball has content
    - Wheels are cached locally
    """
    tarball, resolved_version = download_wheels_tarball(
        package_name="mini-swe-agent",
        version=version,
        platform=platform,
        python_version="312",
    )

    # Version should be resolved
    assert resolved_version is not None
    if version is not None:
        assert resolved_version == version

    # Tarball should have content
    assert len(tarball) > 0

    # Wheels should be cached locally
    cached = _list_cached_wheels("mini-swe-agent")
    assert len(cached) >= 1

    # Find the cache file for this version/platform
    cache_dir = _wheels_cache_dir("mini-swe-agent")
    cache_file = (
        cache_dir / f"mini-swe-agent-{resolved_version}-{platform}-py312.tar.gz"
    )
    assert cache_file.exists()
    assert cache_file.stat().st_size > 0
