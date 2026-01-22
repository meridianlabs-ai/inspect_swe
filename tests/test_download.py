from typing import Literal

import pytest
from inspect_swe import (
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
    wheels_cache_cleanup,
) -> None:
    """Test mini-swe-agent wheels download and caching."""
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

    # Wheels should be cached locally (in isolated temp directory from fixture)
    cache_dir = wheels_cache_cleanup / "mini_swe_agent-wheels"
    cache_file = (
        cache_dir / f"mini-swe-agent-{resolved_version}-{platform}-py312.tar.gz"
    )

    assert cache_file.exists(), f"Cache file not found: {cache_file}"
    assert cache_file.stat().st_size > 0
    # Cleanup handled automatically by wheels_cache_cleanup fixture


@pytest.mark.slow
def test_mini_swe_agent_invalid_version() -> None:
    """Test that invalid version strings raise appropriate errors."""
    with pytest.raises(RuntimeError, match="pip download failed"):
        download_wheels_tarball(
            package_name="mini-swe-agent",
            version="99.99.99",
            platform="linux-x64",
            python_version="312",
        )


def test_mini_swe_agent_unsupported_platform() -> None:
    """Test that unsupported platforms raise appropriate errors."""
    from inspect_swe._util.agentwheel import platform_to_pip_platform

    with pytest.raises(ValueError, match="Unsupported platform"):
        platform_to_pip_platform("unsupported-platform")  # type: ignore[arg-type]


def test_mini_swe_agent_pip_failure_preserves_error(mock_pip_download_failure) -> None:
    """Test that pip download failures include the original error message.

    This verifies that when pip fails, users see the actual pip error
    (e.g., network timeout, package not found) in the exception message.
    """
    # The mock returns this specific error message
    expected_error = "Could not find a version that satisfies the requirement"

    with pytest.raises(RuntimeError, match="pip download failed") as exc_info:
        download_wheels_tarball(
            package_name="mini-swe-agent",
            version="1.17.4",
            platform="linux-x64",
            python_version="312",
        )

    # Verify the original pip error is preserved in the exception
    assert expected_error in str(exc_info.value), (
        f"Expected pip error message to be preserved. Got: {exc_info.value}"
    )


@pytest.mark.slow
def test_mini_swe_agent_cache_hit(wheels_cache_cleanup) -> None:
    """Test that downloading the same version twice uses cache on second request."""
    version = "1.17.4"
    platform: Literal[
        "linux-x64", "linux-arm64", "linux-x64-musl", "linux-arm64-musl"
    ] = "linux-x64"
    python_version = "312"

    # First download
    tarball1, resolved_version1 = download_wheels_tarball(
        package_name="mini-swe-agent",
        version=version,
        platform=platform,
        python_version=python_version,
    )

    # Cache is in isolated temp directory from fixture
    cache_dir = wheels_cache_cleanup / "mini_swe_agent-wheels"
    cache_file = (
        cache_dir / f"mini-swe-agent-{version}-{platform}-py{python_version}.tar.gz"
    )

    # Verify cache file exists
    assert cache_file.exists()
    cache_mtime = cache_file.stat().st_mtime

    # Second download - should hit cache
    tarball2, resolved_version2 = download_wheels_tarball(
        package_name="mini-swe-agent",
        version=version,
        platform=platform,
        python_version=python_version,
    )

    # Verify same tarball returned
    assert tarball1 == tarball2
    assert resolved_version1 == resolved_version2

    # Verify cache file access time was updated (file was touched)
    assert cache_file.stat().st_atime >= cache_mtime
