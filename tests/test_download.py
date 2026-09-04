import asyncio
import sys
from pathlib import Path
from typing import Callable, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from inspect_swe import (
    cached_agent_binaries,
    download_agent_binary,
    download_wheels_tarball,
    resolve_agent_version,
)
from inspect_swe._util.download import download_file


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


def test_opencode_download_routes_to_opencode_source() -> None:
    # the docs' offline-install path (download_agent_binary("opencode", ...))
    # must route to the opencode binary source rather than raising ValueError
    from inspect_swe._tools import download as download_tool

    mock_download = AsyncMock(return_value=(b"", None))
    with patch.object(download_tool, "download_agent_binary_async", mock_download):
        download_agent_binary("opencode", "1.14.30", "linux-x64")

    assert mock_download.await_args is not None
    source, version, platform = mock_download.await_args.args
    assert source.agent == "opencode"
    assert version == "1.14.30"
    assert platform == "linux-x64"


def test_cached_agent_binaries_lists_opencode(tmp_path: Path) -> None:
    from inspect_swe._opencode import agentbinary as opencode_agentbinary

    with patch.object(opencode_agentbinary, "package_cache_dir", return_value=tmp_path):
        (tmp_path / "opencode-package-1.14.30-linux-x64.tar.gz").write_bytes(b"x")
        cached = cached_agent_binaries("opencode")

    assert [(b.agent, b.version) for b in cached] == [("opencode", "1.14.30")]


def test_cached_agent_binaries_lists_kimi_code(tmp_path: Path) -> None:
    from inspect_swe._kimi_code import agentbinary as kimi_agentbinary

    with patch.object(kimi_agentbinary, "package_cache_dir", return_value=tmp_path):
        (tmp_path / "kimi-code-1.2.3-linux-x64").write_bytes(b"x")
        cached = cached_agent_binaries("kimi_code")

    assert [(b.agent, b.version) for b in cached] == [("kimi_code", "1.2.3")]


def test_resolve_agent_version_passes_through_pinned_version() -> None:
    # an explicit version needs no network: it is returned unchanged
    assert resolve_agent_version("codex_cli", "0.50.0") == "0.50.0"


def test_resolve_agent_version_rejects_unknown_platform() -> None:
    with pytest.raises(ValueError, match="Unsupported platform"):
        resolve_agent_version("codex_cli", "0.50.0", "win-x64")  # type: ignore[arg-type]


@pytest.mark.parametrize("version", ["stable", "1.2.3"])
def test_resolve_agent_version_rejects_unknown_agent(version: str) -> None:
    with pytest.raises(ValueError, match="claude_code, codex_cli, kimi_code"):
        resolve_agent_version("gemini_cli", version)  # type: ignore[arg-type]


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
    wheels_cache_cleanup: Path,
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


def test_mini_swe_agent_pip_failure_preserves_error(
    mock_pip_download_failure: MagicMock,
) -> None:
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
def test_mini_swe_agent_cache_hit(wheels_cache_cleanup: Path) -> None:
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


def test_ensure_pip_available_noop_when_present() -> None:
    from inspect_swe._util.agentwheel import _ensure_pip_available

    with (
        patch(
            "inspect_swe._util.agentwheel.importlib.util.find_spec",
            return_value=MagicMock(),
        ),
        patch("inspect_swe._util.agentwheel.subprocess.run") as mock_run,
    ):
        _ensure_pip_available()
        mock_run.assert_not_called()


def test_ensure_pip_available_bootstraps_when_missing() -> None:
    from inspect_swe._util.agentwheel import _ensure_pip_available

    with (
        patch(
            "inspect_swe._util.agentwheel.importlib.util.find_spec", return_value=None
        ),
        patch("inspect_swe._util.agentwheel.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        _ensure_pip_available()

        mock_run.assert_called_once_with(
            [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
            capture_output=True,
            text=True,
        )


def _download_with_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> bytes:
    """Run download_file against a mock transport with zero retry delays."""
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_with_mock_transport(**kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=transport, **kwargs)  # type: ignore[arg-type]

    with (
        patch("httpx.AsyncClient", client_with_mock_transport),
        patch("inspect_swe._util.download._RETRY_DELAYS", (0.0, 0.0, 0.0)),
    ):
        return asyncio.run(download_file("https://example.com/file"))


def test_download_file_retries_transient_errors() -> None:
    """Transient transport errors (e.g. read timeouts) are retried."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, content=b"ok")

    assert _download_with_transport(handler) == b"ok"
    assert calls == 3


def test_download_file_retries_server_errors() -> None:
    """5xx responses are treated as transient and retried."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    assert _download_with_transport(handler) == b"ok"
    assert calls == 2


def test_download_file_does_not_retry_client_errors() -> None:
    """4xx responses are permanent and raise immediately."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        _download_with_transport(handler)
    assert calls == 1


def test_download_file_raises_after_retries_exhausted() -> None:
    """The last error is raised once all attempts are used."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(httpx.ReadTimeout):
        _download_with_transport(handler)
    assert calls == 4


def test_ensure_pip_available_raises_on_failure() -> None:
    from inspect_swe._util.agentwheel import _ensure_pip_available

    with (
        patch(
            "inspect_swe._util.agentwheel.importlib.util.find_spec", return_value=None
        ),
        patch("inspect_swe._util.agentwheel.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            returncode=1, stderr="ensurepip disabled", stdout=""
        )

        with pytest.raises(RuntimeError, match="pip is required") as exc_info:
            _ensure_pip_available()

        assert "ensurepip disabled" in str(exc_info.value)
