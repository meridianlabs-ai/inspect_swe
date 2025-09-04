import json
from pathlib import Path
from typing import Any

from typing_extensions import Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform
from .._util.tarball import extract_tarball


def codex_cli_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("codex-cli-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        # Resolve version alias if needed
        if version in ["stable", "latest"]:
            version = await _fetch_latest_stable_version()

        # Get release information
        release = await _fetch_release_assets(version)

        # Get the platform-specific asset
        arch = _platform_to_codex_arch(platform)
        asset_name = f"codex-{arch}.tar.gz"

        # Find the matching asset
        asset = None
        for a in release.get("assets", []):
            if a["name"] == asset_name:
                asset = a
                break

        if asset is None:
            raise RuntimeError(
                f"No asset found for platform {platform} in version {version}"
            )

        # Extract checksum (format: "sha256:xxx")
        digest = asset.get("digest", "")
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"Invalid digest format: {digest}")
        expected_checksum = digest[7:]  # Remove "sha256:" prefix

        # Get download URL
        download_url = asset["browser_download_url"]

        return AgentBinaryVersion(version, expected_checksum, download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"codex-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("codex-*"))

    return AgentBinarySource(
        agent="codex cli",
        binary="codex",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=extract_tarball,
        post_install=None,
    )


def _platform_to_codex_arch(platform: SandboxPlatform) -> str:
    """Map SandboxPlatform to Codex architecture string.

    Always use musl variants for better compatibility since they're
    statically linked and don't depend on system GLIBC version.
    """
    platform_map = {
        "linux-x64": "x86_64-unknown-linux-musl",
        "linux-x64-musl": "x86_64-unknown-linux-musl",
        "linux-arm64": "aarch64-unknown-linux-musl",
        "linux-arm64-musl": "aarch64-unknown-linux-musl",
    }
    if platform not in platform_map:
        raise ValueError(f"Unsupported platform: {platform}")
    return platform_map[platform]


async def _fetch_latest_stable_version() -> str:
    """Fetch the latest stable version from GitHub releases."""
    releases_url = "https://api.github.com/repos/openai/codex/releases"
    releases_json = await download_text_file(releases_url)
    releases = json.loads(releases_json)

    # Filter out pre-releases and alpha versions
    stable_releases = [
        r
        for r in releases
        if not r.get("prerelease", False) and "-alpha" not in r.get("tag_name", "")
    ]

    if not stable_releases:
        raise RuntimeError("No stable releases found for codex")

    # Get the most recent stable release
    latest = stable_releases[0]
    tag_name = latest["tag_name"]

    # Extract version from tag (e.g., "rust-v0.29.0" -> "0.29.0")
    if tag_name.startswith("rust-v"):
        result: str = tag_name[6:]  # Remove "rust-v" prefix
        return result
    else:
        raise RuntimeError(f"Unexpected tag format: {tag_name}")


async def _fetch_release_assets(version: str) -> dict[str, Any]:
    """Fetch release assets for a specific version."""
    tag = f"rust-v{version}"
    release_url = f"https://api.github.com/repos/openai/codex/releases/tags/{tag}"
    release_json = await download_text_file(release_url)
    result: dict[str, Any] = json.loads(release_json)
    return result
