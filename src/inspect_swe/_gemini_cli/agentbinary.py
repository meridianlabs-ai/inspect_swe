"""Binary source for Google Gemini CLI.

Gemini CLI is distributed as a JavaScript bundle (gemini.js) from GitHub releases.
It requires Node.js to run. If Node.js is not available in the sandbox, we download
a standalone Node.js binary alongside gemini.js.
"""

import json
import lzma
import re
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_file, download_text_file
from .._util.sandbox import SandboxPlatform, bash_command, sandbox_exec

# GitHub API URL for releases
GITHUB_RELEASES_API = "https://api.github.com/repos/google-gemini/gemini-cli/releases"

# Node.js version to download if not available in sandbox
NODE_VERSION = "20.11.0"
NODE_DOWNLOAD_BASE = "https://nodejs.org/dist"


def gemini_cli_binary_source() -> AgentBinarySource:
    """Binary source for Google Gemini CLI.

    Gemini CLI is a JavaScript application distributed as a single bundled file.
    Unlike claude_code which has platform-specific binaries, gemini-cli is
    platform-independent but requires Node.js.
    """
    cached_binary_dir = package_cache_dir("gemini-cli-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        # platform is unused - gemini-cli is platform-independent JavaScript
        _ = platform

        # Get release info from GitHub
        if version in ["stable", "latest"]:
            release = await _get_latest_release()
        else:
            release = await _get_release_by_tag(f"v{version}")

        actual_version = release["tag_name"].lstrip("v")

        # Find the gemini.js asset
        assets = release.get("assets", [])
        gemini_js_asset = None
        for asset in assets:
            if asset["name"] == "gemini.js":
                gemini_js_asset = asset
                break

        if not gemini_js_asset:
            raise RuntimeError(
                f"gemini.js not found in release {release['tag_name']}. "
                f"Available assets: {[a['name'] for a in assets]}"
            )

        download_url = gemini_js_asset["browser_download_url"]

        # Extract checksum from release body if available, use empty string if not found
        # (checksum verification is optional)
        checksum = _extract_checksum_from_body(release.get("body", "")) or ""

        return AgentBinaryVersion(actual_version, checksum, download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        # Platform-independent since it's JavaScript
        _ = platform  # unused
        return cached_binary_dir / f"gemini-{version}.js"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("gemini-*.js"))

    # The actual execution will use "node <path>" directly
    return AgentBinarySource(
        agent="gemini cli",
        binary="gemini",  # Note: This will be a .js file, run with node
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=None,
    )


async def _get_latest_release() -> dict[str, Any]:
    """Get the latest release from GitHub."""
    response = await download_text_file(f"{GITHUB_RELEASES_API}/latest")
    return json.loads(response)


async def _get_release_by_tag(tag: str) -> dict[str, Any]:
    """Get a specific release by tag."""
    response = await download_text_file(f"{GITHUB_RELEASES_API}/tags/{tag}")
    return json.loads(response)


def _extract_checksum_from_body(body: str) -> str | None:
    """Extract SHA256 checksum from release body if present.

    GitHub releases often include checksums in the body text.
    """
    if not body:
        return None

    # Look for SHA256 patterns
    # Common formats:
    # - SHA256: <hash>
    # - sha256sum: <hash>
    # - `<hash>` (64 hex chars)
    patterns = [
        r"SHA256[:\s]+([a-fA-F0-9]{64})",
        r"sha256[:\s]+([a-fA-F0-9]{64})",
        r"`([a-fA-F0-9]{64})`",
    ]

    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    return None


async def ensure_node_available(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Ensure Node.js is available in the sandbox, downloading if necessary.

    Args:
        sandbox: The sandbox environment
        platform: The detected platform
        user: Optional user to run commands as

    Returns:
        Path to the node binary (either system node or downloaded)
    """
    # Check if node is already available
    result = await sandbox.exec(bash_command("which node"), user=user)
    if result.success:
        return result.stdout.strip()

    # Node not found, need to download it
    async with concurrency("node-install", 1, visible=False):
        return await _download_and_install_node(sandbox, platform, user)


async def _download_and_install_node(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Download and install Node.js to the sandbox."""
    # Map platform to Node.js archive name
    node_platform = _platform_to_node_platform(platform)
    archive_name = f"node-v{NODE_VERSION}-{node_platform}.tar.xz"
    download_url = f"{NODE_DOWNLOAD_BASE}/v{NODE_VERSION}/{archive_name}"

    # Check local cache first
    cache_dir = package_cache_dir("node-downloads")
    cache_path = cache_dir / f"node-v{NODE_VERSION}-{node_platform}"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            node_binary = f.read()
    else:
        # Download the archive
        archive_data = await download_file(download_url)

        # Extract the node binary from the tar.xz archive
        node_binary = _extract_node_binary(archive_data, NODE_VERSION, node_platform)

        # Cache the binary
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(node_binary)

    # Write to sandbox
    node_path = "/var/tmp/.5c95f967ca830048/node"
    await sandbox.write_file(node_path, node_binary)
    await sandbox_exec(sandbox, f"chmod +x {node_path}", user="root")

    return node_path


def _platform_to_node_platform(platform: SandboxPlatform) -> str:
    """Map SandboxPlatform to Node.js platform string."""
    platform_map = {
        "linux-x64": "linux-x64",
        "linux-x64-musl": "linux-x64",  # Node.js doesn't have musl-specific builds
        "linux-arm64": "linux-arm64",
        "linux-arm64-musl": "linux-arm64",
    }
    if platform not in platform_map:
        raise ValueError(f"Unsupported platform for Node.js: {platform}")
    return platform_map[platform]


def _extract_node_binary(archive_data: bytes, version: str, node_platform: str) -> bytes:
    """Extract the node binary from a tar.xz archive."""
    # Decompress xz
    decompressed = lzma.decompress(archive_data)

    # Extract from tar
    with tarfile.open(fileobj=BytesIO(decompressed), mode="r") as tar:
        # The node binary is at node-vX.Y.Z-platform/bin/node
        node_path = f"node-v{version}-{node_platform}/bin/node"
        member = tar.getmember(node_path)
        f = tar.extractfile(member)
        if f is None:
            raise RuntimeError(f"Failed to extract {node_path} from archive")
        return f.read()
