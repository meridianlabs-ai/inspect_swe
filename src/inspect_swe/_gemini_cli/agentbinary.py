"""Binary source for Google Gemini CLI.

Gemini CLI is distributed as a JavaScript bundle (gemini.js) from GitHub releases.
It requires Node.js to run.
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform

# GitHub API URL for releases
GITHUB_RELEASES_API = "https://api.github.com/repos/google-gemini/gemini-cli/releases"


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

    # post_install command: make the JS file executable and ensure node is available
    # The actual execution will use "node <path>" directly
    post_install_cmd = "chmod +x $BINARY_PATH"

    return AgentBinarySource(
        agent="gemini cli",
        binary="gemini",  # Note: This will be a .js file, run with node
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=post_install_cmd,
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
