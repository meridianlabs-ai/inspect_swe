import json
from pathlib import Path
from typing import Any

from inspect_ai.util import SandboxEnvironment
from typing_extensions import Literal

from .._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
    ensure_agent_binary_installed,
)
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.node import ensure_node_available
from .._util.ripgrep import ensure_ripgrep_available
from .._util.sandbox import SandboxPlatform, detect_sandbox_platform
from .._util.tarball import extract_tarball


async def ensure_opencode_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, list[str]]:
    """Install OpenCode and return its binary plus dependency bin directories.

    OpenCode ships a standalone, self-contained binary per platform on GitHub
    releases, so it is acquired through the same host-side download-and-stage path
    as claude_code and codex_cli (``ensure_agent_binary_installed``): a ``which``
    probe reuses an already-installed/overlaid binary, otherwise the runner fetches
    the release tarball, verifies its checksum, and writes the binary into the
    sandbox. Nothing runs ``npm`` — and nothing downloads from inside the sandbox —
    so this works in a network-isolated sandbox where an in-sandbox fetch would hang.
    """
    platform = await detect_sandbox_platform(sandbox)

    node_binary = await ensure_node_available(sandbox, platform, user)
    dependency_bin_dirs = [
        node_binary.rsplit("/", 1)[0],
        await ensure_ripgrep_available(sandbox, platform, user),
    ]

    opencode_binary = await ensure_agent_binary_installed(
        opencode_binary_source(), version, user, sandbox
    )
    return opencode_binary, dependency_bin_dirs


def opencode_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("opencode-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        # Resolve version alias if needed
        if version in ["stable", "latest"]:
            version = await _fetch_latest_version()

        # The release asset name is the platform string verbatim, and each variant
        # is dynamically linked against its own libc (glibc for linux-x64, musl for
        # linux-x64-musl, ...) — so the platform-matched asset is correct by
        # construction. detect_sandbox_platform picks the right libc for the sandbox.
        release = await _fetch_release_assets(version)
        asset_name = f"opencode-{platform}.tar.gz"
        assets = {a["name"]: a for a in release.get("assets", [])}
        asset = assets.get(asset_name)
        if asset is None:
            raise RuntimeError(f"No asset {asset_name!r} in opencode release {version}")

        # Extract checksum (format: "sha256:xxx")
        digest = asset.get("digest", "")
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"Invalid digest format: {digest}")
        expected_checksum = digest[7:]  # Remove "sha256:" prefix

        download_url = asset["browser_download_url"]
        return AgentBinaryVersion(version, expected_checksum, download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"opencode-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("opencode-*"))

    return AgentBinarySource(
        agent="opencode",
        binary="opencode",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        # The release asset is a tar.gz wrapping a single binary; unwrap it
        # host-side so the staged file is the executable itself.
        post_download=extract_tarball,
        post_install=None,
    )


async def _fetch_latest_version() -> str:
    """Fetch the latest released opencode version from GitHub."""
    latest_url = "https://api.github.com/repos/anomalyco/opencode/releases/latest"
    latest = json.loads(await download_text_file(latest_url))
    tag_name = str(latest["tag_name"])
    return tag_name.lstrip("v")


async def _fetch_release_assets(version: str) -> dict[str, Any]:
    """Fetch release assets for a specific version."""
    tag = f"v{version}"
    release_url = f"https://api.github.com/repos/anomalyco/opencode/releases/tags/{tag}"
    release_json = await download_text_file(release_url)
    result: dict[str, Any] = json.loads(release_json)
    return result
