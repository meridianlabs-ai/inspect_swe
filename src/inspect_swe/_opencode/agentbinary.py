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
        # Resolve the version and its release assets together:
        # /releases/latest returns both the tag name and the full asset list
        # in one response, so "stable"/"latest" resolves with exactly one
        # API call rather than a tag lookup followed by a second, per-tag
        # lookup. Halving the request count also halves how often a sample
        # queued behind a rate-limited neighbor has to repeat the exact call
        # that just failed. A pinned version still needs its own,
        # unavoidable per-tag lookup.
        if version in ["stable", "latest"]:
            release = await _fetch_latest_release()
            version = str(release["tag_name"]).lstrip("v")
        else:
            release = await _fetch_release_assets(version)

        assets = {a["name"]: a for a in release.get("assets", [])}
        asset = None
        for asset_name in _asset_name_candidates(platform):
            asset = assets.get(asset_name)
            if asset is not None:
                break
        if asset is None:
            raise RuntimeError(
                f"No matching asset for platform {platform!r} in opencode "
                f"release {version}"
            )

        # Extract checksum (format: "sha256:xxx"). GitHub serves
        # "digest": null for assets uploaded before digest support, so
        # normalize None to "" rather than crashing on .startswith.
        digest = asset.get("digest") or ""
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"Invalid digest format: {digest}")
        expected_checksum = digest[7:]  # Remove "sha256:" prefix

        download_url = asset["browser_download_url"]
        # The release asset is a tar.gz wrapping the single opencode binary.
        # Cache it verbatim, checksum and all — as a "package" archive in
        # AgentBinarySource terms — and let ensure_agent_binary_installed
        # extract it in the sandbox at install time, rather than transforming
        # it host-side (post_download) into a blob a later cache hit can no
        # longer verify against the release digest.
        return AgentBinaryVersion(
            version, expected_checksum, download_url, package=True
        )

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        # Never written or read in practice: resolve_version always returns
        # package=True, and opencode never shipped a single-binary cache (the
        # npm era cached bundles under "opencode-bundles"). Defined only
        # because AgentBinarySource requires it.
        return cached_binary_dir / f"opencode-{version}-{platform}"

    def cached_package_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"opencode-package-{version}-{platform}.tar.gz"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("opencode-*"))

    return AgentBinarySource(
        agent="opencode",
        binary="opencode",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=None,
        package_entrypoint="opencode",
        cached_package_path=cached_package_path,
    )


def _asset_name_candidates(platform: SandboxPlatform) -> list[str]:
    """Ordered release-asset name candidates for a platform, most preferred first.

    Every x64 release ships two builds: the default, compiled with AVX2
    instructions, and a "-baseline" build without them. OpenCode's own npm
    postinstall probes the *host's* CPU at install time and picks between
    them; we install host-side, before any in-sandbox probe ever runs, and
    can't reliably read the underlying host's CPU flags from out here — the
    sandbox may be scheduled onto a different physical host than the one
    that eventually runs the binary, and some sandbox providers don't expose
    real CPU flags at all. We deliberately prefer the baseline asset on
    every x64 platform: it trades a little performance on modern hosts for
    never staging a binary that SIGILLs on any host predating AVX2 (~2013),
    which is the safer default for sandbox portability. arm64 has no
    baseline variant (AVX2 is an x86 extension) and needs no fallback.
    """
    if platform in ("linux-x64", "linux-x64-musl"):
        musl = "-musl" if platform.endswith("-musl") else ""
        return [
            f"opencode-linux-x64-baseline{musl}.tar.gz",
            f"opencode-linux-x64{musl}.tar.gz",
        ]
    return [f"opencode-{platform}.tar.gz"]


async def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest opencode release, tag and assets together."""
    latest_url = "https://api.github.com/repos/anomalyco/opencode/releases/latest"
    result: dict[str, Any] = json.loads(await download_text_file(latest_url))
    return result


async def _fetch_release_assets(version: str) -> dict[str, Any]:
    """Fetch release assets for a specific pinned version."""
    tag = f"v{version}"
    release_url = f"https://api.github.com/repos/anomalyco/opencode/releases/tags/{tag}"
    release_json = await download_text_file(release_url)
    result: dict[str, Any] = json.loads(release_json)
    return result
