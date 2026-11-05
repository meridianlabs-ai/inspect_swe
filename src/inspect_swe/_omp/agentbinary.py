import json
import re
from pathlib import Path
from typing import Any

from inspect_ai.util import SandboxEnvironment

from .._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
    ensure_agent_binary_installed,
)
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform


async def ensure_omp_setup(
    sandbox: SandboxEnvironment, version: str, user: str | None
) -> str:
    """Stage OMP's standalone executable using the host download/cache helpers.

    Release binaries embed Bun, the native addon, and core coding tools; no npm,
    Node, or ripgrep installation is needed. OMP extracts its embedded addon into
    its writable HOME at startup. Musl images must already provide libstdc++ and
    libgcc. Optional tools can have additional runtime dependencies; this helper
    neither installs those nor runs a package manager inside the sandbox.
    """
    return await ensure_agent_binary_installed(
        omp_binary_source(), version, user, sandbox
    )


def omp_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("omp-downloads")

    async def resolve_version(
        version: str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        asset_name = _asset_name(platform)
        release_path = (
            "latest" if version in ("stable", "latest") else f"tags/v{version}"
        )
        release_url = (
            f"https://api.github.com/repos/can1357/oh-my-pi/releases/{release_path}"
        )
        release: dict[str, Any] = json.loads(await download_text_file(release_url))
        if version in ("stable", "latest"):
            version = str(release["tag_name"]).removeprefix("v")

        asset = next(
            (a for a in release.get("assets", []) if a["name"] == asset_name), None
        )
        if asset is None:
            raise RuntimeError(
                f"No matching asset for platform {platform!r} in omp release {version}"
            )

        # GitHub's digest covers the executable verbatim, as does the release's
        # SHA256SUMS.txt. Do not install when the API omits or malforms the digest.
        digest = asset.get("digest") or ""
        if not isinstance(digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", digest
        ):
            raise RuntimeError(f"Invalid SHA-256 digest for {asset_name}: {digest!r}")
        return AgentBinaryVersion(
            version, digest.removeprefix("sha256:"), asset["browser_download_url"]
        )

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"omp-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("omp-*"))

    return AgentBinarySource(
        agent="omp",
        binary="omp",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=None,
    )


def _asset_name(platform: SandboxPlatform) -> str:
    # Upstream puts "musl" before the architecture, unlike SandboxPlatform.
    assets: dict[SandboxPlatform, str] = {
        "linux-x64": "omp-linux-x64",
        "linux-arm64": "omp-linux-arm64",
        "linux-x64-musl": "omp-linux-musl-x64",
        "linux-arm64-musl": "omp-linux-musl-arm64",
    }
    try:
        return assets[platform]
    except KeyError:
        raise ValueError(f"Unsupported OMP platform: {platform}") from None
