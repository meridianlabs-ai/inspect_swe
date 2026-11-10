import json
from pathlib import Path
from typing import Any, Literal

from .._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
)
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform


def pi_binary_source() -> AgentBinarySource:
    """Return the host-downloaded, checksum-verified Pi release source."""
    cached_package_dir = package_cache_dir("pi-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        asset_name = _asset_name(platform=platform)
        release = await _fetch_release(version=version)
        assets = {asset["name"]: asset for asset in release.get("assets", [])}
        asset = assets.get(asset_name)
        if asset is None:
            raise RuntimeError(
                f"No matching Pi asset {asset_name!r} in release {version}"
            )
        digest = asset.get("digest") or ""
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"Invalid Pi release digest: {digest}")
        return AgentBinaryVersion(
            version=str(release["tag_name"]).lstrip("v"),
            expected_checksum=digest[7:],
            download_url=asset["browser_download_url"],
            package=True,
        )

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_package_dir / f"pi-{version}-{platform}"

    def cached_package_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_package_dir / f"pi-package-{version}-{platform}.tar.gz"

    return AgentBinarySource(
        agent="pi",
        binary="pi",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=lambda: list(cached_package_dir.glob("pi-*")),
        post_download=None,
        post_install=None,
        package_entrypoint="pi/pi",
        cached_package_path=cached_package_path,
    )


def _asset_name(platform: SandboxPlatform) -> str:
    names = {
        "linux-x64": "pi-linux-x64.tar.gz",
        "linux-arm64": "pi-linux-arm64.tar.gz",
    }
    if platform.endswith("-musl"):
        raise RuntimeError(
            "Pi standalone releases require glibc; musl sandboxes are not supported."
        )
    try:
        return names[platform]
    except KeyError as error:
        raise ValueError(f"Unsupported platform: {platform}") from error


async def _fetch_release(version: str) -> dict[str, Any]:
    if version in ("stable", "latest"):
        url = "https://api.github.com/repos/earendil-works/pi/releases/latest"
    else:
        url = f"https://api.github.com/repos/earendil-works/pi/releases/tags/v{version}"
    return dict(json.loads(await download_text_file(url=url)))
