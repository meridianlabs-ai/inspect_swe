import json
from pathlib import Path
from typing import Any

from typing_extensions import Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform
from .._util.tarball import extract_tarball

# The Antigravity CLI publishes signed, digest-bearing release assets on a public
# GitHub repo, so version resolution is the same shape as codex's: read the
# release, pick the platform asset, take its sha256 digest.
_RELEASES_REPO = "google-antigravity/antigravity-cli"

# Only glibc Linux builds are published: there is no musl asset. Fail loud on a
# musl platform rather than installing the glibc binary, which would die at exec
# time with an unhelpful "not found" from the loader.
_PLATFORM_ASSETS: dict[SandboxPlatform, str] = {
    "linux-x64": "agy_cli_linux_x64.tar.gz",
    "linux-arm64": "agy_cli_linux_arm64.tar.gz",
}


def antigravity_cli_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("antigravity-cli-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        # Resolve the platform first: an unsupported platform is knowable
        # without the network, and reporting it as a failed release lookup
        # (which is what an offline or rate-limited `stable` resolution raises)
        # hides the actual cause.
        asset_name = _PLATFORM_ASSETS.get(platform)
        if asset_name is None:
            raise ValueError(
                f"Unsupported platform for the Antigravity CLI: {platform}. "
                f"Supported: {sorted(_PLATFORM_ASSETS)}."
            )

        if version in ("stable", "latest"):
            version = await _fetch_latest_version()

        release = await _fetch_release(version)
        assets = {
            asset["name"]: asset
            for asset in release.get("assets", [])
            if isinstance(asset, dict)
        }
        asset = assets.get(asset_name)
        if asset is None:
            raise RuntimeError(
                f"No asset {asset_name} in Antigravity CLI release {version}"
            )

        digest = asset.get("digest", "")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise RuntimeError(f"Invalid digest format: {digest!r}")

        download_url = asset.get("browser_download_url")
        if not isinstance(download_url, str):
            raise RuntimeError(
                f"No download URL for asset {asset_name} in Antigravity CLI "
                f"release {version}: {download_url!r}"
            )

        return AgentBinaryVersion(version, digest.removeprefix("sha256:"), download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"agy-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("agy-*"))

    return AgentBinarySource(
        # The release asset is a tar.gz holding exactly one file (named
        # `antigravity`), which is what extract_tarball requires -- and the file
        # is installed under the `agy` name the CLI is invoked as.
        agent="antigravity cli",
        binary="agy",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=extract_tarball,
        post_install=None,
    )


async def _fetch_latest_version() -> str:
    latest = json.loads(
        await download_text_file(
            f"https://api.github.com/repos/{_RELEASES_REPO}/releases/latest"
        )
    )
    tag_name = latest["tag_name"]
    if not isinstance(tag_name, str):
        raise RuntimeError(f"Unexpected tag format: {tag_name!r}")
    # Antigravity CLI tags are bare versions ("1.1.20"), with no prefix to strip.
    return tag_name


async def _fetch_release(version: str) -> dict[str, Any]:
    release_json = await download_text_file(
        f"https://api.github.com/repos/{_RELEASES_REPO}/releases/tags/{version}"
    )
    result: dict[str, Any] = json.loads(release_json)
    return result
