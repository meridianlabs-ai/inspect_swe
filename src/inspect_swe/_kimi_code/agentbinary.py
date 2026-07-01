import json
from pathlib import Path
from typing import Any

from typing_extensions import Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform

# Kimi Code publishes raw per-platform binaries plus a version pointer and a
# per-version manifest of filenames and sha256 checksums.
KIMI_DIST_URL = "https://code.kimi.com/kimi-code/binaries"
KIMI_LATEST_URL = "https://code.kimi.com/kimi-code/latest.json"


def kimi_code_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("kimi-code-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        if version in ["stable", "latest"]:
            version = await _fetch_latest_version()

        manifest = await _fetch_manifest(version)
        platforms = manifest["platforms"]
        kimi_platform = _platform_to_kimi_platform(platform)
        if kimi_platform not in platforms:
            raise RuntimeError(
                f"No Kimi Code binary for platform {platform} in version {version}"
            )
        entry = platforms[kimi_platform]
        download_url = f"{KIMI_DIST_URL}/{version}/{entry['filename']}"
        return AgentBinaryVersion(version, entry["checksum"], download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"kimi-code-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("kimi-code-*"))

    return AgentBinarySource(
        agent="kimi code",
        binary="kimi",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=None,
    )


def _platform_to_kimi_platform(platform: SandboxPlatform) -> str:
    # Kimi ships glibc linux builds keyed by arch only (no musl variants), so
    # collapse musl platforms onto the plain linux key.
    platform_map = {
        "linux-x64": "linux-x64",
        "linux-x64-musl": "linux-x64",
        "linux-arm64": "linux-arm64",
        "linux-arm64-musl": "linux-arm64",
    }
    if platform not in platform_map:
        raise ValueError(f"Unsupported platform: {platform}")
    return platform_map[platform]


async def _fetch_latest_version() -> str:
    latest = json.loads(await download_text_file(KIMI_LATEST_URL))
    return str(latest["version"])


async def _fetch_manifest(version: str) -> dict[str, Any]:
    manifest_url = f"{KIMI_DIST_URL}/{version}/manifest.json"
    return dict(json.loads(await download_text_file(manifest_url)))
