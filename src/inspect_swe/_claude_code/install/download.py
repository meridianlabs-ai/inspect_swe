import re
from typing import Literal

from pydantic import BaseModel

from ..._util._async import run_coroutine
from ..._util.checksum import verify_checksum
from ..._util.download import download_file, download_text_file
from ..._util.sandbox import SandboxPlatform
from ..._util.trace import trace
from .cache import (
    read_cached_claude_code_binary,
    write_cached_claude_code_binary,
)


def download_claude_code(
    version: Literal["stable", "latest"] | str, platform: SandboxPlatform
) -> None:
    """Download Claude Code.

    Download a version of Claude Code. This version will be added to the cache of downloaded versions (which retains the 5 most recently downloaded versions).

    Use this if you need to ensure that a specific version of Claude Code is downloaded in advance (e.g. if you are going to run your evaluations offline). After downloading, explicit requests for the downloaded version (e.g. `claude_code(version="1.0.98")`) will not require network access.

    Args:
        version: Version to download ("stable", "latest", or an explicit version number).
        platform: Target platform ("linux-x64", "linux-arm64", "linux-x64-musl", or "linux-arm64-musl")
    """
    run_coroutine(download_claude_code_async(version, platform))


async def download_claude_code_async(
    version: Literal["stable", "latest"] | str, platform: SandboxPlatform
) -> bytes:
    # determine version and checksum
    gcs_bucket = await _claude_code_gcs_bucket()
    version = await _claude_code_version(gcs_bucket, version)
    manifest = await _claude_code_manifest(gcs_bucket, version)
    expected_checksum = _checksum_for_platform(manifest, platform)

    # check the cache
    binary_data = read_cached_claude_code_binary(version, platform, expected_checksum)
    if binary_data is None:
        # not in cache, download and verify checksum
        binary_url = f"{gcs_bucket}/{version}/{platform}/claude"
        binary_data = await download_file(binary_url)
        if not verify_checksum(binary_data, expected_checksum):
            raise ValueError("Checksum verification failed")

        # save to cache
        write_cached_claude_code_binary(binary_data, version, platform)

        # trace
        trace(f"Downloaded claude code binary: {version} ({platform})")
    else:
        trace(f"Used claude code binary from cache: {version} ({platform})")

    # return data
    return binary_data


async def _claude_code_gcs_bucket() -> str:
    INSTALL_SCRIPT_URL = "https://claude.ai/install.sh"
    script_content = await download_text_file(INSTALL_SCRIPT_URL)
    pattern = r'GCS_BUCKET="(https://storage\.googleapis\.com/[^"]+)"'
    match = re.search(pattern, script_content)
    if match is not None:
        gcs_bucket = match.group(1)
        return gcs_bucket
    else:
        raise RuntimeError("Unable to determine GCS bucket for claude code.")


async def _claude_code_version(gcs_bucket: str, target: str) -> str:
    # validate target
    target_pattern = r"^(stable|latest|[0-9]+\.[0-9]+\.[0-9]+(-[^[:space:]]+)?)$"
    if re.match(target_pattern, target) is None:
        raise RuntimeError(
            "Invalid version target (must be 'stable', 'latest', or a semver version number)"
        )

    # resolve target alias if required
    if target in ["stable", "latest"]:
        version_url = f"{gcs_bucket}/{target}"
        version = await download_text_file(version_url)
        return version
    else:
        return target


class PlatformInfo(BaseModel):
    checksum: str
    size: int


class Manifest(BaseModel):
    version: str
    platforms: dict[str, PlatformInfo]


async def _claude_code_manifest(gcs_bucket: str, version: str) -> Manifest:
    manifest_url = f"{gcs_bucket}/{version}/manifest.json"
    manifest_json = await download_text_file(manifest_url)
    return Manifest.model_validate_json(manifest_json)


def _checksum_for_platform(manifest: Manifest, platform: SandboxPlatform) -> str:
    if platform not in manifest.platforms:
        raise RuntimeError(f"Platform '{platform}' not found in manifest.")
    return manifest.platforms[platform].checksum
