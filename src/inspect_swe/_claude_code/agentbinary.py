import re
from pathlib import Path

from pydantic import BaseModel
from typing_extensions import Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform


def claude_code_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("claude-code-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        gcs_bucket = await _claude_code_gcs_bucket()
        version = await _claude_code_version(gcs_bucket, version)
        manifest = await _claude_code_manifest(gcs_bucket, version)
        expected_checksum = _checksum_for_platform(manifest, platform)
        download_url = f"{gcs_bucket}/{version}/{platform}/claude"
        return AgentBinaryVersion(version, expected_checksum, download_url)

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"claude-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("claude-*"))

    return AgentBinarySource(
        agent="claude code",
        binary="claude",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install="config list",
    )


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
