"""Release metadata and host download boundaries for OMP standalone binaries."""

import hashlib
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_swe import (
    cached_agent_binaries,
    download_agent_binary,
)
from inspect_swe._omp import agentbinary
from inspect_swe._util import agentbinary as binary_util
from inspect_swe._util.checksum import ChecksumMismatchError
from inspect_swe._util.sandbox import SandboxPlatform

# Actual asset names and digests from the v18.6.1 GitHub release, also published
# in its SHA256SUMS.txt. No test downloads the large standalone executables.
_ASSETS = {
    "omp-linux-x64": "c92a6846d02984e84f07c6362d18add3783f528f494ffcf1e0f26e594f327463",
    "omp-linux-arm64": "cb7815330bb117877e4e133562ea82e3001ee470e47393552507b5a7f0407f4a",
    "omp-linux-musl-x64": "6b463bb8d808706bd727df99a0fa938928bb51a83d92964ebad35f62a9a915a6",
    "omp-linux-musl-arm64": "a1f89c25f2cd19a5c47ddefa314302ef610c199e1df2bcf6a375e8712eb0e879",
}
_RELEASE_URL = "https://api.github.com/repos/can1357/oh-my-pi/releases/"


def _release(version: str = "18.6.1") -> dict[str, Any]:
    return {
        "tag_name": f"v{version}",
        "assets": [
            {
                "name": name,
                "digest": f"sha256:{digest}",
                "browser_download_url": (
                    "https://github.com/can1357/oh-my-pi/releases/download/"
                    f"v{version}/{name}"
                ),
            }
            for name, digest in _ASSETS.items()
        ],
    }


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agentbinary, "package_cache_dir", lambda _: tmp_path)


@pytest.mark.parametrize(
    ("platform", "asset_name"),
    [
        ("linux-x64", "omp-linux-x64"),
        ("linux-arm64", "omp-linux-arm64"),
        ("linux-x64-musl", "omp-linux-musl-x64"),
        ("linux-arm64-musl", "omp-linux-musl-arm64"),
    ],
)
def test_pinned_release_selects_exact_standalone_asset(
    platform: SandboxPlatform, asset_name: str
) -> None:
    source = agentbinary.omp_binary_source()
    with patch.object(
        agentbinary,
        "download_text_file",
        AsyncMock(return_value=json.dumps(_release())),
    ) as fetch:
        resolved = anyio.run(source.resolve_version, "18.6.1", platform)

    fetch.assert_awaited_once_with(f"{_RELEASE_URL}tags/v18.6.1")
    assert resolved.version == "18.6.1"
    assert resolved.expected_checksum == _ASSETS[asset_name]
    assert resolved.download_url == (
        f"https://github.com/can1357/oh-my-pi/releases/download/v18.6.1/{asset_name}"
    )
    assert resolved.package is False


@pytest.mark.parametrize("version", ["stable", "latest"])
def test_release_alias_resolves_tag_and_assets_together(version: str) -> None:
    source = agentbinary.omp_binary_source()
    with patch.object(
        agentbinary,
        "download_text_file",
        AsyncMock(return_value=json.dumps(_release())),
    ) as fetch:
        resolved = anyio.run(source.resolve_version, version, "linux-x64")

    fetch.assert_awaited_once_with(f"{_RELEASE_URL}latest")
    assert resolved.version == "18.6.1"
    assert resolved.expected_checksum == _ASSETS["omp-linux-x64"]


@pytest.mark.parametrize("platform", ["darwin-arm64", "windows-x64", "linux-riscv64"])
def test_unsupported_platform_fails_before_network(platform: str) -> None:
    source = agentbinary.omp_binary_source()
    with (
        patch.object(agentbinary, "download_text_file", AsyncMock()) as fetch,
        pytest.raises(ValueError, match="Unsupported OMP platform"),
    ):
        anyio.run(source.resolve_version, "18.6.1", cast(SandboxPlatform, platform))
    fetch.assert_not_awaited()


def test_release_without_target_asset_fails() -> None:
    release = _release()
    release["assets"] = [release["assets"][0]]  # glibc x64 cannot substitute for musl
    with (
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(return_value=json.dumps(release)),
        ),
        pytest.raises(RuntimeError, match="No matching asset.*linux-x64-musl"),
    ):
        anyio.run(
            agentbinary.omp_binary_source().resolve_version, "18.6.1", "linux-x64-musl"
        )


@pytest.mark.parametrize(
    "digest", [None, "", "sha512:" + "a" * 64, "sha256:abc", "sha256:" + "g" * 64, 123]
)
def test_missing_or_invalid_digest_fails_closed(digest: object) -> None:
    release = _release()
    release["assets"][0]["digest"] = digest
    with (
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(return_value=json.dumps(release)),
        ),
        pytest.raises(RuntimeError, match="Invalid SHA-256 digest"),
    ):
        anyio.run(
            agentbinary.omp_binary_source().resolve_version, "18.6.1", "linux-x64"
        )


def test_nonexistent_version_does_not_fall_back_to_latest() -> None:
    with (
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(side_effect=RuntimeError("404")),
        ) as fetch,
        pytest.raises(RuntimeError, match="404"),
    ):
        anyio.run(
            agentbinary.omp_binary_source().resolve_version, "99.0.0", "linux-x64"
        )
    fetch.assert_awaited_once_with(f"{_RELEASE_URL}tags/v99.0.0")


@pytest.mark.parametrize("corrupt", [False, True])
def test_prefetch_verifies_bytes_before_caching(tmp_path: Path, corrupt: bool) -> None:
    # Exercise the public prefetch API and real checksum/cache helpers, replacing
    # only the network responses with small deterministic artifacts.
    version = "18.6.988" if corrupt else "18.6.987"
    binary = b"standalone release bytes"
    release = _release(version)
    release["assets"][0]["digest"] = f"sha256:{hashlib.sha256(binary).hexdigest()}"
    cache_path = tmp_path / f"omp-{version}-linux-x64"
    with (
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(return_value=json.dumps(release)),
        ),
        patch.object(
            binary_util,
            "download_file",
            AsyncMock(return_value=b"tampered" if corrupt else binary),
        ),
    ):
        if corrupt:
            with pytest.raises(ChecksumMismatchError):
                download_agent_binary("omp", version, "linux-x64")
            assert not cache_path.exists()
            assert cached_agent_binaries("omp") == []
        else:
            download_agent_binary("omp", version, "linux-x64")
            assert cache_path.read_bytes() == binary
            assert [
                (b.agent, b.version, b.path) for b in cached_agent_binaries("omp")
            ] == [("omp", version, cache_path)]
