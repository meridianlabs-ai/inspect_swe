"""Tests for the OpenCode agent install/setup utilities."""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironment, sandbox
from inspect_swe._opencode import agentbinary
from inspect_swe._opencode.agentbinary import ensure_opencode_setup
from inspect_swe._util.sandbox import SandboxPlatform

from tests.conftest import skip_if_no_docker

# mirrors the real anomalyco/opencode release shape: every x64 platform ships
# both an AVX2-optimized default build and a "-baseline" build without it;
# arm64 has no baseline variant (AVX2 is an x86 extension).
_RELEASE_ASSETS = {
    "tag_name": "v1.14.30",
    "assets": [
        {
            "name": "opencode-linux-x64.tar.gz",
            "digest": "sha256:default-x64",
            "browser_download_url": "https://example.com/opencode-linux-x64.tar.gz",
        },
        {
            "name": "opencode-linux-x64-baseline.tar.gz",
            "digest": "sha256:baseline-x64",
            "browser_download_url": "https://example.com/opencode-linux-x64-baseline.tar.gz",
        },
        {
            "name": "opencode-linux-x64-musl.tar.gz",
            "digest": "sha256:default-x64-musl",
            "browser_download_url": "https://example.com/opencode-linux-x64-musl.tar.gz",
        },
        {
            "name": "opencode-linux-x64-baseline-musl.tar.gz",
            "digest": "sha256:baseline-x64-musl",
            "browser_download_url": "https://example.com/opencode-linux-x64-baseline-musl.tar.gz",
        },
        {
            "name": "opencode-linux-arm64.tar.gz",
            "digest": "sha256:arm64",
            "browser_download_url": "https://example.com/opencode-linux-arm64.tar.gz",
        },
        {
            "name": "opencode-linux-arm64-musl.tar.gz",
            "digest": "sha256:arm64-musl",
            "browser_download_url": "https://example.com/opencode-linux-arm64-musl.tar.gz",
        },
    ],
}


@pytest.mark.parametrize(
    "platform,expected_checksum,expected_url_suffix",
    [
        ("linux-x64", "baseline-x64", "opencode-linux-x64-baseline.tar.gz"),
        (
            "linux-x64-musl",
            "baseline-x64-musl",
            "opencode-linux-x64-baseline-musl.tar.gz",
        ),
        ("linux-arm64", "arm64", "opencode-linux-arm64.tar.gz"),
        ("linux-arm64-musl", "arm64-musl", "opencode-linux-arm64-musl.tar.gz"),
    ],
)
def test_opencode_resolve_version_prefers_baseline_asset_on_x64(
    platform: SandboxPlatform, expected_checksum: str, expected_url_suffix: str
) -> None:
    # x64 platforms must resolve to the AVX2-baseline asset: we install
    # host-side, before any in-sandbox CPU probe runs, and can't reliably
    # read the underlying host's CPU flags from out here, so the baseline
    # build is the only choice that can't SIGILL crash on a pre-AVX2 host.
    # arm64 has no baseline variant and resolves its single asset unchanged.
    source = agentbinary.opencode_binary_source()
    with patch.object(
        agentbinary,
        "_fetch_release_assets",
        AsyncMock(return_value=_RELEASE_ASSETS),
    ):
        resolved = anyio.run(source.resolve_version, "0.42.0", platform)
    assert resolved.version == "0.42.0"
    assert resolved.expected_checksum == expected_checksum
    assert resolved.download_url.endswith(expected_url_suffix)
    assert resolved.package is True


def test_opencode_resolve_version_falls_back_without_baseline_asset() -> None:
    # an older release that never published a baseline asset still resolves
    # for x64 — the candidate list falls back to the default build rather
    # than failing outright.
    release = {
        "assets": [
            {
                "name": "opencode-linux-x64.tar.gz",
                "digest": "sha256:default-only",
                "browser_download_url": "https://example.com/opencode-linux-x64.tar.gz",
            }
        ]
    }
    source = agentbinary.opencode_binary_source()
    with patch.object(
        agentbinary, "_fetch_release_assets", AsyncMock(return_value=release)
    ):
        resolved = anyio.run(source.resolve_version, "0.10.0", "linux-x64")
    assert resolved.expected_checksum == "default-only"
    assert resolved.download_url.endswith("opencode-linux-x64.tar.gz")


def test_opencode_resolve_version_resolves_alias_with_a_single_api_call() -> None:
    # /releases/latest returns both the tag name and the full asset list in
    # one response, so "stable"/"latest" must resolve with exactly one API
    # call — a tag lookup followed by a second, per-tag lookup would double
    # the request count and how often a sample queued behind a rate-limited
    # neighbor repeats the exact call that just failed.
    source = agentbinary.opencode_binary_source()
    with (
        patch.object(
            agentbinary,
            "_fetch_latest_release",
            AsyncMock(return_value=_RELEASE_ASSETS),
        ) as mock_fetch_latest,
        patch.object(
            agentbinary, "_fetch_release_assets", AsyncMock()
        ) as mock_fetch_assets,
    ):
        resolved = anyio.run(source.resolve_version, "stable", "linux-x64")
    mock_fetch_latest.assert_awaited_once()
    mock_fetch_assets.assert_not_awaited()
    assert resolved.version == "1.14.30"
    assert resolved.expected_checksum == "baseline-x64"


def test_opencode_resolve_version_raises_when_platform_asset_missing() -> None:
    release = {
        "assets": [
            {
                "name": "opencode-linux-x64.tar.gz",
                "digest": "sha256:default-x64",
                "browser_download_url": "https://example.com/opencode-linux-x64.tar.gz",
            }
        ]
    }
    source = agentbinary.opencode_binary_source()
    with patch.object(
        agentbinary, "_fetch_release_assets", AsyncMock(return_value=release)
    ):
        with pytest.raises(RuntimeError, match="No matching asset"):
            anyio.run(source.resolve_version, "0.42.0", "linux-arm64")


def test_opencode_resolve_version_raises_on_malformed_digest() -> None:
    release = {
        "assets": [
            {
                "name": "opencode-linux-x64.tar.gz",
                "digest": "md5:not-sha256",
                "browser_download_url": "https://example.com/opencode-linux-x64.tar.gz",
            }
        ]
    }
    source = agentbinary.opencode_binary_source()
    with patch.object(
        agentbinary, "_fetch_release_assets", AsyncMock(return_value=release)
    ):
        with pytest.raises(RuntimeError, match="Invalid digest format"):
            anyio.run(source.resolve_version, "0.42.0", "linux-x64")


def test_opencode_source_uses_package_archive_caching(tmp_path: Path) -> None:
    # the release asset is a tar.gz wrapping the binary: it is staged as a
    # "package" archive (cached verbatim, checksum and all, extracted in the
    # sandbox at install time) rather than transformed host-side into a blob
    # a later cache hit can no longer verify against the release digest.
    # cache paths stay keyed by version + platform so concurrent samples on
    # different platforms don't collide.
    with patch.object(agentbinary, "package_cache_dir", return_value=tmp_path):
        source = agentbinary.opencode_binary_source()
    assert source.package_entrypoint == "opencode"
    assert source.post_download is None
    assert source.cached_package_path is not None
    package_path = source.cached_package_path("0.42.0", "linux-arm64")
    assert package_path == tmp_path / "opencode-package-0.42.0-linux-arm64.tar.gz"
    # the legacy single-binary layout still resolves, as an offline-fallback
    # read path for caches written before opencode moved to package-archive
    # caching
    legacy_path = source.cached_binary_path("0.42.0", "linux-arm64")
    assert legacy_path == tmp_path / "opencode-0.42.0-linux-arm64"


def test_opencode_list_cached_binaries(tmp_path: Path) -> None:
    with patch.object(agentbinary, "package_cache_dir", return_value=tmp_path):
        source = agentbinary.opencode_binary_source()
    (tmp_path / "opencode-0.42.0-linux-arm64").write_bytes(b"binary")
    (tmp_path / "opencode-package-0.41.0-linux-x64.tar.gz").write_bytes(b"archive")
    (tmp_path / "unrelated-file").write_bytes(b"noise")
    assert {p.name for p in source.list_cached_binaries()} == {
        "opencode-0.42.0-linux-arm64",
        "opencode-package-0.41.0-linux-x64.tar.gz",
    }


def test_ensure_opencode_setup_delegates_to_ensure_agent_binary_installed() -> None:
    # ensure_opencode_setup must install opencode through the shared
    # host-side AgentBinarySource path (ensure_agent_binary_installed), not
    # reimplement its own sandbox-side install/download logic.
    sbox = cast(SandboxEnvironment, object())
    fake_source = object()

    async def fake_detect_sandbox_platform(sandbox: object) -> str:
        return "linux-x64"

    async def fake_ensure_node_available(
        sandbox: object, platform: object, user: object
    ) -> str:
        return "/usr/local/bin/node"

    async def fake_ensure_ripgrep_available(
        sandbox: object, platform: object, user: object
    ) -> str:
        return "/usr/local/bin"

    mock_ensure_installed = AsyncMock(return_value="/opt/opencode/opencode")
    with (
        patch.object(
            agentbinary, "detect_sandbox_platform", fake_detect_sandbox_platform
        ),
        patch.object(agentbinary, "ensure_node_available", fake_ensure_node_available),
        patch.object(
            agentbinary, "ensure_ripgrep_available", fake_ensure_ripgrep_available
        ),
        patch.object(
            agentbinary, "opencode_binary_source", return_value=fake_source
        ) as mock_source,
        patch.object(
            agentbinary, "ensure_agent_binary_installed", mock_ensure_installed
        ),
    ):
        binary, dependency_bin_dirs = anyio.run(
            ensure_opencode_setup, sbox, "stable", None
        )

    mock_source.assert_called_once()
    mock_ensure_installed.assert_awaited_once_with(fake_source, "stable", None, sbox)
    assert binary == "/opt/opencode/opencode"
    assert dependency_bin_dirs == ["/usr/local/bin", "/usr/local/bin"]


@solver
def install_opencode_in_sandbox(version: str = "stable") -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sbox = sandbox()
        opencode_binary, dependency_bin_dirs = await ensure_opencode_setup(
            sbox, version=version, user=None
        )
        state.metadata["opencode_binary"] = opencode_binary
        state.metadata["dependency_bin_dirs"] = dependency_bin_dirs

        path = ":".join([*dependency_bin_dirs, "/usr/local/bin", "/usr/bin", "/bin"])
        version_result = await sbox.exec(
            [opencode_binary, "--version"], env={"PATH": path}, user=None
        )
        state.metadata["version_ok"] = version_result.success
        state.metadata["reported_version"] = version_result.stdout.strip()
        return state

    return solve


@scorer(metrics=[])
def check_install() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        version_ok = state.metadata.get("version_ok")
        binary = state.metadata.get("opencode_binary")
        if not version_ok or not binary:
            return Score(
                value=0,
                explanation=f"Install failed: binary={binary} version_ok={version_ok}",
            )
        return Score(
            value=1,
            explanation=f"Installed at {binary}, --version => {state.metadata.get('reported_version')!r}",
        )

    return score


@skip_if_no_docker
@pytest.mark.slow
def test_install_opencode_in_docker_sandbox() -> None:
    """Verify opencode-ai installs into a docker sandbox and reports a version."""
    task = Task(
        dataset=[Sample(input="install", target="ok")],
        solver=install_opencode_in_sandbox(),
        scorer=check_install(),
        sandbox="docker",
    )
    logs = eval(task, model="mockllm/model", limit=1)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success", f"Task failed: {log.error}"
    assert log.samples and log.samples[0].scores

    score_value = list(log.samples[0].scores.values())[0]
    assert score_value.value == 1, score_value.explanation
