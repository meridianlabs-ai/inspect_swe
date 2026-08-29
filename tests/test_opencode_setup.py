"""Tests for the OpenCode agent install/setup utilities."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from inspect_swe._opencode import agentbinary
from inspect_swe._opencode.agentbinary import ensure_opencode_setup

from tests.conftest import skip_if_no_docker

_RELEASE_ASSETS = {
    "assets": [
        {
            "name": "opencode-linux-x64.tar.gz",
            "digest": "sha256:aaa",
            "browser_download_url": "https://example.com/opencode-linux-x64.tar.gz",
        },
        {
            "name": "opencode-linux-arm64.tar.gz",
            "digest": "sha256:bbb",
            "browser_download_url": "https://example.com/opencode-linux-arm64.tar.gz",
        },
    ]
}


def test_opencode_resolve_version_matches_platform_asset() -> None:
    # each platform has its own dynamically-linked build; the resolved asset
    # must be the one whose name matches the sandbox's detected platform, not
    # just any asset in the release
    source = agentbinary.opencode_binary_source()
    with patch.object(
        agentbinary,
        "_fetch_release_assets",
        AsyncMock(return_value=_RELEASE_ASSETS),
    ):
        resolved = anyio.run(source.resolve_version, "0.42.0", "linux-arm64")
    assert resolved.version == "0.42.0"
    assert resolved.expected_checksum == "bbb"
    assert resolved.download_url.endswith("opencode-linux-arm64.tar.gz")
    assert resolved.package is False


def test_opencode_resolve_version_resolves_alias_before_fetching() -> None:
    # "stable"/"latest" must resolve to a concrete version before the
    # per-version release lookup, so the asset request targets a real tag
    source = agentbinary.opencode_binary_source()
    with (
        patch.object(
            agentbinary, "_fetch_latest_version", AsyncMock(return_value="1.14.30")
        ),
        patch.object(
            agentbinary,
            "_fetch_release_assets",
            AsyncMock(return_value=_RELEASE_ASSETS),
        ) as mock_fetch_assets,
    ):
        resolved = anyio.run(source.resolve_version, "stable", "linux-x64")
    mock_fetch_assets.assert_awaited_once_with("1.14.30")
    assert resolved.version == "1.14.30"
    assert resolved.expected_checksum == "aaa"


def test_opencode_resolve_version_raises_when_platform_asset_missing() -> None:
    source = agentbinary.opencode_binary_source()
    with patch.object(
        agentbinary,
        "_fetch_release_assets",
        AsyncMock(return_value=_RELEASE_ASSETS),
    ):
        with pytest.raises(RuntimeError, match="No asset"):
            anyio.run(source.resolve_version, "0.42.0", "windows-x64")


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


def test_opencode_source_caches_by_version_and_platform(tmp_path: Path) -> None:
    # the standalone binary is not a package archive: no package_entrypoint or
    # cached_package_path, and cached files are keyed by version + platform so
    # concurrent samples on different platforms don't collide
    with patch.object(agentbinary, "package_cache_dir", return_value=tmp_path):
        source = agentbinary.opencode_binary_source()
    assert source.package_entrypoint is None
    assert source.cached_package_path is None
    cache_path = source.cached_binary_path("0.42.0", "linux-arm64")
    assert cache_path == tmp_path / "opencode-0.42.0-linux-arm64"


def test_opencode_list_cached_binaries(tmp_path: Path) -> None:
    with patch.object(agentbinary, "package_cache_dir", return_value=tmp_path):
        source = agentbinary.opencode_binary_source()
    (tmp_path / "opencode-0.42.0-linux-arm64").write_bytes(b"binary")
    (tmp_path / "opencode-0.41.0-linux-x64").write_bytes(b"binary")
    (tmp_path / "unrelated-file").write_bytes(b"noise")
    assert {p.name for p in source.list_cached_binaries()} == {
        "opencode-0.42.0-linux-arm64",
        "opencode-0.41.0-linux-x64",
    }


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
