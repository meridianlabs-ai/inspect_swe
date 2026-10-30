from functools import partial

import anyio
import pytest
from inspect_swe import resolve_agent_version
from inspect_swe._pi.agentbinary import _asset_name, pi_binary_source
from inspect_swe._util.sandbox import SandboxPlatform


@pytest.mark.parametrize(
    ("platform", "asset"),
    [
        ("linux-x64", "pi-linux-x64.tar.gz"),
        ("linux-arm64", "pi-linux-arm64.tar.gz"),
    ],
)
def test_pi_release_asset_mapping(platform: SandboxPlatform, asset: str) -> None:
    assert _asset_name(platform=platform) == asset


@pytest.mark.parametrize("platform", ["linux-x64-musl", "linux-arm64-musl"])
def test_pi_rejects_musl(platform: SandboxPlatform) -> None:
    with pytest.raises(RuntimeError, match="glibc"):
        _asset_name(platform=platform)


@pytest.mark.slow
def test_pi_latest_release_has_verified_linux_assets() -> None:
    source = pi_binary_source()
    for platform in ("linux-x64", "linux-arm64"):
        resolved = anyio.run(partial(source.resolve_version, "latest", platform))
        assert resolved.download_url.startswith(
            "https://github.com/earendil-works/pi/releases/"
        )
        assert len(resolved.expected_checksum) == 64
        assert int(resolved.expected_checksum, 16) > 0
        assert resolved.package


def test_pi_source_uses_standalone_package() -> None:
    source = pi_binary_source()
    assert source.binary == "pi"
    assert source.package_entrypoint == "pi/pi"
    assert source.cached_package_path is not None
    assert (
        source.cached_package_path("0.85.1", "linux-x64").name
        == "pi-package-0.85.1-linux-x64.tar.gz"
    )


@pytest.mark.parametrize("version", ["auto", "sandbox", "0.85.1"])
def test_pi_explicit_version_resolution(version: str) -> None:
    assert resolve_agent_version(agent="pi", version=version) == version
