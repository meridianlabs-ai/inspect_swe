import json
from typing import Any, Literal

from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.download import download_text_file
from .._util.node import (
    create_npm_bundle,
    ensure_node_available,
    install_npm_bundle,
)
from .._util.sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
)


async def ensure_opencode_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, str]:
    """Install node and OpenCode in the sandbox.

    Returns (opencode_binary, node_binary) paths.
    """
    platform = await detect_sandbox_platform(sandbox)
    node_binary = await ensure_node_available(sandbox, platform, user)
    opencode_version = await resolve_opencode_version(version)
    opencode_binary = await ensure_opencode_installed(
        sandbox, node_binary, opencode_version, platform, user
    )
    return opencode_binary, node_binary


async def resolve_opencode_version(
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
) -> str:
    """Resolve version string to an actual semver version."""
    if version in ["auto", "sandbox", "stable", "latest"]:
        release = await _fetch_latest_release()
        return str(release["tag_name"]).lstrip("v")

    return version


async def ensure_opencode_installed(
    sandbox: SandboxEnvironment,
    node_path: str,
    version: str,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install OpenCode via npm and return path to the opencode binary."""
    install_dir = f"{SANDBOX_INSTALL_DIR}/opencode"
    binary = f"{install_dir}/node_modules/.bin/opencode"

    result = await sandbox.exec(bash_command(f"test -x {binary}"), user=user)
    if result.success:
        result = await sandbox.exec(cmd=[node_path, binary, "--version"], user=user)
        if result.success and result.stdout.strip() == version:
            return binary

    async with concurrency("opencode-install", 1, visible=False):
        bundle_data = create_npm_bundle(
            package="opencode-ai",
            version=version,
            platform=platform,
            cache_name="opencode-bundles",
            ignore_scripts=True,
        )
        binary_path = await install_npm_bundle(
            sandbox=sandbox,
            bundle_data=bundle_data,
            install_dir=install_dir,
            binary_name="opencode",
            user=user,
        )
        await _run_opencode_postinstall(
            sandbox, node_path, install_dir, user
        )
        return binary_path


async def _run_opencode_postinstall(
    sandbox: SandboxEnvironment,
    node_path: str,
    install_dir: str,
    user: str | None,
) -> None:
    """Run the opencode-ai postinstall script inside the sandbox.

    The postinstall fetches the platform-specific opencode binary
    (e.g. ``opencode-linux-arm64``). We skip it on the host (host is
    typically darwin/arm64 and the linux-only bundle wouldn't contain
    a matching package) and run it here where the platform matches.
    """
    package_dir = f"{install_dir}/node_modules/opencode-ai"
    postinstall = f"{package_dir}/postinstall.mjs"

    exists = await sandbox.exec(bash_command(f"test -f {postinstall}"), user=user)
    if not exists.success:
        return

    result = await sandbox.exec(
        cmd=[node_path, "postinstall.mjs"],
        cwd=package_dir,
        user=user,
        timeout=120,
    )
    if not result.success:
        raise RuntimeError(
            f"opencode-ai postinstall failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


async def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest release from GitHub."""
    releases_url = "https://api.github.com/repos/anomalyco/opencode/releases"
    release_json = await download_text_file(f"{releases_url}/latest")
    return dict(json.loads(release_json))
