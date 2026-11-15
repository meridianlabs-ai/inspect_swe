from typing import Literal

import anyio
from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.node import (
    create_npm_bundle,
    ensure_node_available,
    install_npm_bundle,
    resolve_npm_package_version,
)
from .._util.sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
)
from .._util.versioncache import cached_version_resolution

_GEMINI_CLI_PACKAGE = "@google/gemini-cli"


async def ensure_gemini_cli_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, str]:
    """Install node and gemini-cli in the sandbox.

    Returns (gemini_binary, node_binary) paths.
    """
    platform = await detect_sandbox_platform(sandbox)
    node_binary = await ensure_node_available(sandbox, platform, user)
    gemini_version = await resolve_gemini_version(version)
    gemini_binary = await ensure_gemini_cli_installed(
        sandbox, node_binary, gemini_version, platform, user
    )
    return gemini_binary, node_binary


async def resolve_gemini_version(
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
) -> str:
    """Resolve version string to an actual semver version.

    The aliases resolve to the newest release past the npm release cooldown
    (see ``resolve_npm_package_version``).
    """
    if version in ["auto", "sandbox", "stable", "latest"]:
        # cached so concurrent samples don't each query the registry — all
        # four aliases resolve to the same release
        return await cached_version_resolution("gemini-cli", _fetch_latest_version)

    return version


async def _fetch_latest_version() -> str:
    return await anyio.to_thread.run_sync(
        resolve_npm_package_version, _GEMINI_CLI_PACKAGE
    )


async def ensure_gemini_cli_installed(
    sandbox: SandboxEnvironment,
    node_path: str,
    version: str,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install Gemini CLI via npm and return path to the gemini binary.

    This installs the full @google/gemini-cli package including all policy files,
    ensuring YOLO mode and other features work correctly.
    """
    gemini_install_dir = f"{SANDBOX_INSTALL_DIR}/gemini-cli"
    gemini_binary = f"{gemini_install_dir}/node_modules/.bin/gemini"

    result = await sandbox.exec(
        bash_command(f"test -x {gemini_binary}"),
        user=user,
    )
    if result.success:
        result = await sandbox.exec(
            cmd=[node_path, gemini_binary, "--version"],
            user=user,
        )
        if result.success:
            installed_version = result.stdout.strip()
            if installed_version == version:
                return gemini_binary

    async with concurrency("gemini-cli-install", 1, visible=False):
        bundle_data = create_npm_bundle(
            package=_GEMINI_CLI_PACKAGE,
            version=version,
            platform=platform,
            cache_name="gemini-cli-bundles",
        )
        return await install_npm_bundle(
            sandbox=sandbox,
            bundle_data=bundle_data,
            install_dir=gemini_install_dir,
            binary_name="gemini",
            user=user,
        )
