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
    """Resolve version string to an actual semver version."""
    if version in ["auto", "sandbox", "stable", "latest"]:
        release = await _fetch_latest_release()
        return str(release["tag_name"]).lstrip("v")

    return version


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
            package="@google/gemini-cli",
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


async def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest release from GitHub."""
    releases_url = "https://api.github.com/repos/google-gemini/gemini-cli/releases"
    release_json = await download_text_file(f"{releases_url}/latest")
    return dict(json.loads(release_json))
