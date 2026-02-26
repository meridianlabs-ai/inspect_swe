"""Build and install the claude-code-acp ACP adapter in sandboxes.

Downloads Node.js if needed, builds the npm bundle on the host (with
caching), and copies both into each sandbox.
"""

import logging

from inspect_ai.util import SandboxEnvironment, concurrency

from inspect_swe._util.node import (
    create_npm_bundle,
    ensure_node_available,
    install_npm_bundle,
    resolve_npm_package_version,
)
from inspect_swe._util.sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
)

logger = logging.getLogger(__name__)

_ACP_ADAPTER_PACKAGE = "@zed-industries/claude-code-acp"


async def ensure_claude_code_acp_setup(
    sandbox: SandboxEnvironment,
    user: str | None = None,
) -> tuple[str, str]:
    """Install node and claude-code-acp in the sandbox.

    Returns (acp_binary, node_binary) paths.
    """
    platform = await detect_sandbox_platform(sandbox)
    node_binary = await ensure_node_available(sandbox, platform, user)
    acp_binary = await _ensure_acp_installed(sandbox, platform, user)
    return acp_binary, node_binary


async def _ensure_acp_installed(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install claude-code-acp via locally-built npm bundle."""
    install_dir = f"{SANDBOX_INSTALL_DIR}/claude-code-acp"
    acp_binary = f"{install_dir}/node_modules/.bin/claude-code-acp"

    result = await sandbox.exec(bash_command(f"test -x {acp_binary}"), user=user)
    if result.success:
        return acp_binary

    async with concurrency("claude-code-acp-install", 1, visible=False):
        version = resolve_npm_package_version(_ACP_ADAPTER_PACKAGE)
        bundle_data = create_npm_bundle(
            package=_ACP_ADAPTER_PACKAGE,
            version=version,
            platform=platform,
            cache_name="claude-code-acp-bundles",
        )
        return await install_npm_bundle(
            sandbox=sandbox,
            bundle_data=bundle_data,
            install_dir=install_dir,
            binary_name="claude-code-acp",
            user=user,
        )
