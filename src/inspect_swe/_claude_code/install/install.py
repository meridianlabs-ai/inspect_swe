from typing import Literal

from inspect_ai.util import SandboxEnvironment

from ..._util.sandbox import detect_sandbox_platform, sandbox_exec
from .download import download_claude_code_binary


async def ensure_claude_code_installed(
    sandbox: SandboxEnvironment, version: Literal["stable", "latest"] | str
) -> str:
    # if claude code is already installed then stand down
    container_claude_binary = await sandbox_exec(sandbox, "which claude")
    if container_claude_binary:
        return container_claude_binary

    # detect the target platform
    platform = await detect_sandbox_platform(sandbox)

    # download the binary
    claude_binary_bytes = await download_claude_code_binary(version, platform)

    # write it into the container and return it
    container_claude_binary = "/opt/claude-{version}-{platform}"
    await sandbox.write_file(container_claude_binary, claude_binary_bytes)
    await sandbox_exec(sandbox, f"chmod +x {container_claude_binary}")
    await sandbox_exec(sandbox, f"{container_claude_binary} config list")
    return container_claude_binary
