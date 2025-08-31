from typing import Literal

from inspect_ai.util import SandboxEnvironment, concurrency
from inspect_ai.util import sandbox as sandbox_env

from ..._util.sandbox import bash_command, detect_sandbox_platform, sandbox_exec
from .download import download_claude_code_binary


async def ensure_claude_code_installed(
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    sandbox: SandboxEnvironment | None = None,
    user: str | None = None,
) -> str:
    # resolve sandbox
    sandbox = sandbox or sandbox_env()

    # look in the sandbox first if we need to
    if version == "auto" or version == "sandbox":
        result = await sandbox.exec(bash_command("which claude"), user=user)
        if result.success:
            return result.stdout.strip()

        # if version == "sandbox" and we don't find it that's an error
        if version == "sandbox":
            raise RuntimeError("unable to locate claude code in sandbox")

    # detect the sandbox target platform
    platform = await detect_sandbox_platform(sandbox)

    # use concurrency so multiple samples don't attempt the same download all at once
    async with concurrency("claude-install", 1):
        # download the binary
        claude_binary_bytes = await download_claude_code_binary(version, platform)

        # write it into the container and return it
        container_claude_binary = "/opt/claude-{version}-{platform}"
        await sandbox.write_file(container_claude_binary, claude_binary_bytes)
        await sandbox_exec(sandbox, f"chmod +x {container_claude_binary}")
        await sandbox_exec(sandbox, f"{container_claude_binary} config list", user=user)
        return container_claude_binary
