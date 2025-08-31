from typing import Literal

from inspect_ai.util import SandboxEnvironment, concurrency
from inspect_ai.util import sandbox as sandbox_env

from inspect_swe._claude_code.install.cache import read_cached_claude_code_binary
from inspect_swe._util.trace import trace

from ..._util.sandbox import bash_command, detect_sandbox_platform, sandbox_exec
from .download import download_claude_code_async


async def ensure_claude_code_installed(
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    user: str | None = None,
    sandbox: SandboxEnvironment | None = None,
) -> str:
    # resolve sandbox
    sandbox = sandbox or sandbox_env()

    # look in the sandbox first if we need to
    if version == "auto" or version == "sandbox":
        result = await sandbox.exec(bash_command("which claude"), user=user)
        if result.success:
            claude_binary = result.stdout.strip()
            trace(f"Using claude code installed in sandbox: {claude_binary}")
            return claude_binary

        # if version == "sandbox" and we don't find it that's an error
        if version == "sandbox":
            raise RuntimeError("unable to locate claude code in sandbox")

        # otherwise set to "stable"
        version = "stable"

    # detect the sandbox target platform
    platform = await detect_sandbox_platform(sandbox)

    # use concurrency so multiple samples don't attempt the same download all at once
    async with concurrency("claude-install", 1, visible=False):
        # if a specific version is requested, first try to read it directly from the cache
        if version not in ["stable", "latest"]:
            claude_binary_bytes: bytes | None = read_cached_claude_code_binary(
                version, platform, None
            )
            if claude_binary_bytes is not None:
                trace(f"Used claude code binary from cache: {version} ({platform})")
        else:
            claude_binary_bytes = None

        # download the binary
        if claude_binary_bytes is None:
            claude_binary_bytes = await download_claude_code_async(version, platform)

        # write it into the container and return it
        claude_binary = f"/opt/claude-{version}-{platform}"
        await sandbox.write_file(claude_binary, claude_binary_bytes)
        await sandbox_exec(sandbox, f"chmod +x {claude_binary}")
        await sandbox_exec(sandbox, f"{claude_binary} config list", user=user)
        return claude_binary
