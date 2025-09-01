from typing import Literal

from .._claude_code.install.download import download_claude_code_async
from .._util._async import run_coroutine
from .._util.sandbox import SandboxPlatform


def download_agent_binary(
    binary: Literal["claude_code"],
    version: Literal["stable", "latest"] | str,
    platform: SandboxPlatform,
) -> None:
    """Download agent binary.

    Download an agent binary. This version will be added to the cache of downloaded versions (which retains the 5 most recently downloaded versions).

    Use this if you need to ensure that a specific version of an agent binary is downloaded in advance (e.g. if you are going to run your evaluations offline). After downloading, explicit requests for the downloaded version (e.g. `claude_code(version="1.0.98")`) will not require network access.

    Args:
        binary: Type of binary to download (currently only "claude_code")
        version: Version to download ("stable", "latest", or an explicit version number).
        platform: Target platform ("linux-x64", "linux-arm64", "linux-x64-musl", or "linux-arm64-musl")
    """
    if binary == "claude_code":
        run_coroutine(download_claude_code_async(version, platform))
    else:
        raise ValueError(f"Unsuported agent binary type: {binary}")
