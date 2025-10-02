import re
from typing import Literal

from .._claude_code.agentbinary import claude_code_binary_source
from .._codex_cli.agentbinary import codex_cli_binary_source
from .._util._async import run_coroutine
from .._util.agentbinary import (
    AgentBinarySource,
    download_agent_binary_async,
)
from .._util.sandbox import SandboxPlatform


def download_agent_binary(
    binary: Literal["claude_code", "codex_cli"],
    version: Literal["stable", "latest"] | str,
    platform: SandboxPlatform,
) -> None:
    """Download agent binary.

    Download an agent binary. This version will be added to the cache of downloaded versions (which retains the 5 most recently downloaded versions).

    Use this if you need to ensure that a specific version of an agent binary is downloaded in advance (e.g. if you are going to run your evaluations offline). After downloading, explicit requests for the downloaded version (e.g. `claude_code(version="1.0.98")`) will not require network access.

    Args:
        binary: Type of binary to download
        version: Version to download ("stable", "latest", or an explicit version number).
        platform: Target platform ("linux-x64", "linux-arm64", "linux-x64-musl", or "linux-arm64-musl")
    """
    match binary:
        case "claude_code":
            source = claude_code_binary_source()
        case "codex_cli":
            source = codex_cli_binary_source()
        case _:
            raise ValueError(f"Unsuported agent binary type: {binary}")

    run_coroutine(download_agent_binary_async(source, version, platform))


def cached_agent_binaries(
    binary: Literal["claude_code", "codex_cli"] | None = None,
) -> list[str]:
    """List the agent binaries which have been cached on this system.

    Args:
       binary: Type of binary to list (lists all of if not specified).

    Returns:
       List of binary names ordered by agent and version (descending).

    """
    if binary is None:
        return cached_agent_binaries("claude_code") + cached_agent_binaries("codex_cli")

    source = _agent_binary_source(binary)
    binaries = source.list_cached_binaries()

    def parse_version(item: str) -> tuple[str, int, int, int]:
        match = re.match(r"([^-]+)-(\d+)\.(\d+)\.(\d+)", item)
        if match:
            return (
                match.group(1),  # type
                int(match.group(2)),  # major
                int(match.group(3)),  # minor
                int(match.group(4)),  # patch
            )
        return ("", 0, 0, 0)

    # Sort by type ascending, then version descending
    return sorted(
        [path.name for path in binaries],
        key=lambda x: (
            parse_version(x)[0],  # Group by type
            -parse_version(x)[1],  # Major version descending
            -parse_version(x)[2],  # Minor version descending
            -parse_version(x)[3],  # Patch version descending
        ),
    )


def _agent_binary_source(
    binary: Literal["claude_code", "codex_cli"],
) -> AgentBinarySource:
    match binary:
        case "claude_code":
            return claude_code_binary_source()
        case "codex_cli":
            return codex_cli_binary_source()
        case _:
            raise ValueError(f"Unsuported agent binary type: {binary}")
