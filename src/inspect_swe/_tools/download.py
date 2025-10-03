import re
from pathlib import Path
from typing import Literal, NamedTuple

from .._claude_code.agentbinary import claude_code_binary_source
from .._codex_cli.agentbinary import codex_cli_binary_source
from .._util._async import run_coroutine
from .._util.agentbinary import (
    AgentBinarySource,
    download_agent_binary_async,
)
from .._util.sandbox import SandboxPlatform


class AgentBinary(NamedTuple):
    """Agent binary."""

    agent: Literal["claude_code", "codex_cli"]
    """Agent type."""

    version: str
    """Agent version."""

    path: Path
    """"Agent path."""

    def __str__(self) -> str:
        return f"{self.agent} {self.version}: {self.path}"

    def __repr__(self) -> str:
        return f"{self.agent} {self.version}: {self.path}"


class AgentBinaries(list[AgentBinary]):
    def __str__(self) -> str:
        if not self:
            return ""

        max_agent = max(len(b.agent) for b in self)
        max_version = max(len(b.version) for b in self)

        lines = []
        for b in self:
            lines.append(
                f"{b.agent:<{max_agent}}  {b.version:<{max_version}}  {b.path}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


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
    binary: Literal["claude_code", "codex_cli"] | None = None, quiet: bool = False
) -> AgentBinaries:
    """List the agent binaries which have been cached on this system.

    Args:
       binary: Type of binary to list (lists all of if not specified).
       quiet: Do not print the binaries as a side effect

    Returns:
       List of AgentBinary tuples ordered by agent and version (descending).

    """
    if binary is None:
        return AgentBinaries(
            cached_agent_binaries("claude_code") + cached_agent_binaries("codex_cli")
        )

    source = _agent_binary_source(binary)
    binaries = source.list_cached_binaries()

    def parse_name(name: str) -> tuple[str, str, int, int, int]:
        match = re.match(r"([a-z_]+)-(\d+)\.(\d+)\.(\d+)", name)
        if match:
            return (
                match.group(1),  # agent type
                f"{match.group(2)}.{match.group(3)}.{match.group(4)}",  # version string
                int(match.group(2)),  # major
                int(match.group(3)),  # minor
                int(match.group(4)),  # patch
            )
        return ("", "", 0, 0, 0)

    # Sort by type ascending, then version descending
    result = []
    for path in binaries:
        _, version, major, minor, patch = parse_name(path.name)
        if version:
            result.append(
                AgentBinary(
                    agent=binary,
                    version=version,
                    path=path,
                )
            )

    return AgentBinaries(
        sorted(
            result,
            key=lambda x: (
                x.agent,
                -int(x.version.split(".")[0]),
                -int(x.version.split(".")[1]),
                -int(x.version.split(".")[2]),
            ),
        )
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
