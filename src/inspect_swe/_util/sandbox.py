from logging import getLogger
from pathlib import PurePosixPath
from typing import Literal, TypeAlias, cast

from inspect_ai.util import SandboxEnvironment

logger = getLogger(__name__)

SandboxPlatform: TypeAlias = Literal[
    "linux-x64", "linux-arm64", "linux-x64-musl", "linux-arm64-musl"
]
"""Target platform identifier for sandbox binary and wheel downloads."""

SANDBOX_INSTALL_DIR = "/var/tmp/.5c95f967ca830048"


async def detect_sandbox_platform(sandbox: SandboxEnvironment) -> SandboxPlatform:
    # Get OS
    os_name = await sandbox_exec(sandbox, "uname -s")
    if os_name == "Linux":
        os_type = "linux"
    else:
        raise ValueError(f"Unsupported OS: {os_name}")

    # Get architecture
    arch = await sandbox_exec(sandbox, "uname -m")
    if arch in ["x86_64", "amd64"]:
        arch_type = "x64"
    elif arch in ["arm64", "aarch64"]:
        arch_type = "arm64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Check for musl on Linux
    if os_type == "linux":
        # Check for musl libc
        musl_check_cmd = (
            "if [ -f /lib/libc.musl-x86_64.so.1 ] || "
            "[ -f /lib/libc.musl-aarch64.so.1 ] || "
            "ldd /bin/ls 2>&1 | grep -q musl; then "
            "echo 'musl'; else echo 'glibc'; fi"
        )
        libc_type = await sandbox_exec(sandbox, musl_check_cmd)
        if libc_type == "musl":
            platform = f"linux-{arch_type}-musl"
        else:
            platform = f"linux-{arch_type}"
    else:
        platform = f"{os_type}-{arch_type}"

    return cast(SandboxPlatform, platform)


def bash_command(cmd: str) -> list[str]:
    return ["bash", "-c", cmd]


async def resolve_agent_cwd(
    sandbox: SandboxEnvironment, user: str | None, cwd: str | None
) -> str:
    """Resolve the working directory to run an agent within.

    An explicit `cwd` is always honored: absolute paths are returned as-is,
    relative paths are canonicalized to absolute inside the sandbox (the
    result feeds env vars like CODEX_HOME that resolve against the agent
    process's cwd, where a relative path would point elsewhere). Otherwise
    the sandbox's default working directory is used — except when it is `/`,
    which almost always means the image has no WORKDIR (docker's default)
    rather than a deliberate choice, in which case the user's home directory
    is used (agents writing to the container root is never what the task
    intended, and typically fails outright for non-root users).

    This fallback would arguably be better placed in inspect_ai's sandbox
    working-directory resolution, but changing it there would affect every
    `sandbox.exec()` caller in every eval (bash tool, setup scripts,
    scorers, ...) rather than just sandbox agents, so the policy is
    applied here at the agent layer instead.
    """
    if cwd is not None:
        if PurePosixPath(cwd).is_absolute():
            return cwd
        return await sandbox_exec(sandbox, "pwd", user=user, cwd=cwd)
    working_dir = await sandbox_exec(sandbox, "pwd", user=user)
    if working_dir != "/":
        return working_dir
    home_dir = await sandbox_exec(
        sandbox, 'cd ~ 2>/dev/null && pwd || echo "/"', user=user
    )
    if home_dir != "/":
        logger.info(
            f"Sandbox default working directory is '/' (image likely has no "
            f"WORKDIR); running agent in home directory '{home_dir}' instead."
        )
    return home_dir


async def sandbox_exec(
    sandbox: SandboxEnvironment,
    cmd: str,
    user: str | None = None,
    cwd: str | None = None,
) -> str:
    result = await sandbox.exec(bash_command(cmd), user=user, cwd=cwd)
    if not result.success:
        raise RuntimeError(f"Error executing sandbox command {cmd}: {result.stderr}")
    return result.stdout.strip()
