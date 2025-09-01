from typing import Literal, TypeAlias, cast

from inspect_ai.util import SandboxEnvironment

SandboxPlatform: TypeAlias = Literal[
    "linux-x64", "linux-arm64", "linux-x64-musl", "linux-arm64-musl"
]


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


async def sandbox_exec(
    sandbox: SandboxEnvironment, cmd: str, user: str | None = None
) -> str:
    result = await sandbox.exec(bash_command(cmd), user=user)
    if not result.success:
        raise RuntimeError(f"Error executing sandbox command {cmd}: {result.stderr}")
    return result.stdout.strip()
