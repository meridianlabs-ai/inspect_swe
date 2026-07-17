"""mitmproxy provisioning and launch support for the Antigravity CLI."""

import asyncio  # noqa: ANYIO_OK -- pre-existing; see _monitor_interceptor's create_task
import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Final

import anyio
from inspect_ai.util import SandboxEnvironment, concurrency
from inspect_ai.util._sandbox.exec_remote import (
    ExecCompleted,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    ExecStderr,
)

from .._util.download import download_file
from .._util.sandbox import SANDBOX_INSTALL_DIR, SandboxPlatform, bash_command

logger = getLogger(__name__)

MITMPROXY_VERSION: Final = "11.1.3"
MITMPROXY_INSTALL_DIR: Final = f"{SANDBOX_INSTALL_DIR}/mitmproxy"
MITMDUMP_PATH: Final = f"{MITMPROXY_INSTALL_DIR}/mitmdump"
MITMPROXY_ARCHIVE_PATH: Final = f"{SANDBOX_INSTALL_DIR}/mitmproxy.tar.gz"
ADDON_PATH: Final = f"{MITMPROXY_INSTALL_DIR}/antigravity_addon.py"
DEFAULT_MITMPROXY_TARBALL_URL: Final = (
    "https://downloads.mitmproxy.org/11.1.3/"
    "mitmproxy-11.1.3-linux-x86_64.tar.gz"
)


@dataclass(slots=True, eq=False)  # noqa: MUTABLE_OK -- frozen=True breaks __traceback__
class MitmproxyStartupTimeoutError(RuntimeError):
    """contextlib's @asynccontextmanager sets __traceback__ on caught
    exceptions via a plain attribute assignment; combined with slots=True,
    frozen's generated __setattr__ raises ``TypeError: super(type, obj): obj
    must be an instance or subtype of type`` when re-raised through an async
    context manager (e.g. sandbox_agent_bridge). ``eq=False`` keeps identity-
    based ``__hash__`` (inspect_ai's exception-group flattening hashes
    exceptions); dataclass sets ``__hash__ = None`` for eq=True, frozen=False.
    Exceptions are mutated by the runtime (traceback/context/cause)
    regardless of dataclass frozen semantics, so this class stays mutable.
    """

    detail: str

    detail: str

    def __str__(self) -> str:
        return self.detail


def _mitmproxy_tarball_url(platform: SandboxPlatform) -> str:
    """Resolve the standalone mitmproxy archive URL for a sandbox platform."""
    configured_url = os.environ.get("ANTIGRAVITY_MITMPROXY_TARBALL_URL")
    if configured_url is not None:
        return configured_url

    architecture = {
        "linux-x64": "linux-x86_64",
        "linux-x64-musl": "linux-x86_64",
        "linux-arm64": "linux-aarch64",
        "linux-arm64-musl": "linux-aarch64",
    }[platform]
    if architecture == "linux-x86_64":
        return DEFAULT_MITMPROXY_TARBALL_URL
    return (
        f"https://downloads.mitmproxy.org/{MITMPROXY_VERSION}/"
        f"mitmproxy-{MITMPROXY_VERSION}-{architecture}.tar.gz"
    )


async def ensure_mitmproxy(
    sandbox: SandboxEnvironment, platform: SandboxPlatform, user: str | None
) -> str:
    """Install mitmdump in the sandbox and return its absolute path."""
    result = await sandbox.exec(bash_command(f"test -x {MITMDUMP_PATH}"), user=user)
    if result.success:
        return MITMDUMP_PATH

    async with concurrency("mitmproxy-install", 1, visible=False):
        archive_data = await download_file(_mitmproxy_tarball_url(platform))
        await sandbox.exec(
            bash_command(f"mkdir -p {MITMPROXY_INSTALL_DIR}"), user="root"
        )
        await sandbox.write_file(MITMPROXY_ARCHIVE_PATH, archive_data)
        result = await sandbox.exec(
            bash_command(
                f"tar -xzf {MITMPROXY_ARCHIVE_PATH} -C {MITMPROXY_INSTALL_DIR} && rm -f {MITMPROXY_ARCHIVE_PATH}"
            ),
            user="root",
        )
        if not result.success:
            raise RuntimeError(f"Unable to extract mitmproxy: {result.stderr}")

    result = await sandbox.exec(bash_command(f"test -x {MITMDUMP_PATH}"), user=user)
    if not result.success:
        raise RuntimeError(f"mitmdump binary not found at {MITMDUMP_PATH}")
    return MITMDUMP_PATH


async def start_interceptor(
    sandbox: SandboxEnvironment,
    *,
    listen_port: int,
    bridge_port: int,
    confdir: str,
    user: str | None,
) -> tuple[ExecRemoteProcess, str, asyncio.Task[None]]:
    """Start mitmdump and return its process, CA certificate, and monitor task."""
    addon_data = Path(__file__).with_name("_addon.py").read_bytes()
    await sandbox.write_file(ADDON_PATH, addon_data)
    cmd = [
        MITMDUMP_PATH,
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(listen_port),
        "--set",
        f"confdir={confdir}",
        "-s",
        ADDON_PATH,
        "-q",
    ]
    # When the sandbox exposes an upstream egress-allowlist proxy (set by the
    # compose egress profile as SANDBOX_EGRESS_PROXY), chain mitmdump's
    # pass-through traffic (agy's OAuth / cloudcode bootstrap calls) through it.
    # Inference is still short-circuited to the local Inspect bridge by the addon
    # and never egresses; in an isolated sandbox this proxy is the ONLY route out.
    egress = await sandbox.exec(
        bash_command('printf %s "${SANDBOX_EGRESS_PROXY:-}"'), user=user
    )
    upstream = egress.stdout.strip() if egress.success else ""
    if upstream:
        cmd += ["--mode", f"upstream:{upstream}"]
    process = await sandbox.exec_remote(
        cmd=cmd,
        options=ExecRemoteStreamingOptions(
            concurrency=False,
            env={"ANTIGRAVITY_BRIDGE_PORT": str(bridge_port)},
        ),
    )
    monitor_task = asyncio.create_task(_monitor_interceptor(process))

    ca_cert_path = f"{confdir}/mitmproxy-ca-cert.pem"
    for _ in range(30):
        result = await sandbox.exec(bash_command(f"test -f {ca_cert_path}"), user=user)
        if result.success:
            break
        await anyio.sleep(0.5)
    else:
        await process.kill()
        raise MitmproxyStartupTimeoutError(
            detail=f"mitmproxy CA certificate was not created at {ca_cert_path}"
        )

    await _wait_for_listening_port(sandbox, process, listen_port, user)
    return process, ca_cert_path, monitor_task


async def _wait_for_listening_port(
    sandbox: SandboxEnvironment,
    process: ExecRemoteProcess,
    port: int,
    user: str | None,
) -> None:
    """Poll until mitmdump's listen socket accepts connections.

    The CA certificate file materializes as mitmdump starts, but the actual
    listen socket may not be accepting connections yet -- agy's very first
    outbound call (silent-auth / token refresh) can otherwise race a
    not-yet-ready listener and fail with a hard ``connection refused``.
    """
    probe = bash_command(f"echo -n '' > /dev/tcp/127.0.0.1/{port}")
    for _ in range(30):
        result = await sandbox.exec(probe, user=user)
        if result.success:
            return
        await anyio.sleep(0.5)

    await process.kill()
    raise MitmproxyStartupTimeoutError(
        detail=f"mitmproxy did not start accepting connections on port {port}"
    )


async def _monitor_interceptor(process: ExecRemoteProcess) -> None:
    stderr: list[str] = []
    async for event in process:
        if isinstance(event, ExecStderr):
            stderr.append(event.data)
            logger.debug("mitmdump stderr: %s", event.data.rstrip())
        if isinstance(event, ExecCompleted):
            if not event.success:
                raise RuntimeError(
                    f"mitmdump process exited unexpectedly with failure: {''.join(stderr)}."
                )
            if stderr:
                logger.warning(
                    "mitmdump stderr output on clean exit:\n%s", "".join(stderr).rstrip()
                )
            return
    raise RuntimeError(
        f"mitmdump process stream ended unexpectedly: {''.join(stderr)}."
    )
