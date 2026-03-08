"""Default ACP Client: auto-approves permissions, rejects fs/terminal methods."""

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncIterator
from functools import wraps
from typing import Any, Awaitable, Callable

import anyio
from acp import RequestError, connect_to_agent
from acp.client.connection import ClientSideConnection
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EnvVariable,
    KillTerminalCommandResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    SessionInfoUpdate,
    TerminalOutputResponse,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from inspect_ai.util import ExecRemoteProcess

from .transport import ErrorInfo, create_exec_remote_streams

logger = logging.getLogger(__name__)


class ACPError(RuntimeError):
    """ACP failure surfaced to inspect_swe users."""


def _unsupported_capability_request(capability: str) -> RequestError:
    """Return a clear ACP error for unsupported host capabilities."""
    return RequestError(
        -32601,
        f"ACP adapter requested unsupported client capability {capability}",
        {"capability": capability},
    )


class DefaultClient(Client):
    """ACP Client that auto-approves permissions and rejects fs/terminal methods."""

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        for kind in ("allow_always", "allow_once"):
            for opt in options:
                if opt.kind == kind:
                    logger.debug(
                        "Auto-approving permission [%s]: option=%s kind=%s",
                        session_id,
                        opt.option_id,
                        opt.kind,
                    )
                    return RequestPermissionResponse(
                        outcome=AllowedOutcome(
                            outcome="selected", option_id=opt.option_id
                        ),
                    )
        available = [(opt.option_id, opt.kind) for opt in options]
        raise RuntimeError(
            "ACP adapter requested permission, but inspect_swe could not auto-approve "
            f"it because no allow option was provided [session={session_id}]: {available}"
        )

    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate
        | ConfigOptionUpdate
        | SessionInfoUpdate
        | UsageUpdate,
        **kwargs: Any,
    ) -> None:
        logger.debug("ACP update [%s]: %s", session_id, type(update).__name__)

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        raise _unsupported_capability_request("fs/read_text_file")

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        raise _unsupported_capability_request("fs/write_text_file")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise _unsupported_capability_request("terminal/create")

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        raise _unsupported_capability_request("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise _unsupported_capability_request("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise _unsupported_capability_request("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse | None:
        raise _unsupported_capability_request("terminal/kill")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise _unsupported_capability_request(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass

    def on_connect(self, conn: Any) -> None:
        return None


async def _await_stderr_flush(feeder: asyncio.Task[None]) -> None:
    """Give the feeder a moment to collect trailing stderr."""
    done, _ = await asyncio.wait({feeder}, timeout=0.5)
    if feeder in done:
        with contextlib.suppress(RuntimeError, asyncio.CancelledError):
            await feeder


def format_acp_failure(
    *,
    phase: str,
    error_info: ErrorInfo | None,
    acp_error: BaseException | str | None,
) -> str:
    """Format a user-facing ACP failure without adapter-specific parsing."""
    parts = [f"ACP {phase} failed"]

    acp_error_text = (
        str(acp_error)
        if acp_error is not None
        else "ACP adapter process exited unexpectedly."
    )
    stderr_text = error_info.stderr.strip() if error_info is not None else ""

    parts.append("")
    parts.append(f"ACP error:\n{acp_error_text}")
    if stderr_text:
        parts.append("")
        parts.append(f"Adapter stderr:\n{stderr_text}")
    return "\n".join(parts)


_UNWRAPPED_CONNECTION_METHODS = {
    "close",
    "__aenter__",
    "__aexit__",
    "on_connect",
}


def _wrap_connection_method(
    conn: ClientSideConnection,
    method_name: str,
    method: Callable[..., Awaitable[Any]],
    feeder: asyncio.Task[None],
    error_info: ErrorInfo,
) -> Callable[..., Awaitable[Any]]:
    """Wrap a connection method so failures always include stderr."""

    @wraps(method)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return await method(*args, **kwargs)
        except Exception as ex:
            await _await_stderr_flush(feeder)
            message = format_acp_failure(
                phase=method_name,
                error_info=error_info,
                acp_error=ex,
            )
            raise ACPError(message) from ex

    return wrapped


def _wrap_connection_methods(
    conn: ClientSideConnection,
    feeder: asyncio.Task[None],
    error_info: ErrorInfo,
) -> ClientSideConnection:
    """Wrap public async connection methods on the real ACP connection object."""
    for method_name, method_obj in inspect.getmembers(type(conn)):
        if method_name.startswith("_") or method_name in _UNWRAPPED_CONNECTION_METHODS:
            continue
        if not callable(method_obj):
            continue
        if not (
            inspect.iscoroutinefunction(method_obj)
            or inspect.iscoroutinefunction(getattr(method_obj, "__wrapped__", None))
        ):
            continue
        method = getattr(conn, method_name)
        setattr(
            conn,
            method_name,
            _wrap_connection_method(conn, method_name, method, feeder, error_info),
        )
    return conn


@contextlib.asynccontextmanager
async def acp_connection(
    proc: ExecRemoteProcess,
) -> AsyncIterator[tuple[ClientSideConnection, asyncio.Task[None], ErrorInfo]]:
    """Bridge an ``ExecRemoteProcess`` to ACP.  Yield ``(conn, feeder, error_info)``.

    Bridges ``ExecRemoteProcess`` to the SDK's ``connect_to_agent()`` via
    a transport wrapper, then cleans up on exit.

    *feeder* is a background task that reads process stdout and feeds it
    into the ACP reader.  It completes when the process exits, so callers
    can ``await feeder`` to detect unexpected process termination.

    *proc_info* collects stderr output and the exit code as the process
    runs.  Inspect after ``await feeder`` for full diagnostics.

    Usage::

        async with acp_connection(proc) as (conn, feeder, proc_info):
            await conn.initialize(...)
            session = await conn.new_session(...)
            await conn.prompt(...)
    """
    reader, writer, feeder, proc_info = await create_exec_remote_streams(proc)
    conn = connect_to_agent(DefaultClient(), writer, reader)
    wrapped_conn = _wrap_connection_methods(conn, feeder, proc_info)
    try:
        yield wrapped_conn, feeder, proc_info
    finally:
        with anyio.CancelScope(shield=True):  # ensure cleanup on cancel
            await conn.close()
            writer.close()
            feeder.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feeder
            with contextlib.suppress(RuntimeError):
                await proc.close_stdin()
            await proc.kill()
