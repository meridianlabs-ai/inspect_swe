"""Default ACP Client: auto-approves permissions, rejects fs/terminal methods."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

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
            f"No allow option in permission request [session={session_id}]: {available}"
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
        raise RequestError.method_not_found("fs/read_text_file")

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        raise RequestError.method_not_found("fs/write_text_file")

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
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass


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
    try:
        yield conn, feeder, proc_info
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
