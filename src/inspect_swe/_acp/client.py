"""Minimal ACP Client implementation + ACPClient wrapper.

Follows the `SDK example client
<https://github.com/agentclientprotocol/python-sdk/blob/main/examples/client.py>`_
pattern: reject fs/terminal methods, log session updates.
"""

import asyncio
import contextlib
import logging
from typing import Any

from acp import RequestError, connect_to_agent
from acp.client.connection import ClientSideConnection
from acp.interfaces import Agent
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AvailableCommandsUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    KillTerminalCommandResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from inspect_ai.util._sandbox.exec_remote import ExecRemoteProcess

from .transport import create_exec_remote_streams

logger = logging.getLogger(__name__)


class _MinimalClient:
    """ACP Client that rejects fs/terminal methods and logs session updates.

    Agents running with ``--dangerously-skip-permissions`` (or equivalent)
    handle all tool execution internally and won't call these methods.
    """

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        raise RequestError.method_not_found("session/request_permission")

    async def session_update(
        self,
        session_id: str,
        update: (
            UserMessageChunk
            | AgentMessageChunk
            | AgentThoughtChunk
            | ToolCallStart
            | ToolCallProgress
            | AgentPlanUpdate
            | AvailableCommandsUpdate
            | CurrentModeUpdate
        ),
        **kwargs: Any,
    ) -> None:
        logger.debug("ACP update [%s]: %s", session_id, type(update).__name__)

    async def read_text_file(
        self, path: str, session_id: str, **kwargs: Any
    ) -> ReadTextFileResponse:
        raise RequestError.method_not_found("fs/read_text_file")

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def create_terminal(
        self, command: str, session_id: str, **kwargs: Any
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

    async def ext_method(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(
        self, method: str, params: dict[str, Any]
    ) -> None:
        pass

    def on_connect(self, conn: Agent) -> None:
        pass


class ACPClient:
    """High-level ACP client backed by the SDK's ``ClientSideConnection``.

    Wraps ``ExecRemoteProcess`` via the transport bridge so that the
    SDK's ``connect_to_agent()`` works.  Usage::

        acp_client = ACPClient(proc)
        conn = await acp_client.start()   # returns ClientSideConnection
        await conn.initialize(...)
        await conn.new_session(...)
        await conn.prompt(...)
        await acp_client.close()
    """

    def __init__(self, proc: ExecRemoteProcess) -> None:
        self._proc = proc
        self.conn: ClientSideConnection | None = None
        self._feeder: asyncio.Task[None] | None = None

    async def start(self) -> ClientSideConnection:
        """Create the transport bridge and return a ``ClientSideConnection``."""
        reader, writer, self._feeder = await create_exec_remote_streams(
            self._proc
        )
        self.conn = connect_to_agent(_MinimalClient(), writer, reader)
        return self.conn

    async def close(self) -> None:
        """Shut down the connection, feeder task, and process."""
        if self.conn:
            await self.conn.close()
        if self._feeder:
            self._feeder.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._feeder
        with contextlib.suppress(RuntimeError):
            await self._proc.close_stdin()
        await self._proc.kill()
