"""Transport bridge: ExecRemoteProcess <-> asyncio.StreamReader/StreamWriter.

Follows the exact pattern from ``acp.stdio._WritePipeProtocol`` and
``acp.stdio._StdoutTransport`` to create synthetic streams that satisfy
the ``isinstance`` check in ``ClientSideConnection.__init__``.
"""

import asyncio
import logging
from asyncio import transports as aio_transports
from typing import cast

from inspect_ai.util._sandbox.exec_remote import (
    Completed,
    ExecRemoteProcess,
    StderrChunk,
    StdoutChunk,
)

logger = logging.getLogger(__name__)


class _WritePipeProtocol(asyncio.BaseProtocol):
    """Handles ``drain()`` flow control.

    Same logic as ``acp.stdio._WritePipeProtocol``.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._paused = False
        self._drain_waiter: asyncio.Future[None] | None = None

    def pause_writing(self) -> None:
        self._paused = True
        if self._drain_waiter is None:
            self._drain_waiter = self._loop.create_future()

    def resume_writing(self) -> None:
        self._paused = False
        if self._drain_waiter is not None and not self._drain_waiter.done():
            self._drain_waiter.set_result(None)
        self._drain_waiter = None

    async def _drain_helper(self) -> None:
        if self._paused and self._drain_waiter is not None:
            await self._drain_waiter


class _WriteStdinTransport(asyncio.BaseTransport):
    """Routes ``StreamWriter.write()`` to ``proc.write_stdin()`` via an async queue."""

    def __init__(self, proc: ExecRemoteProcess) -> None:
        super().__init__()
        self._proc = proc
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closing = False
        self._task = asyncio.create_task(self._flush_loop())

    def write(self, data: bytes) -> None:  # type: ignore[override]
        if not self._closing:
            logger.debug("STDIN write (%d bytes): %s", len(data), data[:500])
            self._queue.put_nowait(data)

    async def _flush_loop(self) -> None:
        while True:
            data = await self._queue.get()
            if data is None:
                break
            await self._proc.write_stdin(data)

    def is_closing(self) -> bool:  # type: ignore[override]
        return self._closing

    def close(self) -> None:  # type: ignore[override]
        self._closing = True
        self._queue.put_nowait(None)

    def get_extra_info(self, name: str, default: object = None) -> object:  # type: ignore[override]
        return default


async def create_exec_remote_streams(
    proc: ExecRemoteProcess,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, asyncio.Task[None]]:
    """Create a ``StreamReader``/``StreamWriter`` pair bridged to *proc*.

    Returns ``(reader, writer, feeder_task)``.

    * **reader** receives data from the agent's stdout (via ``StdoutChunk`` events).
    * **writer** sends data to the agent's stdin (via ``proc.write_stdin()``).
    * **feeder_task** is a background task that reads ``StdoutChunk`` events
      and feeds them into *reader*.  The caller should cancel it on cleanup.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()

    protocol = _WritePipeProtocol()
    transport = _WriteStdinTransport(proc)
    writer = asyncio.StreamWriter(
        cast(aio_transports.WriteTransport, transport),
        protocol,
        None,
        loop,
    )

    async def _feed_stdout() -> None:
        logger.debug("_feed_stdout: starting event loop on proc")
        async for event in proc:
            if isinstance(event, StdoutChunk):
                logger.debug("STDOUT chunk (%d chars): %s", len(event.data), event.data[:500])
                reader.feed_data(event.data.encode())
            elif isinstance(event, StderrChunk):
                logger.debug("ACP stderr: %s", event.data.rstrip())
            elif isinstance(event, Completed):
                logger.debug("Process completed (exit=%s)", getattr(event, 'returncode', '?'))
                reader.feed_eof()
        logger.debug("_feed_stdout: event loop ended")

    feeder = asyncio.create_task(_feed_stdout())
    return reader, writer, feeder
