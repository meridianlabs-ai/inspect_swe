"""Transport bridge: ExecRemoteProcess <-> asyncio.StreamReader/StreamWriter.

Mirrors the ``acp.stdio`` transport pattern so the synthetic streams
pass ``ClientSideConnection.__init__``'s type checks.
"""

import asyncio
import logging
from asyncio import transports as aio_transports
from asyncio.streams import FlowControlMixin
from typing import cast

from inspect_ai.util import ExecRemoteEvent, ExecRemoteProcess

logger = logging.getLogger(__name__)


class _WriteStdinTransport(asyncio.BaseTransport):
    """Routes ``StreamWriter.write()`` to ``proc.write_stdin()`` via an async queue."""

    def __init__(self, proc: ExecRemoteProcess) -> None:
        super().__init__()
        self._proc = proc
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closing = False
        self._task = asyncio.create_task(self._flush_loop())

    def write(self, data: bytes) -> None:
        if not self._closing:
            self._queue.put_nowait(data)

    async def _flush_loop(self) -> None:
        try:
            while True:
                data = await self._queue.get()
                if data is None:
                    break
                await self._proc.write_stdin(data)
        except Exception:
            logger.exception("ACP stdin flush loop failed")
            self._closing = True

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True
        self._queue.put_nowait(None)

    def get_extra_info(self, name: str, default: object = None) -> object:
        return default


class ErrorInfo:
    """Exit code and stderr collected from an ``ExecRemoteProcess``."""

    def __init__(self) -> None:
        self.exit_code: int | None = None
        self.stderr: str = ""

    def summary(self) -> str:
        """Human-readable summary of the process exit."""
        parts = [f"exit_code={self.exit_code}"]
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


async def create_exec_remote_streams(
    proc: ExecRemoteProcess,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, asyncio.Task[None], ErrorInfo]:
    """Create ``StreamReader``/``StreamWriter`` bridged to *proc*'s stdin/stdout."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()

    protocol = FlowControlMixin()
    transport = _WriteStdinTransport(proc)
    writer = asyncio.StreamWriter(
        cast(aio_transports.WriteTransport, transport),
        protocol,
        None,
        loop,
    )

    info = ErrorInfo()

    async def _feed_stdout() -> None:
        stderr_parts: list[str] = []
        try:
            async for event in proc:
                if isinstance(event, ExecRemoteEvent.Stdout):
                    reader.feed_data(event.data.encode())
                elif isinstance(event, ExecRemoteEvent.Stderr):
                    stderr_parts.append(event.data)
                    logger.warning("ACP stderr: %s", event.data.rstrip())
                elif isinstance(event, ExecRemoteEvent.Completed):
                    info.exit_code = event.exit_code
                    if event.exit_code != 0:
                        logger.warning(
                            "ACP process exited with code %d",
                            event.exit_code,
                        )
                        raise RuntimeError(
                            f"ACP process exited with code {event.exit_code}"
                        )
                    else:
                        logger.debug("ACP process completed successfully")
        finally:
            info.stderr = "".join(stderr_parts)
            reader.feed_eof()

    feeder = asyncio.create_task(_feed_stdout())
    return reader, writer, feeder, info
