"""ACPAgent -- base class for ACP-based agents.

Satisfies the ``Agent`` protocol (has ``__call__``).  Handles both
interactive mode (caller drives prompts) and non-interactive mode
(single prompt from ``state.messages``, then return).

Subclasses implement ``_start_agent()`` to provide agent-specific setup
(bridge, env vars, adapter command, etc.) and return the running
``ExecRemoteProcess`` plus a bridge handle.
"""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from acp import PROTOCOL_VERSION, text_block
from acp.client.connection import ClientSideConnection

from inspect_ai.agent import AgentState
from inspect_ai.util._sandbox.exec_remote import ExecRemoteProcess

from .client import ACPClient

logger = logging.getLogger(__name__)


class ACPAgent:
    """Base class for ACP-based agents.

    Two modes controlled by ``interactive``:

    ``interactive=False`` (default)
        ``__call__`` sends one prompt built from ``state.messages``,
        waits for the response, and returns.

    ``interactive=True``
        ``__call__`` sets up the ACP lifecycle, exposes ``.conn`` and
        ``.session_id``, signals ``._ready``, then waits on ``._done``.
        The caller drives **all** prompts via ``conn.prompt()`` /
        ``conn.cancel()``.

    Subclasses must implement :meth:`_start_agent` to launch the
    ACP adapter process in a sandbox and return the process handle
    plus a bridge object for collecting conversation history.
    """

    def __init__(self, *, interactive: bool = False, cwd: str | None = None) -> None:
        self.conn: ClientSideConnection | None = None
        """The ACP connection.  Only valid during ``__call__``."""

        self.session_id: str | None = None
        """The ACP session ID.  Only valid during ``__call__``."""

        self.interactive = interactive
        self.cwd = cwd or "/home/user"
        self._ready = asyncio.Event()
        self._done = asyncio.Event()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    @asynccontextmanager
    async def _start_agent(
        self,
        state: AgentState,
    ) -> AsyncIterator[tuple[ExecRemoteProcess, object]]:
        """Launch the ACP adapter process.  Yield ``(proc, bridge)``.

        *proc* is the ``ExecRemoteProcess`` with ``stdin_open=True``.
        *bridge* must have a ``.state`` attribute with ``.messages``
        and ``.output`` for collecting the conversation history.

        The base class handles the ACP lifecycle (ACPClient, initialize,
        new_session, close) around whatever this method yields.
        """
        yield  # type: ignore[misc]

    def _build_prompt(self, state: AgentState) -> str:
        """Build the initial prompt from *state*.

        Override for agent-specific prompt building logic.
        """
        if state.messages:
            return state.messages[0].text
        return ""

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    async def __call__(self, state: AgentState, **kwargs: object) -> AgentState:
        async with self._start_agent(state) as (proc, bridge):
            acp_client = ACPClient(proc)
            conn = await acp_client.start()

            try:
                logger.info("ACP: initializing...")
                await conn.initialize(protocol_version=PROTOCOL_VERSION)

                logger.info("ACP: creating session (cwd=%s)", self.cwd)
                session = await conn.new_session(cwd=self.cwd)

                self.conn = conn
                self.session_id = session.session_id

                if self.interactive:
                    self._ready.set()
                    await self._done.wait()
                else:
                    prompt = self._build_prompt(state)
                    logger.info(
                        "ACPAgent: sending prompt (%d chars)", len(prompt)
                    )
                    await conn.prompt(
                        prompt=[text_block(prompt)],
                        session_id=session.session_id,
                    )
            finally:
                self.conn = None
                self.session_id = None
                self._ready.clear()
                await acp_client.close()

        state.messages = bridge.state.messages  # type: ignore[union-attr]
        state.output = bridge.state.output  # type: ignore[union-attr]
        return state
