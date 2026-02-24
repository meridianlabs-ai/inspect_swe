"""Base class for ACP-based agents running in sandboxes.

Subclasses implement ``_start_agent()`` to provide agent-specific
setup and return the running ``ExecRemoteProcess`` plus bridge handle.
"""

import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from acp import PROTOCOL_VERSION
from acp.client.connection import ClientSideConnection
from acp.schema import HttpMcpServer
from inspect_ai.agent import Agent, AgentState, BridgedToolsSpec, SandboxAgentBridge
from inspect_ai.log._samples import sample_active
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model, get_model
from inspect_ai.tool import MCPServerConfig, MCPServerConfigHTTP
from inspect_ai.util import ExecRemoteProcess
from typing_extensions import TypedDict, Unpack

from .client import acp_connection

logger = logging.getLogger(__name__)


def bridge_mcp_to_acp(configs: list[MCPServerConfigHTTP]) -> list[HttpMcpServer]:
    """Convert bridge ``MCPServerConfigHTTP`` objects to ACP ``HttpMcpServer``."""
    result: list[HttpMcpServer] = []
    for cfg in configs:
        result.append(
            HttpMcpServer(
                type="http",
                name=cfg.name,
                url=cfg.url,
                headers=[],
            )
        )
        logger.info("Bridge MCP -> ACP: %s @ %s", cfg.name, cfg.url)
    return result


class ACPAgentParams(TypedDict, total=False):
    """Keyword arguments accepted by :class:`ACPAgent`.

    Attributes:
        model: Model name or instance (defaults to the task's main model).
        filter: Filter for intercepting bridged model API requests.
        bridged_tools: Host-side Inspect tools exposed to the agent via MCP.
        mcp_servers: Additional MCP servers (HTTP configs are converted
            to ACP format for ``new_session``).
        system_prompt: Appended to the agent's default system prompt.
        retry_refusals: Number of times to retry on model refusals.
        model_map: Canonical model name -> Model or model name overrides.
        cwd: Working directory inside the sandbox.
        env: Extra environment variables for the agent process.
        user: User to execute the agent as in the sandbox.
        sandbox: Sandbox environment name.
    """

    model: str | Model | None
    filter: GenerateFilter | None
    bridged_tools: list[BridgedToolsSpec] | None
    mcp_servers: list[MCPServerConfig] | None
    system_prompt: str | None
    retry_refusals: int | None
    model_map: dict[str, str | Model] | None
    cwd: str | None
    env: dict[str, str] | None
    user: str | None
    sandbox: str | None


class ACPAgent(Agent):
    """Base class for ACP-based agents running in sandboxes.

    Manages the ACP lifecycle (connection, session, MCP announcement,
    cleanup).  Subclasses implement :meth:`_start_agent` for
    agent-specific setup.

    Sets up the ACP lifecycle, exposes ``.conn`` and ``.session_id``,
    signals ``.ready``, then blocks until the task is cancelled.  The
    caller drives all prompts via ``conn.prompt()`` / ``conn.cancel()``.
    """

    def __init__(self, **kwargs: Unpack[ACPAgentParams]) -> None:
        if sample_active() is None:
            raise RuntimeError(
                f"{type(self).__name__} must be created within an active sample "
                f"(e.g. inside a solver's solve() function), not at task "
                f"definition time. Each sample needs its own ACP agent instance "
                f"to avoid sharing connection state across samples."
            )

        self.conn: ClientSideConnection | None = None
        self.session_id: str | None = None

        self.model: str | Model = kwargs.get("model") or get_model()
        self.filter = kwargs.get("filter")
        self.bridged_tools: list[BridgedToolsSpec] = kwargs.get("bridged_tools") or []
        self.mcp_servers: list[MCPServerConfig] = kwargs.get("mcp_servers") or []
        self.system_prompt = kwargs.get("system_prompt")
        self.retry_refusals = kwargs.get("retry_refusals")
        self.cwd = kwargs.get("cwd") or "/home/user"
        self.env: dict[str, str] = kwargs.get("env") or {}
        self.user = kwargs.get("user")
        self.sandbox = kwargs.get("sandbox")

        self.model_map: dict[str, str | Model] = self._build_model_map()
        if kwargs.get("model_map"):
            self.model_map.update(kwargs["model_map"])

        self.ready = anyio.Event()  # signals conn/session_id are usable

    def _build_model_map(self) -> dict[str, str | Model]:
        """Map canonical model name -> Model instance. Subclasses can override to add entries."""
        model = get_model(self.model)
        return {model.canonical_name(): model}

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    @asynccontextmanager
    async def _start_agent(
        self,
        state: AgentState,
    ) -> AsyncIterator[tuple[ExecRemoteProcess, SandboxAgentBridge]]:
        """Launch the ACP adapter process.  Yield ``(proc, bridge)``.

        *proc* is the ``ExecRemoteProcess`` with ``stdin_open=True``.
        *bridge* tracks conversation history and provides
        ``mcp_server_configs`` for bridged tools.

        The base class handles the ACP lifecycle (connection, initialize,
        new_session, close) around whatever this method yields.
        """
        yield

    def _resolve_system_prompt(self, state: AgentState) -> str | None:
        """Merge system messages from *state* with ``self.system_prompt``."""
        parts = [m.text for m in state.messages if isinstance(m, ChatMessageSystem)]
        if self.system_prompt is not None:
            parts.append(self.system_prompt)
        return "\n\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    async def __call__(
        self, state: AgentState, *args: Any, **kwargs: Any
    ) -> AgentState:
        self.ready = anyio.Event()
        try:
            async with self._start_agent(state) as (proc, bridge):
                try:
                    all_configs: list[MCPServerConfigHTTP] = [
                        c
                        for c in self.mcp_servers
                        if isinstance(c, MCPServerConfigHTTP)
                    ]
                    all_configs.extend(bridge.mcp_server_configs)
                    acp_mcp_servers = bridge_mcp_to_acp(all_configs)

                    # Wait for the bridge proxy MCP endpoints to be
                    # reachable before starting the ACP session.  Some
                    # agents (e.g. gemini CLI) connect to MCP servers
                    # synchronously during new_session and will silently
                    # skip tools if the proxy isn't ready yet.
                    if all_configs:
                        await _wait_for_mcp_endpoints(all_configs, bridge)

                    async with acp_connection(proc) as (conn, feeder, error_info):
                        logger.info("ACP: initializing...")
                        await conn.initialize(protocol_version=PROTOCOL_VERSION)

                        logger.info(
                            "ACP: creating session (cwd=%s, mcp_servers=%d)",
                            self.cwd,
                            len(acp_mcp_servers),
                        )

                        session = await conn.new_session(
                            cwd=self.cwd,
                            mcp_servers=acp_mcp_servers or None,
                        )

                        self.conn = conn
                        self.session_id = session.session_id

                        try:
                            self.ready.set()
                            # Block until either the caller cancels the task or
                            # the ACP adapter process exits.
                            await feeder
                            if error_info.exit_code != 0:
                                raise RuntimeError(
                                    "ACP adapter process exited with an error "
                                    "while session was still active\n"
                                    + error_info.summary()
                                )
                        finally:
                            self.conn = None
                            self.session_id = None
                finally:
                    state.messages = bridge.state.messages
                    state.output = bridge.state.output
        except anyio.get_cancelled_exc_class():
            # Return partial state instead of propagating -- cleanup
            # already happened in the finally blocks above.
            logger.info("ACPAgent: cancelled, returning partial state")

        return state


async def _wait_for_mcp_endpoints(
    configs: list[MCPServerConfigHTTP],
    bridge: SandboxAgentBridge,
    timeout: float = 30.0,
    interval: float = 0.5,
) -> None:
    """Wait until bridge MCP HTTP endpoints are reachable from the sandbox.

    The bridge proxy starts asynchronously and may not be listening yet
    when ``_start_agent`` yields.  This polls the first MCP endpoint
    until it responds.
    """
    from inspect_ai.util import sandbox as sandbox_env

    sbox = sandbox_env()
    url = configs[0].url
    elapsed = 0.0

    while elapsed < timeout:
        result = await sbox.exec(
            ["bash", "-c", f"curl -sf -o /dev/null --max-time 2 -X POST {url} 2>/dev/null && echo OK || echo FAIL"],
        )
        if "OK" in result.stdout:
            logger.info("Bridge MCP endpoint ready at %s (%.1fs)", url, elapsed)
            return
        await anyio.sleep(interval)
        elapsed += interval

    logger.warning(
        "Bridge MCP endpoint at %s not ready after %.0fs — proceeding anyway",
        url,
        timeout,
    )
