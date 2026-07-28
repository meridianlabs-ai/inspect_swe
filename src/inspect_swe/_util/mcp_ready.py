"""Wait for bridge MCP HTTP endpoints to become reachable from the sandbox.

The bridge proxy starts asynchronously, so its MCP endpoints may not be
listening yet at the moment an agent subprocess is launched. Agents that
connect to MCP servers synchronously at startup will then come up with NO MCP
tools and — critically — will not report an error: they simply behave as though
the tools do not exist. This is a silent-failure mode, so every agent launch
that depends on bridged MCP tools should gate on this.
"""

import logging

import anyio
from inspect_ai.agent import SandboxAgentBridge
from inspect_ai.tool import MCPServerConfigHTTP

logger = logging.getLogger(__name__)

DEFAULT_MCP_READY_TIMEOUT = 30.0
DEFAULT_MCP_READY_INTERVAL = 0.5


class MCPEndpointsUnreachableError(RuntimeError):
    """Bridged MCP endpoints never became reachable from the sandbox.

    Raised so the sample ERRORS rather than proceeding. An agent launched
    without its bridged tools does not fail visibly -- it reports "No such tool
    available", produces an empty response, and that response gets scored as a
    normal trajectory. An errored sample is retryable and honest; a scored
    toolless one silently corrupts results.
    """


async def wait_for_mcp_endpoints(
    configs: list[MCPServerConfigHTTP],
    bridge: SandboxAgentBridge,
    timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    interval: float = DEFAULT_MCP_READY_INTERVAL,
    required: bool = True,
) -> bool:
    """Wait until bridge MCP HTTP endpoints are reachable from the sandbox.

    The bridge proxy starts asynchronously and may not be listening yet when an
    agent subprocess is launched. This polls the first MCP endpoint until it
    responds.

    Returns True once an endpoint is reachable. On timeout, raises
    MCPEndpointsUnreachableError when ``required`` (the default), else returns
    False. Proceeding past an unreachable endpoint means launching an agent that
    will silently have no tools, so callers should have a specific reason to
    pass ``required=False``.
    """
    from inspect_ai.util import sandbox as sandbox_env

    if not configs:
        return True

    sbox = sandbox_env()
    url = configs[0].url
    elapsed = 0.0

    while elapsed < timeout:
        result = await sbox.exec(
            [
                "bash",
                "-c",
                f"curl -sf -o /dev/null --max-time 2 -X POST {url} 2>/dev/null && echo OK || echo FAIL",
            ],
        )
        if "OK" in result.stdout:
            logger.info("Bridge MCP endpoint ready at %s (%.1fs)", url, elapsed)
            return True
        await anyio.sleep(interval)
        elapsed += interval

    message = (
        f"Bridge MCP endpoint at {url} not reachable after {timeout:.0f}s. "
        "Refusing to launch the agent: it would start with no bridged MCP tools "
        "and report no error, so its output would be scored as a valid toolless "
        "trajectory."
    )
    if required:
        logger.error(message)
        raise MCPEndpointsUnreachableError(message)
    logger.warning("%s Proceeding anyway (required=False).", message)
    return False
