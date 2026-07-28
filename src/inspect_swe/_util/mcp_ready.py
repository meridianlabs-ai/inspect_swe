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


async def wait_for_mcp_endpoints(
    configs: list[MCPServerConfigHTTP],
    bridge: SandboxAgentBridge,
    timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    interval: float = DEFAULT_MCP_READY_INTERVAL,
) -> bool:
    """Wait until bridge MCP HTTP endpoints are reachable from the sandbox.

    The bridge proxy starts asynchronously and may not be listening yet when an
    agent subprocess is launched. This polls the first MCP endpoint until it
    responds.

    Returns True if an endpoint became reachable, False on timeout. Callers that
    can fail loudly should prefer doing so over launching an agent that will
    silently have no tools.
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

    logger.warning(
        "Bridge MCP endpoint at %s not ready after %.0fs — proceeding anyway",
        url,
        timeout,
    )
    return False
