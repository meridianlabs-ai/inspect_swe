"""Wait for bridge MCP endpoints to actually serve tools, from inside the sandbox.

The bridge proxy starts asynchronously, so its MCP endpoints may not be
listening yet at the moment an agent subprocess is launched. Agents that
connect to MCP servers synchronously at startup will then come up with NO MCP
tools and — critically — will not report an error: they simply behave as though
the tools do not exist. This is a silent-failure mode, so every agent launch
that depends on bridged MCP tools should gate on this.

Reachability is not readiness. The bridge serves `/mcp/{name}` from the
in-sandbox proxy, but only `tools/list` crosses back to the host, and that hop
is a file-based RPC the proxy polls with no timeout. So the endpoint answers
long before the host can answer a tool listing, and a bodyless probe cannot
tell the two apart: the proxy replies to an unknown JSON-RPC method with a
well-formed error carried over **HTTP 200**, which any `curl -f` treats as
success. Under load an agent whose own MCP startup timer is bounded then gives
up on a server this gate just declared ready, and the sample proceeds toolless.

So the probe here is a real `tools/list` and the success condition is a
non-empty tool array. That exercises the whole path the agent depends on --
proxy, file RPC, host service -- and leaves it warm before the agent's timer
starts.
"""

import json
import logging

import anyio
from inspect_ai.agent import SandboxAgentBridge
from inspect_ai.tool import MCPServerConfigHTTP

logger = logging.getLogger(__name__)

DEFAULT_MCP_READY_TIMEOUT = 120.0
DEFAULT_MCP_READY_INTERVAL = 0.5

# Per-probe HTTP budget. Generous relative to the poll interval because a cold
# tools/list has to make the sandbox->host round trip; a probe that times out is
# retried by the loop, so this bounds one attempt rather than the whole wait.
_PROBE_MAX_TIME = 15

# Agent-side placeholders some CLIs expose while MCP servers are still
# connecting. They are not bridged tools, so a listing containing only these is
# not a ready endpoint.
_NON_ENVIRONMENT_TOOLS = frozenset({"WaitForMcpServers"})


class MCPEndpointsUnreachableError(RuntimeError):
    """Bridged MCP endpoints never served a tool listing from the sandbox.

    Raised so the sample ERRORS rather than proceeding. An agent launched
    without its bridged tools does not fail visibly -- it reports "No such tool
    available", produces an empty response, and that response gets scored as a
    normal trajectory. An errored sample is retryable and honest; a scored
    toolless one silently corrupts results.
    """


def _tools_from_probe(stdout: str) -> list[str] | None:
    """Tool names from a `tools/list` response, or None if it wasn't one.

    None means "no usable answer yet" (empty body, non-JSON, or a JSON-RPC
    error) and is indistinguishable from not-ready for gating purposes. An
    empty list means the endpoint answered and genuinely has no tools, which is
    also not ready -- the caller treats both as keep-waiting.
    """
    body = stdout.strip()
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or parsed.get("error") is not None:
        return None
    result = parsed.get("result")
    if not isinstance(result, dict):
        return None
    tools = result.get("tools")
    if not isinstance(tools, list):
        return None
    names: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


async def wait_for_mcp_endpoints(
    configs: list[MCPServerConfigHTTP],
    bridge: SandboxAgentBridge,
    timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    interval: float = DEFAULT_MCP_READY_INTERVAL,
    required: bool = True,
) -> bool:
    """Wait until every bridge MCP endpoint serves a non-empty tool listing.

    Polls each endpoint with a real JSON-RPC ``tools/list`` and requires at
    least one environment tool back. A listening endpoint is not enough: only
    ``tools/list`` crosses to the host, so it is the only probe that proves the
    agent will actually receive tools (see module docstring).

    Returns True once all endpoints serve tools. On timeout, raises
    MCPEndpointsUnreachableError when ``required`` (the default), else returns
    False. Proceeding past an endpoint that serves no tools means launching an
    agent that will silently have none, so callers should have a specific
    reason to pass ``required=False``.
    """
    from inspect_ai.util import sandbox as sandbox_env

    if not configs:
        return True

    sbox = sandbox_env()
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    pending = {config.name: config.url for config in configs}
    last_seen: dict[str, str] = {}

    # Wall-clock deadline: each probe's own runtime (up to _PROBE_MAX_TIME while
    # curl waits on a cold endpoint) counts against the budget. Accumulating the
    # sleep interval instead would let a "120s" timeout stretch to roughly
    # timeout/interval * _PROBE_MAX_TIME of real time when probes hang.
    start = anyio.current_time()
    deadline = start + timeout

    while True:
        remaining = deadline - anyio.current_time()
        if remaining <= 0:
            break
        for name, url in list(pending.items()):
            probe_budget = max(1, min(_PROBE_MAX_TIME, int(remaining)))
            # No shell: the URL is passed as a plain argv element, so query
            # strings or metacharacters in a config URL cannot alter the
            # command. -f is deliberately omitted: the proxy returns JSON-RPC
            # errors over HTTP 200, so the status code carries no information
            # and the body is what has to be read.
            result = await sbox.exec(
                [
                    "curl",
                    "-s",
                    "--max-time",
                    str(probe_budget),
                    "-H",
                    "Content-Type: application/json",
                    "-H",
                    "Accept: application/json, text/event-stream",
                    "-X",
                    "POST",
                    "--data-binary",
                    "@-",
                    url,
                ],
                input=request,
            )
            tools = _tools_from_probe(result.stdout)
            if tools is None:
                last_seen[name] = "no tool listing yet"
                continue
            environment_tools = [t for t in tools if t not in _NON_ENVIRONMENT_TOOLS]
            if not environment_tools:
                last_seen[name] = f"listing served {len(tools)} tools, none usable"
                continue
            logger.info(
                "Bridge MCP endpoint %s ready at %s with %d tools (%.1fs)",
                name,
                url,
                len(environment_tools),
                anyio.current_time() - start,
            )
            del pending[name]
        if not pending:
            return True
        await anyio.sleep(interval)

    unready = ", ".join(
        f"{name} ({last_seen.get(name, 'never answered')})" for name in pending
    )
    message = (
        f"Bridge MCP endpoint(s) served no tools after {timeout:.0f}s: {unready}. "
        "Refusing to launch the agent: it would start with no bridged MCP tools "
        "and report no error, so its output would be scored as a valid toolless "
        "trajectory."
    )
    if required:
        logger.error(message)
        raise MCPEndpointsUnreachableError(message)
    logger.warning("%s Proceeding anyway (required=False).", message)
    return False
