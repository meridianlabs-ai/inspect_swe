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

So the probe here is a real `tools/list` and the success condition is that at
least one of the tools the bridge is *supposed* to serve for that server comes
back. That exercises the whole path the agent depends on -- proxy, file RPC,
host service -- and leaves it warm before the agent's timer starts.
"""

import json
import logging

import anyio
from inspect_ai.agent import SandboxAgentBridge
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_ai.util import SandboxEnvironment

logger = logging.getLogger(__name__)

DEFAULT_MCP_READY_TIMEOUT = 120.0
DEFAULT_MCP_READY_INTERVAL = 0.5

# Per-probe HTTP budget. Generous relative to the poll interval because a cold
# tools/list has to make the sandbox->host round trip; a probe that times out is
# retried by the loop, so this bounds one attempt rather than the whole wait.
_PROBE_MAX_TIME = 15.0


class MCPEndpointsUnreachableError(RuntimeError):
    """Bridged MCP endpoints never served a tool listing from the sandbox.

    Raised so the sample ERRORS rather than proceeding. An agent launched
    without its bridged tools does not fail visibly -- it reports "No such tool
    available", produces an empty response, and that response gets scored as a
    normal trajectory. An errored sample is retryable and honest; a scored
    toolless one silently corrupts results.
    """


class MCPProbeExecutableMissingError(RuntimeError):
    """The probe executable (``curl``) is missing from the sandbox image.

    Raised immediately rather than after the full timeout. Without ``curl``,
    the gate has no way to check readiness, and every sample would otherwise
    wait the full ``timeout`` before failing with a misleading "endpoints
    never answered" message. The sandbox image needs ``curl`` for this gate
    to run; bridged-tools documentation notes the dependency.
    """


def _tools_from_probe(stdout: str) -> list[str] | None:
    """Tool names from a ``tools/list`` response, or ``None`` if it wasn't one.

    ``None`` means "no usable answer yet" (empty body, non-JSON, or a JSON-RPC
    error) and is indistinguishable from not-ready for gating purposes. An
    empty list means the endpoint answered and genuinely has no tools, which
    is also treated as keep-waiting; callers who legitimately expect an empty
    server should not probe it in the first place.
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


def _probe_executable_missing(returncode: int, stderr: str) -> bool:
    """True if the probe executable itself couldn't run.

    curl uses ``127`` when the shell can't find it. We also match the
    ``sh``/``bash`` "command not found" message and the exec-time
    "No such file or directory" for defensiveness across sandbox shells.
    Connection failures (return code 7, 28, etc.) are handled elsewhere as
    "endpoint not ready yet" and must NOT short-circuit through this path.
    """
    if returncode == 127:
        return True
    lowered = stderr.lower()
    return (
        "command not found" in lowered
        or "curl: not found" in lowered
        or "no such file or directory" in lowered
    )


async def wait_for_mcp_endpoints(
    configs: list[MCPServerConfigHTTP],
    bridge: SandboxAgentBridge,
    *,
    sandbox: SandboxEnvironment | str | None = None,
    timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    interval: float = DEFAULT_MCP_READY_INTERVAL,
    required: bool = True,
) -> bool:
    """Wait until every bridge MCP endpoint serves its expected tools.

    Polls each endpoint with a real JSON-RPC ``tools/list`` and requires a
    tool the bridge registry says that server should expose to come back. A
    listening endpoint is not enough: only ``tools/list`` crosses to the host,
    so it is the only probe that proves the agent will actually receive tools
    (see module docstring).

    Args:
      configs: Bridged HTTP MCP configs to gate on. Any config whose bridge
        registry entry is empty is skipped: a zero-tool bridge is a valid
        configuration, and probing it forever would fail a healthy gate.
      bridge: The sandbox agent bridge whose ``bridged_tools`` registry is
        the source of truth for what ``tools/list`` should return. The gate
        succeeds for a server as soon as any expected tool comes back.
      sandbox: The sandbox environment the agent will run in, or its name.
        Passed through to ``inspect_ai.util.sandbox`` resolution. Must match
        the sandbox the caller uses to create the bridge and launch the
        CLI, otherwise the probe runs in the wrong container -- typically
        the task's default env when the agent's is different -- and either
        waits out the full timeout or validates an unrelated listener.
      timeout: Total wall-clock seconds to wait across all endpoints. Probe
        runtime counts against this deadline: with hanging probes, an outer
        computation that fixed ``remaining`` once could run well past the
        advertised deadline, so it is recomputed before every probe.
      interval: Seconds to sleep between poll passes.
      required: Raise on timeout when true (default). False returns ``False``
        instead of raising; callers should have a specific reason to opt out,
        because proceeding past an unready endpoint means launching an agent
        that will silently have no tools.

    Returns:
      True once every configured endpoint serves at least one expected tool,
      or the configs list is empty / every server has zero expected tools.

    Raises:
      MCPProbeExecutableMissingError: ``curl`` cannot be executed in the
        sandbox. Raised on the first probe, not after ``timeout``, so a
        misconfigured image fails visibly instead of after two silent minutes.
      MCPEndpointsUnreachableError: ``timeout`` elapsed with at least one
        endpoint still unready and ``required`` is true.
    """
    from inspect_ai.util import sandbox as sandbox_env

    if not configs:
        return True

    # Resolve the sandbox the agent actually uses. sandbox_env() with no name
    # returns the default env, which is not what a task with multiple envs
    # (and a caller-selected `sandbox=`) points its agent at. Callers pass
    # the same sandbox they build the bridge and CLI against.
    sbox: SandboxEnvironment
    if isinstance(sandbox, SandboxEnvironment):
        sbox = sandbox
    else:
        sbox = sandbox_env(sandbox)

    # Skip bridged servers configured with no tools: the bridge registry is
    # the source of truth for what tools/list should return, and a zero-tool
    # bridge is a valid configuration. Probing one forever would fail the
    # gate on a healthy bridge, so short-circuit before polling.
    to_probe: list[tuple[str, str, set[str]]] = []
    bridged_tools = bridge.bridged_tools or {}
    for config in configs:
        expected = set(bridged_tools.get(config.name, {}))
        if not expected:
            logger.info(
                "Bridge MCP endpoint %s registered with zero tools; skipping probe.",
                config.name,
            )
            continue
        to_probe.append((config.name, config.url, expected))

    if not to_probe:
        return True

    pending: dict[str, tuple[str, set[str]]] = {
        name: (url, expected) for name, url, expected in to_probe
    }
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    last_seen: dict[str, str] = {}

    # Wall-clock deadline: each probe's own runtime (up to _PROBE_MAX_TIME while
    # curl waits on a cold endpoint) counts against the budget. Accumulating the
    # sleep interval instead would let a "120s" timeout stretch to roughly
    # timeout/interval * _PROBE_MAX_TIME of real time when probes hang.
    start = anyio.current_time()
    deadline = start + timeout

    while pending:
        if deadline - anyio.current_time() <= 0:
            break
        for name, (url, expected) in list(pending.items()):
            # Recompute before every probe: a slow first endpoint eats the
            # budget of the ones behind it in the same pass, so a fixed
            # `remaining` in the outer scope leaves later probes running
            # well past `timeout`.
            remaining = deadline - anyio.current_time()
            if remaining <= 0:
                break
            probe_budget = min(_PROBE_MAX_TIME, max(0.1, remaining))
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
                    f"{probe_budget:.3f}",
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
                # Bound sbox.exec itself with the exact remaining budget so
                # a hung sandbox transport cannot outlive the deadline. +1
                # gives curl's own --max-time room to fire first and report.
                timeout=max(1, int(probe_budget) + 1),
            )
            if _probe_executable_missing(result.returncode, result.stderr):
                message = (
                    f"MCP readiness probe cannot run: curl returned "
                    f"{result.returncode}. stderr: {result.stderr!r}. "
                    "The sandbox image must ship curl for bridged-tools "
                    "readiness gating."
                )
                logger.error(message)
                raise MCPProbeExecutableMissingError(message)
            tools = _tools_from_probe(result.stdout)
            if tools is None:
                last_seen[name] = "no tool listing yet"
                continue
            served = set(tools)
            hit = served & expected
            if not hit:
                sample_expected = ", ".join(sorted(expected)[:3])
                last_seen[name] = (
                    f"listing served {len(tools)} tools, "
                    f"none matched expected ({sample_expected}...)"
                )
                continue
            logger.info(
                "Bridge MCP endpoint %s ready at %s with %d/%d expected tools (%.1fs)",
                name,
                url,
                len(hit),
                len(expected),
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
