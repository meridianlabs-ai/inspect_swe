"""Regression tests for MCP readiness gating on agent (re)launch.

Context: the claude_code retry loop restarts the Claude Code subprocess with
``--resume``. The bridge proxy starts asynchronously, so its MCP endpoints may
not be reachable at that moment. A resumed session that comes up without its
MCP tools fails SILENTLY: the agent sees "No such tool available", produces an
empty response, and the sample is scored as an ordinary toolless trajectory with
no error field set. Measured in production: 209 samples across six collection
arms, with a session restart preceding every one.

The gate probes ``tools/list`` rather than merely opening a connection, because
reachability and readiness are different facts here: the in-sandbox proxy serves
``/mcp/{name}`` immediately, but only ``tools/list`` crosses to the host, and the
proxy answers an unknown JSON-RPC method with a well-formed error over HTTP 200.
A status-code probe therefore passes while the agent still receives nothing. The
tests below pin that distinction.

These drive coroutines via ``anyio.run`` rather than a pytest async plugin, so
they need no new test dependency or pytest configuration.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_swe._util.mcp_ready import (
    MCPEndpointsUnreachableError,
    wait_for_mcp_endpoints,
)


def _http_config(
    url: str = "http://localhost:13337/mcp/taiga-mcp", name: str = "taiga-mcp"
) -> MCPServerConfigHTTP:
    return MCPServerConfigHTTP(type="http", name=name, url=url)


def _tools_listing(*names: str) -> str:
    """A JSON-RPC tools/list success response advertising the given tools."""
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": n, "inputSchema": {}} for n in names]},
        }
    )


def _jsonrpc_error_over_http_200() -> str:
    """What the bridge proxy actually returns for a probe it cannot service.

    This is the shape that made the original status-code gate useless: the proxy
    replies 200 with a JSON-RPC error body, so `curl -f` reports success.
    """
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32601, "message": "Unknown method: None"},
        }
    )


class _FakeExecResult:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.success = True
        self.returncode = 0


def _sandbox_returning(*stdouts: str) -> Any:
    """A fake sandbox whose exec() yields the given stdouts in order."""
    sbox = AsyncMock()
    sbox.exec = AsyncMock(side_effect=[_FakeExecResult(s) for s in stdouts])
    return sbox


def test_returns_true_once_endpoint_serves_tools() -> None:
    sbox = _sandbox_returning(_tools_listing("browser", "read_file"))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([_http_config()], bridge=AsyncMock())

    assert anyio.run(run) is True
    assert sbox.exec.await_count == 1


def test_probe_sends_a_real_tools_list_request() -> None:
    """The probe must exercise the host round trip, not just the local proxy.

    `tools/list` is the only method that crosses back to the host; `initialize`
    is answered inside the sandbox and proves nothing about whether the agent
    will receive tools.
    """
    sbox = _sandbox_returning(_tools_listing("browser"))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([_http_config()], bridge=AsyncMock())

    assert anyio.run(run) is True
    sent = sbox.exec.await_args.kwargs["input"]
    assert json.loads(sent)["method"] == "tools/list"


def test_jsonrpc_error_over_http_200_is_not_ready() -> None:
    """The bug this gate exists to catch, in its exact wire form.

    The proxy returns JSON-RPC errors with HTTP 200, so the previous
    `curl -sf -X POST` probe (no body, unknown method) always succeeded and
    declared the endpoint ready while `tools/list` would still have failed. The
    agent then launched, got nothing, and was scored as a toolless trajectory.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult(_jsonrpc_error_over_http_200()))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="served no tools"):
        anyio.run(run)


def test_empty_tool_listing_is_not_ready() -> None:
    """An endpoint that answers with zero tools is not ready either.

    This is the shape a healthy-looking-but-unprovisioned bridge produces, and
    it is precisely what must not reach an agent launch.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult(_tools_listing()))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="served no tools"):
        anyio.run(run)


def test_readiness_placeholder_alone_is_not_ready() -> None:
    """A listing of only agent-side placeholders is not an environment."""
    sbox = AsyncMock()
    sbox.exec = AsyncMock(
        return_value=_FakeExecResult(_tools_listing("WaitForMcpServers"))
    )

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="none usable"):
        anyio.run(run)


def test_polls_until_endpoint_serves_tools() -> None:
    """The proxy is up but the host cannot answer yet -- the restart case.

    Empty body, then a JSON-RPC error, then a real listing. Only the last one
    is ready, and the gate must wait rather than launching on either of the
    first two.
    """
    sbox = _sandbox_returning(
        "", _jsonrpc_error_over_http_200(), _tools_listing("browser")
    )

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), interval=0.001
            )

    assert anyio.run(run) is True
    assert sbox.exec.await_count == 3


def test_non_json_body_is_not_ready() -> None:
    """A proxy error page or partial write must not be read as a listing."""
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult("<html>502 Bad Gateway</html>"))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="served no tools"):
        anyio.run(run)


def test_every_configured_endpoint_must_serve_tools() -> None:
    """Gating only the first config leaves later servers silently toolless.

    The previous implementation probed ``configs[0]`` only, so a second bridged
    server could serve nothing and the agent would still launch.
    """

    def exec_for(cmd: list[str], **kwargs: Any) -> _FakeExecResult:
        # Dispatch on the endpoint under probe: 'a' is ready, 'b' never is.
        body = _tools_listing("browser") if "/mcp/a" in cmd[-1] else _tools_listing()
        return _FakeExecResult(body)

    sbox = AsyncMock()
    sbox.exec = AsyncMock(side_effect=exec_for)

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [
                    _http_config(url="http://localhost:1/mcp/a", name="a"),
                    _http_config(url="http://localhost:1/mcp/b", name="b"),
                ],
                bridge=AsyncMock(),
                timeout=0.005,
                interval=0.001,
                required=False,
            )

    assert anyio.run(run) is False


def test_raises_on_timeout_so_the_sample_errors_instead_of_being_scored() -> None:
    """The whole point: never proceed into a silently-toolless launch.

    Proceeding is what produced the original bug -- the agent starts without
    bridged tools, reports no error, and its empty output is scored as a valid
    trajectory. An errored sample is retryable; a scored toolless one is poison.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult(""))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="served no tools"):
        anyio.run(run)


def test_required_false_still_allows_opting_out() -> None:
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult(""))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()],
                bridge=AsyncMock(),
                timeout=0.01,
                interval=0.001,
                required=False,
            )

    assert anyio.run(run) is False


def test_no_configs_is_a_no_op() -> None:
    """Agents with no bridged MCP servers must not pay for a sandbox exec."""
    sbox = AsyncMock()

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([], bridge=AsyncMock())

    assert anyio.run(run) is True
    sbox.exec.assert_not_awaited()


def test_claude_code_gates_every_launch_on_mcp_readiness() -> None:
    """Guard the wiring, not just the helper.

    The helper existing is not the fix -- it already existed in acp/agent.py
    while claude_code launched without it. This asserts the call sits INSIDE the
    retry loop, so it covers ``--resume`` relaunches, which is the actual defect.
    """
    import inspect_swe._claude_code.claude_code as cc

    src = Path(cc.__file__).read_text(encoding="utf-8")
    # Check the CALL, not the import: an unused import would satisfy a bare
    # name check while leaving every launch unguarded.
    assert "await wait_for_mcp_endpoints" in src, (
        "claude_code must await wait_for_mcp_endpoints before launching the "
        "agent, otherwise a resumed session can start with no MCP tools and "
        "fail silently"
    )

    # The await must precede the subprocess launch and follow the per-attempt
    # consumer.reset() -- i.e. inside the retry loop rather than in one-time
    # setup, otherwise resumed launches remain unguarded.
    reset_at = src.index("consumer.reset()")
    wait_at = src.index("await wait_for_mcp_endpoints")
    launch_at = src.index("await sbox.exec_remote")
    assert reset_at < wait_at < launch_at, (
        "wait_for_mcp_endpoints must be awaited inside the retry loop, "
        "between consumer.reset() and the exec_remote launch"
    )


def test_every_self_launching_bridged_agent_gates_on_mcp_readiness() -> None:
    """The narrow-fix guard.

    The original fix covered only claude_code -- the one agent we had a failing
    transcript for -- while four siblings consumed bridge.mcp_server_configs and
    launched their own subprocess with the identical exposure. Any new agent that
    does the same must gate too, so assert it structurally rather than trusting
    reviewers to notice.

    acp/_agents/* are exempt: their MCP connection happens in the ACP
    new_session, which acp/agent.py already gates.
    """
    root = Path(__file__).parent.parent / "src" / "inspect_swe"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "mcp_ready.py" or "acp/_agents" in path.as_posix():
            continue
        src = path.read_text(encoding="utf-8")
        if "bridge.mcp_server_configs" not in src:
            continue
        if "wait_for_mcp_endpoints" not in src:
            offenders.append(path.relative_to(root).as_posix())
    assert not offenders, (
        "these agents consume bridged MCP configs but never wait for the "
        f"endpoints to be reachable: {offenders}"
    )
