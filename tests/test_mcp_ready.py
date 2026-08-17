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

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_swe._util.mcp_ready import (
    MCPEndpointsUnreachableError,
    MCPProbeExecutableMissingError,
    wait_for_mcp_endpoints,
)


def _http_config(
    url: str = "http://localhost:13337/mcp/taiga-mcp", name: str = "taiga-mcp"
) -> MCPServerConfigHTTP:
    return MCPServerConfigHTTP(type="http", name=name, url=url)


def _bridge_with(**servers_and_tools: list[str]) -> Any:
    """A bridge stub whose ``bridged_tools`` registry advertises the given tools.

    ``bridged_tools`` is a nested dict ``{server_name: {tool_name: Tool}}``; the
    gate only reads keys, so the inner values can be arbitrary sentinels.
    """
    return SimpleNamespace(
        bridged_tools={
            name: {tool: object() for tool in tools}
            for name, tools in servers_and_tools.items()
        }
    )


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
    def __init__(
        self, stdout: str = "", *, stderr: str = "", returncode: int = 0
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.success = returncode == 0
        self.returncode = returncode


def _sandbox_returning(*stdouts: str) -> Any:
    """A fake sandbox whose exec() yields the given stdouts in order."""
    sbox = AsyncMock()
    sbox.exec = AsyncMock(side_effect=[_FakeExecResult(s) for s in stdouts])
    return sbox


def test_returns_true_once_endpoint_serves_tools() -> None:
    sbox = _sandbox_returning(_tools_listing("browser", "read_file"))
    bridge = _bridge_with(**{"taiga-mcp": ["browser", "read_file"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([_http_config()], bridge=bridge)

    assert anyio.run(run) is True
    assert sbox.exec.await_count == 1


def test_probe_sends_a_real_tools_list_request() -> None:
    """The probe must exercise the host round trip, not just the local proxy.

    `tools/list` is the only method that crosses back to the host; `initialize`
    is answered inside the sandbox and proves nothing about whether the agent
    will receive tools.
    """
    sbox = _sandbox_returning(_tools_listing("browser"))
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([_http_config()], bridge=bridge)

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
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, timeout=0.01, interval=0.001
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
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="served no tools"):
        anyio.run(run)


def test_listing_without_any_expected_tool_is_not_ready() -> None:
    """A listing that returns tools the bridge did not register still fails.

    The old ``_NON_ENVIRONMENT_TOOLS`` filter special-cased ``WaitForMcpServers``,
    which is a client-side placeholder some CLIs expose while MCP is still
    connecting -- the bridge proxy cannot return it from ``tools/list``. The
    correct rule is stricter and uses the bridge's own registry: an endpoint is
    ready only when it serves at least one of the tools that server was
    configured to expose, regardless of which name is on the response.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult(_tools_listing("unexpected")))
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="none matched expected"):
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
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, interval=0.001
            )

    assert anyio.run(run) is True
    assert sbox.exec.await_count == 3


def test_non_json_body_is_not_ready() -> None:
    """A proxy error page or partial write must not be read as a listing."""
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult("<html>502 Bad Gateway</html>"))
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, timeout=0.01, interval=0.001
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
    bridge = _bridge_with(a=["browser"], b=["browser"])

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [
                    _http_config(url="http://localhost:1/mcp/a", name="a"),
                    _http_config(url="http://localhost:1/mcp/b", name="b"),
                ],
                bridge=bridge,
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
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="served no tools"):
        anyio.run(run)


def test_required_false_still_allows_opting_out() -> None:
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult(""))
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()],
                bridge=bridge,
                timeout=0.01,
                interval=0.001,
                required=False,
            )

    assert anyio.run(run) is False


def test_no_configs_is_a_no_op() -> None:
    """Agents with no bridged MCP servers must not pay for a sandbox exec."""
    sbox = AsyncMock()
    bridge = _bridge_with()

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([], bridge=bridge)

    assert anyio.run(run) is True
    sbox.exec.assert_not_awaited()


def test_zero_tool_bridged_server_is_skipped_not_probed() -> None:
    """A bridged server registered with no tools is a valid config, not a bug.

    The gate previously polled such an endpoint for the full timeout and then
    failed, because ``tools/list`` cannot return a non-empty listing for a
    server that has none. The bridge's own registry is the source of truth for
    what to expect, and a zero-tool bridge should short-circuit before the
    first probe rather than fail a healthy launch.
    """
    sbox = AsyncMock()
    bridge = _bridge_with(**{"taiga-mcp": []})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, timeout=0.01, interval=0.001
            )

    assert anyio.run(run) is True
    sbox.exec.assert_not_awaited()


def test_missing_probe_executable_fails_fast_with_stderr() -> None:
    """A sandbox without ``curl`` must fail on the first probe, not after 120s.

    Treating command-not-found identically to endpoint-unready made a
    misconfigured image wait the full timeout and then blame the bridge.
    Distinguishing the two is what makes the failure legible from one probe.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(
        return_value=_FakeExecResult(
            stderr="bash: curl: command not found", returncode=127
        )
    )
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()],
                bridge=bridge,
                timeout=60.0,  # deliberately generous so a wait would show
                interval=0.001,
            )

    with pytest.raises(MCPProbeExecutableMissingError, match="curl returned 127"):
        anyio.run(run)
    # One probe: no wait-then-time-out.
    assert sbox.exec.await_count == 1


def test_missing_probe_executable_matches_exec_error_message() -> None:
    """Match exec-time missing-binary messages, not just exit code 127.

    Some shells report the missing binary with a nonzero exit and stderr but
    not exit code 127; the message match keeps the fast-fail behaviour for those.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(
        return_value=_FakeExecResult(stderr="exec: curl: not found", returncode=1)
    )
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()],
                bridge=bridge,
                timeout=60.0,
                interval=0.001,
            )

    with pytest.raises(MCPProbeExecutableMissingError):
        anyio.run(run)
    assert sbox.exec.await_count == 1


def test_exec_timeout_is_treated_as_a_failed_probe_not_a_bare_TimeoutError() -> None:
    """Wrap ``sbox.exec`` in a ``TimeoutError`` catch, not let it escape.

    ``SandboxEnvironment.exec`` raises ``TimeoutError`` when its own
    ``timeout=`` expires. If that escapes, callers see a bare ``TimeoutError``
    instead of ``MCPEndpointsUnreachableError`` (bypassing ``required=False``
    and losing the explanatory message), and the internal ``timeout_retry``
    on the sandbox transport can stretch one probe to ~3x its budget before
    the deadline logic notices. The gate wraps ``sbox.exec`` in a
    ``TimeoutError`` catch and passes ``timeout_retry=False``, so the poll
    loop stays the retry mechanism and the error path stays uniform.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(side_effect=TimeoutError("sandbox transport wedged"))
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()],
                bridge=bridge,
                timeout=0.01,
                interval=0.001,
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="probe exec timed out"):
        anyio.run(run)
    # sbox.exec is called with timeout_retry=False so the poll loop, not the
    # sandbox transport, decides when to give up.
    for call in sbox.exec.await_args_list:
        assert call.kwargs.get("timeout_retry") is False, (
            "wait_for_mcp_endpoints must pass timeout_retry=False; "
            "the poll loop is the retry mechanism"
        )


def test_sandbox_argument_selects_the_agent_environment() -> None:
    """Multi-environment tasks must probe the sandbox the agent will run in.

    Resolving ``sandbox_env()`` with no name returns the task's DEFAULT env, not
    the one the caller passed as ``sandbox=``. The gate then either waits out
    its full timeout in the wrong container or accidentally validates an
    unrelated listener on the same port. Passing the sandbox name through so
    ``sandbox_env(<name>)`` selects the same container the CLI is about to
    launch in is what fixes it.
    """
    sbox = _sandbox_returning(_tools_listing("browser"))
    resolved: dict[str, str | None] = {}

    def resolve(name: str | None = None) -> Any:
        resolved["name"] = name
        return sbox

    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", side_effect=resolve):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, sandbox="agent-env"
            )

    assert anyio.run(run) is True
    assert resolved == {"name": "agent-env"}


def test_bridged_tools_registry_gates_expected_names() -> None:
    """The registry defines what tools/list should return; a match is required.

    The old placeholder-based filter (``_NON_ENVIRONMENT_TOOLS``) was misleading
    because the client-side placeholders it filtered are never returned by the
    bridge. The correct check is that at least one tool the bridge registered
    for this server comes back, which cannot false-positive on a placeholder
    the proxy could not have emitted.
    """
    sbox = _sandbox_returning(_tools_listing("browser", "read_file"))
    bridge = _bridge_with(**{"taiga-mcp": ["read_file"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([_http_config()], bridge=bridge)

    assert anyio.run(run) is True


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
    wait_at = src.index("await wait_for_mcp_endpoints", reset_at)
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

    def has_awaited_gate(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                func = node.value.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else (func.id if isinstance(func, ast.Name) else None)
                )
                if name == "wait_for_mcp_endpoints":
                    return True
        return False

    def consumes_bridged_configs(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "mcp_server_configs"
                and isinstance(node.value, ast.Name)
                and node.value.id == "bridge"
            ):
                return True
        return False

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "mcp_ready.py" or "acp/_agents" in path.as_posix():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not consumes_bridged_configs(tree):
            continue
        # An AST-level awaited call is required: a comment, docstring, or unused
        # import mentioning the gate does not satisfy this (substring checks did).
        if not has_awaited_gate(tree):
            offenders.append(path.relative_to(root).as_posix())
    assert not offenders, (
        "these agents consume bridged MCP configs but never await "
        f"wait_for_mcp_endpoints: {offenders}"
    )


def test_every_self_launching_agent_gates_before_the_centaur_branch() -> None:
    """The nesting fix: the readiness wait must cover the centaur path too.

    Before this round of review, each agent that dispatches through
    ``if centaur: ... else: ...`` gated only inside the ``else`` retry loop, so
    invoking the CLI alias in centaur mode raced the bridge every time. Kimi is
    especially affected because it has no client-side blocking mechanism, but
    Claude Code, Codex, Gemini CLI, and OpenCode share the shape.

    Assert structurally that every agent with a top-level ``if centaur:`` branch
    that consumes bridged configs awaits ``wait_for_mcp_endpoints`` at a
    lexical position BEFORE that ``if centaur:`` block. Retaining a per-attempt
    gate inside the retry loop is fine; this test is about the initial gate.
    """
    root = Path(__file__).parent.parent / "src" / "inspect_swe"

    def _centaur_dispatch_lineno(tree: ast.AST) -> int | None:
        """Line of the `await _run_<agent>_centaur(...)` dispatch, if any."""
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Await) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else None)
            )
            if name and name.endswith("_centaur"):
                return node.lineno
        return None

    def _gate_line_before(tree: ast.AST, dispatch_lineno: int) -> bool:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Await) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else None)
            )
            if name == "wait_for_mcp_endpoints" and node.lineno < dispatch_lineno:
                return True
        return False

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "mcp_ready.py" or "acp/_agents" in path.as_posix():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Only agents that both dispatch through centaur and consume bridged
        # configs are subject to this rule.
        dispatch_lineno = _centaur_dispatch_lineno(tree)
        if dispatch_lineno is None:
            continue
        if not any(
            isinstance(node, ast.Attribute)
            and node.attr == "mcp_server_configs"
            and isinstance(node.value, ast.Name)
            and node.value.id == "bridge"
            for node in ast.walk(tree)
        ):
            continue
        if not _gate_line_before(tree, dispatch_lineno):
            offenders.append(path.relative_to(root).as_posix())
    assert not offenders, (
        "these agents dispatch to centaur without awaiting "
        f"wait_for_mcp_endpoints beforehand: {offenders}"
    )


def test_remaining_deadline_is_recomputed_per_endpoint() -> None:
    """A slow first endpoint must not exhaust the wall-clock for the ones behind.

    A fixed ``remaining`` computed once per pass let later probes run with the
    initial budget even after earlier ones burned it. The recomputation makes
    the wall-clock deadline hold across the whole pending set, not just the
    first endpoint.

    Two endpoints, both never ready, poll interval near zero, timeout ~0.2s,
    each probe burns 0.05s of real time. Interval-counting semantics would
    poll ~200 * 2 times before tripping the deadline; per-probe recomputation
    trips it after ~4 probes across both endpoints.
    """
    calls = 0

    async def slow_never_ready(*args: Any, **kwargs: Any) -> _FakeExecResult:
        nonlocal calls
        calls += 1
        await anyio.sleep(0.05)
        return _FakeExecResult("")

    sbox = AsyncMock()
    sbox.exec = AsyncMock(side_effect=slow_never_ready)
    bridge = _bridge_with(a=["browser"], b=["browser"])

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [
                    _http_config(url="http://localhost:1/mcp/a", name="a"),
                    _http_config(url="http://localhost:1/mcp/b", name="b"),
                ],
                bridge=bridge,
                timeout=0.2,
                interval=0.001,
            )

    with pytest.raises(MCPEndpointsUnreachableError):
        anyio.run(run)
    # A generous headroom that still fails the fixed-``remaining`` loop, which
    # burnt ~400 probes on this input.
    assert calls <= 30, (
        f"per-probe deadline recomputation regressed: {calls} probes ran, "
        "should have stopped after ~4 across both endpoints"
    )


def test_slow_probes_count_against_the_wall_clock_timeout() -> None:
    """The timeout is a wall-clock deadline, not a count of sleep intervals.

    The original loop accumulated only the poll interval, so time spent inside
    a hanging probe (up to curl's --max-time per attempt) was free: a "0.2s"
    timeout with 0.05s probes and a tiny interval could poll for minutes. The
    deadline must include probe time, so with probes that each burn 0.05s of
    real time this must raise after ~0.2s and, decisively, after only a
    handful of probe attempts rather than dozens.
    """
    calls = 0

    async def slow_never_ready(*args: Any, **kwargs: Any) -> _FakeExecResult:
        nonlocal calls
        calls += 1
        await anyio.sleep(0.05)
        return _FakeExecResult("")

    sbox = AsyncMock()
    sbox.exec = AsyncMock(side_effect=slow_never_ready)
    bridge = _bridge_with(**{"taiga-mcp": ["browser"]})

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=bridge, timeout=0.2, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError):
        anyio.run(run)
    # Interval-counting semantics would need ~200 sleeps of 0.001s to trip the
    # timeout, taking ~200 probes; deadline semantics trips after ~4 probes
    # (0.05s each). Allow generous headroom while still failing the old loop.
    assert calls <= 20, (
        f"timeout ignored probe duration: {calls} probes ran, wall-clock "
        "deadline should have stopped after ~4"
    )
