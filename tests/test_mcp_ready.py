"""Regression tests for MCP readiness gating on agent (re)launch.

Context: the claude_code retry loop restarts the Claude Code subprocess with
``--resume``. The bridge proxy starts asynchronously, so its MCP endpoints may
not be reachable at that moment. A resumed session that comes up without its
MCP tools fails SILENTLY: the agent sees "No such tool available", produces an
empty response, and the sample is scored as an ordinary toolless trajectory with
no error field set. Measured in production: 209 samples across six collection
arms, with a session restart preceding every one.

These drive coroutines via ``anyio.run`` rather than a pytest async plugin, so
they need no new test dependency or pytest configuration.
"""

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


def _http_config(url: str = "http://localhost:13337/mcp") -> MCPServerConfigHTTP:
    return MCPServerConfigHTTP(type="http", name="taiga-mcp", url=url)


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


def test_returns_true_once_endpoint_is_reachable() -> None:
    sbox = _sandbox_returning("OK\n")

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints([_http_config()], bridge=AsyncMock())

    assert anyio.run(run) is True
    assert sbox.exec.await_count == 1


def test_polls_until_endpoint_becomes_reachable() -> None:
    """The proxy is not up on the first probe -- this is the restart case."""
    sbox = _sandbox_returning("FAIL\n", "FAIL\n", "OK\n")

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), interval=0.001
            )

    assert anyio.run(run) is True
    assert sbox.exec.await_count == 3


def test_raises_on_timeout_so_the_sample_errors_instead_of_being_scored() -> None:
    """The whole point: never proceed into a silently-toolless launch.

    Proceeding is what produced the original bug -- the agent starts without
    bridged tools, reports no error, and its empty output is scored as a valid
    trajectory. An errored sample is retryable; a scored toolless one is poison.
    """
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult("FAIL\n"))

    async def run() -> bool:
        with patch("inspect_ai.util.sandbox", return_value=sbox):
            return await wait_for_mcp_endpoints(
                [_http_config()], bridge=AsyncMock(), timeout=0.01, interval=0.001
            )

    with pytest.raises(MCPEndpointsUnreachableError, match="not reachable"):
        anyio.run(run)


def test_required_false_still_allows_opting_out() -> None:
    sbox = AsyncMock()
    sbox.exec = AsyncMock(return_value=_FakeExecResult("FAIL\n"))

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
