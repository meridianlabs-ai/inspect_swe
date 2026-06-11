"""Unit tests for ACPAgent session-resume branching (no live CLI needed).

Exercises ``ACPAgent._open_session``: new-vs-load selection, the ``loadSession``
capability gate, and that ``_prepare_resume`` runs before ``load_session``.
"""

from typing import Any

import anyio
import pytest
from inspect_swe.acp.agent import ACPAgent
from inspect_swe.acp.client import ACPError


class _FakeCaps:
    def __init__(self, load_session: bool | None) -> None:
        self.load_session = load_session


class _FakeInit:
    def __init__(self, load_session: bool | None, *, has_caps: bool = True) -> None:
        self.agent_capabilities = _FakeCaps(load_session) if has_caps else None


class _FakeNewResponse:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _FakeConn:
    def __init__(self) -> None:
        self.new_calls: list[Any] = []
        self.load_calls: list[Any] = []

    async def new_session(self, cwd: str, mcp_servers: Any = None) -> _FakeNewResponse:
        self.new_calls.append((cwd, mcp_servers))
        return _FakeNewResponse("new-session-id")

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: Any = None
    ) -> object:
        self.load_calls.append((cwd, session_id, mcp_servers))
        return object()


class _ProbeAgent(ACPAgent):
    """Concrete ACPAgent that records _prepare_resume; built bypassing __init__."""

    prepared: str | None

    def _start_agent(self, state: Any) -> Any:  # satisfies abstractmethod; unused
        raise NotImplementedError

    async def _prepare_resume(self, session_id: str) -> None:
        self.prepared = session_id


def _agent(resume_session_id: str | None) -> _ProbeAgent:
    # Bypass __init__ (it requires an active sample); set only what _open_session reads.
    agent = object.__new__(_ProbeAgent)
    agent.cwd = "/work"
    agent.resume_session_id = resume_session_id
    agent.prepared = None
    return agent


async def _open(agent: _ProbeAgent, conn: _FakeConn, init: _FakeInit) -> str:
    return await agent._open_session(conn, init, [])  # type: ignore[arg-type]


def test_new_session_when_no_resume_id() -> None:
    agent, conn = _agent(None), _FakeConn()

    async def run() -> None:
        sid = await _open(agent, conn, _FakeInit(load_session=True))
        assert sid == "new-session-id"
        assert len(conn.new_calls) == 1
        assert conn.load_calls == []
        assert agent.prepared is None

    anyio.run(run)


def test_resume_calls_load_session_and_prepare() -> None:
    agent, conn = _agent("prior-session"), _FakeConn()

    async def run() -> None:
        sid = await _open(agent, conn, _FakeInit(load_session=True))
        # load_session response carries no id, so we keep the one we passed.
        assert sid == "prior-session"
        assert conn.new_calls == []
        assert conn.load_calls == [("/work", "prior-session", None)]
        assert agent.prepared == "prior-session"  # materialized before load

    anyio.run(run)


def test_resume_without_capability_raises() -> None:
    agent, conn = _agent("prior-session"), _FakeConn()

    async def run() -> None:
        with pytest.raises(ACPError, match="loadSession"):
            await _open(agent, conn, _FakeInit(load_session=False))
        assert conn.load_calls == []
        assert agent.prepared is None  # never materialized when unsupported

    anyio.run(run)


def test_resume_with_no_capabilities_block_raises() -> None:
    agent, conn = _agent("prior-session"), _FakeConn()

    async def run() -> None:
        with pytest.raises(ACPError, match="loadSession"):
            await _open(agent, conn, _FakeInit(load_session=None, has_caps=False))
        assert conn.load_calls == []

    anyio.run(run)
