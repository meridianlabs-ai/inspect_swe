"""Unit tests for ACPAgent session-resume branching (no live CLI needed).

Exercises ``ACPAgent._open_session``: new-vs-load selection, the ``loadSession``
capability gate, that ``_resolve_resume_session`` runs before ``load_session``,
that ``_load_session_meta`` reaches the request, and that the base class rejects
``resume_messages`` it can't serialize.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast

import anyio
import pytest
from inspect_ai.model import ChatMessageUser
from inspect_swe.acp import agent as agent_mod
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
        self, cwd: str, session_id: str, mcp_servers: Any = None, **kwargs: Any
    ) -> object:
        self.load_calls.append((cwd, session_id, mcp_servers, kwargs))
        return object()


class _ProbeAgent(ACPAgent):
    """Concrete ACPAgent that records resume resolution; built bypassing __init__."""

    resolved: bool

    def _start_agent(self, state: Any) -> Any:  # satisfies abstractmethod; unused
        raise NotImplementedError

    async def _resolve_resume_session(self) -> str:
        self.resolved = True
        return await super()._resolve_resume_session()


def _agent(
    resume_session_id: str | None = None,
    resume_messages: list[Any] | None = None,
) -> _ProbeAgent:
    # Bypass __init__ (it requires an active sample); set only what _open_session reads.
    agent = object.__new__(_ProbeAgent)
    agent.cwd = "/work"
    agent.resume_session_id = resume_session_id
    agent.resume_messages = resume_messages
    agent.resolved = False
    return agent


async def _open(agent: _ProbeAgent, conn: _FakeConn, init: _FakeInit) -> str:
    return await agent._open_session(conn, init, [])  # type: ignore[arg-type]


def test_new_session_when_not_resuming() -> None:
    agent, conn = _agent(), _FakeConn()

    async def run() -> None:
        sid = await _open(agent, conn, _FakeInit(load_session=True))
        assert sid == "new-session-id"
        assert len(conn.new_calls) == 1
        assert conn.load_calls == []
        assert agent.resolved is False

    anyio.run(run)


def test_resume_calls_load_session_after_resolving() -> None:
    agent, conn = _agent("prior-session"), _FakeConn()

    async def run() -> None:
        sid = await _open(agent, conn, _FakeInit(load_session=True))
        # load_session response carries no id, so we keep the one we passed.
        assert sid == "prior-session"
        assert conn.new_calls == []
        assert conn.load_calls == [("/work", "prior-session", None, {})]
        assert agent.resolved is True  # materialized before load

    anyio.run(run)


def test_resume_without_capability_raises() -> None:
    agent, conn = _agent("prior-session"), _FakeConn()

    async def run() -> None:
        with pytest.raises(ACPError, match="loadSession"):
            await _open(agent, conn, _FakeInit(load_session=False))
        assert conn.load_calls == []
        assert agent.resolved is False  # never materialized when unsupported

    anyio.run(run)


def test_resume_with_no_capabilities_block_raises() -> None:
    agent, conn = _agent("prior-session"), _FakeConn()

    async def run() -> None:
        with pytest.raises(ACPError, match="loadSession"):
            await _open(agent, conn, _FakeInit(load_session=None, has_caps=False))
        assert conn.load_calls == []

    anyio.run(run)


def test_resume_messages_unsupported_by_base_class() -> None:
    # An agent that doesn't override _resolve_resume_session can't serialize a
    # synthetic prior; that must fail loudly rather than start a fresh session.
    agent = _agent(resume_messages=[ChatMessageUser(content="hello")])
    conn = _FakeConn()

    async def run() -> None:
        with pytest.raises(ACPError, match="resume_messages"):
            await _open(agent, conn, _FakeInit(load_session=True))
        assert conn.new_calls == []
        assert conn.load_calls == []

    anyio.run(run)


def test_resume_messages_makes_agent_resuming() -> None:
    agent = _agent(resume_messages=[ChatMessageUser(content="hello")])
    assert agent.is_resuming is True
    assert _agent().is_resuming is False
    assert _agent("sid").is_resuming is True


def test_resolve_resume_session_runs_strictly_before_load_session() -> None:
    # _resolve_resume_session materializes the on-disk session; it MUST complete
    # before load_session, or the server's session/load has nothing to read.
    events: list[str] = []

    class _OrderingConn(_FakeConn):
        async def load_session(
            self, cwd: str, session_id: str, mcp_servers: Any = None, **kwargs: Any
        ) -> object:
            events.append("load")
            return await super().load_session(cwd, session_id, mcp_servers, **kwargs)

    agent, conn = _agent("prior-session"), _OrderingConn()

    async def _resolve() -> str:
        events.append("resolve")
        return "prior-session"

    agent._resolve_resume_session = _resolve  # type: ignore[method-assign]

    async def run() -> None:
        await _open(agent, conn, _FakeInit(load_session=True))
        assert events == ["resolve", "load"]

    anyio.run(run)


def test_load_session_meta_is_forwarded() -> None:
    # Subclasses reach CLI-specific resume options through the request's _meta,
    # which the ACP python client builds from surplus kwargs.
    agent, conn = _agent("prior-session"), _FakeConn()
    meta = {"claudeCode": {"options": {"resumeSessionAt": "row-uuid"}}}
    agent._load_session_meta = lambda: meta  # type: ignore[method-assign]

    async def run() -> None:
        await _open(agent, conn, _FakeInit(load_session=True))
        assert conn.load_calls == [("/work", "prior-session", None, meta)]

    anyio.run(run)


def test_open_session_failure_clears_connection_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent("prior-session")
    agent.mcp_servers = []
    agent.sandbox = None
    agent.mcp_ready_timeout = 1
    agent.conn = cast(Any, object())  # stale state from an earlier invocation
    agent.session_id = "stale-session"
    conn = _FakeConn()

    async def initialize(protocol_version: Any) -> _FakeInit:
        return _FakeInit(load_session=True)

    conn.initialize = initialize  # type: ignore[attr-defined]

    @asynccontextmanager
    async def fake_start(state: Any) -> AsyncIterator[tuple[object, Any]]:
        bridge = SimpleNamespace(
            mcp_server_configs=[],
            state=SimpleNamespace(messages=[], output=None),
        )
        yield object(), bridge

    @asynccontextmanager
    async def fake_connection(proc: object) -> AsyncIterator[tuple[Any, Any, Any]]:
        yield conn, object(), SimpleNamespace(exit_code=0)

    async def fail_open(*args: Any, **kwargs: Any) -> str:
        raise ACPError("load failed")

    agent._start_agent = fake_start  # type: ignore[method-assign]
    agent._open_session = fail_open  # type: ignore[method-assign]
    monkeypatch.setattr(agent_mod, "acp_connection", fake_connection)
    state = SimpleNamespace(messages=[], output=None)

    async def run() -> None:
        with pytest.raises(ACPError, match="load failed"):
            await agent(state)  # type: ignore[arg-type]

    anyio.run(run)
    assert agent.conn is None
    assert agent.session_id is None
