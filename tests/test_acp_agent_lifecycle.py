"""ACPAgent lifecycle: the `ready` event a caller waits on, and cwd resolution."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from inspect_ai.agent import AgentState
from inspect_swe.acp import agent as agent_module
from inspect_swe.acp.agent import ACPAgent


class _FakeAgent(ACPAgent):
    start_cwds: list[str]

    @asynccontextmanager
    async def _start_agent(self, state: AgentState) -> AsyncIterator[Any]:
        # subclasses build paths from self.cwd here, so it must be resolved
        self.start_cwds = [*getattr(self, "start_cwds", []), self.cwd]
        bridge = SimpleNamespace(
            mcp_server_configs=[], state=SimpleNamespace(messages=[], output=None)
        )
        yield object(), bridge


@pytest.fixture
def fake_acp(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the sandbox and ACP connection; record the session cwd."""
    session_cwds: list[str] = []

    class _Conn:
        async def initialize(self, **kwargs: Any) -> None:
            pass

        async def new_session(self, cwd: str, **kwargs: Any) -> Any:
            await anyio.sleep(0)  # a real round trip suspends
            session_cwds.append(cwd)
            return SimpleNamespace(session_id="sess-1")

    @asynccontextmanager
    async def _connection(proc: Any) -> AsyncIterator[Any]:
        yield _Conn(), None, None

    async def _resolve(sandbox: Any, user: str | None, cwd: str | None) -> str:
        await anyio.sleep(0)  # a real resolution execs in the sandbox
        return "/root" if cwd is None else cwd

    async def _wait_forever(self: ACPAgent, feeder: Any, error_info: Any) -> None:
        await anyio.sleep_forever()

    monkeypatch.setattr(agent_module, "sample_active", lambda: object())
    monkeypatch.setattr(
        agent_module,
        "get_model",
        lambda m=None: SimpleNamespace(canonical_name=lambda: "m"),
    )
    monkeypatch.setattr(agent_module, "sandbox_env", lambda name=None: object())
    monkeypatch.setattr(agent_module, "resolve_agent_cwd", _resolve)
    monkeypatch.setattr(agent_module, "acp_connection", _connection)
    monkeypatch.setattr(ACPAgent, "_wait_for_active_session", _wait_forever)
    return session_cwds


async def _start(agent: ACPAgent) -> None:
    async with anyio.create_task_group() as tg:
        tg.start_soon(agent, AgentState(messages=[]))
        # waiting on `ready` right after start_soon, before __call__ runs
        with anyio.fail_after(5):
            await agent.ready.wait()
        assert agent.session_id == "sess-1"
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_ready_set_for_caller_waiting_before_call(fake_acp: list[str]) -> None:
    await _start(_FakeAgent(model="m"))


@pytest.mark.anyio
async def test_ready_resets_for_a_second_run(fake_acp: list[str]) -> None:
    agent = _FakeAgent(model="m")
    await _start(agent)
    await _start(agent)


@pytest.mark.anyio
async def test_cwd_defaults_to_sandbox_resolution(fake_acp: list[str]) -> None:
    agent = _FakeAgent(model="m")
    await _start(agent)
    assert fake_acp == ["/root"]
    assert agent.start_cwds == ["/root"]


@pytest.mark.anyio
async def test_explicit_cwd_is_kept(fake_acp: list[str]) -> None:
    agent = _FakeAgent(model="m", cwd="/work")
    await _start(agent)
    assert fake_acp == ["/work"]
    assert agent.start_cwds == ["/work"]


@pytest.mark.anyio
async def test_empty_cwd_is_resolved(fake_acp: list[str]) -> None:
    agent = _FakeAgent(model="m", cwd="")
    await _start(agent)
    assert fake_acp == ["/root"]
