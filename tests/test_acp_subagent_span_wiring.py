"""The ACP agents install their consumer on the bridge and reset it after."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from inspect_ai.agent import AgentState
from inspect_swe._claude_code._events.bridge_only import BridgeOnlyLiveConsumer
from inspect_swe._claude_code._events.live_consumer import (
    LiveConsumer,
)
from inspect_swe._codex_cli._events.consumer import CodexConsumer
from inspect_swe.acp import agent as acp_agent_module
from inspect_swe.acp._agents.claude_code import claude_code as cc_module
from inspect_swe.acp._agents.codex_cli import codex_cli as codex_module


class _Stop(Exception):
    """Raised inside the fake bridge to end _start_agent early."""


class _Store(dict[str, Any]):
    def set(self, key: str, value: Any) -> None:
        self[key] = value


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, module: Any, events: list[str]
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_bridge(state: AgentState, **kwargs: Any) -> AsyncIterator[Any]:
        captured.update(kwargs)
        events.append("bridge enter")
        try:
            yield object()
        finally:
            events.append("bridge exit")

    async def fake_setup(*args: Any, **kwargs: Any) -> None:
        raise _Stop

    store = _Store()
    monkeypatch.setattr(acp_agent_module, "sample_active", lambda: object())
    monkeypatch.setattr(module, "sandbox_agent_bridge", fake_bridge)
    monkeypatch.setattr(module, "sandbox_env", lambda name=None: object())
    monkeypatch.setattr(module, "store", lambda: store)
    for name in ("ensure_claude_code_acp_setup", "ensure_codex_acp_setup"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, fake_setup)
    return captured


async def _run_start(agent: Any) -> None:
    with pytest.raises(_Stop):
        async with agent._start_agent(AgentState(messages=[])):
            pass


@pytest.mark.anyio
async def test_acp_claude_code_installs_bridge_only_live_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    captured = _install_fakes(monkeypatch, cc_module, events)
    monkeypatch.setattr(
        LiveConsumer, "reset", lambda self: events.append("reset"), raising=True
    )
    await _run_start(cc_module.ClaudeCode(model="mockllm/model"))

    sink = captured["model_event_sink"]
    # ACP has no stream-json, so spans must close from bridge traffic
    assert isinstance(sink, BridgeOnlyLiveConsumer)
    assert events == ["bridge enter", "bridge exit", "reset"]


@pytest.mark.anyio
async def test_acp_codex_installs_codex_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    captured = _install_fakes(monkeypatch, codex_module, events)
    monkeypatch.setattr(
        CodexConsumer, "reset", lambda self: events.append("reset"), raising=True
    )
    await _run_start(codex_module.CodexCli(model="mockllm/model"))

    assert isinstance(captured["model_event_sink"], CodexConsumer)
    assert events == ["bridge enter", "bridge exit", "reset"]
