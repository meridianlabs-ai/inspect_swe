"""Unit tests for CodexCli resume wiring (no live codex / sandbox needed).

Covers the codex-specific resume glue: constructor fail-fast, that a
``resume_rollout`` wires ``resume_session_id`` for the base class, and that
``_prepare_resume`` writes the rollout into ``CODEX_HOME`` (warning on a
model mismatch).
"""

import logging
from datetime import UTC, datetime
from typing import Any

import anyio
import pytest
from inspect_swe._util.path import join_path
from inspect_swe.acp._agents.codex_cli import build_rollout
from inspect_swe.acp._agents.codex_cli import codex_cli as mod
from inspect_swe.acp._agents.codex_cli.rollout import RolloutSpec
from inspect_swe.acp.agent import ACPAgent

_TS = datetime(2026, 6, 11, 12, 30, 0, tzinfo=UTC)


class _FakeSbox:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    async def write_file(self, path: str, content: str) -> None:
        self.writes.append((path, content))


class _FakeModel:
    def __init__(self, name: str) -> None:
        self._name = name

    def canonical_name(self) -> str:
        return self._name


def _spec(model: str) -> RolloutSpec:
    return build_rollout(cwd="/w", prior=[], model=model, timestamp=_TS)


def test_resume_session_id_without_rollout_raises() -> None:
    # A bare resume_session_id can't materialize codex's on-disk rollout.
    with pytest.raises(ValueError, match="resume_rollout"):
        mod.CodexCli(resume_session_id="some-id")


def test_resume_rollout_wires_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init(self: Any, **kwargs: Any) -> None:  # bypass active-sample req
        captured.update(kwargs)

    monkeypatch.setattr(ACPAgent, "__init__", fake_init)
    spec = _spec("gpt-5.5")
    agent = mod.CodexCli(resume_rollout=spec)
    assert captured["resume_session_id"] == spec.session_id
    assert agent._resume_rollout is spec


def test_interactive_codex_cli_forwards_resume_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the public factory must forward resume_rollout through to CodexCli
    captured: dict[str, Any] = {}

    def fake_init(self: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(ACPAgent, "__init__", fake_init)
    spec = _spec("gpt-5.5")
    agent = mod.interactive_codex_cli(resume_rollout=spec)
    assert isinstance(agent, mod.CodexCli)
    assert agent._resume_rollout is spec
    assert captured["resume_session_id"] == spec.session_id


def _prepared_agent(model: str, spec: RolloutSpec) -> mod.CodexCli:
    agent = object.__new__(mod.CodexCli)  # skip __init__ (needs an active sample)
    agent._resume_rollout = spec
    agent._codex_home = "/home/user/.codex"
    agent.sandbox = None
    agent.model = model
    return agent


def test_prepare_resume_writes_rollout_into_codex_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sbox = _FakeSbox()
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: sbox)
    monkeypatch.setattr(mod, "get_model", lambda name=None: _FakeModel("gpt-5.5"))

    spec = _spec("gpt-5.5")
    agent = _prepared_agent("gpt-5.5", spec)

    async def run() -> None:
        await agent._prepare_resume(spec.session_id)

    anyio.run(run)
    assert sbox.writes == [
        (join_path("/home/user/.codex", spec.relative_path), spec.content)
    ]


def test_prepare_resume_warns_on_model_mismatch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: _FakeSbox())
    monkeypatch.setattr(mod, "get_model", lambda name=None: _FakeModel("gpt-5.5"))

    # rollout built for a different model than the agent resolves to
    spec = _spec("gpt-4.1")
    agent = _prepared_agent("gpt-5.5", spec)

    async def run() -> None:
        await agent._prepare_resume(spec.session_id)

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        anyio.run(run)
    assert any("differs from" in r.message for r in caplog.records)


def test_prepare_resume_no_warning_on_model_match(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: _FakeSbox())
    monkeypatch.setattr(mod, "get_model", lambda name=None: _FakeModel("gpt-5.5"))

    spec = _spec("gpt-5.5")
    agent = _prepared_agent("gpt-5.5", spec)

    async def run() -> None:
        await agent._prepare_resume(spec.session_id)

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        anyio.run(run)
    assert not [r for r in caplog.records if "differs from" in r.message]
