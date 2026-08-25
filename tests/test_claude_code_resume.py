"""Unit tests for ClaudeCode resume wiring (no live CLI / sandbox needed).

Covers the claude-specific resume glue: constructor fail-fast, that a
``resume_transcript`` wires ``resume_session_id``, that
``_resolve_resume_session`` writes the transcript under
``CLAUDE_CONFIG_DIR/projects/...``, and that ``resume_message_uuid`` reaches
``session/load`` as the Agent SDK's ``resumeSessionAt``.
"""

import logging
from typing import Any

import anyio
import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_swe._util.path import join_path
from inspect_swe.acp._agents.claude_code import claude_code as mod
from inspect_swe.acp._agents.claude_code.transcript import (
    AssistantText,
    TranscriptSpec,
    UserText,
    build_transcript,
    parse_transcript,
)
from inspect_swe.acp.agent import ACPAgent


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


def _spec(cwd: str = "/home/user") -> TranscriptSpec:
    return build_transcript(
        cwd=cwd,
        items=[UserText(text="go"), AssistantText(text="done")],
        model="claude-opus-5",
    )


def test_resume_transcript_with_session_id_raises() -> None:
    with pytest.raises(ValueError, match="resume_transcript"):
        mod.ClaudeCode(resume_transcript=_spec(), resume_session_id="other-id")


def test_resume_transcript_with_messages_raises() -> None:
    with pytest.raises(ValueError, match="resume_transcript"):
        mod.ClaudeCode(
            resume_transcript=_spec(),
            resume_messages=[ChatMessageUser(content="hi")],
        )


def test_resume_message_uuid_without_a_resume_input_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Truncating needs something to truncate.
    def fake_init(self: Any, **kwargs: Any) -> None:
        self.resume_session_id = kwargs.get("resume_session_id")
        self.resume_messages = kwargs.get("resume_messages")

    monkeypatch.setattr(ACPAgent, "__init__", fake_init)
    with pytest.raises(ValueError, match="resume_message_uuid"):
        mod.ClaudeCode(resume_message_uuid="row-uuid")


def test_resume_message_uuid_with_messages_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_init(self: Any, **kwargs: Any) -> None:
        self.resume_session_id = kwargs.get("resume_session_id")
        self.resume_messages = kwargs.get("resume_messages")

    monkeypatch.setattr(ACPAgent, "__init__", fake_init)
    with pytest.raises(ValueError, match="fresh row uuids"):
        mod.ClaudeCode(
            resume_message_uuid="row-uuid",
            resume_messages=[ChatMessageUser(content="hi")],
        )


def test_resume_transcript_wires_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init(self: Any, **kwargs: Any) -> None:  # bypass active-sample req
        captured.update(kwargs)
        self.resume_session_id = kwargs.get("resume_session_id")
        self.resume_messages = kwargs.get("resume_messages")

    monkeypatch.setattr(ACPAgent, "__init__", fake_init)
    spec = _spec()
    agent = mod.ClaudeCode(resume_transcript=spec)
    assert captured["resume_session_id"] == spec.session_id
    assert agent._resume_transcript is spec


def test_interactive_claude_code_forwards_resume_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_init(self: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        self.resume_session_id = kwargs.get("resume_session_id")
        self.resume_messages = kwargs.get("resume_messages")

    monkeypatch.setattr(ACPAgent, "__init__", fake_init)
    spec = _spec()
    agent = mod.interactive_claude_code(
        resume_transcript=spec,
        resume_message_uuid=spec.item_uuids[0],
        config_dir="/opt/claude",
    )
    assert isinstance(agent, mod.ClaudeCode)
    assert agent._resume_transcript is spec
    assert agent._resume_message_uuid == spec.item_uuids[0]
    assert agent._config_dir == "/opt/claude"
    assert captured["resume_session_id"] == spec.session_id


def _prepared_agent(
    spec: TranscriptSpec | None,
    *,
    messages: list[Any] | None = None,
    session_id: str | None = None,
    config_dir: str | None = "/root/.claude",
    cwd: str = "/home/user",
    resume_message_uuid: str | None = None,
) -> mod.ClaudeCode:
    agent = object.__new__(mod.ClaudeCode)  # skip __init__ (needs an active sample)
    agent._resume_transcript = spec
    agent._resume_message_uuid = resume_message_uuid
    agent._config_dir = config_dir
    agent._resolved_config_dir = config_dir
    agent.sandbox = None
    agent.user = None
    agent.env = {}
    agent.model = "claude-opus-5"
    agent.cwd = cwd
    agent.resume_messages = messages
    agent.resume_session_id = session_id if spec is None else spec.session_id
    return agent


def _fake_realpath(monkeypatch: pytest.MonkeyPatch, resolved: str) -> None:
    async def fake_exec(sbox: Any, cmd: str, **kwargs: Any) -> str:
        assert cmd.startswith("realpath ")
        return resolved

    monkeypatch.setattr(mod, "sandbox_exec", fake_exec)


def test_resolve_resume_writes_transcript_into_config_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sbox = _FakeSbox()
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: sbox)
    _fake_realpath(monkeypatch, "/home/user")

    spec = _spec()
    agent = _prepared_agent(spec)

    async def run() -> str:
        return await agent._resolve_resume_session()

    session_id = anyio.run(run)
    assert session_id == spec.session_id
    assert sbox.writes == [
        (join_path("/root/.claude", spec.relative_path), spec.content)
    ]
    # the path the SDK will look in: projects/<cwd-slug>/<session>.jsonl
    assert sbox.writes[0][0] == (
        f"/root/.claude/projects/-home-user/{spec.session_id}.jsonl"
    )


def test_resolve_config_dir_honors_env_and_quotes_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[str] = []

    async def fake_exec(sbox: Any, cmd: str, **kwargs: Any) -> str:
        commands.append(cmd)
        return ""

    monkeypatch.setattr(mod, "sandbox_exec", fake_exec)
    agent = _prepared_agent(None, config_dir=None, session_id="sid")
    agent.env = {"CLAUDE_CONFIG_DIR": "/custom claude"}

    async def run() -> str:
        return await agent._resolve_config_dir(_FakeSbox())  # type: ignore[arg-type]

    assert anyio.run(run) == "/custom claude"
    assert commands == ["mkdir -p '/custom claude'"]


def test_explicit_config_dir_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[str] = []

    async def fake_exec(sbox: Any, cmd: str, **kwargs: Any) -> str:
        commands.append(cmd)
        return ""

    monkeypatch.setattr(mod, "sandbox_exec", fake_exec)
    agent = _prepared_agent(None, config_dir="/explicit dir", session_id="sid")
    agent.env = {"CLAUDE_CONFIG_DIR": "/env-dir"}

    async def run() -> str:
        return await agent._resolve_config_dir(_FakeSbox())  # type: ignore[arg-type]

    assert anyio.run(run) == "/explicit dir"
    assert commands == ["mkdir -p '/explicit dir'"]


def test_resolve_resume_builds_transcript_from_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sbox = _FakeSbox()
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: sbox)
    monkeypatch.setattr(mod, "get_model", lambda name=None: _FakeModel("claude-opus-5"))
    _fake_realpath(monkeypatch, "/home/user")

    agent = _prepared_agent(
        None,
        messages=[
            ChatMessageUser(content="add a test"),
            ChatMessageAssistant(content="done"),
        ],
    )

    async def run() -> str:
        return await agent._resolve_resume_session()

    session_id = anyio.run(run)
    assert len(sbox.writes) == 1
    path, content = sbox.writes[0]
    assert path.endswith(f"{session_id}.jsonl")
    parsed = parse_transcript(content)
    assert parsed.session_id == session_id
    assert parsed.items == [UserText(text="add a test"), AssistantText(text="done")]


def test_resolve_resume_session_id_only_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sbox = _FakeSbox()
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: sbox)

    agent = _prepared_agent(None, session_id="11111111-2222-3333-4444-555555555555")

    async def run() -> str:
        return await agent._resolve_resume_session()

    assert anyio.run(run) == "11111111-2222-3333-4444-555555555555"
    assert sbox.writes == []


def test_transcript_lands_under_the_resolved_cwd_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SDK resolves symlinks before slugging the cwd; writing under the
    # unresolved spelling puts the transcript where session/load never looks,
    # and resume then silently starts a fresh conversation.
    sbox = _FakeSbox()
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: sbox)
    _fake_realpath(monkeypatch, "/private/tmp/work")

    spec = _spec(cwd="/tmp/work")
    agent = _prepared_agent(spec, cwd="/tmp/work")

    async def run() -> str:
        return await agent._resolve_resume_session()

    session_id = anyio.run(run)
    assert sbox.writes[0][0] == (
        f"/root/.claude/projects/-private-tmp-work/{session_id}.jsonl"
    )


def test_resolve_resume_warns_when_transcript_lands_in_cwd(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: _FakeSbox())
    _fake_realpath(monkeypatch, "/home/user")
    agent = _prepared_agent(_spec(), config_dir="/home/user/.claude", cwd="/home/user")

    async def run() -> None:
        await agent._resolve_resume_session()

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        anyio.run(run)
    assert any("working directory" in record.message for record in caplog.records)


def test_resolve_resume_rejects_cwd_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Claude Code locates a session by cwd slug, so a transcript built for a
    # different cwd would be written where session/load will never look.
    monkeypatch.setattr(mod, "sandbox_env", lambda name=None: _FakeSbox())
    agent = _prepared_agent(_spec(cwd="/other"), cwd="/home/user")

    async def run() -> str:
        return await agent._resolve_resume_session()

    with pytest.raises(ValueError, match="cwd"):
        anyio.run(run)


def test_load_session_meta_carries_resume_session_at() -> None:
    agent = _prepared_agent(None, session_id="sid", resume_message_uuid="row-uuid")
    assert agent._load_session_meta() == {
        "claudeCode": {"options": {"resumeSessionAt": "row-uuid"}}
    }


def test_load_session_meta_empty_without_truncation() -> None:
    assert _prepared_agent(None, session_id="sid")._load_session_meta() == {}
