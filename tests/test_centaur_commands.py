"""Centaur must forward commands filters only to compatible human CLI versions.

Centaur makes a CLI available to an Inspect ``human_cli()`` session. The supplied
user must reach that session so it runs as the intended non-root account. A custom
commands filter requires the optional Inspect AI API, while sessions without one
must retain compatibility with the declared lower dependency bound.
"""

import asyncio
import inspect
from collections.abc import Callable

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.agent._human.commands.command import HumanAgentCommand
from inspect_swe._codex_cli.codex_cli import codex_cli
from inspect_swe._gemini_cli.gemini_cli import gemini_cli
from inspect_swe._kimi_code.kimi_code import kimi_code
from inspect_swe._opencode.opencode import opencode
from inspect_swe._util import centaur as centaur_mod
from inspect_swe._util.centaur import CentaurOptions, run_centaur


def _commands_filter(commands: list[HumanAgentCommand]) -> list[HumanAgentCommand]:
    return commands


def test_run_centaur_forwards_user_and_commands_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_human_cli(
        *,
        answer: object,
        intermediate_scoring: bool,
        record_session: bool,
        instructions: str,
        bashrc: str,
        user: str | None,
        commands_filter: object,
    ) -> str:
        captured.update(
            {
                "user": user,
                "commands_filter": commands_filter,
            }
        )
        return "human-cli-agent"

    async def fake_run(agent: object, state: object) -> None:
        captured["ran"] = agent

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", fake_run)

    asyncio.run(
        run_centaur(
            CentaurOptions(),
            instructions="instr",
            bashrc="bashrc",
            state=AgentState(messages=[]),
            user="agent",
            commands_filter=_commands_filter,
        )
    )

    assert captured["user"] == "agent"
    assert captured["commands_filter"] is _commands_filter
    assert captured["ran"] == "human-cli-agent"


def test_run_centaur_omits_commands_filter_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_human_cli(
        *,
        answer: object,
        intermediate_scoring: bool,
        record_session: bool,
        instructions: str,
        bashrc: str,
        user: str | None,
    ) -> str:
        captured["user"] = user
        return "human-cli-agent"

    async def fake_run(agent: object, state: object) -> None:
        return None

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", fake_run)

    asyncio.run(
        run_centaur(
            CentaurOptions(),
            instructions="instr",
            bashrc="bashrc",
            state=AgentState(messages=[]),
        )
    )

    assert captured["user"] is None


def test_run_centaur_requires_human_cli_command_filter_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_human_cli(
        *,
        answer: object,
        intermediate_scoring: bool,
        record_session: bool,
        instructions: str,
        bashrc: str,
        user: str | None,
    ) -> str:
        return "human-cli-agent"

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)

    with pytest.raises(RuntimeError, match="inspect_ai.*human_cli.*commands_filter"):
        asyncio.run(
            run_centaur(
                CentaurOptions(),
                instructions="instr",
                bashrc="bashrc",
                state=AgentState(messages=[]),
                commands_filter=_commands_filter,
            )
        )


@pytest.mark.parametrize(
    "factory",
    [codex_cli, gemini_cli, kimi_code, opencode],
)
def test_centaur_commands_filter_is_keyword_only(
    factory: Callable[..., object],
) -> None:
    parameters = inspect.signature(factory).parameters

    assert parameters["attempts"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["commands_filter"].kind is inspect.Parameter.KEYWORD_ONLY
