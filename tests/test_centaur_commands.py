"""`run_centaur` must forward `user` and `commands_filter` to `human_cli`.

Centaur makes Claude Code available to an Inspect `human_cli()` session. Without these
two forwarded, a hosted human+agent session lands as root and exposes only the stock
task commands -- so a project cannot install a non-root worker or its own submit/score
commands. `human_cli` accepts both; centaur used to drop them.

`human_cli` is mocked here, so the test does not depend on the host `inspect_ai`
carrying the `commands_filter` parameter (added by inspect_ai's human-cli
customize-commands change); it asserts only that `run_centaur` passes both through.
"""

import asyncio

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.agent._human.commands.command import HumanAgentCommand
from inspect_swe._util import centaur as centaur_mod
from inspect_swe._util.centaur import CentaurOptions, run_centaur


def _commands_filter(commands: list[HumanAgentCommand]) -> list[HumanAgentCommand]:
    return commands


def test_run_centaur_forwards_user_and_commands_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_human_cli(**kwargs: object) -> str:
        captured.update(kwargs)
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


def test_run_centaur_defaults_leave_user_and_commands_filter_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_human_cli(**kwargs: object) -> str:
        captured.update(kwargs)
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
    assert captured["commands_filter"] is None
