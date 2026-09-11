import inspect
from collections.abc import Callable
from typing import cast

from inspect_ai.agent import Agent, AgentState, human_cli, run
from inspect_ai.agent._human.commands.command import HumanAgentCommand
from pydantic import BaseModel, Field

CommandsFilter = Callable[[list[HumanAgentCommand]], list[HumanAgentCommand]]


class CentaurOptions(BaseModel):
    """Options for centaur mode."""

    answer: bool | str = Field(default=True)
    """
    Is an explicit answer required for this task or is it scored
    based on files in the container? Pass a `str` with a regex to validate
    that the answer matches the expected format.
    """

    intermediate_scoring: bool = Field(default=False)
    """Allow the human agent to check their score while working."""

    record_session: bool = Field(default=True)
    """Record all user commands and outputs in the sandbox bash session."""


async def run_centaur(
    options: CentaurOptions,
    instructions: str,
    bashrc: str,
    state: AgentState,
    user: str | None = None,
    commands_filter: CommandsFilter | None = None,
) -> None:
    if commands_filter is not None:
        try:
            signature = inspect.signature(human_cli)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "commands_filter requires an inspect_ai human_cli() signature "
                "that can be inspected."
            ) from error

        commands_filter_parameter = signature.parameters.get("commands_filter")
        supports_commands_filter = (
            commands_filter_parameter is not None
            and commands_filter_parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        ) or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not supports_commands_filter:
            raise RuntimeError(
                "commands_filter requires an inspect_ai version whose "
                "human_cli() accepts commands_filter=."
            )

        agent = _human_cli_with_commands_filter(
            options, instructions, bashrc, user, commands_filter
        )
    else:
        agent = human_cli(
            answer=options.answer,
            intermediate_scoring=options.intermediate_scoring,
            record_session=options.record_session,
            instructions=instructions,
            bashrc=bashrc,
            user=user,
        )

    await run(agent, state)


def _human_cli_with_commands_filter(
    options: CentaurOptions,
    instructions: str,
    bashrc: str,
    user: str | None,
    commands_filter: CommandsFilter,
) -> Agent:
    return cast(Callable[..., Agent], human_cli)(
        answer=options.answer,
        intermediate_scoring=options.intermediate_scoring,
        record_session=options.record_session,
        instructions=instructions,
        bashrc=bashrc,
        user=user,
        commands_filter=commands_filter,
    )
