from inspect_ai.agent import AgentState, human_cli, run
from pydantic import BaseModel, Field


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
    options: CentaurOptions, instructions: str, bashrc: str, state: AgentState
) -> None:
    agent = human_cli(
        answer=options.answer,
        intermediate_scoring=options.intermediate_scoring,
        record_session=options.record_session,
        instructions=instructions,
        bashrc=bashrc,
    )
    await run(agent, state)
