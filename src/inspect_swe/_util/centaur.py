from contextlib import AbstractContextManager, nullcontext

from inspect_ai.agent import AgentState, human_cli, run
from inspect_ai.util import sandbox_default
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
    options: CentaurOptions,
    instructions: str,
    bashrc: str,
    state: AgentState,
    *,
    user: str | None = None,
    sandbox: str | None = None,
) -> None:
    """Hand the session to the human, in the environment the agent resolved.

    Args:
        options: Options for centaur mode.
        instructions: Instructions beyond the default task command instructions.
        bashrc: Additional content for the human cli shell's .bashrc.
        state: Agent state to run the human session against.
        user: User to open the session as, or `None` for the sandbox
            environment's own default.
        sandbox: Name of the sandbox to run the session in, or `None` to leave
            the ambient default alone.

    A named sandbox has to become the default for the whole session, not just
    for one call within it: `human_cli` installs the task tools and offers the
    login through unnamed lookups of its own, so a wrapper that resolved a
    named sandbox for its own launch would otherwise hand the human a terminal
    in a different container from the one it installed into.
    """
    selected: AbstractContextManager[None] = (
        sandbox_default(sandbox) if sandbox is not None else nullcontext()
    )
    with selected:
        agent = human_cli(
            answer=options.answer,
            intermediate_scoring=options.intermediate_scoring,
            record_session=options.record_session,
            user=user,
            instructions=instructions,
            bashrc=bashrc,
        )
        await run(agent, state)
