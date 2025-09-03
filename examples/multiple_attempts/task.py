from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_swe import claude_code, codex_cli


@task
def multiple_attempts(
    agent: Literal["claude_code", "codex_cli"] = "claude_code",
) -> Task:
    # setup agent
    system_prompt = "You will be given two attempts and you should take only a short time (maximum 3 tool calls) to come with each attempt. If you haven't yet found the magic number after 3 tool calls you should just make a guess."
    attempts = 2
    match agent:
        case "claude_code":
            solver = claude_code(system_prompt=system_prompt, attempts=attempts)
        case "codex_cli":
            solver = codex_cli(system_prompt=system_prompt, attempts=attempts)

    # create task
    return Task(
        dataset=[
            Sample(
                input="Try to guess the magic number (it's hidden somewhere on this system)",
                target="56198347654",
            )
        ],
        solver=solver,
        scorer=includes(),
        sandbox="docker",
    )
