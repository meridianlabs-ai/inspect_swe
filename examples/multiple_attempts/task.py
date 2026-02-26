from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli, gemini_cli, mini_swe_agent


@task
def multiple_attempts(
    agent: Literal["claude_code", "codex_cli", "gemini_cli", "mini_swe_agent"] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    # setup agent
    system_prompt = "You will be given two attempts to guess a magic number and you should not make any tools calls in attempting to make your guess -- you just need to do it based on the information you already have."
    attempts = 2
    match agent:
        case "claude_code":
            solver = claude_code(system_prompt=system_prompt, attempts=attempts)
        case "codex_cli":
            solver = codex_cli(system_prompt=system_prompt, attempts=attempts)
        case "gemini_cli":
            solver = gemini_cli(system_prompt=system_prompt, attempts=attempts)
        case "mini_swe_agent":
            solver = mini_swe_agent(system_prompt=system_prompt, attempts=attempts)

    # create task
    return Task(
        dataset=[
            Sample(
                input="Try to guess the magic number",
                target="56198347654",
            )
        ],
        solver=solver,
        scorer=includes(),
        sandbox=sandbox,
    )
