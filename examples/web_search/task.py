from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_swe import claude_code, codex_cli


@task
def web_search(agent: Literal["claude_code", "codex_cli"] = "claude_code") -> Task:
    # setup agent
    system_prompt = "Please use the WebSearch tool to research this question"
    match agent:
        case "claude_code":
            solver = claude_code(system_prompt=system_prompt)
        case "codex_cli":
            solver = codex_cli(system_prompt=system_prompt)

    # create task
    return Task(
        dataset=[
            Sample(
                input="What transport protocols are supported in "
                + " the 2025-03-26 version of the MCP spec?"
            )
        ],
        solver=solver,
        sandbox="docker",
    )
