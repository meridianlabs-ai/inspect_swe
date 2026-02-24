from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli, gemini_cli, mini_swe_agent


@task
def system_explorer(
    agent: Literal["claude_code", "codex_cli", "gemini_cli", "mini_swe_agent"] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    match agent:
        case "claude_code":
            solver = claude_code()
        case "codex_cli":
            solver = codex_cli()
        case "gemini_cli":
            solver = gemini_cli()
        case "mini_swe_agent":
            solver = mini_swe_agent()

    return Task(
        dataset=json_dataset("dataset.json"),
        solver=solver,
        scorer=model_graded_qa(),
        sandbox=sandbox,
    )
