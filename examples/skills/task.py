from pathlib import Path
from textwrap import dedent
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli, gemini_cli


@task
def agent_skills(
    agent: Literal["claude_code", "codex_cli", "gemini_cli"] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    # setup agent
    system_prompt = dedent("""
        You have access to a skill that will tell you how to find the secret code.
        Use the skill tool first to get instructions, then follow them exactly.
        You must read the asset file AND run the script as instructed.
        """)
    skills = [Path(__file__).parent / "secret-code"]
    match agent:
        case "claude_code":
            solver = claude_code(system_prompt=system_prompt, skills=skills, attempts=2)
        case "codex_cli":
            solver = codex_cli(system_prompt=system_prompt, skills=skills, attempts=2)
        case "gemini_cli":
            solver = gemini_cli(system_prompt=system_prompt, skills=skills, attempts=2)

    # create task
    return Task(
        dataset=[
            Sample(
                input=(
                    "What is the secret code? You MUST first read the asset file "
                    "and tell me what it contains, then run the script to get the answer."
                ),
                target=["ALPHA-BRAVO-CHARLIE", "DELTA-ECHO-FOXTROT"],
            ),
        ],
        solver=solver,
        scorer=includes(),
        sandbox=sandbox,
    )
