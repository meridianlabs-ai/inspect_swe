from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import run
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli, gemini_cli


@solver
def multi_call_solver(agent_type: Literal["claude_code", "codex_cli", "gemini_cli"]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # create agent
        system_prompt = "Answer simple questions concisely"
        match agent_type:
            case "claude_code":
                agent = claude_code(system_prompt=system_prompt)
            case "codex_cli":
                agent = codex_cli(system_prompt=system_prompt)
            case "gemini_cli":
                agent = gemini_cli(system_prompt=system_prompt)

        # first run
        agent_state = await run(agent, state.messages)

        # run 3 more times with additional questions
        questions = [
            "Great! Now what is 2+2?",
            "Perfect! What color is the sky?",
            "Excellent! What is the capital of France?",
        ]

        for question in questions:
            # append message and run again
            agent_state.messages.append(ChatMessageUser(content=question))
            agent_state = await run(agent, agent_state)

        # transfer state and return
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@task
def multi_call(
    agent: Literal["claude_code", "codex_cli", "gemini_cli"] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    return Task(
        dataset=[Sample(input="What is 1+1?")],
        solver=multi_call_solver(agent),
        sandbox=sandbox,
    )
