from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.tool import MCPServerConfigStdio
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli


@task
def mcp_memory(
    agent: Literal["claude_code", "codex_cli"] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    # setup agent
    system_prompt = "You MUST use the memory tools to keep track of your work. Please note all findings using the memory tools."
    mcp_servers = [
        MCPServerConfigStdio(
            name="memory",
            command="npx",
            args=["--offline", "@modelcontextprotocol/server-memory"],
        )
    ]
    match agent:
        case "claude_code":
            solver = claude_code(system_prompt=system_prompt, mcp_servers=mcp_servers)
        case "codex_cli":
            solver = codex_cli(system_prompt=system_prompt, mcp_servers=mcp_servers)

    # create task
    return Task(
        dataset=[
            Sample(
                input=f"List the contents of the current directory, then use the memory tools to record what you found. Then, on the next turn, read from memory your findings and report them. {system_prompt}"
            )
        ],
        solver=solver,
        sandbox=sandbox,
    )
