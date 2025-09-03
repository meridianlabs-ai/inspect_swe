from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.tool import MCPServerConfigStdio
from inspect_swe import claude_code, codex_cli


@task
def mcp_memory(agent: Literal["claude_code", "codex_cli"] = "claude_code") -> Task:
    # setup agent
    system_prompt = "You MUST use the memory tools to keep track of your work as you invesigate the system configuration. Please note all findings using the memory tools."
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
                input="Without using the internet, investigate the network configuration and report: 1) What network interfaces are present on the system, 2) What is the IP address of the loopback interface, and 3) What port does SSH typically listen on according to its configuration file?"
            )
        ],
        solver=solver,
        sandbox="docker",
    )
