from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.tool import MCPServerConfigStdio
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli, gemini_cli, opencode


@task
def mcp_memory(
    agent: Literal[
        "claude_code", "codex_cli", "gemini_cli", "opencode"
    ] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    # setup agent
    # Name the MCP server explicitly: Claude Code's own system prompt describes
    # a file-based "auto memory" written with the Write tool, so a bare "memory
    # tools" instruction is ambiguous and the model may write files instead.
    system_prompt = "You MUST use the `memory` MCP server's tools (e.g. create_entities, add_observations, read_graph) to keep track of your work. Record all findings with those tools, not in files."
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
        case "gemini_cli":
            solver = gemini_cli(system_prompt=system_prompt, mcp_servers=mcp_servers)
        case "opencode":
            solver = opencode(system_prompt=system_prompt, mcp_servers=mcp_servers)

    # create task
    return Task(
        dataset=[
            Sample(
                input=f"List the contents of the current directory, then record what you found using the `memory` MCP server's tools. Then, on the next turn, read your findings back from the `memory` MCP server and report them. {system_prompt}"
            )
        ],
        solver=solver,
        sandbox=sandbox,
    )
