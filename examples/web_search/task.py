from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.tool import MCPServerConfigStdio
from inspect_swe import ClaudeCodeOptions, claude_code


@task
def web_search() -> Task:
    return Task(
        dataset=[
            Sample(
                input="What transport protocols are supported in "
                + " the 2025-03-26 version of the MCP spec?"
            )
        ],
        solver=claude_code(
            options=ClaudeCodeOptions(
                system_prompt="Please use the WebSearch tool to "
                + "research this question and the memory tools "
                + "to keep track of your research.",
                mcp_servers=[
                    MCPServerConfigStdio(
                        name="memory",
                        command="npx",
                        args=["--offline", "@modelcontextprotocol/server-memory"],
                    )
                ],
            )
        ),
        sandbox=("docker", "Dockerfile"),
    )
