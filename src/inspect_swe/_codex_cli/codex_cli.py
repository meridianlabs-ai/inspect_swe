from textwrap import dedent
from typing import Sequence

from inspect_ai.agent import Agent, AgentAttempts, AgentState, agent
from inspect_ai.tool import MCPServerConfig


@agent
def codex_cli(
    name: str = "Codex CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    attempts: int | AgentAttempts = 1,
) -> Agent:
    """Codex CLI.

    Agent that uses OpenAI [Codex CLI](https://github.com/openai/codex) running in a sandbox.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)
        system_prompt: Additional system prompt to append to default system prompt.
        mcp_servers: MCP servers to make available to the agent.
        attempts: Configure agent to make multiple attempts.
    """

    async def execute(state: AgentState) -> AgentState:
        return state

    return execute
