from textwrap import dedent
from typing import Literal, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.tool import MCPServerConfig
from inspect_ai.util import sandbox as sandbox_env

from .._util.agentbinary import ensure_agent_binary_installed
from .agentbinary import codex_cli_binary_source

# TODO: attempts
# TODO: mcp_servers
# TODO: search / disallowed_tools
# TODO: other codex-specific options
# TODO: tests (move web_search and disallowed to general)


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
    model: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "latest"] | str = "auto",
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
        model: Model name to use (defaults to main model for task).
        cwd: Working directory to run codex cli within.
        env: Environment variables to set for codex cli
        user: User to execute codex cli with.
        sandbox: Optional sandbox environment name.
        version: Version of codex cli to use. One of:
            - "auto": Use any available version of codex cli in the sandbox, otherwise download the latest version.
            - "sandbox": Use the version of codex cli in the sandbox (raises `RuntimeError` if codex is not available in the sandbox)
            - "latest": Download and use the very latest version of codex cli.
            - "x.x.x": Download and use a specific version of codex cli.
    """
    # resolve model
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        async with sandbox_agent_bridge(state) as bridge:
            # ensure codex is installed and get binary location
            codex_binary = await ensure_agent_binary_installed(
                codex_cli_binary_source(), version, user, sandbox_env(sandbox)
            )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            # built full promot
            prompt = "\n\n".join(
                system_messages
                + [
                    message.text
                    for message in state.messages
                    if isinstance(message, ChatMessageUser)
                ]
            )

            # execute the agent
            result = await sandbox_env(sandbox).exec(
                cmd=[
                    codex_binary,
                    "exec",
                    "--model",
                    model,
                    "--skip-git-repo-check",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--color",
                    "never",
                    prompt,
                ],
                cwd=cwd,
                env={
                    "OPENAI_BASE_URL": f"http://localhost:{bridge.port}/v1",
                }
                | (env or {}),
            )

        if result.success:
            return bridge.state
        else:
            raise RuntimeError(f"Error executing codex agent: {result.stderr}")

    return agent_with(execute, name=name, description=description)
