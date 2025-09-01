import uuid
from textwrap import dedent
from typing import Any, Literal, Sequence

from inspect_ai.agent import (
    Agent,
    AgentState,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.tool import MCPServerConfig
from inspect_ai.util import sandbox as sandbox_env
from pydantic import BaseModel, Field
from pydantic_core import to_json

from .install.install import ensure_claude_code_installed


class ClaudeCodeOptions(BaseModel):
    """Claude Code options."""

    system_prompt: str | None = Field(default=None)
    """Additional system prompt to append to default system prompt."""

    mcp_servers: Sequence[MCPServerConfig] | None = Field(default=None)
    """MCP servers to make available to the agent."""

    model: str | None = Field(default=None)
    """ Model name to use for Opus and Sonnet calls (defaults to main model for task)."""

    small_model: str | None = Field(default=None)
    """Model to use for Haiku calls (defaults to main model for task)."""

    env: dict[str, str] | None = Field(default=None)
    """Environment variables to set for claude code."""


@agent
def claude_code(
    name: str = "Claude Code",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    options: ClaudeCodeOptions | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    user: str | None = None,
    sandbox: str | None = None,
) -> Agent:
    """Claude Code agent.

    Agent that uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) running in a sandbox.

    The agent can either use a version of Claude Code installed in the sandbox, or can download a version and install it in the sandbox (see docs on `version` option below for details).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)
        options: Claude code options.
        version: Version of claude code to use. One of:
            - "auto": Use any available version of claude code in the sandbox, otherwise download the current stable version.
            - "sandbox": Use the version of claude code in the sandbox (raises `RuntimeError` if claude is not available in the sandbox)
            - "stable": Download and use the current stable version of claude code.
            - "latest": Download and use the very latest version of claude code.
            - "x.x.x": Download and use a specific version of claude code.
        user: User to execute claude code with.
        sandbox: Optional sandbox environment name.
    """
    # provide default options if none specified
    options = options or ClaudeCodeOptions()

    # resolve models
    model = f"inspect/{options.model}" if options.model is not None else "inspect"
    small_model = (
        f"inspect/{options.small_model}"
        if options.small_model is not None
        else "inspect"
    )

    async def execute(state: AgentState) -> AgentState:
        async with sandbox_agent_bridge(state) as bridge:
            # ensure claude is installed and get binary location
            claude_binary = await ensure_claude_code_installed(
                version, user, sandbox_env(sandbox)
            )

            # allocate session_id
            session_id = str(uuid.uuid4())

            # base options
            cmd = [
                claude_binary,
                "--session-id",
                session_id,
                "--print",  # run without interactions
                "--dangerously-skip-permissions",
                "--model",
                model,
            ]

            # system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if options.system_prompt is not None:
                system_messages.append(options.system_prompt)
            if system_messages:
                cmd.extend(["--append-system-prompt", "\n\n".join(system_messages)])

            # mcp servers
            if options.mcp_servers:
                cmd.extend(mcp_server_args(options.mcp_servers))

            # user prompt
            prompt = "\n\n".join(
                [m.text for m in state.messages if isinstance(m, ChatMessageUser)]
            )
            cmd.append("--")
            cmd.append(prompt)

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # execute the agent
            result = await sbox.exec(
                cmd=cmd,
                env={
                    "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                    "ANTHROPIC_API_KEY": "sk-ant-api03-DOq5tyLPrk9M4hPE",
                    "ANTHROPIC_MODEL": model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                    "CLAUDE_CODE_SUBAGENT_MODEL": model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": small_model,
                    "ANTHROPIC_SMALL_FAST_MODEL": small_model,
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "IS_SANDBOX": "1",
                }
                | (options.env or {}),
                user=user,
            )

        if result.success:
            return bridge.state
        else:
            raise RuntimeError(
                f"Error executing claude code agent: {result.stdout}\n{result.stderr}"
            )

    # return agent with specified name and descritpion
    return agent_with(execute, name=name, description=description)


def mcp_server_args(mcp_servers: Sequence[MCPServerConfig]) -> list[str]:
    # build servers and allowed tools
    mcp_servers_json: dict[str, dict[str, Any]] = {}
    allowed_tools: list[str] = []
    for mcp_server in mcp_servers:
        mcp_servers_json[mcp_server.name] = mcp_server.model_dump(
            exclude={"name", "tools"}, exclude_none=True
        )
        if mcp_server.tools == "all":
            allowed_tools.append(f"mcp__{mcp_server.name}_*")
        elif isinstance(mcp_server.tools, list):
            allowed_tools.extend(
                [f"mcp__{mcp_server.name}__{tool}" for tool in mcp_server.tools]
            )
        else:
            raise ValueError(
                f"Unexpected value for mcp server tools: {mcp_server.tools}"
            )

    # map to cli args
    cmds: list[str] = []
    if len(mcp_servers_json) > 0:
        cmds.append("--mcp-config")
        cmds.append(
            to_json({"mcpServers": mcp_servers_json}, exclude_none=True).decode()
        )
    if len(allowed_tools):
        cmds.append("--allowed-tools")
        cmds.append(",".join(allowed_tools))

    return cmds
