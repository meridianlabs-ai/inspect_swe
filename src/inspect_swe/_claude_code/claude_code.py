from typing import Literal

from inspect_ai.agent import (
    Agent,
    AgentState,
    agent,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.util import sandbox as sandbox_env

from inspect_swe._claude_code.install.install import ensure_claude_code_installed

# TODO: CLAUDE.md
# TODO: Subagents
# TODO: System message
# TODO: Configuration
# TODO: MCP Servers
# TODO: AgentAttempts
# TODO: AgentContinue
# TODO: name and description (a la react())
# TODO: model and small_model


@agent
def claude_code(
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    user: str | None = None,
    sandbox: str | None = None,
) -> Agent:
    """Claude Code agent.

    Agent that uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) running in a sandbox.

    The agent can either use a version of Claude Code installed in the sandbox, or can download a version and install it in the sandbox (see docs on `version` option below for details).

    Args:
        version: Version of claude code to use. One of:
            - "auto": Use any available version of claude code in the sandbox, otherwise download the current stable version.
            - "sandbox": Use the version of claude code in the sandbox (raises `RuntimeError` if claude is not available in the sandbox)
            - "stable": Download and use the current stable version of claude code.
            - "latest": Download and use the very latest version of claude code.
            - "x.x.x": Download and use a specific version of claude code.
        user: User to execute claude code with.
        sandbox: Optional sandbox environment name.
    """

    async def execute(state: AgentState) -> AgentState:
        async with sandbox_agent_bridge(state) as bridge:
            # ensure claude is installed and get binary location
            claude_binary = await ensure_claude_code_installed(
                version, user, sandbox_env(sandbox)
            )

            # base options
            cmd = [
                claude_binary,
                "--print",  # run without interactions
                "--dangerously-skip-permissions",
                "--model",  # use current inspect model
                "inspect",
            ]

            # system message
            system_message = "\n\n".join(
                [m.text for m in state.messages if isinstance(m, ChatMessageSystem)]
            )
            if system_message:
                cmd.extend(["--append-system-prompt", system_message])

            # user prompt
            prompt = "\n\n".join(
                [m.text for m in state.messages if isinstance(m, ChatMessageUser)]
            )
            cmd.append(prompt)

            # execute the agent
            result = await sandbox_env(sandbox).exec(
                cmd=cmd,
                env={
                    "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                    "ANTHROPIC_API_KEY": "sk-ant-api03-DOq5tyLPrk9M4hPE",
                    "ANTHROPIC_SMALL_FAST_MODEL": "inspect",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "IS_SANDBOX": "1",
                },
                user=user,
            )

        if result.success:
            return bridge.state
        else:
            raise RuntimeError(
                f"Error executing claude code agent: {result.stdout}\n{result.stderr}"
            )

    return execute
