import os
from textwrap import dedent
from typing import Any, Literal, Sequence

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
from inspect_swe._util.sandbox import sandbox_exec
from inspect_swe._util.toml import to_toml
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from .agentbinary import codex_cli_binary_source

# TODO: attempts

# TODO: docs


@agent
def codex_cli(
    name: str = "Codex CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    disallowed_tools: list[Literal["web_search"]] | None = None,
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
        disallowed_tools: Optionally disallow tools (currently only web_search).
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

    # ensure disallowed_tools list
    disallowed_tools = disallowed_tools or []

    async def execute(state: AgentState) -> AgentState:
        async with sandbox_agent_bridge(state, model=model) as bridge:
            # ensure codex is installed and get binary location
            codex_binary = await ensure_agent_binary_installed(
                codex_cli_binary_source(), version, user, sandbox_env(sandbox)
            )

            # helper to create codex cwd relative paths
            def codex_path(file: str) -> str:
                return (
                    file if cwd is None else os.path.join(cwd, file).replace("\\", "/")
                )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # determine CODEX_HOME (we want this to be whatever sandbox working dir is)
            working_dir = (await sandbox_exec(sbox, "pwd", user=user, cwd=cwd)).strip()
            codex_home = f"{working_dir}/.codex"

            # write system messages to AGENTS.md
            if system_messages:
                await sbox.write_file(
                    codex_path("AGENTS.md"), "\n\n".join(system_messages)
                )

            # built full promot
            prompt = "\n\n".join(
                [
                    message.text
                    for message in state.messages
                    if isinstance(message, ChatMessageUser)
                ]
            )

            # build agent cmd
            agent_cmd = [
                codex_binary,
                "exec",
                "--model",
                "gpt-5",  # real model is passed to the bridge above
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--color",
                "never",
            ]

            # include the plan and apply patch tools.
            # NOTE: update_plan not currently working in 'exec' mode:
            # https://github.com/openai/codex/issues/1952
            agent_cmd.extend(["-c", "include_plan_tool=true"])
            agent_cmd.extend(["-c", "include_apply_patch_tool=true"])

            # include web search if appropriate
            if "web_search" not in disallowed_tools:
                agent_cmd.extend(["-c", "tools.web_search=true"])

            # append the prompt
            agent_cmd.append(prompt)

            # register mcp servers
            if mcp_servers:
                mcp_config: dict[str, Any] = {}
                for mcp_server in mcp_servers or []:
                    mcp_config[f"mcp_servers.{mcp_server.name}"] = (
                        mcp_server.model_dump(
                            exclude={"name", "tools"}, exclude_none=True
                        )
                    )
                await sandbox_exec(
                    sbox, cmd=f"mkdir -p {codex_path('.codex')}", user=user
                )
                await sbox.write_file(
                    codex_path(".codex/config.toml"), to_toml(mcp_config)
                )

            # capture stdout and stderr
            debug_output: list[str] = []

            # execute the agent
            result = await sbox.exec(
                cmd=["bash", "-c", 'exec "$@"', "bash"] + agent_cmd,
                cwd=cwd,
                env={
                    "CODEX_HOME": codex_home,
                    "OPENAI_BASE_URL": f"http://localhost:{bridge.port}/v1",
                    "RUST_LOG": "debug",
                }
                | (env or {}),
            )

            # record output for debug
            debug_output.append(result.stdout)
            debug_output.append(result.stderr)

            # trace debug info
            debug_output.insert(0, "Codex CLI Debug Output:")
            trace("\n".join(debug_output))

            # raise for error
            if not result.success:
                raise RuntimeError(
                    f"Error executing claude code agent: {result.stdout}\n{result.stderr}"
                )

            # return success
            return bridge.state

    return agent_with(execute, name=name, description=description)
