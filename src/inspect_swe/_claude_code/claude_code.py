import uuid
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
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateFilter
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig
from inspect_ai.util import sandbox as sandbox_env
from pydantic_core import to_json

from .._util._async import is_callable_coroutine
from .._util.agentbinary import ensure_agent_binary_installed
from .._util.trace import trace
from .agentbinary import claude_code_binary_source


@agent
def claude_code(
    name: str = "Claude Code",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    disallowed_tools: list[str] | None = None,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    small_model: str | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
) -> Agent:
    """Claude Code agent.

    Agent that uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) running in a sandbox.

    The agent can either use a version of Claude Code installed in the sandbox, or can download a version and install it in the sandbox (see docs on `version` option below for details).

    Use `disallowed_tools` to control access to tools. See [Tools available to Claude](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for the list of built-in tools which can be disallowed.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)
        system_prompt: Additional system prompt to append to default system prompt.
        mcp_servers: MCP servers to make available to the agent.
        disallowed_tools: List of tool names to disallow entirely.
        attempts: Configure agent to make multiple attempts.
        model: Model name to use for Opus and Sonnet calls (defaults to main model for task).
        small_model: Model to use for Haiku calls (defaults to main model for task).
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run claude code within.
        env: Environment variables to set for claude code.
        user: User to execute claude code with.
        sandbox: Optional sandbox environment name.
        version: Version of claude code to use. One of:
            - "auto": Use any available version of claude code in the sandbox, otherwise download the current stable version.
            - "sandbox": Use the version of claude code in the sandbox (raises `RuntimeError` if claude is not available in the sandbox)
            - "stable": Download and use the current stable version of claude code.
            - "latest": Download and use the very latest version of claude code.
            - "x.x.x": Download and use a specific version of claude code.
    """
    # resolve models
    model = f"inspect/{model}" if model is not None else "inspect"
    small_model = f"inspect/{small_model}" if small_model is not None else "inspect"

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        async with sandbox_agent_bridge(
            state, filter=filter, retry_refusals=retry_refusals
        ) as bridge:
            # ensure claude is installed and get binary location
            claude_binary = await ensure_agent_binary_installed(
                claude_code_binary_source(), version, user, sandbox_env(sandbox)
            )

            # allocate session_id
            session_id = str(uuid.uuid4())

            # base options
            cmd = [
                "--print",  # run without interactions
                "--dangerously-skip-permissions",
                "--debug",
                "--verbose",
                "--model",
                model,
            ]

            # system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)
            if system_messages:
                cmd.extend(["--append-system-prompt", "\n\n".join(system_messages)])

            # mcp servers
            cmd_allowed_tools: list[str] = []
            if mcp_servers:
                mcp_server_args, mcp_allowed_tools = resolve_mcp_servers(mcp_servers)
                cmd.extend(mcp_server_args)
                cmd_allowed_tools.extend(mcp_allowed_tools)

            # add allowed and disallowed tools
            if len(cmd_allowed_tools) > 0:
                cmd.append("--allowed-tools")
                cmd.append(",".join(cmd_allowed_tools))
            if disallowed_tools is not None and len(disallowed_tools) > 0:
                cmd.append("--disallowed-tools")
                cmd.append(",".join(disallowed_tools))

            # user prompt
            prompt = "\n\n".join(
                [m.text for m in state.messages if isinstance(m, ChatMessageUser)]
            )

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # execute the agent (track debug output)
            debug_output: list[str] = []
            agent_prompt = prompt
            attempt_count = 0
            while True:
                # either starting a new session or resuming one
                id_param = "--session-id" if attempt_count == 0 else "--resume"
                agent_cmd = (
                    [claude_binary, id_param, session_id] + cmd + ["--", agent_prompt]
                )

                # run agent
                result = await sbox.exec(
                    cmd=["bash", "-c", 'exec 0<&- "$@"', "bash"] + agent_cmd,
                    cwd=cwd,
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
                    | (env or {}),
                    user=user,
                    concurrency=False,
                )

                # track debug output
                debug_output.append(result.stdout)
                debug_output.append(result.stderr)

                # raise for error
                if not result.success:
                    raise RuntimeError(
                        f"Error executing claude code agent: {result.stdout}\n{result.stderr}"
                    )

                # exit if we are at max_attempts
                attempt_count += 1
                if attempt_count >= attempts.attempts:
                    break

                # score this attempt
                answer_scores = await score(state)

                # break if we score 'correct'
                if attempts.score_value(answer_scores[0].value) == 1.0:
                    break

                # otherwise update prompt with incorrect message and continue
                else:
                    if callable(attempts.incorrect_message):
                        if not is_callable_coroutine(attempts.incorrect_message):
                            raise ValueError(
                                "The incorrect_message function must be async."
                            )
                        agent_prompt = await attempts.incorrect_message(
                            state, answer_scores
                        )
                    else:
                        agent_prompt = attempts.incorrect_message

            # trace debug info
            debug_output.insert(0, "Claude Code Debug Output:")
            trace("\n".join(debug_output))

        return bridge.state

    # return agent with specified name and descritpion
    return agent_with(execute, name=name, description=description)


def resolve_mcp_servers(
    mcp_servers: Sequence[MCPServerConfig],
) -> tuple[list[str], list[str]]:
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
    mcp_config_cmds: list[str] = []
    if len(mcp_servers_json) > 0:
        mcp_config_cmds.append("--mcp-config")
        mcp_config_cmds.append(
            to_json({"mcpServers": mcp_servers_json}, exclude_none=True).decode()
        )

    return mcp_config_cmds, allowed_tools
