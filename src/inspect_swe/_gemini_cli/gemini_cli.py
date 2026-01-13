"""Gemini CLI agent for Inspect.

This agent uses Google's Gemini CLI running in a sandbox with
model requests bridged to Inspect's model infrastructure.
"""

from logging import getLogger
from textwrap import dedent
from typing import Literal, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig
from inspect_ai.util import store
from inspect_ai.util import sandbox as sandbox_env

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from .agentbinary import gemini_cli_binary_source

logger = getLogger(__file__)


@agent
def gemini_cli(
    name: str = "Gemini CLI",
    description: str = dedent("""
       AI agent powered by Google Gemini for software engineering tasks.
       Capable of writing, testing, debugging, and iterating on code.
    """),
    system_prompt: str | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
) -> Agent:
    """Gemini CLI agent.

    Agent that uses Google [Gemini CLI](https://github.com/google-gemini/gemini-cli)
    running in a sandbox with Inspect model bridging.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description
        system_prompt: Additional system prompt to append
        mcp_servers: MCP servers to make available to the agent
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP
        attempts: Configure agent to make multiple attempts
        model: Model name to use (defaults to main model for task)
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run gemini cli within
        env: Environment variables to set for gemini cli
        user: User to execute gemini cli with
        sandbox: Optional sandbox environment name
        version: Version of gemini cli to use. One of:
            - "auto": Use any available version in sandbox, otherwise download latest
            - "sandbox": Use sandbox version (raises RuntimeError if not available)
            - "stable"/"latest": Download and use the latest version
            - "x.x.x": Download and use a specific version
    """
    # resolve model - use "inspect/" prefix pattern for bridge
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution)
        MODEL_PORT = "gemini_cli_model_port"
        port = store().get(MODEL_PORT, 13131) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=model,
            filter=filter,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
        ) as bridge:
            # ensure gemini-cli is installed
            gemini_binary = await ensure_agent_binary_installed(
                gemini_cli_binary_source(), version, user, sandbox_env(sandbox)
            )

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # build system prompt from messages
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            # build user prompt
            prompt, has_assistant_response = build_user_prompt(state.messages)

            # Prepend system prompt to user prompt if provided
            # (gemini-cli doesn't have a separate --system-prompt flag)
            if system_messages:
                combined_system = "\n\n".join(system_messages)
                prompt = f"{combined_system}\n\n{prompt}"

            # build base command
            # NOTE: gemini-cli is a JavaScript file, run with node
            # Use --max-old-space-size to increase heap size for large prompts
            cmd = [
                "node",
                "--max-old-space-size=4096",  # 4GB heap size
                gemini_binary,
                "--yolo",  # Auto-approve all actions (YOLO mode)
                "--output-format", "text",  # Text output format
            ]

            # Configure MCP server names if provided
            if mcp_servers or bridge.mcp_server_configs:
                all_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)
                server_names = [server.name for server in all_servers]
                for name in server_names:
                    cmd.extend(["--allowed-mcp-server-names", name])

            # build environment variables
            agent_env = {
                "GOOGLE_GEMINI_BASE_URL": f"http://localhost:{bridge.port}",
                "GEMINI_API_KEY": "sk-inspect-bridge",  # Dummy key - proxy handles auth
            }
            if env:
                agent_env.update(env)

            # execute agent with retry loop
            debug_output: list[str] = []
            agent_prompt = prompt
            attempt_count = 0

            while True:
                agent_cmd = cmd.copy()

                # resume previous conversation if continuing
                if has_assistant_response or attempt_count > 0:
                    agent_cmd.extend(["--resume", "latest"])

                # add prompt as positional argument at the end
                agent_cmd.append(agent_prompt)

                # run agent - close stdin to prevent interactive mode
                result = await sbox.exec(
                    cmd=["bash", "-c", 'exec 0<&- "$@"', "bash"] + agent_cmd,
                    cwd=cwd,
                    env=agent_env,
                    user=user,
                    concurrency=False,
                )

                debug_output.append(result.stdout)
                debug_output.append(result.stderr)

                if not result.success:
                    raise RuntimeError(
                        f"Error executing gemini cli: {result.stdout}\n{result.stderr}"
                    )

                attempt_count += 1
                if attempt_count >= attempts.attempts:
                    break

                # score and check for success
                answer_scores = await score(bridge.state)
                if attempts.score_value(answer_scores[0].value) == 1.0:
                    break

                # update prompt for retry
                agent_prompt = await _get_incorrect_message(
                    attempts, bridge.state, answer_scores
                )

            # trace debug output
            debug_output.insert(0, "Gemini CLI Debug Output:")
            trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def _build_mcp_config(mcp_servers: Sequence[MCPServerConfig]) -> str:
    """Build MCP configuration JSON for gemini-cli."""
    import json

    config: dict[str, dict[str, dict]] = {"mcpServers": {}}
    for server in mcp_servers:
        server_config = server.model_dump(exclude={"name", "tools"}, exclude_none=True)
        config["mcpServers"][server.name] = server_config
    return json.dumps(config)


async def _get_incorrect_message(
    attempts: AgentAttempts, state: AgentState, answer_scores: list
) -> str:
    """Get the incorrect message for retry attempts."""
    if callable(attempts.incorrect_message):
        if not is_callable_coroutine(attempts.incorrect_message):
            raise ValueError("The incorrect_message function must be async.")
        return await attempts.incorrect_message(state, answer_scores)
    return attempts.incorrect_message
