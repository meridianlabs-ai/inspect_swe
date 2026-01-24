import json
from logging import getLogger
from textwrap import dedent
from typing import Any, Literal, Sequence

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
from inspect_ai.scorer import Score, score
from inspect_ai.tool import MCPServerConfig
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import detect_sandbox_platform
from inspect_swe._util.trace import trace

from .agentbinary import (
    ensure_gemini_cli_installed,
    ensure_node_and_npm_available,
    resolve_gemini_version,
)

logger = getLogger(__name__)

# Gemini CLI home directory
GEMINI_HOME = "/tmp"
GEMINI_SETTINGS_PATH = f"{GEMINI_HOME}/.gemini/settings.json"


def _build_mcp_server_config(server: MCPServerConfig) -> dict[str, Any]:
    """Build Gemini CLI settings.json MCP server config from MCPServerConfig.

    Gemini CLI (v0.24+) supports transport configuration via:
    - url + type: "http" → StreamableHTTPClientTransport (preferred)
    - url + type: "sse" → SSEClientTransport
    - httpUrl → StreamableHTTPClientTransport (deprecated but supported)
    - command → StdioClientTransport (for stdio servers)
    """
    config: dict[str, Any] = {}

    # Handle HTTP-based MCP servers (including bridged tools)
    if isinstance(server, MCPServerConfigHTTP):
        # Use url + type: "http" for streamable HTTP transport
        # This is the preferred format in newer Gemini CLI versions (v0.24+)
        config["url"] = server.url
        config["type"] = "http"

        if server.headers:
            config["headers"] = server.headers
    else:
        # For stdio servers, pass command and args
        config["command"] = server.command  # type: ignore[attr-defined]
        if hasattr(server, "args") and server.args:  # type: ignore[attr-defined]
            config["args"] = server.args  # type: ignore[attr-defined]
        if hasattr(server, "env") and server.env:  # type: ignore[attr-defined]
            config["env"] = server.env  # type: ignore[attr-defined]

    return config


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
    gemini_model: str = "gemini-2.5-pro",
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
        model: Model name to use for inspect bridge (defaults to main model for task)
        gemini_model: Gemini model name to pass to CLI. This bypasses the auto-router.
            Use "gemini-2.5-pro" (default) or "gemini-2.5-flash". The actual model
            calls still go through the inspect bridge, but this disables the router.
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
            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # detect platform for node download if needed
            platform = await detect_sandbox_platform(sbox)

            # ensure node and npm are available (downloads full distribution if needed)
            node_binary, _ = await ensure_node_and_npm_available(sbox, platform, user)

            # resolve gemini version
            gemini_version = await resolve_gemini_version(version)

            # ensure gemini-cli is installed via npm bundle
            # This gives us the full package including policy files for YOLO mode
            gemini_binary = await ensure_gemini_cli_installed(
                sbox, node_binary, gemini_version, platform, user
            )

            # Write MCP server configs to settings.json
            # Gemini CLI discovers MCP servers from settings.json, not CLI args
            all_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            if all_servers:
                mcp_servers_config = {
                    server.name: _build_mcp_server_config(server)
                    for server in all_servers
                }
                settings = {"mcpServers": mcp_servers_config}
                settings_json = json.dumps(settings, indent=2)

                # Create .gemini directory and write settings.json
                await sbox.exec(["mkdir", "-p", f"{GEMINI_HOME}/.gemini"], user=user)
                await sbox.write_file(GEMINI_SETTINGS_PATH, settings_json)

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
            # The gemini binary from npm install is a shell script that invokes node
            cmd = [
                gemini_binary,
                "--model",
                gemini_model,  # Specify model to bypass auto-router
                "--yolo",  # Auto-approve all actions (YOLO mode)
                "--output-format",
                "text",  # Text output format
            ]

            # Configure MCP server names if provided
            # (all_servers defined earlier when writing settings.json)
            for server in all_servers:
                cmd.extend(["--allowed-mcp-server-names", server.name])

            # build environment variables
            # Add node to PATH so the gemini shell script can find it
            node_dir = "/".join(node_binary.split("/")[:-1])  # Get bin directory
            agent_env = {
                "GOOGLE_GEMINI_BASE_URL": f"http://localhost:{bridge.port}",
                "GEMINI_API_KEY": "sk-inspect-bridge",  # Dummy key - proxy handles auth
                "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",  # Gemini CLI needs a home directory
            }
            if env:
                agent_env.update(env)

            # execute agent with retry loop
            debug_output: list[str] = []
            agent_prompt = prompt
            attempt_count = 0
            cli_error_msg: str | None = None

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
                    cli_error_msg = _clean_gemini_error(result.stdout, result.stderr)
                    raise RuntimeError(
                        f"Error executing gemini cli agent: {cli_error_msg}"
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

            # Debug: log bridge state before returning
            last_msg = bridge.state.messages[-1] if bridge.state.messages else None
            has_pending = (
                last_msg
                and last_msg.role == "assistant"
                and getattr(last_msg, "tool_calls", None)
            )
            logger.info(
                f"Gemini CLI completed: attempt_count={attempt_count}, "
                f"result.success={result.success}, "
                f"num_messages={len(bridge.state.messages)}, "
                f"has_pending_tool_calls={has_pending}"
            )
            if has_pending:
                logger.warning(
                    f"Gemini CLI exited with pending tool calls: "
                    f"{[tc.function for tc in last_msg.tool_calls]}"
                )

        return bridge.state

    return agent_with(execute, name=name, description=description)


async def _get_incorrect_message(
    attempts: AgentAttempts, state: AgentState, answer_scores: list[Score]
) -> str:
    if callable(attempts.incorrect_message):
        if not is_callable_coroutine(attempts.incorrect_message):
            raise ValueError("The incorrect_message function must be async.")
        return await attempts.incorrect_message(state, answer_scores)
    return attempts.incorrect_message


def _clean_gemini_error(stdout: str, stderr: str) -> str:
    """Clean up Gemini CLI error output by removing noise.

    The Gemini CLI can output __THOUGHT_SIG__ tokens (internal reasoning signatures)
    that clutter error messages. This function strips them out to make errors readable.
    """
    combined = f"{stdout}\n{stderr}"

    # Remove __THOUGHT_SIG__ lines (they're base64-encoded internal data)
    cleaned_lines = []
    for line in combined.split("\n"):
        if not line.startswith("__THOUGHT_SIG__:"):
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    # Truncate if still too long (some errors can be very verbose)
    max_len = 2000
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "... (truncated)"

    return cleaned if cleaned else "Unknown error (no output)"
