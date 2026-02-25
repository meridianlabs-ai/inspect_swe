import json
import shlex
from pathlib import Path
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
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.path import join_path
from inspect_swe._util.trace import trace

from .agentbinary import ensure_gemini_cli_setup


@agent
def gemini_cli(
    name: str = "Gemini CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
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
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP
        centaur: Run in 'centaur' mode, which makes Gemini CLI available to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models.
            When a model name in the mapping is referenced, the corresponding Model/string is used.
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
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve model
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "gemini_cli_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=model,
            model_aliases=model_aliases,
            filter=filter,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # install skills
            if resolved_skills is not None:
                GEMINI_SKILLS = ".gemini/skills"
                skills_dir = (
                    join_path(cwd, GEMINI_SKILLS) if cwd is not None else GEMINI_SKILLS
                )
                await install_skills(resolved_skills, sbox, user, skills_dir)

            # install node and gemini-cli in sandbox
            gemini_binary, node_binary = await ensure_gemini_cli_setup(
                sbox, version, user
            )

            # mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            # detect sandbox home directory
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"

            # write MCP server configs to settings.json in actual home
            # (not /tmp, so MCP servers can use npm cache from the real home)
            if all_mcp_servers:
                settings_json = resolve_mcp_servers(all_mcp_servers)
                gemini_settings_dir = f"{sandbox_home}/.gemini"
                await sbox.exec(["mkdir", "-p", gemini_settings_dir], user=user)
                await sbox.write_file(
                    f"{gemini_settings_dir}/settings.json", settings_json
                )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

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
                "--output-format",
                "text",  # Text output format
            ]

            # Add --yolo only for non-centaur mode (let user approve actions in centaur)
            if centaur is False:
                cmd.append("--yolo")

            # Configure MCP server names if provided
            # (all_mcp_servers defined earlier when writing settings.json)
            for server in all_mcp_servers:
                cmd.extend(["--allowed-mcp-server-names", server.name])

            # setup agent env (add node to PATH so the gemini shell script can find it)
            node_dir = str(Path(node_binary).parent)
            agent_env = {
                "GOOGLE_GEMINI_BASE_URL": f"http://localhost:{bridge.port}",
                "GEMINI_API_KEY": "api-key",
                "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
                "HOME": sandbox_home,  # Use detected sandbox home for config + npm cache
            } | (env or {})

            if centaur:
                await _run_gemini_cli_centaur(
                    options=centaur,
                    gemini_cmd=cmd,
                    agent_env=agent_env,
                    state=state,
                )
            else:
                # execute the agent (track debug output)
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    agent_cmd = cmd.copy()

                    # resume previous conversation
                    if has_assistant_response or attempt_count > 0:
                        agent_cmd.extend(["--resume", "latest"])

                    # add prompt as positional argument at the end
                    agent_cmd.append(agent_prompt)

                    # run agent
                    result = await sbox.exec_remote(
                        cmd=["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
                        + agent_cmd,
                        options=ExecRemoteAwaitableOptions(
                            cwd=cwd,
                            env=agent_env,
                            user=user,
                            concurrency=False,
                        ),
                        stream=False,
                    )

                    # track debug output
                    debug_output.append(result.stdout)
                    debug_output.append(result.stderr)

                    # raise for error
                    if not result.success:
                        cli_error_msg = _clean_gemini_error(
                            result.stdout, result.stderr
                        )
                        raise RuntimeError(
                            f"Error executing gemini cli agent {result.returncode}: {cli_error_msg}"
                        )

                    # exit if we are at max_attempts
                    attempt_count += 1
                    if attempt_count >= attempts.attempts:
                        break

                    # score and check for success
                    answer_scores = await score(bridge.state)
                    # break if we score 'correct'
                    if attempts.score_value(answer_scores[0].value) == 1.0:
                        break

                    # update prompt for retry
                    if callable(attempts.incorrect_message):
                        if not is_callable_coroutine(attempts.incorrect_message):
                            raise ValueError(
                                "The incorrect_message function must be async."
                            )
                        agent_prompt = await attempts.incorrect_message(
                            bridge.state, answer_scores
                        )
                    else:
                        agent_prompt = attempts.incorrect_message

                # trace debug output
                debug_output.insert(0, "Gemini CLI Debug Output:")
                trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def resolve_mcp_servers(mcp_servers: Sequence[MCPServerConfig]) -> str:
    """Build Gemini CLI settings.json content from MCP server configs."""
    mcp_servers_config: dict[str, Any] = {}
    for server in mcp_servers:
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        # For HTTP transport, Gemini CLI uses 'httpUrl' field
        if isinstance(server, MCPServerConfigHTTP) and "url" in config:
            config["httpUrl"] = config.pop("url")
        if "cwd" in config and not isinstance(config["cwd"], str):
            config["cwd"] = str(config["cwd"])
        mcp_servers_config[server.name] = config
    return json.dumps({"mcpServers": mcp_servers_config}, indent=2)


def _clean_gemini_error(stdout: str, stderr: str) -> str:
    """Clean up Gemini CLI error output by removing noise.

    The Gemini CLI output can include embedded <think> tags (reasoning content
    preserved by the bridge) that clutter error messages. This function strips
    them out to make errors readable.
    """
    combined = f"{stdout}\n{stderr}"

    cleaned_lines = [
        line for line in combined.split("\n") if not line.strip().startswith("<think")
    ]

    cleaned = "\n".join(cleaned_lines).strip()

    max_len = 2000
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "... (truncated)"

    return cleaned if cleaned else "Unknown error (no output)"


async def _run_gemini_cli_centaur(
    options: CentaurOptions,
    gemini_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    instructions = "Gemini CLI:\n\n - You may also use Gemini CLI via the 'gemini' command.\n - Use 'gemini --resume latest' if you need to resume a previous gemini session."

    # build .bashrc content - only export vars needed for the gemini alias,
    # not HOME which would break human_cli (PATH is needed for node)
    centaur_env = {k: v for k, v in agent_env.items() if k != "HOME"}
    agent_env_vars = [f'export {k}="{v}"' for k, v in centaur_env.items()]
    alias_cmd = shlex.join(gemini_cmd)
    alias_cmd = "alias gemini='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(agent_env_vars + ["", alias_cmd])

    # run the human cli
    await run_centaur(options, instructions, bashrc, state)
