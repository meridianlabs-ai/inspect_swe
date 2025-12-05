import os
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
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig
from inspect_ai.util import SandboxEnvironment, store
from inspect_ai.util import sandbox as sandbox_env

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import sandbox_exec
from inspect_swe._util.toml import to_toml
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from .agentbinary import codex_cli_binary_source

logger = getLogger(__file__)


@agent
def codex_cli(
    name: str = "Codex CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    model_config: str = "gpt-5.1",
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    disallowed_tools: list[Literal["web_search"]] | None = None,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    home_dir: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "latest"] | str = "auto",
    config_overrides: dict[str, str] | None = None,
) -> Agent:
    """Codex CLI.

    Agent that uses OpenAI [Codex CLI](https://github.com/openai/codex) running in a sandbox.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)
        system_prompt: Additional system prompt to append to default system prompt.
        model_config: Model configuration profile (e.g. used to determine the system prompt).
        mcp_servers: MCP servers to make available to the agent.
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
            Each BridgedToolsSpec creates an MCP server that makes the specified
            tools available to the agent running in the sandbox.
        disallowed_tools: Optionally disallow tools (currently only web_search).
        attempts: Configure agent to make multiple attempts.
        model: Model name to use (defaults to main model for task).
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        home_dir: Home directory to use for codex cli. If set, AGENTS.md and the MCP configuration will be written here.
        cwd: Working directory to run codex cli within.
        env: Environment variables to set for codex cli
        user: User to execute codex cli with.
        sandbox: Optional sandbox environment name.
        version: Version of codex cli to use. One of:
            - "auto": Use any available version of codex cli in the sandbox, otherwise download the latest version.
            - "sandbox": Use the version of codex cli in the sandbox (raises `RuntimeError` if codex is not available in the sandbox)
            - "latest": Download and use the very latest version of codex cli.
            - "x.x.x": Download and use a specific version of codex cli.
        config_overrides: Additional Codex CLI configuration overrides.
            Each key-value pair is passed as `-c key=value` to the CLI.
    """
    # resolve model
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # ensure disallowed_tools list
    disallowed_tools = disallowed_tools or []

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "codex_cli_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=model,
            filter=filter,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
        ) as bridge:
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

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # determine CODEX_HOME (default to whatever sandbox working dir is)
            if home_dir is None:
                working_dir = await sandbox_exec(sbox, "pwd", user=user, cwd=cwd)
                codex_home = _join_path(working_dir, ".codex")
            else:
                # Resolve ~ and $VARS inside the sandbox
                codex_home = await sandbox_exec(
                    sbox, f'eval echo "{home_dir}"', user=user, cwd=cwd
                )
            await sandbox_exec(sbox, cmd=f"mkdir -p {codex_home}", user=user)

            # location for agents_md
            def codex_agents_md() -> str:
                AGENTS_MD = "AGENTS.md"
                if home_dir is not None:
                    return _join_path(codex_home, AGENTS_MD)
                elif cwd is not None:
                    return _join_path(cwd, AGENTS_MD)
                else:
                    return AGENTS_MD

            # location for config_toml (either codex_home or cwd/.codex )
            async def codex_config_toml() -> str:
                CONFIG_TOML = "config.toml"
                if home_dir is not None:
                    return _join_path(codex_home, CONFIG_TOML)
                else:
                    dir = ".codex" if cwd is None else _join_path(cwd, ".codex")
                    await sandbox_exec(sbox, cmd=f"mkdir -p {dir}", user=user)
                    return _join_path(dir, CONFIG_TOML)

            # write system messages to AGENTS.md
            if system_messages:
                await sbox.write_file(codex_agents_md(), "\n\n".join(system_messages))

            prompt, has_assistant_response = build_user_prompt(state.messages)

            # build agent cmd
            cmd = [
                codex_binary,
                "exec",
                # real model is passed to the bridge above, this just affects defaults e.g. system prompt
                "--model",
                model_config,
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--color",
                "never",
            ]

            # include web search if appropriate
            if "web_search" not in disallowed_tools:
                cmd.extend(["-c", "tools.web_search=true"])

            # apply config overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    cmd.extend(["-c", f"{key}={value}"])

            # register mcp servers (combine static configs with bridged tools)
            all_mcp_servers = list(mcp_servers or []) + bridge.mcp_server_configs
            if all_mcp_servers:
                mcp_config: dict[str, Any] = {}
                for mcp_server in all_mcp_servers:
                    mcp_config[f"mcp_servers.{mcp_server.name}"] = (
                        mcp_server.model_dump(
                            exclude={"name", "tools"}, exclude_none=True
                        )
                    )
                await sbox.write_file(await codex_config_toml(), to_toml(mcp_config))

            # execute the agent (track debug output)
            debug_output: list[str] = []
            agent_prompt = prompt
            attempt_count = 0
            while True:
                # append prompt
                agent_cmd = cmd.copy()
                agent_cmd.append(agent_prompt)

                # resume previous conversation
                if has_assistant_response or attempt_count > 0:
                    agent_cmd.extend(["resume", "--last"])

                # run agent
                result = await sbox.exec(
                    cmd=["bash", "-c", 'exec 0<&- "$@"', "bash"] + agent_cmd,
                    cwd=cwd,
                    env={
                        "CODEX_HOME": codex_home,
                        "OPENAI_BASE_URL": f"http://localhost:{bridge.port}/v1",
                        "RUST_LOG": "warning",
                    }
                    | (env or {}),
                    user=user,
                )

                # record output for debug
                debug_output.append(result.stdout)
                debug_output.append(result.stderr)

                # raise for error
                if not result.success:
                    raise RuntimeError(
                        f"Error executing codex cli agent: {result.stdout}\n{result.stderr}"
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
            debug_output.insert(0, "Codex CLI Debug Output:")
            trace("\n".join(debug_output))

        # return success
        return bridge.state

    return agent_with(execute, name=name, description=description)


def _join_path(base: str, path: str) -> str:
    return os.path.join(base, path).replace("\\", "/")


async def _last_rollout(
    sandbox: SandboxEnvironment, codex_home: str, user: str | None
) -> str | None:
    try:
        rollout = await sandbox_exec(
            sandbox,
            f"find '{codex_home}/sessions' -type f -name 'rollout-*.jsonl' -exec ls -t -- {{}} + | head -n 1",
            user=user,
        )
        return rollout
    except RuntimeError as ex:
        logger.warning(f"Error attempting to read rollout file: {ex}")
        return None
