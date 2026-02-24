import json
import shlex
import uuid
from collections.abc import AsyncIterator
from contextlib import nullcontext
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
from inspect_ai.event import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageSystem, GenerateFilter
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.util import (
    ExecCompleted,
    ExecStderr,
    ExecStdout,
    store,
)
from inspect_ai.util import (
    sandbox as sandbox_env,
)
from inspect_ai.util._sandbox import (
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
)
from pydantic_core import to_json

from inspect_swe._claude_code._events import claude_code_events
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.events import capture_model_events
from inspect_swe._util.path import join_path

from .._util._async import is_callable_coroutine
from .._util.agentbinary import ensure_agent_binary_installed
from .._util.messages import build_user_prompt
from .._util.model import inspect_model
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
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    disallowed_tools: list[str] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
    subagent_model: str | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    debug: bool | None = None,
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
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent.
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
            Each BridgedToolsSpec creates an MCP server that makes the specified
            tools available to the agent running in the sandbox.
        disallowed_tools: List of tool names to disallow entirely.
        centaur: Run in 'centaur' mode, which makes Claude Code available to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts. When this is specified, the task will be scored when the agent stops calling tools. If the scoring is successful, execution will stop. Otherwise, the agent will be prompted to pick up where it left off for another attempt.
        model: Model name to use for Opus and Sonnet calls (defaults to main model for task).
        opus_model: The model to use for `opus`, or for `opusplan` when Plan Mode is active. Defaults to `model`.
        sonnet_model: The model to use for `sonnet`, or for `opusplan` when Plan Mode is not active. Defaults to `model`.
        haiku_model: The model to use for haiku, or [background functionality](https://code.claude.com/docs/en/costs#background-token-usage). Defaults to `model`.
        subagent_model: The model to use for [subagents](https://code.claude.com/docs/en/sub-agents). Defaults to `model`.
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run claude code within.
        env: Environment variables to set for claude code.
        user: User to execute claude code with.
        debug: Add `--debug` cli flag. Verbose logging is always enabled.
        sandbox: Optional sandbox environment name.
        version: Version of claude code to use. One of:
            - "auto": Use any available version of claude code in the sandbox, otherwise download the current stable version.
            - "sandbox": Use the version of claude code in the sandbox (raises `RuntimeError` if claude is not available in the sandbox)
            - "stable": Download and use the current stable version of claude code.
            - "latest": Download and use the very latest version of claude code.
            - "x.x.x": Download and use a specific version of claude code.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve models
    model = f"inspect/{model}" if model is not None else "inspect"
    opus_model = inspect_model(opus_model)
    sonnet_model = inspect_model(sonnet_model)
    haiku_model = inspect_model(haiku_model)
    subagent_model = inspect_model(subagent_model)

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "claude_code_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        bridge_event_ctx = (
            nullcontext(lambda _: None) if centaur else capture_model_events()
        )
        async with (
            sandbox_agent_bridge(
                state,
                model=model,
                filter=filter,
                retry_refusals=retry_refusals,
                port=port,
                bridged_tools=bridged_tools,
            ) as bridge,
            bridge_event_ctx as apply_bridge_event,
        ):
            # ensure claude is installed and get binary location
            claude_binary = await ensure_agent_binary_installed(
                claude_code_binary_source(), version, user, sandbox_env(sandbox)
            )

            # allocate session_id
            session_id = str(uuid.uuid4())

            # base options
            cmd = [
                "--dangerously-skip-permissions",
                "--model",
                model,
            ]

            # add interactive options if not running as centaur
            if centaur is False:
                cmd.extend(["--print", "--output-format", "stream-json", "--verbose"])
                if debug:
                    cmd.append("--debug")

            # system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)
            if system_messages:
                cmd.extend(["--append-system-prompt", "\n\n".join(system_messages)])

            # mcp servers (combine static configs with bridged tools)
            cmd_allowed_tools: list[str] = []
            all_mcp_servers = list(mcp_servers or []) + bridge.mcp_server_configs
            if all_mcp_servers:
                mcp_server_args, mcp_allowed_tools = resolve_mcp_servers(
                    all_mcp_servers
                )
                cmd.extend(mcp_server_args)
                cmd_allowed_tools.extend(mcp_allowed_tools)

            # add allowed and disallowed tools
            if len(cmd_allowed_tools) > 0:
                cmd.append("--allowed-tools")
                cmd.append(",".join(cmd_allowed_tools))
            if disallowed_tools is not None and len(disallowed_tools) > 0:
                cmd.append("--disallowed-tools")
                cmd.append(",".join(disallowed_tools))

            prompt, has_assistant_response = build_user_prompt(state.messages)

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # install skills
            if resolved_skills is not None:
                CLAUDE_SKILLS = ".claude/skills"
                skills_dir = (
                    join_path(cwd, CLAUDE_SKILLS) if cwd is not None else CLAUDE_SKILLS
                )
                await install_skills(resolved_skills, sbox, user, skills_dir)

            # define agent env
            agent_env = {
                "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                "ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-DOq5tyLPrk9M4hPE",
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": opus_model or model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": sonnet_model or model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": haiku_model or model,
                "CLAUDE_CODE_SUBAGENT_MODEL": subagent_model or model,
                "ANTHROPIC_SMALL_FAST_MODEL": haiku_model or model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                "IS_SANDBOX": "1",
            } | (env or {})

            # Claude Code 2.1.37 reports "has Authorization header: false"
            # despite ANTHROPIC_AUTH_TOKEN being set in the environment,
            # then enters an OAuth flow that silently fails (rc=0, no
            # output).  Providing an apiKeyHelper in settings.json
            # supplies a key through a path that does work.
            api_key = agent_env.get("ANTHROPIC_AUTH_TOKEN", "dummy-key-for-bridge")
            await _seed_claude_config(sbox, api_key, user, cwd)

            # centaur mode uses human_cli with custom instructions and bash rc
            if centaur:
                await run_claude_code_centaur(
                    options=centaur,
                    claude_cmd=[claude_binary] + cmd,
                    agent_env=agent_env,
                    state=state,
                )
            else:
                # execute the agent (track debug output)
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0
                timeout_count = 0
                while True:
                    # resume previous conversation
                    if has_assistant_response or attempt_count > 0 or timeout_count > 0:
                        agent_cmd = (
                            [claude_binary, "--continue"] + cmd + ["--", agent_prompt]
                        )
                    else:
                        agent_cmd = (
                            [claude_binary, "--session-id", session_id]
                            + cmd
                            + ["--", agent_prompt]
                        )

                    # run agent
                    proc = await sbox.exec_remote(
                        cmd=["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
                        + agent_cmd,
                        options=ExecRemoteStreamingOptions(
                            cwd=cwd,
                            env=agent_env,
                            user=user,
                            concurrency=False,
                        ),
                    )
                    try:
                        async for event in claude_code_events(
                            _jsonl_stream(proc, debug_output)
                        ):
                            if isinstance(event, ModelEvent):
                                apply_bridge_event(event)
                            else:
                                transcript()._event(event)
                    finally:
                        await proc.kill()

                    # reset timeout counter
                    timeout_count = 0

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


async def _seed_claude_config(
    sbox: Any,
    api_key: str,
    user: str | None,
    cwd: str | None,
) -> None:
    """Write ~/.claude/settings.json with an apiKeyHelper.

    Claude Code 2.1.37 does not use ANTHROPIC_AUTH_TOKEN from the
    environment for API requests.  Providing an apiKeyHelper in
    settings.json supplies the key through a path it does use.
    """
    await sbox.exec(
        cmd=[
            "bash",
            "-c",
            'mkdir -p "$HOME/.claude"'
            " && echo '"
            '{"apiKeyHelper": "echo ' + api_key + '"}'
            '\' > "$HOME/.claude/settings.json"',
        ],
        user=user,
        cwd=cwd,
    )


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




async def run_claude_code_centaur(
    options: CentaurOptions,
    claude_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    instructions = "Claude Code:\n\n - You may also use Claude Code via the 'claude' command.\n - Use 'claude --resume' if you need to resume a previous claude session."

    # build .bashrc content
    agent_env_vars = [f'export {k}="{v}"' for k, v in agent_env.items()]
    claude_config = """echo '{"hasCompletedOnboarding":true,"bypassPermissionsModeAccepted":true}' > "$HOME"/.claude.json"""
    path_config = [
        'mkdir -p "$HOME/.local/bin"',
        'export PATH="$HOME/.local/bin:$PATH"',
        f'ln -sf {claude_cmd[0]} "$HOME/.local/bin/claude"',
    ]
    alias_cmd = shlex.join(claude_cmd)
    alias_cmd = "alias claude='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(
        agent_env_vars + path_config + ["", claude_config, "", alias_cmd]
    )

    # run the human cli
    await run_centaur(options, instructions, bashrc, state)


async def _jsonl_stream(
    proc: ExecRemoteProcess,
    debug_output: list[str],
) -> AsyncIterator[dict[str, Any]]:
    """Line-buffer stdout chunks from exec_remote, yield parsed JSONL dicts."""
    line_buffer = ""
    exit_code = 0
    async for event in proc:
        if isinstance(event, ExecStdout):
            line_buffer += event.data
            while "\n" in line_buffer:
                line, line_buffer = line_buffer.split("\n", 1)
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        debug_output.append(f"JSONL parse error: {line}")
        elif isinstance(event, ExecStderr):
            debug_output.append(event.data)
        elif isinstance(event, ExecCompleted):
            exit_code = event.exit_code
    # Handle trailing partial line
    if line_buffer.strip():
        try:
            yield json.loads(line_buffer.strip())
        except json.JSONDecodeError:
            debug_output.append(f"JSONL parse error (trailing): {line_buffer}")
    if exit_code != 0:
        tail = debug_output[-100:] if len(debug_output) > 100 else debug_output
        raise RuntimeError(
            f"Error executing claude code agent {exit_code}: {' '.join(tail)}"
        )
