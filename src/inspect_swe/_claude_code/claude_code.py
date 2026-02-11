import json
import shlex
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Awaitable, Callable, Literal, Sequence

import anyio
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
from inspect_ai.tool import MCPServerConfig, Skill, Tool, install_skills, read_skills, tool
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from pydantic_core import to_json

from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.path import join_path

from .._util._async import is_callable_coroutine
from .._util.agentbinary import ensure_agent_binary_installed
from .._util.messages import build_user_prompt
from .._util.trace import trace
from .agentbinary import claude_code_binary_source

# Name of the internal MCP server used for hook dispatch.
_HOOKS_SERVER_NAME = "_hooks"


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
    retry_timeouts: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    hooks: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] | None = None,
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
        attempts: Configure agent to make multiple attempts.
        model: Model name to use for Opus and Sonnet calls (defaults to main model for task).
        opus_model: The model to use for `opus`, or for `opusplan` when Plan Mode is active. Defaults to `model`.
        sonnet_model: The model to use for `sonnet`, or for `opusplan` when Plan Mode is not active. Defaults to `model`.
        haiku_model: The model to use for haiku, or [background functionality](https://code.claude.com/docs/en/costs#background-token-usage). Defaults to `model`.
        subagent_model: The model to use for [subagents](https://code.claude.com/docs/en/sub-agents). Defaults to `model`.
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        retry_timeouts: Should timeouts be retried? (pass number of times to retry)
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
        hooks: Dict mapping Claude Code hook event names (e.g. "Stop",
            "PreToolUse") to async callables. Each callable receives the
            hook input dict and returns a decision dict.
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

    # build effective bridged_tools and disallowed_tools lists
    effective_bridged_tools = list(bridged_tools or [])
    effective_disallowed_tools = list(disallowed_tools or [])

    if hooks is not None:
        # Register the hook dispatch tool with the bridge so hook scripts
        # can call it via the sandbox service.  The hooks server is
        # excluded from --mcp-config below so Claude Code never sees it.
        effective_bridged_tools.append(
            BridgedToolsSpec(
                name=_HOOKS_SERVER_NAME,
                tools=[_hook_dispatch(hooks)],
            )
        )

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "claude_code_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=model,
            filter=filter,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=effective_bridged_tools or None,
        ) as bridge:
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
                cmd.extend(
                    [
                        "--print",
                        "--debug",
                        "--verbose",
                    ]
                )

            # system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)
            if system_messages:
                cmd.extend(["--append-system-prompt", "\n\n".join(system_messages)])

            # mcp servers (combine static configs with bridged tools).
            # The hooks server is registered with the bridge (so hook
            # scripts can call it via the sandbox service) but must NOT
            # appear in --mcp-config
            cc_mcp_servers = [
                s for s in
                (list(mcp_servers or []) + bridge.mcp_server_configs)
                if s.name != _HOOKS_SERVER_NAME
            ]

            cmd_allowed_tools: list[str] = []
            if cc_mcp_servers:
                mcp_server_args, mcp_allowed_tools = resolve_mcp_servers(
                    cc_mcp_servers
                )
                cmd.extend(mcp_server_args)
                cmd_allowed_tools.extend(mcp_allowed_tools)

            # add allowed and disallowed tools
            if len(cmd_allowed_tools) > 0:
                cmd.append("--allowed-tools")
                cmd.append(",".join(cmd_allowed_tools))
            if len(effective_disallowed_tools) > 0:
                cmd.append("--disallowed-tools")
                cmd.append(",".join(effective_disallowed_tools))

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

            # install hook scripts into the sandbox
            if hooks is not None:
                await _install_hooks(sbox, user, hooks)

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
                    result = await sbox.exec(
                        cmd=["bash", "-c", 'exec 0<&- "$@"', "bash"] + agent_cmd,
                        cwd=cwd,
                        env=agent_env,
                        user=user,
                        concurrency=False,
                    )

                    # track debug output
                    debug_output.append(result.stdout)
                    debug_output.append(result.stderr)

                    # raise for error
                    if not result.success:
                        # see if this is a timeout and we are retrying timeouts
                        if (
                            "request timed out"
                            in (result.stdout.lower() + result.stderr.lower())
                            and retry_timeouts is not None
                            and timeout_count < retry_timeouts
                        ):
                            timeout_count += 1
                            delay = min(2**timeout_count, 60)
                            trace(
                                f"Retrying timed out request (retry {timeout_count}, waiting {delay} seconds)."
                            )
                            await anyio.sleep(delay)
                            continue

                        raise RuntimeError(
                            f"Error executing claude code agent: {result.stdout}\n{result.stderr}"
                        )

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


# ---------------------------------------------------------------------------
# Hook dispatch via MCP bridge
# ---------------------------------------------------------------------------

@tool
def _hook_dispatch(
    hooks: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]],
) -> Tool:
    """Dispatch a Claude Code hook event to the registered handler."""

    async def run(hook_event: str, hook_input: str) -> str:
        """Dispatch a hook event.

        Args:
            hook_event: The hook event name (e.g. "Stop", "PreToolUse").
            hook_input: JSON-encoded hook input from Claude Code.

        Returns:
            JSON-encoded decision dict.
        """
        handler = hooks.get(hook_event)
        if handler is None:
            raise ValueError(
                f"No handler registered for hook event {hook_event!r}. "
                f"Registered events: {list(hooks.keys())}"
            )
        input_dict = json.loads(hook_input)
        result = await handler(input_dict)
        return json.dumps(result)

    return run


async def _install_hooks(
    sbox: Any,
    user: str | None,
    hooks: dict[str, Any],
) -> None:
    """Write hook scripts and Claude Code settings into the sandbox."""
    # Write a hook script for each event
    hook_settings: dict[str, list] = {}
    for event_name in hooks:
        script_path = f"/tmp/_hook_{event_name}.py"
        script_content = _generate_hook_script(event_name)
        await sbox.write_file(script_path, script_content)
        await sbox.exec(cmd=["chmod", "+x", script_path], user="root")
        hook_settings[event_name] = [{
            "hooks": [{
                "type": "command",
                "command": f"python3 {script_path}",
            }]
        }]

    # Write Claude Code settings
    settings = {"hooks": hook_settings}
    home_result = await sbox.exec(
        cmd=["bash", "-c", 'echo "$HOME"'], user=user,
    )
    home_dir = home_result.stdout.strip() or "/root"
    claude_dir = f"{home_dir}/.claude"
    await sbox.exec(cmd=["mkdir", "-p", claude_dir], user=user)
    await sbox.write_file(
        f"{claude_dir}/settings.json", json.dumps(settings, indent=2),
    )


def _generate_hook_script(event_name: str) -> str:
    """Generate a Python hook script that calls the bridge service."""
    return dedent(f"""\
        #!/usr/bin/env python3
        import glob, json, os, sys
        hook_input = json.load(sys.stdin)
        dirs = glob.glob("/var/tmp/sandbox-services/bridge_model_service/*/")
        assert dirs, "No bridge_model_service instance found"
        instance_dir = dirs[0]
        instance_name = instance_dir.rstrip("/").split("/")[-1]
        os.environ["BRIDGE_MODEL_SERVICE_INSTANCE"] = instance_name
        sys.path.insert(0, instance_dir)
        from bridge_model_service import call_bridge_model_service
        result = call_bridge_model_service(
            "call_tool",
            server="{_HOOKS_SERVER_NAME}",
            tool="_hook_dispatch",
            arguments={{"hook_event": "{event_name}", "hook_input": json.dumps(hook_input)}},
        )
        if result:
            print(result if isinstance(result, str) else json.dumps(result))
        else:
            sys.exit(0)
    """)


# ---------------------------------------------------------------------------
# Utilities (unchanged from upstream)
# ---------------------------------------------------------------------------

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


def inspect_model(model: str | None) -> str | None:
    """Ensure that model name is prefaced with 'inspect/'."""
    if model is not None:
        if model != "inspect" and not model.startswith("inspect/"):
            return f"inspect/{model}"

    return model


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
