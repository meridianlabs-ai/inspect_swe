import shlex
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_channel,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateFilter,
    Model,
)
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.util import (
    ExecRemoteStreamingOptions,
    store,
    store_as,
)
from inspect_ai.util import (
    sandbox as sandbox_env,
)
from inspect_ai.util._span import current_span_id
from pydantic_core import to_json

from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._claude_code._events.stream import claude_code_event_stream
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.path import join_path

from .._util.agentbinary import ensure_agent_binary_installed
from .._util.messages import build_user_prompt
from .._util.trace import trace
from .agentbinary import claude_code_binary_source
from .loop import (
    ClaudeCodeDebug,
    Completed,
    Policy,
    RunOutcome,
    _user_text,
    attempts_policy,
    consume_outcome,
    error_retry_policy,
    operator_policies,
    run_agent_loop,
)
from .model import resolve_claude_code_models


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
    model_config: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
    subagent_model: str | None = None,
    filter: GenerateFilter | None = None,
    auto_mode: bool = False,
    retry_refusals: int | None = 3,
    retry_uncaught_errors: int | None = 3,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
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
        model_config: Model id used to select the identity Claude Code presents
            to itself (its "You are powered by the model ..." system prompt) and
            any model-gated client behavior. Defaults to `None`, which derives it
            from the real served model so the presented identity matches what's
            actually running. Purely the displayed identity — calls are still
            bridged to the served Inspect model regardless. (Claude Code renders
            the genuine name/cutoff for recognized Anthropic ids and shows other
            ids verbatim.)
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models.
            When a model name in the mapping is referenced, the corresponding Model/string is used.
        opus_model: The model to use for `opus`, or for `opusplan` when Plan Mode is active. Defaults to `model`.
        sonnet_model: The model to use for `sonnet`, or for `opusplan` when Plan Mode is not active. Defaults to `model`.
        haiku_model: The model to use for haiku, or [background functionality](https://code.claude.com/docs/en/costs#background-token-usage). Defaults to `model`.
        subagent_model: The model to use for [subagents](https://code.claude.com/docs/en/sub-agents). Defaults to `model`.
        filter: Filter for intercepting bridged model requests.
        auto_mode: Use `auto` permission mode rather than `--dangerously-skip-permissions`. Note that this can result in rejected tool calls so only enable if your evaluation can tolerate this.
        retry_refusals: Should refusals be retried? Defaults to retrying up to 3 times.
        retry_uncaught_errors: Should uncaught errors (unexpected crashes of Claude Code) be retried. Defaults to retrying up to 3 times.
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
        debug: Add `--debug` cli flag and trace all debug output.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # allocate session_id once per agent instance so that all calls to execute()
    # for the same sample share the same session. this enables --resume <id> to
    # replay the full conversation history through the bridge on continuation runs,
    # giving the model proper context (unlike --continue which only sends the new turn).
    session_id = str(uuid.uuid4())

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "claude_code_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        # Real-time consumer of Claude Code JSONL output. Doubles as the
        # bridge's ModelEventSink — the bridge hands us every ModelEvent
        # instead of emitting it to the transcript, and we attribute each
        # to the correct agent span using parent_tool_use_id from the JSONL
        # stream. Captures the outer span_id (this @agent's span) so
        # sub-agent spans we discover from JSONL can be parented correctly.
        # See live_consumer.py for full mechanism.
        consumer = LiveConsumer(outer_span_id=current_span_id())

        # Resolve the (cosmetic) model identities Claude Code presents to itself
        # and the bridge aliases that route them to the real served model. The
        # per-role env vars below carry the opus/sonnet/haiku/subagent names.
        models = resolve_claude_code_models(
            model,
            model_config,
            opus_model=opus_model,
            sonnet_model=sonnet_model,
            haiku_model=haiku_model,
            subagent_model=subagent_model,
            model_aliases=model_aliases,
        )

        async with sandbox_agent_bridge(
            state,
            model=models.bridge_model,
            model_aliases=models.aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
            model_event_sink=consumer,
        ) as bridge:
            # ensure claude is installed and get binary location
            claude_binary = await ensure_agent_binary_installed(
                claude_code_binary_source(), version, user, sandbox_env(sandbox)
            )

            # base options — auto_mode uses --permission-mode auto (monitor active);
            # otherwise --dangerously-skip-permissions (no permission gating).
            permission_flag = (
                ["--permission-mode", "auto"]
                if auto_mode
                else ["--dangerously-skip-permissions"]
            )
            cmd = [
                *permission_flag,
                "--model",
                models.presented,
            ]

            # add interactive options if not running as centaur
            if centaur is False:
                cmd.extend(["--print", "--output-format", "stream-json", "--verbose"])
                if debug:
                    cmd.append("--debug")

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
                "ANTHROPIC_MODEL": models.presented,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": models.opus,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": models.sonnet,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": models.haiku,
                "CLAUDE_CODE_SUBAGENT_MODEL": models.subagent,
                "ANTHROPIC_SMALL_FAST_MODEL": models.haiku,
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
                # Open the agent intervention channel. `live` is True if this
                # eval was launched with --acp-server
                async with agent_channel() as ch:
                    live = ch.is_live

                    # debug capture across all launches in this execution
                    debug_output: list[str] = []
                    cc_debug = store_as(ClaudeCodeDebug) if debug else None

                    async def pending_operator_message() -> str | None:
                        return _user_text(await ch.before_turn(state.messages))

                    async def interrupt_redirect() -> str | None:
                        # Drop after_cancel's repair messages (Claude Code
                        # repairs its own conversation on --resume).
                        return _user_text(await ch.after_cancel(state.messages))

                    async def launch(resume: bool, agent_prompt: str) -> RunOutcome:
                        agent_cmd = _build_agent_cmd(
                            claude_binary=claude_binary,
                            session_id=session_id,
                            cmd=cmd,
                            messages=state.messages,
                            system_prompt=system_prompt,
                            resume=resume,
                            prompt=agent_prompt,
                        )

                        # Fresh consumer state per launch: agent-tree maps
                        # don't survive a subprocess restart, and reset()
                        # closes any spans the prior launch left open (keeping
                        # SpanBegin/End balanced).
                        consumer.reset()

                        # Stream so the consumer emits spans and the bridge
                        # sees Task prompts in real time.
                        proc = await sbox.exec_remote(
                            cmd=["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
                            + agent_cmd,
                            options=ExecRemoteStreamingOptions(
                                cwd=cwd,
                                env=agent_env,
                                user=user,
                                concurrency=False,
                            ),
                            stream=True,
                        )

                        outcome = await consume_outcome(
                            claude_code_event_stream(proc),
                            proc.kill,
                            on_jsonl=consumer.process_jsonl_line,
                            turn_scope=ch.turn_scope if live else None,
                            pending_message=pending_operator_message if live else None,
                            cc_debug=cc_debug,
                            debug_output=debug_output if debug else None,
                        )
                        if debug and isinstance(outcome, Completed):
                            debug_output.append(outcome.stderr)
                        return outcome

                    # ORDER IS LOAD-BEARING: operator continuation (same
                    # attempt) beats error retry, which beats attempt scoring.
                    policies: list[Policy] = [
                        *(
                            operator_policies(
                                pending=pending_operator_message,
                                interrupted=interrupt_redirect,
                                continue_prompt="Please proceed to the next step using your best judgement. If you believe you have completed the task, please print the results of the task in your next message",
                            )
                            if live
                            else []
                        ),
                        error_retry_policy(retry_uncaught_errors),
                        attempts_policy(attempts, state),
                    ]

                    try:
                        await run_agent_loop(
                            launch,
                            policies,
                            prompt=prompt,
                            resume=has_assistant_response,
                        )
                    finally:
                        # Close any spans the final launch left open — covers
                        # both normal exit (last subprocess ran cleanly but a
                        # sub-agent never returned its tool_result) and
                        # exception exit (error_retry_policy's RuntimeError, or
                        # anything else raised inside the loop). Without this,
                        # the agent span tree leaks past the @agent boundary.
                        consumer.reset()

                    # trace debug info
                    if debug:
                        debug_output.insert(0, "Claude Code Debug Output:")
                        trace("\n".join(debug_output))

        return bridge.state

    # return agent with specified name and descritpion
    return agent_with(execute, name=name, description=description)


def _build_agent_cmd(
    *,
    claude_binary: str,
    session_id: str,
    cmd: list[str],
    messages: Sequence[ChatMessage],
    system_prompt: str | None,
    resume: bool,
    prompt: str,
) -> list[str]:
    """Assemble the Claude Code argv for one launch.

    On a fresh session (``resume=False``) the system prompt is passed via
    ``--append-system-prompt``. On resume it is omitted: the session already
    contains system messages (the bridge round-trips Claude Code's own system
    prompt back into ``messages`` as a ``ChatMessageSystem``), and re-passing
    the flag would duplicate the entire system prompt on every resumed turn
    (it is applied per-invocation, not persisted). See the
    "system prompt duplicated on resumed turns" fix (#64).
    """
    system_args: list[str] = []
    if not resume:
        system_texts = [m.text for m in messages if isinstance(m, ChatMessageSystem)]
        if system_prompt is not None:
            system_texts.append(system_prompt)
        if system_texts:
            system_args = ["--append-system-prompt", "\n\n".join(system_texts)]
    session_flag = ["--resume", session_id] if resume else ["--session-id", session_id]
    return [claude_binary, *session_flag, *cmd, *system_args, "--", prompt]


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
