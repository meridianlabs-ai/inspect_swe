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
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.util import (
    ExecRemoteStreamingOptions,
    StoreModel,
    store,
    store_as,
)
from inspect_ai.util import (
    sandbox as sandbox_env,
)
from inspect_ai.util._span import current_span_id
from pydantic import Field
from pydantic_core import to_json

from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._claude_code._events.stream import (
    ExitEvent,
    JsonlEvent,
    JsonlParseError,
    StderrEvent,
    claude_code_event_stream,
)
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.path import join_path

from .._util._async import is_callable_coroutine
from .._util.agentbinary import ensure_agent_binary_installed
from .._util.messages import build_user_prompt
from .._util.trace import trace
from .agentbinary import claude_code_binary_source
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
                # execute the agent (track debug output)
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0
                uncaught_error_count = 0
                try:
                    while True:
                        is_resume = (
                            has_assistant_response
                            or attempt_count > 0
                            or uncaught_error_count > 0
                        )

                        # System prompt is sent only when creating the session.
                        # On resume the session already contains system messages, so send
                        # nothing: the bridge round-trips Claude Code's own
                        # system prompt back into state.messages as a
                        # ChatMessageSystem, and re-passing it via
                        # --append-system-prompt would duplicate the entire
                        # system prompt on every resumed turn (the flag is
                        # applied per-invocation, not persisted anyway).
                        system_args: list[str] = []
                        if not is_resume:
                            system_texts = [
                                m.text
                                for m in state.messages
                                if isinstance(m, ChatMessageSystem)
                            ]
                            if system_prompt is not None:
                                system_texts.append(system_prompt)
                            if system_texts:
                                system_args = [
                                    "--append-system-prompt",
                                    "\n\n".join(system_texts),
                                ]

                        # resume previous conversation
                        if is_resume:
                            agent_cmd = (
                                [claude_binary, "--resume", session_id]
                                + cmd
                                + system_args
                                + ["--", agent_prompt]
                            )
                        else:
                            agent_cmd = (
                                [claude_binary, "--session-id", session_id]
                                + cmd
                                + system_args
                                + ["--", agent_prompt]
                            )

                        # Fresh consumer state per attempt — agent-tree maps
                        # don't carry across Claude Code subprocess restarts.
                        # reset() also closes any spans the previous attempt
                        # left open (e.g. Claude exited mid-Task before the
                        # tool_result), so SpanBegin/End stay balanced.
                        consumer.reset()

                        # launch Claude Code in streaming mode; drain stdout in
                        # real time so the consumer emits agent spans and the
                        # bridge resolver sees Task prompts as they appear.
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

                        cc_debug = store_as(ClaudeCodeDebug) if debug else None
                        stderr_data = ""
                        exit_code = 0

                        async for cc_event in claude_code_event_stream(proc):
                            if isinstance(cc_event, JsonlEvent):
                                consumer.process_jsonl_line(cc_event.raw)
                                if cc_debug is not None:
                                    cc_debug.stdout.append(cc_event.line)
                            elif isinstance(cc_event, JsonlParseError):
                                if debug:
                                    debug_output.append(
                                        f"JSONL parse error: {cc_event.line}"
                                    )
                            elif isinstance(cc_event, StderrEvent):
                                stderr_data += cc_event.data
                                if cc_debug is not None:
                                    cc_debug.stderr.append(cc_event.data)
                            elif isinstance(cc_event, ExitEvent):
                                exit_code = cc_event.code

                        if debug:
                            debug_output.append(stderr_data)

                        # raise for error
                        if exit_code != 0:
                            # if claude code exits with code 1 and no stderr,
                            # this means an uncaught exception reached the top
                            # of its main loop -- we treat this as a scaffold
                            # bug and retry/resume a configurable number of
                            # times
                            if (
                                exit_code == 1
                                and len(stderr_data.strip()) == 0
                                and retry_uncaught_errors is not None
                                and uncaught_error_count < retry_uncaught_errors
                            ):
                                uncaught_error_count += 1
                                continue

                            # otherwise this is a hard failure
                            raise RuntimeError(
                                f"Error executing claude code agent {exit_code}: {stderr_data}"
                            )

                        # reset uncaught error counter
                        uncaught_error_count = 0

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
                                if not is_callable_coroutine(
                                    attempts.incorrect_message
                                ):
                                    raise ValueError(
                                        "The incorrect_message function must be async."
                                    )
                                agent_prompt = await attempts.incorrect_message(
                                    state, answer_scores
                                )
                            else:
                                agent_prompt = attempts.incorrect_message
                finally:
                    # Close any spans the final attempt left open — covers
                    # both normal exit (last subprocess ran cleanly but a
                    # sub-agent never returned its tool_result) and
                    # exception exit (RuntimeError above, or anything else
                    # raised inside the loop). Without this, the agent
                    # span tree leaks past the @agent boundary.
                    consumer.reset()

                # trace debug info
                if debug:
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


class ClaudeCodeDebug(StoreModel):
    stderr: list[str] = Field(default_factory=list)
    stdout: list[str] = Field(default_factory=list)
