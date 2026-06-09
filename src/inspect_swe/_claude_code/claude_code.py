import shlex
import uuid
from contextlib import nullcontext
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Literal, Sequence

import anyio
from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentInterrupted,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_channel,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.agent._types import DEFAULT_CONTINUE_PROMPT_NO_SUBMIT
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateFilter,
    Model,
)
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.util import (
    ExecRemoteProcess,
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
    ClaudeCodeStreamEvent,
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
                # Open the agent intervention channel. `live` is True if this
                # eval was launched with --acp-server
                async with agent_channel() as ch:
                    live = ch.is_live

                    # track debug output across all launches in this execution
                    debug_output: list[str] = []

                    async def consume(
                        proc: ExecRemoteProcess,
                    ) -> tuple[int, str, str | None, bool]:
                        # Drain Claude Code's JSONL stream. In live mode, at the
                        # next safe delivery seam (`_operator_delivery_gate`) a
                        # queued operator message kills the process and is
                        # returned as the redirect for a --resume relaunch.
                        # `any_tool_uses` reports whether this run made any
                        # top-level tool call (used to nudge a conversational-only
                        # operator interjection back to work).
                        # Returns (exit_code, stderr, redirect|None, any_tool_uses).
                        cc_debug = store_as(ClaudeCodeDebug) if debug else None
                        stderr_data = ""
                        exit_code = 0
                        any_tool_uses = False
                        is_delivery_seam = _operator_delivery_gate()

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

                            if not any_tool_uses and _top_level_tool_use_ids(cc_event):
                                any_tool_uses = True

                            if live and is_delivery_seam(cc_event):
                                pending = _user_text(
                                    await ch.before_turn(state.messages)
                                )
                                if pending is not None:
                                    await proc.kill()
                                    return (
                                        exit_code,
                                        stderr_data,
                                        pending,
                                        any_tool_uses,
                                    )

                        return exit_code, stderr_data, None, any_tool_uses

                    async def run_prompt(
                        resume: bool, agent_prompt: str
                    ) -> tuple[int, str]:
                        # Run one prompt to completion. In live mode, Esc
                        # interrupts and queued operator messages relaunch Claude
                        # Code via --resume with the operator text. Non-live mode
                        # launches once and returns.
                        # Tracks whether the current launch delivered an operator
                        # message, so a conversational-only reply can be nudged
                        # back to work (once per interjection).
                        launched_for_operator = False
                        while True:
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

                            try:
                                with ch.turn_scope() if live else nullcontext():
                                    (
                                        exit_code,
                                        stderr_data,
                                        redirect,
                                        any_tool_uses,
                                    ) = await consume(proc)
                            except AgentInterrupted:
                                # Esc. The scope cancel may land away from
                                # exec_remote's kill handler, so kill explicitly
                                # (shielded + idempotent).
                                with anyio.CancelScope(shield=True):
                                    await proc.kill()
                                # Block for the redirect; drop after_cancel's
                                # repair messages (Claude Code repairs its own
                                # conversation on --resume).
                                follow = _user_text(
                                    await ch.after_cancel(state.messages)
                                )
                                resume, agent_prompt = True, follow or agent_prompt
                                launched_for_operator = True
                                continue

                            if redirect is not None:
                                # Plain message delivered at a seam; relaunch via
                                # --resume.
                                resume, agent_prompt = True, redirect
                                launched_for_operator = True
                                continue

                            if live:
                                # Backstop: a message that landed during the
                                # final result.
                                pending = _user_text(
                                    await ch.before_turn(state.messages)
                                )
                                if pending is not None:
                                    resume, agent_prompt = True, pending
                                    launched_for_operator = True
                                    continue

                            if (
                                live
                                and launched_for_operator
                                and not any_tool_uses
                                and exit_code == 0
                            ):
                                # An operator interjection the model answered
                                # conversationally (no tools) would otherwise end
                                # the run. Nudge it once to resume work via
                                # --resume. Clearing launched_for_operator bounds
                                # this to once per interjection (only a real
                                # operator message sets it True again).
                                resume, agent_prompt = (
                                    True,
                                    DEFAULT_CONTINUE_PROMPT_NO_SUBMIT.strip(),
                                )
                                launched_for_operator = False
                                continue

                            return exit_code, stderr_data

                    # execute the agent
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

                            exit_code, stderr_data = await run_prompt(
                                is_resume, agent_prompt
                            )

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
                        # Close any spans the final launch left open — covers
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


def _user_text(msgs: Sequence[ChatMessage]) -> str | None:
    """Join operator user-message text drained from the agent channel.

    Filters out any synthetic ``ChatMessageTool`` repair messages (e.g. those
    ``after_cancel`` prepends) — Claude Code repairs its own conversation when
    we relaunch with ``--resume``, so appending ours would double-repair.
    Returns ``None`` when there are no user messages to forward.
    """
    texts = [m.text for m in msgs if isinstance(m, ChatMessageUser)]
    return "\n\n".join(texts) if texts else None


def _is_turn_boundary(cc_event: ClaudeCodeStreamEvent) -> bool:
    """True at the start of a new top-level Claude Code assistant turn.

    A top-level ``assistant`` event (``parent_tool_use_id is None``) only
    arrives once every prior top-level tool call — including any ``Task``
    sub-agents — has resolved (the top-level model can't start a new turn
    until its tool results are back). So it is a safe seam to abort at: no
    open sub-agent spans and no unresolved top-level tool calls. A sub-agent's
    own ``assistant`` events carry a ``parent_tool_use_id`` and are excluded.
    """
    return (
        isinstance(cc_event, JsonlEvent)
        and cc_event.raw.get("type") == "assistant"
        and cc_event.raw.get("parent_tool_use_id") is None
    )


def _operator_delivery_gate() -> Callable[[ClaudeCodeStreamEvent], bool]:
    """Build the per-run predicate for when a queued operator message may be delivered.

    A *plain* operator message is delivered at the next safe seam — one where
    no top-level tool call is unresolved and no sub-agent span is open, so the
    process can be killed and resumed cleanly. The returned predicate returns
    True at the first of two such seams:

    1. **Tools just completed** — every tool call from the CURRENT top-level
       turn has resolved. This lands BEFORE the next generation starts, so we
       don't wait for (and then discard on --resume) a full wasted turn. The
       fast path for tool-heavy turns.
    2. **Next top-level assistant turn** (:func:`_is_turn_boundary`) — the
       backstop for turns with no tool calls (the outstanding set never
       populates) and for a message that lands during the final result.

    Closes over the set of outstanding *top-level* tool calls: a ``Task``
    sub-agent's own tool calls carry a non-null ``parent_tool_use_id`` and are
    excluded, so the Task's top-level ``tool_result`` (which only arrives once
    the sub-agent has finished) is what clears it — preserving the "all
    top-level tools resolved, no open sub-agent spans" guarantee.
    """
    # Mutated in place (``update`` / ``difference_update``) so the closure never
    # rebinds it and no ``nonlocal`` is needed.
    outstanding: set[str] = set()

    def is_delivery_seam(cc_event: ClaudeCodeStreamEvent) -> bool:
        outstanding.update(_top_level_tool_use_ids(cc_event))
        resolved = _tool_result_ids(cc_event)
        tools_just_completed = False
        if resolved and outstanding:
            outstanding.difference_update(resolved)
            tools_just_completed = not outstanding
        return tools_just_completed or _is_turn_boundary(cc_event)

    return is_delivery_seam


def _top_level_tool_use_ids(cc_event: ClaudeCodeStreamEvent) -> set[str]:
    """IDs of the ``tool_use`` blocks in a *top-level* assistant event.

    Empty for sub-agent assistant events (non-null ``parent_tool_use_id``),
    non-assistant events, and assistant turns with no tool calls.
    """
    if not (
        isinstance(cc_event, JsonlEvent)
        and cc_event.raw.get("type") == "assistant"
        and cc_event.raw.get("parent_tool_use_id") is None
    ):
        return set()
    return _tool_block_ids(cc_event.raw, "tool_use", "id")


def _tool_result_ids(cc_event: ClaudeCodeStreamEvent) -> set[str]:
    """``tool_use_id``s of the ``tool_result`` blocks in a *top-level* user event.

    These mark top-level tool calls whose results have come back. Empty for
    sub-agent user events (non-null ``parent_tool_use_id``) and non-user
    events.
    """
    if not (
        isinstance(cc_event, JsonlEvent)
        and cc_event.raw.get("type") == "user"
        and cc_event.raw.get("parent_tool_use_id") is None
    ):
        return set()
    return _tool_block_ids(cc_event.raw, "tool_result", "tool_use_id")


def _tool_block_ids(raw: dict[str, Any], block_type: str, id_field: str) -> set[str]:
    """Collect ``id_field`` from each ``block_type`` block in ``raw.message.content``."""
    message = raw.get("message")
    if not isinstance(message, dict):
        return set()
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    ids: set[str] = set()
    for block in content:
        if isinstance(block, dict) and block.get("type") == block_type:
            block_id = block.get(id_field)
            if isinstance(block_id, str):
                ids.add(block_id)
    return ids


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
