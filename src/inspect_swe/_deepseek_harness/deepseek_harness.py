import inspect
import json
import shlex
from logging import getLogger
from pathlib import Path
from textwrap import dedent
from typing import Any, Awaitable, Callable, Literal, Sequence, cast

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.agent._bridge.util import resolve_inspect_model
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateConfig,
    GenerateFilter,
    GenerateInput,
    Model,
    ModelOutput,
    get_model_info,
)
from inspect_ai.scorer import score
from inspect_ai.tool import (
    MCPServerConfig,
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
    Skill,
    ToolChoice,
    ToolInfo,
    install_skills,
    read_skills,
)
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.mcp_ready import (
    DEFAULT_MCP_READY_TIMEOUT,
    wait_for_mcp_endpoints,
)
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from .._util.sandbox import resolve_agent_cwd
from .agentbinary import deepseek_harness_binary_source
from .reasoning import ReasoningRestorer

# The `dsh` profile the agent runs: one task per invocation, final answer and
# a newline-delimited JSON event stream on stdout, same tools/prompt as dsh's
# other surfaces.
DSH_PROFILE = "headless"

# dsh's conversation adapter is a DeepSeek "Messages" (Anthropic-compatible)
# client registered under this provider route.
DSH_PROVIDER = "deepseek-official"

# context window dsh's DeepSeek adapter assumes for a model without one
DSH_DEFAULT_CONTEXT_WINDOW = 1_000_000

logger = getLogger(__name__)

# GenerateFilter is a union of Model-first and (deprecated) str-first callables;
# the bridge dispatches on the user filter's first-parameter annotation. Our
# combined_filter wrapper hides the user filter from that dispatch, so we
# replicate it: Model-first filters get the Model, str-first get model.name.
_ModelFilter = Callable[
    [Model, list[ChatMessage], list[ToolInfo], "ToolChoice | None", GenerateConfig],
    Awaitable["ModelOutput | GenerateInput | None"],
]
_StrFilter = Callable[
    [str, list[ChatMessage], list[ToolInfo], "ToolChoice | None", GenerateConfig],
    Awaitable["ModelOutput | GenerateInput | None"],
]


def _is_legacy_str_filter(fn: GenerateFilter) -> bool:
    first = next(iter(inspect.signature(fn).parameters.values()), None)
    return first is not None and first.annotation is str


@agent
def deepseek_harness(
    name: str = "DeepSeek Harness",
    description: str = dedent("""
        DeepSeek Harness (dsh) coding agent — reads and edits code, runs shell
        commands, searches files, and iterates based on feedback.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    mcp_ready_timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    context_window: int | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool = False,
) -> Agent:
    """DeepSeek Harness agent.

    Agent that uses [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
    (`dsh`) running in a sandbox with Inspect model bridging.

    dsh runs its `headless` profile (its full coding-agent composition: system
    prompt, tools, subagents, compaction) with its DeepSeek Messages adapter
    pointed at the Inspect bridge, so the returned transcript is the bridge's
    record of those calls. MCP servers are wired natively as `dsh-mcp-client`
    rows of a per-run profile patch.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description
        system_prompt: Additional system prompt to append
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP
        mcp_ready_timeout: Seconds to wait for bridged MCP endpoints to serve
            tools before the agent launch errors.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        context_window: Context window to configure for dsh (drives its
            automatic compaction). Defaults to the bridged model's context
            length from Inspect model metadata, falling back to dsh's own
            default (1,000,000 tokens, sized for DeepSeek V4 models) when
            that metadata is unavailable.
        model_aliases: Optional mapping of model names to Model instances or model name strings.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run dsh within
        env: Environment variables to set for dsh
        user: User to execute dsh with
        sandbox: Optional sandbox environment name
        version: Version of dsh to use. One of:
            - "auto": Use any available version in sandbox, otherwise download latest
            - "sandbox": Use sandbox version (raises RuntimeError if not available)
            - "stable"/"latest": Download and use the latest version
            - "x.x.x": Download and use a specific version (a
              `deepseek-harness-runtime-bin` PyPI version, e.g. "0.2.1rc2")
        debug: Trace all debug output.
    """
    # resolve the name dsh sends and the bridge fallback independently: the
    # bridge matches aliases against the raw requested name before stripping
    # the inspect/ prefix from its fallback.
    requested_model = model or "inspect"
    bridge_model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills / attempts
    resolved_skills = read_skills(skills) if skills is not None else None
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # dsh's mcp-client has no per-server tool filter and no legacy SSE
    # transport, so reject configs we can't honor rather than silently widen
    # or drop them.
    for server in mcp_servers or []:
        _validate_mcp_server(server)

    filter_is_legacy = filter is not None and _is_legacy_str_filter(filter)

    async def execute(state: AgentState) -> AgentState:
        # present the served model's real name to dsh (its system prompt says
        # "powered by the {{model}} model") and alias that name back to the
        # served model; an explicit alias name is presented as-is
        served_model = resolve_inspect_model(
            requested_model, model_aliases, bridge_model
        )
        if model_aliases is not None and requested_model in model_aliases:
            presented_model = requested_model
            aliases = model_aliases
        else:
            presented_model = served_model.name
            aliases = {presented_model: served_model, **(model_aliases or {})}

        resolved_context_window = _resolve_context_window(
            model=served_model, context_window=context_window
        )

        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "deepseek_harness_model_port"
        port = store().get(MODEL_PORT, 3200) + 1
        store().set(MODEL_PORT, port)

        reasoning_restorer = ReasoningRestorer()

        async def combined_filter(
            model: Model,
            messages: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice | None,
            config: GenerateConfig,
        ) -> ModelOutput | GenerateInput | None:
            # in place (not only via GenerateInput) so the restored reasoning
            # persists in both the recorded ModelEvent input and the bridge
            # state messages
            changed = reasoning_restorer.restore(messages)
            if filter is not None:
                if filter_is_legacy:
                    result = await cast(_StrFilter, filter)(
                        model.name, messages, tools, tool_choice, config
                    )
                else:
                    result = await cast(_ModelFilter, filter)(
                        model, messages, tools, tool_choice, config
                    )
                if result is not None:
                    return result
            if changed:
                return GenerateInput(
                    input=messages, tools=tools, tool_choice=tool_choice, config=config
                )
            return None

        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases=aliases,
            filter=combined_filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
            # dsh's native web_search tool issues Anthropic `web_search`
            # server-tool requests, which the bridge serves via this grant
            # (for providers with internal search, as for kimi/claude code)
            web_search=True,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # install dsh in sandbox
            dsh_binary = await ensure_agent_binary_installed(
                deepseek_harness_binary_source(), version, user, sbox
            )

            # combine static mcp configs with bridged tools' mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)
            for server in bridge.mcp_server_configs:
                _validate_mcp_server(server)

            # detect sandbox home directory; dsh never falls back to ~/.dsh
            # when DSH_HOME is set, so give each run an isolated home
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"
            dsh_home = f"{sandbox_home}/.dsh-inspect"
            await sbox.exec(["mkdir", "-p", dsh_home], user=user)

            # install skills (dsh scans $DSH_HOME/skills)
            if resolved_skills is not None:
                await install_skills(resolved_skills, sbox, user, f"{dsh_home}/skills")

            # per-run profile patch (model route, telemetry, mcp servers)
            patch_file = f"{dsh_home}/inspect.patch.yml"
            await sbox.write_file(
                patch_file,
                _profile_patch(
                    model=presented_model,
                    context_window=resolved_context_window,
                    mcp_servers=all_mcp_servers,
                ),
            )

            prompt, has_assistant_response = build_user_prompt(state.messages)

            # continue the sample's dsh session when re-invoked on a state
            # that already carries an assistant turn
            SESSION_ID = "deepseek_harness_session_id"
            session_id: str | None = (
                store().get(SESSION_ID, None) if has_assistant_response else None
            )

            # headless takes a single task and has no separate system-prompt
            # flag, so prepend system messages to a new session's task. A
            # continued session already carries them, and the state's system
            # message is now dsh's own (round-tripped by the bridge).
            if session_id is None:
                system_messages = [
                    m.text for m in state.messages if isinstance(m, ChatMessageSystem)
                ]
                if system_prompt is not None:
                    system_messages.append(system_prompt)
                if system_messages:
                    prompt = "\n\n".join(system_messages) + "\n\n" + prompt

            agent_env = {
                "DSH_HOME": dsh_home,
                "HOME": sandbox_home,
                # dsh sends this as its x-api-key; the bridge accepts any key
                # and real provider credentials never enter the sandbox
                "DEEPSEEK_API_KEY": "sk-inspect-bridge",
                # conversation adapter (appends /v1/messages) and the
                # auxiliary native web search (appends /messages)
                "DEEPSEEK_BASE_URL": f"http://localhost:{bridge.port}",
                "DEEPSEEK_SEARCH_BASE_URL": f"http://localhost:{bridge.port}/v1",
                # the sandbox is the isolation boundary; dsh's own policy
                # would otherwise ask for approval no one can answer
                "DSH_PERMISSION_MODE": "danger-full-access",
                "DSH_TELEMETRY_MODE": "DISABLED",
            } | (env or {})

            cmd = [
                dsh_binary,
                "--profile",
                DSH_PROFILE,
                "--patch",
                patch_file,
                "--json",
            ]

            # dsh connects MCP servers before its first turn, but gate on the
            # bridged endpoints serving so a slow bridge can't race startup
            http_mcp_configs = [
                c
                for c in bridge.mcp_server_configs
                if isinstance(c, MCPServerConfigHTTP)
            ]

            debug_output: list[str] = []
            agent_prompt = prompt
            attempt_count = 0
            prompt_file = f"{dsh_home}/prompt.txt"

            while True:
                agent_cmd = cmd.copy()
                if session_id is not None:
                    agent_cmd += ["--session-id", session_id]

                if http_mcp_configs:
                    await wait_for_mcp_endpoints(
                        http_mcp_configs,
                        bridge,
                        sandbox=sandbox,
                        timeout=mcp_ready_timeout,
                        required=True,
                    )

                # the task is read from stdin (no positional task), which
                # keeps a prompt starting with "-" from parsing as an option
                await sbox.write_file(prompt_file, agent_prompt)
                result = await sbox.exec_remote(
                    cmd=[
                        "bash",
                        "-c",
                        f'exec "$@" < {shlex.quote(prompt_file)}',
                        "bash",
                    ]
                    + agent_cmd,
                    options=ExecRemoteAwaitableOptions(
                        cwd=agent_cwd,
                        env=agent_env,
                        user=user,
                        concurrency=False,
                    ),
                    stream=False,
                )

                if debug:
                    debug_output.append(result.stdout)
                    debug_output.append(result.stderr)

                events = _parse_events(result.stdout)
                session_id = _session_id(events) or session_id
                if session_id is not None:
                    store().set(SESSION_ID, session_id)

                if not result.success:
                    raise RuntimeError(
                        f"Error executing deepseek harness agent {result.returncode}:\n"
                        f"turn_end: {_turn_end_reason(events)}\n"
                        f"stderr: {result.stderr}"
                    )

                attempt_count += 1
                if attempt_count >= attempts.attempts:
                    break

                answer_scores = await score(bridge.state)
                if attempts.score_value(answer_scores[0].value) == 1.0:
                    break

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

            if debug:
                debug_output.insert(0, "DeepSeek Harness Debug Output:")
                trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def _validate_mcp_server(server: MCPServerConfig) -> None:
    if server.tools != "all":
        raise ValueError(
            f"MCP server {server.name!r}: deepseek_harness does not support "
            "per-server tool filtering (dsh exposes every tool a server lists)."
        )
    if isinstance(server, MCPServerConfigHTTP) and server.type != "http":
        raise ValueError(
            f"MCP server {server.name!r}: deepseek_harness supports only stdio "
            "and Streamable HTTP MCP servers (not SSE)."
        )


def _resolve_context_window(*, model: Model, context_window: int | None) -> int | None:
    if context_window is not None:
        if context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        return context_window

    model_info = get_model_info(model)
    if model_info is None or not model_info.context_length:
        logger.warning(
            f"Context length metadata is unavailable for model {model.name!r}; "
            f"dsh will assume its default ({DSH_DEFAULT_CONTEXT_WINDOW:,} tokens). "
            "Pass context_window to set it explicitly."
        )
        return None
    return model_info.context_length


def _profile_patch(
    *,
    model: str,
    context_window: int | None,
    mcp_servers: Sequence[MCPServerConfig],
) -> str:
    # dsh profile patches are YAML patch lists; JSON is valid YAML and keeps
    # every value (prompts, urls, headers) safely quoted. A row patch's
    # `config` replaces that row's complete config.
    rows: list[dict[str, Any]] = [
        # route the default agent model through the bridge
        {
            "id": "agent-default-model",
            "config": {"provider": DSH_PROVIDER, "model": model},
        },
        # a plain catalog entry: the default `deepseek-flash` entry opts into
        # DeepSeek-API-only wire features (in-history system prompt updates,
        # deferred tool loading) that the bridge does not speak, and images
        # are uploaded through DeepSeek's Files API, so keep it text-only
        {
            "id": "llm-deepseek",
            "config": {"models": [_catalog_model(model, context_window)]},
        },
        # these attach the session log / plugin inventory to every request
        # (diagnostics for DeepSeek's official API, meaningless via the bridge)
        {"id": "session-log-deepseek", "disabled": True},
        {"id": "plugin-package-inventory-deepseek", "disabled": True},
        # an auxiliary model call that only names the session
        {"id": "session-title-llm", "disabled": True},
        # never export telemetry from an eval sandbox
        {"id": "session-telemetry-otel", "disabled": True},
    ]

    mcp_rows = [_mcp_row(server) for server in mcp_servers]
    if mcp_rows:
        rows.append({"insert": mcp_rows})

    return json.dumps(rows, indent=2) + "\n"


def _catalog_model(model: str, context_window: int | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": model}
    if context_window is not None:
        entry["contextWindow"] = context_window
    return entry


def _mcp_row(server: MCPServerConfig) -> dict[str, Any]:
    config: dict[str, Any]
    if isinstance(server, MCPServerConfigStdio):
        config = {
            "serverName": server.name,
            "transport": "stdio",
            "command": server.command,
            "args": list(server.args),
            "env": dict(server.env or {}),
        }
        if server.cwd is not None:
            config["cwd"] = str(server.cwd)
    elif isinstance(server, MCPServerConfigHTTP):
        config = {
            "serverName": server.name,
            "transport": "streamable-http",
            "url": server.url,
            "headers": dict(server.headers or {}),
        }
    else:
        raise ValueError(f"Unsupported MCP server config type: {type(server).__name__}")
    # a server that fails to connect should fail the run, not silently run
    # the task without its tools
    config["failOnStartupError"] = True
    return {
        "id": f"mcp-{server.name}",
        "name": "@deepseek-ai/dsh-mcp-client",
        "config": config,
    }


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _session_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") == "session" and isinstance(event.get("sessionId"), str):
            return str(event["sessionId"])
    return None


def _turn_end_reason(events: list[dict[str, Any]]) -> str | None:
    reason: str | None = None
    for event in events:
        if event.get("type") == "status" and event.get("phase") == "turn_end":
            reason = str(event.get("reason"))
        elif event.get("type") == "error":
            reason = f"error: {event.get('message')}"
    return reason
