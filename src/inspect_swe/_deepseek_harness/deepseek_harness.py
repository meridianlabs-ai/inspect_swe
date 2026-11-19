import json
import re
import shlex
from logging import getLogger
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit

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
from inspect_ai.model._generate_config import active_generate_config
from inspect_ai.scorer import score
from inspect_ai.tool import (
    MCPServerConfig,
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
    Skill,
    Tool,
    ToolChoice,
    ToolInfo,
    install_skills,
    read_skills,
)
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.agentcontext import ModelFilter, classify_filter
from inspect_swe._util.bridged_tools import resolve_bridged_tools
from inspect_swe._util.mcp_ready import (
    DEFAULT_MCP_READY_TIMEOUT,
    wait_for_mcp_endpoints,
)
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from .._util.identity import refuse_sandbox_identity
from .._util.sandbox import resolve_agent_cwd
from ._events.consumer import DshConsumer
from .agentbinary import deepseek_harness_binary_source
from .reasoning import ReasoningRestorer

# one task per run, same prompt and tools as dsh's other surfaces
DSH_PROFILE = "headless"

DSH_PROVIDER = "deepseek-official"

# dsh's Messages-based web search provider (it shares the LLM provider's id)
DSH_SEARCH_PROVIDER = "deepseek-official"

# the model name that provider's request asks for, whatever the agent's model
DSH_SEARCH_MODEL = "deepseek-v4-flash"

# context window dsh's DeepSeek adapter assumes for a model without one
DSH_DEFAULT_CONTEXT_WINDOW = 1_000_000

logger = getLogger(__name__)

# server names dsh's mcp-client accepts
_MCP_SERVER_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")


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
    bridged_tools: Sequence[Tool | BridgedToolsSpec] | None = None,
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
        mcp_servers: MCP servers to make available to the agent. They run in,
            or are reached from, the sandbox, where dsh's own tools can read
            anything dsh can, so everything in a server's configuration is
            sandbox-exposed (its URL, command, arguments and `env`). An HTTP
            server with `headers`, or with a user or password in its URL, is
            rejected: use `bridged_tools` for a server that needs credentials.
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
            Each BridgedToolsSpec creates an MCP server; plain Tools are
            grouped into one MCP server named `tools`.
        mcp_ready_timeout: Seconds to wait for bridged MCP endpoints to serve
            tools before the agent launch errors.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        context_window: Context window to configure for dsh (drives its
            automatic compaction). Defaults to the bridged model's context
            length from Inspect model metadata, falling back to dsh's own
            default (1,000,000 tokens, sized for DeepSeek V4 models) when
            that metadata is unavailable. dsh compacts only while the window
            leaves room for the model's output limit plus a 65,536-token
            reserve.
        model_aliases: Optional mapping of model names to Model instances or model name strings.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run dsh within
        env: Environment variables to set for dsh (sandbox-exposed)
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

    for server in mcp_servers or []:
        _validate_mcp_server(server)

    # resolve plain Tools into specs once, so bad input fails at construction
    bridged_tools_specs = resolve_bridged_tools(bridged_tools)
    for spec in bridged_tools_specs or []:
        _validate_mcp_server_name(spec.name)

    async def execute(state: AgentState) -> AgentState:
        refuse_sandbox_identity("deepseek_harness()", sandbox)

        # give dsh the served model's real name (its prompt names the model)
        served_model = resolve_inspect_model(
            requested_model, model_aliases, bridge_model
        )
        aliases: dict[str, str | Model]
        if model_aliases is not None and requested_model in model_aliases:
            presented_model = requested_model
            aliases = dict(model_aliases)
        else:
            presented_model = served_model.name
            aliases = {presented_model: served_model, **(model_aliases or {})}
        # the model request behind dsh's web_search tool is the agent's too
        aliases.setdefault(DSH_SEARCH_MODEL, served_model)

        resolved_context_window = _resolve_context_window(
            model=served_model, context_window=context_window
        )
        resolved_max_tokens = _resolve_max_tokens(
            model=served_model, context_window=resolved_context_window
        )

        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "deepseek_harness_model_port"
        port = store().get(MODEL_PORT, 3200) + 1
        store().set(MODEL_PORT, port)

        # Classifies every bridged request (root / sub-agent / compaction) and,
        # as the bridge's ModelEventSink, files each model call under the span
        # of the dsh agent that made it. See _events/consumer.py.
        consumer = DshConsumer()

        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases=aliases,
            filter=build_dsh_filter(filter, consumer),
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools_specs,
            # serves the model request behind dsh's web_search tool
            # (providers with internal search)
            web_search=True,
            model_event_sink=consumer,
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

            # detect sandbox home directory
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"
            dsh_home = f"{sandbox_home}/.dsh-inspect"
            await sbox.exec(["mkdir", "-p", dsh_home], user=user)

            # install skills (dsh scans $DSH_HOME/skills)
            if resolved_skills is not None:
                await install_skills(resolved_skills, sbox, user, f"{dsh_home}/skills")

            # per-run profile patch (model route, telemetry, mcp servers).
            # dsh's tools run as dsh's own user, so nothing dsh can read is
            # hidden from the model: the patch holds no credential (see
            # `_validate_mcp_server`)
            patch_file = f"{dsh_home}/inspect.patch.yml"
            await sbox.write_file(
                patch_file,
                _profile_patch(
                    model=presented_model,
                    context_window=resolved_context_window,
                    max_tokens=resolved_max_tokens,
                    mcp_servers=all_mcp_servers,
                ),
            )

            prompt, has_assistant_response = build_user_prompt(state.messages)

            # continue the sample's dsh session on re-invocation
            SESSION_ID = "deepseek_harness_session_id"
            session_id: str | None = (
                store().get(SESSION_ID, None) if has_assistant_response else None
            )

            # headless has no system-prompt flag, so prepend system messages to
            # a new session's task (a continued one already has them)
            if session_id is None:
                system_messages = [
                    m.text for m in state.messages if isinstance(m, ChatMessageSystem)
                ]
                if system_prompt is not None:
                    system_messages.append(system_prompt)
                if system_messages:
                    prompt = "\n\n".join(system_messages) + "\n\n" + prompt

            agent_env = _agent_env(
                dsh_home=dsh_home, home=sandbox_home, port=bridge.port, env=env
            )

            cmd = [
                dsh_binary,
                "--profile",
                DSH_PROFILE,
                "--patch",
                patch_file,
                "--json",
            ]

            http_mcp_configs = [
                c
                for c in bridge.mcp_server_configs
                if isinstance(c, MCPServerConfigHTTP)
            ]

            debug_output: list[str] = []
            agent_prompt = prompt
            attempt_count = 0
            prompt_file = f"{dsh_home}/prompt.txt"
            events_file = f"{dsh_home}/events.jsonl"

            try:
                while True:
                    agent_cmd = cmd.copy()
                    if session_id is not None:
                        agent_cmd += ["--session-id", session_id]

                    # Fresh state per dsh process: reset() also closes any
                    # span the previous one left open (dsh exited with a
                    # child still running), so SpanBegin/End stay balanced.
                    consumer.reset()

                    if http_mcp_configs:
                        await wait_for_mcp_endpoints(
                            http_mcp_configs,
                            bridge,
                            sandbox=sandbox,
                            timeout=mcp_ready_timeout,
                            required=True,
                        )

                    # task on stdin, byte for byte (a leading "-" can't parse
                    # as an option); events to a file so a long run keeps its
                    # first `session` line
                    await sbox.write_file(prompt_file, agent_prompt)
                    result = await sbox.exec_remote(
                        cmd=[
                            "bash",
                            "-c",
                            f'exec "$@" < {shlex.quote(prompt_file)} '
                            f"> {shlex.quote(events_file)}",
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

                    events_result = await sbox.exec(
                        [
                            "sh",
                            "-c",
                            'head -n 1 "$1"; tail -n 20 "$1"',
                            "sh",
                            events_file,
                        ],
                        user=user,
                    )
                    events = _parse_events(events_result.stdout)

                    if debug:
                        debug_output.append(events_result.stdout)
                        debug_output.append(result.stderr)

                    session_id = _session_id(events) or session_id
                    if session_id is not None:
                        store().set(SESSION_ID, session_id)

                    _apply_root_generation(bridge.state, consumer)

                    # dsh exits 1 when its turn ends in a refusal, but the
                    # bridge has already recorded that refusal as the
                    # sample's answer, which can be scored
                    if not result.success and not _is_refusal_exit(consumer):
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
            finally:
                # close any span the last dsh process left open, also when
                # the loop raised, so none leaks past the @agent boundary
                consumer.reset()

            if debug:
                debug_output.insert(0, "DeepSeek Harness Debug Output:")
                trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def build_dsh_filter(
    filter: GenerateFilter | None, consumer: DshConsumer
) -> ModelFilter:
    """Bridge filter for dsh: reasoning repair, default effort, and agent context.

    The user's filter runs last, with the request classified (see
    `DshConsumer.classify`) and the repaired messages and effective config.
    """
    delegate = classify_filter(
        filter, consumer.classify, consumer.on_filter_output, consumer
    )
    reasoning_restorer = ReasoningRestorer()

    async def dsh_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        # in place, so the change also lands in the transcript and state
        changed = reasoning_restorer.restore(messages)
        if _reasoning_unset(model, config):
            config = config.merge(GenerateConfig(reasoning_effort="high"))
            changed = True
        result = await delegate(model, messages, tools, tool_choice, config)
        if result is not None:
            return result
        if changed:
            return GenerateInput(
                input=messages, tools=tools, tool_choice=tool_choice, config=config
            )
        return None

    return dsh_filter


def _apply_root_generation(state: AgentState, consumer: DshConsumer) -> None:
    """Make the root agent's last generation the sample's conversation and answer.

    The bridge picks the conversation it reports (`state.messages`, and
    `state.output` as the answer) by following threads of requests, and a
    sub-agent's thread can take that over: the root wins it back only by
    making two further requests, or one longer than the sub-agent's thread. A
    root that hands a long task to a child and then answers in a single short
    request would leave the child's last message as the sample's answer. dsh
    names the sender of every request, so the consumer knows the root's last
    generation and it is used instead. Nothing changes when the bridge
    already tracks it, or when no request was positively the root's.
    """
    root = consumer.root_generation
    if root is None or state.output is root.output:
        return
    state.messages = [*root.messages, root.output.message]
    state.output = root.output


def _is_refusal_exit(consumer: DshConsumer) -> bool:
    # a refusal surfaces as Inspect's "content_filter" stop reason, the value
    # the native react agent keys refusal handling on. Only the root's last
    # generation counts: a child's refusal reaches the root as a notice and
    # the run goes on.
    return consumer.last_stop_reason == "content_filter"


def _agent_env(
    *, dsh_home: str, home: str, port: int, env: dict[str, str] | None
) -> dict[str, str]:
    return {
        "DSH_HOME": dsh_home,
        "HOME": home,
        # dummy key for the bridge: no provider credential enters the sandbox
        "DEEPSEEK_API_KEY": "sk-inspect-bridge",
        "DEEPSEEK_BASE_URL": f"http://localhost:{port}",
        "DEEPSEEK_SEARCH_BASE_URL": f"http://localhost:{port}/v1",
        # no approval prompts; the sandbox is the boundary
        "DSH_PERMISSION_MODE": "danger-full-access",
        "DSH_TELEMETRY_MODE": "DISABLED",
    } | (env or {})


def _reasoning_unset(model: Model, config: GenerateConfig) -> bool:
    # dsh requests high effort natively, but the bridge drops its request-level
    # config; apply that default unless the eval sets reasoning itself (including
    # a raw `extra_body` reasoning field, which a provider would otherwise overwrite)
    return all(
        c.reasoning_effort is None
        and c.reasoning_tokens is None
        and "reasoning" not in (c.extra_body or {})
        for c in (config, model.config, active_generate_config())
    )


def _validate_mcp_server_name(name: str) -> None:
    if not _MCP_SERVER_NAME_RE.fullmatch(name):
        raise ValueError(
            f"MCP server {name!r}: deepseek_harness requires server names "
            "of 1-32 letters, digits, '_' or '-'."
        )


def _validate_mcp_server(server: MCPServerConfig) -> None:
    _validate_mcp_server_name(server.name)
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
    # dsh's bash tool runs as dsh's own user, so it can read whatever dsh
    # reads its MCP configuration from (a file, an inherited descriptor
    # through /proc/<pid>/fd, the environment). A header sent to dsh would be
    # a credential the model can print.
    if isinstance(server, MCPServerConfigHTTP) and (
        server.headers or _url_has_credentials(server.url)
    ):
        raise ValueError(
            f"MCP server {server.name!r}: deepseek_harness does not accept MCP "
            "servers with headers or with a user or password in their URL. "
            "dsh's own tools can read anything dsh can, so a credential (e.g. "
            "an Authorization token) placed in the sandbox would be readable "
            "by the model. Connect to the server on the host and pass its "
            "tools with bridged_tools instead."
        )


def _url_has_credentials(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return bool(parts.username or parts.password)


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


def _resolve_max_tokens(*, model: Model, context_window: int | None) -> int | None:
    # dsh starts compacting at min(0.8 * window, window - output cap - 65,536)
    # and assumes a 256,000-token output cap for a model without one: on a
    # 200k-window model that leaves no budget and dsh never compacts. Tell it
    # the model's real output limit when Inspect knows it.
    model_info = get_model_info(model)
    if model_info is None or not model_info.output_tokens:
        return None
    if context_window is not None and model_info.output_tokens >= context_window:
        return None
    return model_info.output_tokens


def _profile_patch(
    *,
    model: str,
    context_window: int | None,
    mcp_servers: Sequence[MCPServerConfig],
    max_tokens: int | None = None,
) -> str:
    # JSON is valid YAML and keeps every value safely quoted
    rows: list[dict[str, Any]] = [
        {
            "id": "agent-default-model",
            "config": {"provider": DSH_PROVIDER, "model": model},
        },
        # plain text-only entry: the built-in ones use DeepSeek-API-only
        # features the bridge doesn't speak
        {
            "id": "llm-deepseek",
            "config": {"models": [_catalog_model(model, context_window, max_tokens)]},
        },
        # web_search through DeepSeek's Messages search: a model request with
        # Anthropic's `web_search` server tool, sent to the bridge
        # (DEEPSEEK_SEARCH_BASE_URL), which serves it when the model's
        # provider has internal web search. dsh's default changed between
        # releases to DeepSeek's Search API, a path the bridge doesn't have.
        {
            "id": "web",
            "config": {"searchProvider": DSH_SEARCH_PROVIDER, "fetchProvider": "http"},
        },
        # goals keep a session working across automatic continuation rounds.
        # The one-shot headless runner doesn't wait for them: with a goal
        # active it exits non-zero as the next round starts, after a turn
        # that completed and answered. So the model isn't offered goals.
        {"id": "tool-goal", "disabled": True},
        {"id": "goal-round-driver", "disabled": True},
        # request fields and calls meant for DeepSeek's own API, and telemetry
        {"id": "session-log-deepseek", "disabled": True},
        {"id": "plugin-package-inventory-deepseek", "disabled": True},
        {"id": "session-title-llm", "disabled": True},
        {"id": "session-telemetry-otel", "disabled": True},
    ]

    mcp_rows = [_mcp_row(server) for server in mcp_servers]
    if mcp_rows:
        rows.append({"insert": mcp_rows})

    return json.dumps(rows, indent=2) + "\n"


def _catalog_model(
    model: str, context_window: int | None, max_tokens: int | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": model}
    if context_window is not None:
        entry["contextWindow"] = context_window
    if max_tokens is not None:
        entry["maxTokens"] = max_tokens
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
        # no headers: `_validate_mcp_server` rejects a caller's, and the
        # bridge's own endpoints have none
        config = {
            "serverName": server.name,
            "transport": "streamable-http",
            "url": server.url,
        }
    else:
        raise ValueError(f"Unsupported MCP server config type: {type(server).__name__}")
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
