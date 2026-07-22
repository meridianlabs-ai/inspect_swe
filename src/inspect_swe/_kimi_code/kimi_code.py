import inspect
import json
import re
import shlex
from pathlib import Path
from textwrap import dedent
from typing import Awaitable, Callable, Literal, Sequence, cast

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
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
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
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from .._util.sandbox import resolve_agent_cwd
from .._util.toml import _format_value
from .agentbinary import kimi_code_binary_source

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


# Kimi Code's CLI appends escalating <system-reminder> nags to tool results when
# it detects an identical tool call repeated N times in a row (tool-dedup: tier 1
# at streak >= 3, tier 2 at >= 5, tier 3 "respond now" at >= 8, forced turn stop
# at >= 12). Re-polling the same read tool with the same arguments is legitimate
# (a build hasn't finished, a status is still pending), so tiers 1-2 are harness
# artifacts rather than part of the task; strip them before the bridged model
# sees them. Tier 3 is deliberately NOT stripped: kimi force-stops the turn at
# streak 12 regardless of what the model saw, so hiding the "write your final
# response now" instruction would only turn a graceful wrap-up into an abrupt
# cutoff. Markers cover both wordings (rewritten in kimi-code 0.23.4, PR #1518).
_REPEAT_REMINDER_MARKERS = (
    # kimi-code >= 0.23.4
    "The same tool call has been repeated",
    "The same tool call has now been issued",
    # kimi-code < 0.23.4
    "You are repeating the exact same tool call",
    "Repeated tool call detected",
)
_REPEAT_REMINDER_RE = re.compile(
    r"<system-reminder>(?:(?!</system-reminder>).)*?(?:"
    + "|".join(re.escape(marker) for marker in _REPEAT_REMINDER_MARKERS)
    + r").*?</system-reminder>",
    re.DOTALL,
)


@agent
def kimi_code(
    name: str = "Kimi Code",
    description: str = dedent("""
        MoonshotAI Kimi Code terminal coding agent — reads and edits code, runs
        shell commands, searches files, and iterates based on feedback.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    max_context_size: int | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    disallowed_tools: Sequence[str] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool = False,
) -> Agent:
    """Kimi Code agent.

    Agent that uses [Kimi Code](https://github.com/MoonshotAI/kimi-code)
    running in a sandbox with Inspect model bridging.

    Kimi's model calls are routed through a generated `config.toml` whose
    `openai`-protocol provider points at the Inspect bridge, so the returned
    transcript is the bridge's record of those calls (Kimi CLI stdout is not
    parsed). MCP servers are wired natively via `mcp.json`.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description
        system_prompt: Additional system prompt to append
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP
        centaur: Run in 'centaur' mode, which makes Kimi Code available to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        max_context_size: Context window to configure for Kimi. Defaults to the
            bridged model's context length from Inspect model metadata; required
            when that metadata is unavailable.
        model_aliases: Optional mapping of model names to Model instances or model name strings.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        disallowed_tools: Tool names to deny via Kimi permission rules
        cwd: Working directory to run kimi within
        env: Environment variables to set for kimi
        user: User to execute kimi with
        sandbox: Optional sandbox environment name
        version: Version of kimi to use. One of:
            - "auto": Use any available version in sandbox, otherwise download latest
            - "sandbox": Use sandbox version (raises RuntimeError if not available)
            - "stable"/"latest": Download and use the latest version
            - "x.x.x": Download and use a specific version
        debug: Trace all debug output.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve the name Kimi sends and the bridge fallback independently: the
    # bridge matches aliases against the raw requested name before stripping
    # the inspect/ prefix from its fallback.
    requested_model = model or "inspect"
    bridge_model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills / attempts
    resolved_skills = read_skills(skills) if skills is not None else None
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    resolved_disallowed = list(disallowed_tools or [])
    filter_is_legacy = filter is not None and _is_legacy_str_filter(filter)

    async def execute(state: AgentState) -> AgentState:
        resolved_model = _resolve_model(model=model, model_aliases=model_aliases)
        resolved_max_context_size = _resolve_max_context_size(
            model=resolved_model, max_context_size=max_context_size
        )

        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "kimi_code_model_port"
        port = store().get(MODEL_PORT, 3100) + 1
        store().set(MODEL_PORT, port)

        async def combined_filter(
            model: Model,
            messages: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice | None,
            config: GenerateConfig,
        ) -> ModelOutput | GenerateInput | None:
            # in place (not via GenerateInput) so the rewritten ids persist in
            # both the recorded ModelEvent input and bridge.state.messages
            _dedupe_tool_call_ids(messages)
            cleaned, changed = _strip_repeat_reminders(messages)
            if filter is not None:
                if filter_is_legacy:
                    result = await cast(_StrFilter, filter)(
                        model.name, cleaned, tools, tool_choice, config
                    )
                else:
                    result = await cast(_ModelFilter, filter)(
                        model, cleaned, tools, tool_choice, config
                    )
                if result is not None:
                    return result
            if changed:
                return GenerateInput(
                    input=cleaned, tools=tools, tool_choice=tool_choice, config=config
                )
            return None

        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases=model_aliases,
            filter=combined_filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # install kimi in sandbox
            kimi_binary = await ensure_agent_binary_installed(
                kimi_code_binary_source(), version, user, sbox
            )

            # combine static mcp configs with bridged tools' mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            # detect sandbox home directory
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"
            kimi_home = f"{sandbox_home}/.kimi-code"
            await sbox.exec(["mkdir", "-p", kimi_home], user=user)

            # install skills (kimi discovers them via extra_skill_dirs)
            skills_dir = f"{kimi_home}/skills"
            if resolved_skills is not None:
                await install_skills(resolved_skills, sbox, user, skills_dir)

            # write config.toml (route the bridge provider) and mcp.json
            await sbox.write_file(
                f"{kimi_home}/config.toml",
                _config_toml(
                    port=bridge.port,
                    mcp_servers=all_mcp_servers,
                    disallowed_tools=resolved_disallowed,
                    extra_skill_dirs=[skills_dir]
                    if resolved_skills is not None
                    else [],
                    model=requested_model,
                    max_context_size=resolved_max_context_size,
                ),
            )
            await sbox.write_file(f"{kimi_home}/mcp.json", _mcp_json(all_mcp_servers))

            # build the prompt (kimi -p takes a single message and has no
            # separate system-prompt flag, so we prepend system messages)
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)
            prompt, has_assistant_response = build_user_prompt(state.messages)
            if system_messages:
                prompt = "\n\n".join(system_messages) + "\n\n" + prompt

            cmd = [kimi_binary]
            # kimi rejects --output-format outside -p (prompt) mode, so keep the
            # centaur alias (interactive) free of it
            if centaur is False:
                cmd.extend(["--output-format", "text"])

            agent_env = {
                "KIMI_CODE_HOME": kimi_home,
                "HOME": sandbox_home,
            } | (env or {})

            if centaur:
                await _run_kimi_code_centaur(
                    options=centaur,
                    kimi_cmd=cmd,
                    agent_env=agent_env,
                    state=state,
                )
            else:
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    agent_cmd = cmd.copy()

                    # continue previous conversation between attempts (or when
                    # the inbound state already carries an assistant turn)
                    if has_assistant_response or attempt_count > 0:
                        agent_cmd.append("--continue")
                    agent_cmd += ["-p", agent_prompt]

                    result = await sbox.exec_remote(
                        cmd=["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
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

                    if not result.success:
                        raise RuntimeError(
                            f"Error executing kimi code agent {result.returncode}:\n"
                            f"stdout: {result.stdout}\nstderr: {result.stderr}"
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
                    debug_output.insert(0, "Kimi Code Debug Output:")
                    trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def _contains_repeat_reminder(text: str) -> bool:
    return any(marker in text for marker in _REPEAT_REMINDER_MARKERS)


def _strip_repeat_reminders(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], bool]:
    if not any(_contains_repeat_reminder(m.text) for m in messages):
        return messages, False
    cleaned: list[ChatMessage] = []
    for message in messages:
        if not _contains_repeat_reminder(message.text):
            cleaned.append(message)
            continue
        if isinstance(message.content, str):
            new_text = _REPEAT_REMINDER_RE.sub("", message.content).strip()
            # Dropping a tool/assistant message would orphan a tool-call pairing;
            # only user/system messages are safe to drop when they become empty.
            if not new_text and isinstance(
                message, (ChatMessageUser, ChatMessageSystem)
            ):
                continue
            cleaned.append(message.model_copy(update={"content": new_text}))
        else:
            new_content = [
                item.model_copy(
                    update={"text": _REPEAT_REMINDER_RE.sub("", item.text).strip()}
                )
                if isinstance(item, ContentText)
                else item
                for item in message.content
            ]
            cleaned.append(message.model_copy(update={"content": new_content}))
    return cleaned, True


_DEDUPE_SUFFIX_RE = re.compile(r"__u\d+$")


def _dedupe_tool_call_ids(messages: list[ChatMessage]) -> None:
    # Kimi Code assigns tool_call_ids of the form `functions_<tool>_<n>` whose
    # counter resets, so the same id (e.g. `functions_Bash_0`) recurs within a
    # session — and, in a long enough session, within a single request, which a
    # provider can reject or mispair. Transcript viewers that thread
    # assistant-call<->tool-result on tool_call_id alone also collapse every
    # reuse of an id into one node, rendering the linear chain as a spurious
    # tree. Rewrite ids in place to be unique within the request while
    # preserving pairing: the k-th tool result for an original id maps (FIFO)
    # to the k-th assistant call bearing that id. Kimi resends the full history
    # each request, so the rewrite is deterministic across requests; stripping
    # any prior `__u<n>` suffix keeps it idempotent across bridge retries.
    pending: dict[str, list[str]] = {}
    counter = 0
    for message in messages:
        if isinstance(message, ChatMessageAssistant) and message.tool_calls:
            for tool_call in message.tool_calls:
                original = tool_call.id
                base = _DEDUPE_SUFFIX_RE.sub("", original)
                unique_id = f"{base}__u{counter}"
                counter += 1
                tool_call.id = unique_id
                pending.setdefault(original, []).append(unique_id)
        elif isinstance(message, ChatMessageTool) and message.tool_call_id is not None:
            queue = pending.get(message.tool_call_id)
            if queue:
                message.tool_call_id = queue.pop(0)


def _mcp_json(mcp_servers: Sequence[MCPServerConfig]) -> str:
    # kimi's mcp.json accepts stdio, http, and sse servers, discriminated by a
    # `transport` field that maps directly from the inspect config types.
    servers: dict[str, dict[str, object]] = {}
    for server in mcp_servers:
        entry: dict[str, object]
        if isinstance(server, MCPServerConfigStdio):
            entry = {
                "transport": "stdio",
                "command": server.command,
                "args": list(server.args),
            }
            if server.env:
                entry["env"] = dict(server.env)
            if server.cwd is not None:
                entry["cwd"] = str(server.cwd)
        elif isinstance(server, MCPServerConfigHTTP):
            entry = {"transport": server.type, "url": server.url}
            if server.headers:
                entry["headers"] = dict(server.headers)
        else:
            raise ValueError(
                f"Unsupported MCP server config type: {type(server).__name__}"
            )
        if server.tools != "all":
            entry["enabledTools"] = list(server.tools)
        servers[server.name] = entry
    return json.dumps({"mcpServers": servers}, indent=2)


def _resolve_model(
    *, model: str | None, model_aliases: dict[str, str | Model] | None
) -> Model:
    requested_model = model or "inspect"
    bridge_model = f"inspect/{model}" if model is not None else "inspect"
    return resolve_inspect_model(requested_model, model_aliases, bridge_model)


def _resolve_max_context_size(*, model: Model, max_context_size: int | None) -> int:
    if max_context_size is not None:
        if max_context_size <= 0:
            raise ValueError("max_context_size must be a positive integer")
        return max_context_size

    model_info = get_model_info(model)
    if model_info is None or model_info.context_length is None:
        raise ValueError(
            f"Context length metadata is unavailable for model "
            f"{model.name!r}; pass max_context_size explicitly."
        )
    if model_info.context_length <= 0:
        raise ValueError(
            f"Context length metadata for model {model.name!r} must be "
            "positive; pass max_context_size explicitly."
        )
    return model_info.context_length


def _config_toml(
    *,
    port: int,
    mcp_servers: Sequence[MCPServerConfig],
    disallowed_tools: Sequence[str],
    extra_skill_dirs: Sequence[str],
    model: str,
    max_context_size: int,
) -> str:
    lines = ['default_model = "bridge"', "telemetry = false"]
    if extra_skill_dirs:
        dirs = ", ".join(_format_value(d) for d in extra_skill_dirs)
        lines.append(f"extra_skill_dirs = [{dirs}]")
    lines += [
        "",
        "[providers.bridge]",
        'type = "openai"',
        f'base_url = "http://localhost:{port}/v1"',
        'api_key = "sk-none"',
        "",
        "[models.bridge]",
        'provider = "bridge"',
        f"model = {_format_value(model)}",
        f"max_context_size = {max_context_size}",
    ]
    # -p (prompt) mode auto-approves regular tool calls under the auto policy;
    # make MCP tools explicitly allowed and honor disallowed_tools as static
    # deny rules (deny wins). MCP tools are named mcp__<server>__<tool>; rules
    # match MCP/custom tools by name.
    for server in mcp_servers:
        lines += [
            "",
            "[[permission.rules]]",
            'decision = "allow"',
            f"pattern = {_format_value(f'mcp__{server.name}__*')}",
        ]
    for tool in disallowed_tools:
        lines += [
            "",
            "[[permission.rules]]",
            'decision = "deny"',
            f"pattern = {_format_value(tool)}",
        ]
    return "\n".join(lines) + "\n"


async def _run_kimi_code_centaur(
    options: CentaurOptions,
    kimi_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    instructions = (
        "Kimi Code:\n\n"
        " - You may also use Kimi Code via the 'kimi' command.\n"
        " - Use 'kimi --continue' if you need to resume a previous kimi session."
    )
    # only export vars needed for the kimi alias, not HOME which would break
    # human_cli.
    centaur_env = {k: v for k, v in agent_env.items() if k != "HOME"}
    agent_env_vars = [f'export {k}="{v}"' for k, v in centaur_env.items()]
    alias_cmd = shlex.join(kimi_cmd)
    alias_cmd = "alias kimi='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(agent_env_vars + ["", alias_cmd])

    await run_centaur(options, instructions, bashrc, state)
