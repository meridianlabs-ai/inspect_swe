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
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
    GenerateConfig,
    GenerateFilter,
    Model,
    ModelOutput,
)
from inspect_ai.model._model import GenerateInput
from inspect_ai.scorer import score
from inspect_ai.tool import (
    MCPServerConfig,
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
from .agentbinary import kimi_code_binary_source

# GenerateFilter is a union of Model-first and str-first callables; the bridge
# dispatches Model-first filters (as used here), so we call through this form.
_ModelFilter = Callable[
    [Model, list[ChatMessage], list[ToolInfo], "ToolChoice | None", GenerateConfig],
    Awaitable["ModelOutput | GenerateInput | None"],
]

KIMI_MAX_CONTEXT_SIZE = 262144

# Kimi Code's CLI injects a <system-reminder> nagging the model not to repeat an
# identical tool call. Re-polling the same read tool with the same arguments is
# legitimate (a build hasn't finished, a status is still pending), so the nag is
# a harness artifact rather than part of the task; strip it before the bridged
# model sees it.
_REPEAT_REMINDER_MARKER = "Repeated tool call detected"
_REPEAT_REMINDER_RE = re.compile(
    r"<system-reminder>(?:(?!</system-reminder>).)*?Repeated tool call detected.*?</system-reminder>",
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

    # resolve model
    bridge_model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills / attempts
    resolved_skills = read_skills(skills) if skills is not None else None
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    resolved_disallowed = list(disallowed_tools or [])

    async def execute(state: AgentState) -> AgentState:
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
            cleaned, changed = _strip_repeat_reminders(messages)
            if filter is not None:
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
                ),
            )
            if all_mcp_servers:
                await sbox.write_file(
                    f"{kimi_home}/mcp.json", _mcp_json(all_mcp_servers)
                )

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

            cmd = [kimi_binary, "--output-format", "text"]

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

        _dedupe_tool_call_ids(bridge.state.messages)
        return bridge.state

    return agent_with(execute, name=name, description=description)


def _strip_repeat_reminders(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], bool]:
    if not any(_REPEAT_REMINDER_MARKER in m.text for m in messages):
        return messages, False
    cleaned: list[ChatMessage] = []
    for message in messages:
        if _REPEAT_REMINDER_MARKER not in message.text:
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


def _dedupe_tool_call_ids(messages: list[ChatMessage]) -> None:
    # Kimi Code assigns tool_call_ids of the form `functions_<tool>_<n>` whose
    # counter resets, so the same id (e.g. `functions_Bash_0`) recurs across a
    # session. The bridge records those ids verbatim, so a transcript viewer that
    # threads assistant-call<->tool-result on tool_call_id alone collapses every
    # reuse of an id into one node — the linear chain renders as a spurious tree.
    # Rewrite ids in place to be globally unique while preserving pairing: the
    # k-th tool result for an original id maps (FIFO) to the k-th assistant call
    # bearing that id. Generation is already complete, so the model never sees
    # these ids.
    pending: dict[str, list[str]] = {}
    counter = 0
    for message in messages:
        if isinstance(message, ChatMessageAssistant) and message.tool_calls:
            for tool_call in message.tool_calls:
                original = tool_call.id
                unique_id = f"{original}__{counter}"
                counter += 1
                tool_call.id = unique_id
                pending.setdefault(original, []).append(unique_id)
        elif isinstance(message, ChatMessageTool) and message.tool_call_id is not None:
            queue = pending.get(message.tool_call_id)
            if queue:
                message.tool_call_id = queue.pop(0)


def _mcp_json(mcp_servers: Sequence[MCPServerConfig]) -> str:
    servers: dict[str, dict[str, object]] = {}
    for server in mcp_servers:
        if not isinstance(server, MCPServerConfigStdio):
            raise ValueError(
                f"kimi_code only supports stdio MCP servers, got {type(server).__name__}"
            )
        entry: dict[str, object] = {
            "command": server.command,
            "args": list(server.args),
        }
        if server.env:
            entry["env"] = dict(server.env)
        servers[server.name] = entry
    return json.dumps({"mcpServers": servers}, indent=2)


def _config_toml(
    *,
    port: int,
    mcp_servers: Sequence[MCPServerConfig],
    disallowed_tools: Sequence[str],
    extra_skill_dirs: Sequence[str],
) -> str:
    lines = ['default_model = "bridge"']
    if extra_skill_dirs:
        dirs = ", ".join(f'"{d}"' for d in extra_skill_dirs)
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
        'model = "inspect"',
        f"max_context_size = {KIMI_MAX_CONTEXT_SIZE}",
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
            f'pattern = "mcp__{server.name}__*"',
        ]
    for tool in disallowed_tools:
        lines += [
            "",
            "[[permission.rules]]",
            'decision = "deny"',
            f'pattern = "{tool}"',
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
