import json
import re
import shlex
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Literal, Mapping, Sequence

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
    ChatMessageSystem,
    GenerateFilter,
    Model,
)
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP, MCPServerConfigStdio
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from .._util._async import is_callable_coroutine
from .._util.agentbinary import ensure_agent_binary_installed
from .._util.centaur import CentaurOptions, run_centaur
from .._util.mcp_ready import DEFAULT_MCP_READY_TIMEOUT, wait_for_mcp_endpoints
from .._util.messages import build_user_prompt
from .._util.path import join_path
from .._util.sandbox import resolve_agent_cwd
from .._util.trace import trace
from .agentbinary import antigravity_cli_binary_source

# Reasoning efforts the CLI accepts. Narrower than most agents' scales: `agy
# --effort` takes exactly these three, and REQUIRES one for the Gemini 3.6/3.7
# Flash families (it exits 1 with "requires --effort" otherwise), because it
# resolves `<model>` + `<effort>` into one catalog id (`gemini-3.6-flash-low`).
AntigravityEffort = Literal["low", "medium", "high"]

# Where the CLI keeps its persistent settings and its global MCP registry. These
# are two different files under two different directories -- settings live with
# the CLI's own state, MCP servers in the shared `~/.gemini/config` tree.
_SETTINGS_DIR = ".gemini/antigravity-cli"
_MCP_CONFIG_DIR = ".gemini/config"
_ONBOARDING_CACHE_DIR = f"{_SETTINGS_DIR}/cache"
_ONBOARDING_FILE = f"{_ONBOARDING_CACHE_DIR}/onboarding.json"

# The CLI states the identity of its own conversation in every primary
# request: official 1.1.27/1.1.28 requests carry exactly one system
# `<user_information>` block containing `Conversation ID: <UUID>`, while the
# title-generation conversation the CLI opens alongside carries no such block.
# That declaration is what separates the task's conversation from the CLI's own
# auxiliary traffic -- not title text, and not whether tools were offered.
_USER_INFORMATION_BLOCK = re.compile(
    r"<user_information>(.*?)</user_information>", re.DOTALL
)
_CONVERSATION_ID_LINE = re.compile(r"^[ \t]*Conversation ID:[ \t]*(\S+)[ \t]*$", re.M)
_USER_INFORMATION_DELIMITER = re.compile(r"</?user_information>")


def _framed_blocks(text: str) -> list[str]:
    """The bodies of the complete `<user_information>` blocks in one message.

    The delimiters are counted and ordered before any content is read. Pairing
    an opener with a closer cannot by itself tell an unterminated declaration
    from an absent one, and absence is the positive signal for the CLI's
    auxiliary title request -- so an unbalanced or out-of-order delimiter has
    to raise rather than resolve to a block, or to nothing.
    """
    delimiters = _USER_INFORMATION_DELIMITER.findall(text)
    alternating = [
        "<user_information>" if index % 2 == 0 else "</user_information>"
        for index in range(len(delimiters))
    ]
    if len(delimiters) % 2 or delimiters != alternating:
        raise ValueError(
            f"antigravity cli framed its <user_information> with {len(delimiters)} "
            "delimiters that do not open and close complete blocks"
        )
    return _USER_INFORMATION_BLOCK.findall(text)


def _validated_conversation_id(value: str) -> str:
    # Returned verbatim rather than normalised: this string is compared against
    # the id the CLI prints in its own result, and against the id a later
    # attempt resumes by.
    try:
        uuid.UUID(value)
    except ValueError as ex:
        raise ValueError(
            f"antigravity cli declared a malformed conversation id {value!r}"
        ) from ex
    return value


def _native_conversation_id(messages: Sequence[ChatMessage]) -> str | None:
    """Read the conversation id the CLI declared for a request.

    Args:
        messages: One request, as the model received it.

    Returns:
        The UUID from the request's single system `<user_information>` block,
        or `None` when no system message carries such a block -- which is how
        the CLI's auxiliary title request presents.

    Raises:
        ValueError: The declaration cannot be read unambiguously -- delimiters
            that do not frame complete blocks, more than one block, a block
            carrying no id, or an id that is not a UUID. Guessing would bind
            the run to an identity the CLI's own result can never match,
            turning a parse fault into an unattributable score.
    """
    blocks = [
        block
        for message in messages
        if isinstance(message, ChatMessageSystem)
        for block in _framed_blocks(message.text)
    ]
    if not blocks:
        return None
    if len(blocks) > 1:
        raise ValueError(
            f"antigravity cli declared {len(blocks)} <user_information> blocks "
            "in one request; expected exactly one carrying its conversation id"
        )

    declared = _CONVERSATION_ID_LINE.findall(blocks[0])
    if len(declared) != 1:
        raise ValueError(
            "antigravity cli declared a <user_information> block carrying "
            f"{len(declared)} conversation ids; expected exactly one"
        )
    return _validated_conversation_id(declared[0])


class _NativeConversation:
    """Whose traffic is the canonical `AgentState`, and which id to resume.

    `accept` is handed to the bridge as its `state_filter`, so its signature is
    the bridge's: one request's messages in, a verdict out. Excluding a request
    changes nothing else about it -- it still generates, still emits its own
    producer ids and events, and still counts its usage. All exclusion does is
    keep the CLI's auxiliary title conversation from displacing the task's.

    Args:
        unattended: Keep canonical state on ONE conversation. In centaur mode
            the human may start or resume several, and all of them are theirs.
        bound_id: A conversation this invocation is resuming, bound before any
            request is filtered. Without it the first native conversation to
            arrive takes the binding, which on a resume could be a fresh one
            the CLI substituted rather than the one being continued.
    """

    def __init__(self, *, unattended: bool, bound_id: str | None = None) -> None:
        self._unattended = unattended
        self.bound_id: str | None = bound_id

    def accept(self, messages: Sequence[ChatMessage]) -> bool:
        conversation_id = _native_conversation_id(messages)
        if conversation_id is None:
            # The CLI's own auxiliary request. It must not become canonical,
            # and it must not consume the binding the primary request needs.
            return False

        if self.bound_id is None:
            self.bound_id = conversation_id
        if not self._unattended:
            # A human may start or resume several conversations in one session,
            # and every one of them is theirs. The existing accumulation
            # contract is what preserves them.
            return True
        # One unattended invocation, one canonical conversation.
        return self.bound_id == conversation_id


@agent
def antigravity_cli(
    name: str = "Antigravity CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    mcp_ready_timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    agy_model: str = "gemini-3.6-flash",
    effort: AntigravityEffort | None = "low",
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
) -> Agent:
    """Antigravity CLI agent.

    Agent that uses Google's [Antigravity CLI](https://antigravity.google/docs/cli/overview)
    (`agy`) running in a sandbox with Inspect model bridging.

    Model calls are bridged the same way `gemini_cli`'s are: the CLI's direct
    Gemini API route (`modelProvider: "gemini"`, added in `agy` 1.1.13) is
    selected and `GOOGLE_GEMINI_BASE_URL` is pointed at the loopback
    `sandbox_agent_bridge`, so generation never leaves the sandbox and no Google
    sign-in happens. `GEMINI_API_KEY` is set to a placeholder purely to satisfy
    the CLI's credential check -- the bridge does not read it.

    This is a different agent from `antigravity`, which runs the
    `google-antigravity` **SDK** rather than the shipped CLI.

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
        centaur: Run in 'centaur' mode, which makes the Antigravity CLI available to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g. wrapped Agents) instead of standard models.
        agy_model: Model name to pass to the CLI. The actual model calls still go
            through the Inspect bridge; this selects the CLI's own client-side
            model configuration (context window, effort handling, tool schema).
        effort: Reasoning effort to pass to the CLI. Passed explicitly by default:
            the Gemini 3.6/3.7 Flash families that `agy` defaults to require it,
            and leaving it implicit would let the CLI's own default decide what
            an eval measured. Pass `None` for a model that rejects the flag.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run the CLI within
        env: Environment variables to set for the CLI
        user: User to execute the CLI with
        sandbox: Optional sandbox environment name
        version: Version of the Antigravity CLI to use. One of:
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
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "antigravity_cli_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        # On re-entry an unattended run continues the conversation the previous
        # invocation ran, and its id is in the canonical system block that
        # invocation tracked -- no separate record of it is kept anywhere.
        # Seeded here, BEFORE the bridge starts filtering, so the CLI's own
        # first request cannot rebind canonical state to a fresh conversation.
        # A fresh run has no such block and binds whatever the CLI declares.
        #
        # Only unattended runs resume by id, so only they read the incoming
        # state as a declaration. A centaur session accumulates a conversation
        # per thing the human started or resumed, and reading that as one
        # declaration would make its own accumulation contract ambiguous --
        # centaur filters per request instead, which needs no prior id.
        prior_conversation_id = (
            _native_conversation_id(state.messages) if centaur is False else None
        )
        conversation = _NativeConversation(
            unattended=centaur is False, bound_id=prior_conversation_id
        )

        async with sandbox_agent_bridge(
            state,
            model=model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
            state_filter=conversation.accept,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # resolve working directory (home dir if sandbox default is '/')
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # install the CLI in the sandbox
            agy_binary = await ensure_agent_binary_installed(
                antigravity_cli_binary_source(), version, user, sbox
            )

            # detect sandbox home directory (the CLI resolves both its settings
            # and its global MCP registry relative to $HOME)
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"

            # install skills
            if resolved_skills is not None:
                skills_dir = join_path(agent_cwd, ".agents/skills")
                await install_skills(resolved_skills, sbox, user, skills_dir)

            # mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            settings_dir = join_path(sandbox_home, _SETTINGS_DIR)
            onboarding_cache_dir = join_path(sandbox_home, _ONBOARDING_CACHE_DIR)
            mcp_config_dir = join_path(sandbox_home, _MCP_CONFIG_DIR)
            await sbox.exec(
                ["mkdir", "-p", settings_dir, onboarding_cache_dir, mcp_config_dir],
                user=user,
            )
            await sbox.write_file(
                join_path(settings_dir, "settings.json"),
                _workspace_settings(unattended=centaur is False, workspace=agent_cwd),
            )
            await sbox.write_file(
                join_path(sandbox_home, _ONBOARDING_FILE),
                _completed_onboarding(),
            )
            await sbox.write_file(
                join_path(mcp_config_dir, "mcp_config.json"),
                build_antigravity_mcp_config(
                    all_mcp_servers, eager_tools=bridge.bridged_tools
                ),
            )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            prompt, has_assistant_response = build_user_prompt(state.messages)

            # A caller can hand this agent assistant turns the CLI never
            # produced -- a synthesised or replayed transcript -- and then
            # there is nothing to resume. Both silent options are wrong:
            # `--continue` resumes whatever conversation the CLI has cached,
            # which may be unrelated to this run, and starting fresh drops the
            # context those turns carry, because the prompt built above is only
            # the user turns AFTER the last assistant message.
            if centaur is False and has_assistant_response:
                if prior_conversation_id is None:
                    raise RuntimeError(
                        "antigravity cli cannot resume: the conversation handed "
                        "to this agent carries assistant turns but no native "
                        "conversation id, so there is no conversation to "
                        "continue"
                    )

            # Prepend the system prompt to the user prompt: the CLI has no
            # separate --system-prompt flag (same as gemini_cli).
            if system_messages:
                combined_system = "\n\n".join(system_messages)
                prompt = f"{combined_system}\n\n{prompt}"

            cmd = [
                agy_binary,
                "--model",
                agy_model,
                # Omitted only when explicitly disabled: models outside the
                # 3.6/3.7 Flash families reject --effort as not adjustable.
                *(["--effort", effort] if effort is not None else []),
            ]

            # Headless prompts stay literal, tool calls proceed without
            # prompting, and the run's own result comes back as JSON to be
            # verified before it is scored. All three are print-mode concerns:
            # the human at the terminal reads the CLI's ordinary output, and
            # there is no result envelope for us to parse on their behalf.
            if centaur is False:
                cmd.extend(
                    [
                        "--disable-slash-commands",
                        "--dangerously-skip-permissions",
                        "--output-format",
                        "json",
                    ]
                )

            agent_env = build_antigravity_agent_env(
                bridge_port=bridge.port, sandbox_home=sandbox_home, env=env
            )

            # Gate the launch on the bridged MCP endpoints actually serving
            # tools: the CLI blocks its first turn on MCP connect for headless
            # runs, but only after the endpoint answers `tools/list`.
            _http_mcp_configs = [
                c
                for c in bridge.mcp_server_configs
                if isinstance(c, MCPServerConfigHTTP)
            ]
            if _http_mcp_configs:
                await wait_for_mcp_endpoints(
                    _http_mcp_configs,
                    bridge,
                    sandbox=sandbox,
                    timeout=mcp_ready_timeout,
                    required=True,
                )

            if centaur:
                await _run_antigravity_cli_centaur(
                    options=centaur,
                    agy_cmd=cmd,
                    agent_env=agent_env,
                    state=bridge.state,
                    user=user,
                    sandbox=sandbox,
                    cwd=agent_cwd,
                )
            else:
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    agent_cmd = cmd.copy()

                    # Resume by id, never with `--continue`: the id names the
                    # conversation this run is continuing or has already
                    # verified, rather than whatever the CLI would select on
                    # our behalf. A first attempt with nothing to resume names
                    # nothing -- preassigning an id the CLI does not know makes
                    # it warn and create a different conversation.
                    if has_assistant_response or attempt_count > 0:
                        if conversation.bound_id is None:
                            raise RuntimeError(
                                "antigravity cli never declared a conversation "
                                "id, so there is nothing to resume"
                            )
                        agent_cmd.extend(["--conversation", conversation.bound_id])

                    agent_cmd.extend(["--print", agent_prompt])

                    if _http_mcp_configs and attempt_count > 0:
                        await wait_for_mcp_endpoints(
                            _http_mcp_configs,
                            bridge,
                            sandbox=sandbox,
                            timeout=mcp_ready_timeout,
                            required=True,
                        )
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
                            f"Error executing antigravity cli agent {result.returncode}: "
                            f"{_clean_antigravity_error(result.stdout, result.stderr)}"
                        )

                    # Exit zero says nothing about whether the conversation
                    # succeeded, or about which conversation ran: the CLI
                    # reports a failed conversation while exiting zero. Both
                    # are checked here, before this attempt can be scored or
                    # its state returned.
                    _verify_native_result(result.stdout, conversation.bound_id)

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
                    debug_output.insert(0, "Antigravity CLI Debug Output:")
                    trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def build_antigravity_settings(*, unattended: bool = True) -> str:
    """Build Antigravity CLI settings.json content.

    `modelProvider` is the load-bearing key: it selects the direct Gemini API
    route (`GEMINI_API_KEY` + `GOOGLE_GEMINI_BASE_URL`) instead of the OAuth
    sign-in the CLI otherwise blocks on. Everything else here removes a way for
    a headless run to stall or to reach outside the sandbox.

    Args:
        unattended: Write the policies that keep a headless run from stalling
            on an approval it cannot answer. Withheld in centaur mode, where
            the human at the terminal is the approver.
    """
    settings: dict[str, Any] = {
        "modelProvider": "gemini",
        # The CLI's own terminal sandbox. Isolation is the Inspect sandbox's job;
        # the CLI's needs unprivileged user namespaces, which container policy
        # typically denies.
        #
        # The four booleans below MUST be JSON booleans, not the "on"/"off"
        # strings the settings documentation uses for them. A string is dropped
        # on load with no error and no rewrite of the file, so `settings.json`
        # keeps saying "off" while `/config` reports the default -- which is how
        # the first cut of this shipped with telemetry still enabled.
        "enableTerminalSandbox": False,
        # No usage statistics or crash reports off-box from an eval run.
        "enableTelemetry": False,
        "showTips": False,
        "showFeedbackSurvey": False,
        # Sequential stdout rather than the alternate screen buffer: the run is
        # captured as text, and alt-screen escape sequences corrupt it.
        "altScreenMode": "never",
    }
    if unattended:
        # Nothing is watching this run, so neither policy can be left to
        # prompt. `--dangerously-skip-permissions` already covers tool
        # permissions for a print-mode run -- the CLI says as much when it
        # auto-denies one ("Settings allow-rules do not apply; re-run with
        # --dangerously-skip-permissions to auto-approve all tools") -- but
        # the persisted artifact-review policy is still honored, and its
        # default ("asks-for-review") is a prompt nobody is there to answer.
        #
        # Centaur mode gets neither. That flag is withheld there so the human
        # approves, exactly as gemini_cli withholds --yolo, and a settings
        # file that auto-approves everything would quietly take that back.
        settings["toolPermission"] = "always-proceed"
        settings["artifactReviewPolicy"] = "always-proceed"
    return json.dumps(settings, indent=2)


def build_antigravity_agent_env(
    *,
    bridge_port: int,
    sandbox_home: str,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for the Antigravity CLI subprocess.

    Args:
        bridge_port: Port of the in-sandbox bridge the agent's model calls
            go to.
        sandbox_home: Detected sandbox $HOME (the CLI resolves both its
            settings and its global MCP registry relative to this).
        env: Caller overrides. Applied for everything except the two keys
            that define the credential boundary itself -- caller values win
            on conflict there, same contract as `claude_code_agent_env` --
            but `GOOGLE_GEMINI_BASE_URL` and `GEMINI_API_KEY` are forced
            unconditionally. Unlike `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`
            in `claude_code_agent_env`, where AGENTS.md documents caller
            override of the model-provider route/credential as an accepted,
            deliberate escape hatch, a caller here (e.g. one passing
            `env=os.environ.copy()`) must not be able to silently redirect
            generation off the bridge onto Google's real endpoint, or inject
            a real Google credential into the sandbox: those two keys ARE the
            boundary, not a convenience default.

    Returns:
        The merged environment. `GOOGLE_GEMINI_BASE_URL` and `GEMINI_API_KEY`
        always point at the bridge; every other key follows the caller's
        `env` when supplied.
    """
    return (
        {
            # The CLI self-updates from its auto-updater service on startup. The
            # actual native switch is the literal string "true"; "1" is ignored
            # and lets a pinned binary replace itself.
            "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
            # Keep logo art out of captured transcripts.
            "AGY_CLI_HIDE_LOGO": "1",
            "HOME": sandbox_home,
            # No PATH: the CLI is a self-contained binary launched by absolute
            # path, and overriding PATH would hide the image's toolchain from
            # every command the agent runs.
        }
        | (env or {})
        | {
            # The CLI's direct-Gemini-API route, pointed at the bridge. Both
            # halves are required: the base URL alone leaves the CLI on its
            # sign-in path, and the key alone leaves generation on Google's
            # endpoint. Applied AFTER the caller's `env` so neither key can be
            # silently overridden -- these two ARE the credential boundary.
            "GOOGLE_GEMINI_BASE_URL": f"http://localhost:{bridge_port}",
            "GEMINI_API_KEY": "api-key",
        }
    )


def _workspace_settings(*, unattended: bool, workspace: str) -> str:
    """Build settings for exactly the workspace the CLI process will use."""
    settings: dict[str, Any] = json.loads(
        build_antigravity_settings(unattended=unattended)
    )
    settings["trustedWorkspaces"] = [workspace]
    return json.dumps(settings, indent=2)


def _completed_onboarding() -> str:
    """Native onboarding state that keeps a fresh sandbox at its CLI prompt."""
    return json.dumps(
        {
            "consumerOnboardingComplete": True,
            "enterpriseOnboardingComplete": False,
            "onboardingComplete": True,
        },
        indent=2,
    )


def build_antigravity_mcp_config(
    mcp_servers: Sequence[MCPServerConfig],
    eager_tools: Mapping[str, Iterable[str]],
) -> str:
    """Build Antigravity CLI mcp_config.json content.

    The CLI's remote-server schema names the endpoint `serverUrl`; the `url` and
    `httpUrl` spellings other agents accept are silently ignored, which presents
    as a server that is configured but never connects.

    MCP tools are otherwise loaded LAZILY: the CLI declares one native
    dispatcher (`call_mcp_tool`) and routes every MCP tool through it, so a
    bridged Inspect tool is never a tool the model can call by name. Listing a
    tool under the server's `tools` map with `eager: true` registers it as a
    native tool instead. The key is the name the server itself serves -- the
    same namespace as the CLI's `disabledTools` and its `mcp(server/tool)`
    permission syntax; the model-facing `mcp_<server>_<tool>` spelling is
    derived by the CLI, so writing that here would mark nothing.

    Args:
        mcp_servers: Servers to write into the registry.
        eager_tools: Bridged tool names by server name (the bridge's
            `bridged_tools` registry). Only these servers are marked; anything
            the caller configured itself keeps the CLI's native lazy loading.

    Raises:
        ValueError: If a server carries transport credentials -- HTTP
            `headers` or stdio `env` -- that would be written verbatim into
            `$HOME/.gemini/config/mcp_config.json`. That file sits inside the
            sandboxed CLI's own $HOME, where the evaluated agent's own tools
            can read it just as readily as the CLI does, so a real credential
            placed there crosses the sandbox credential boundary (AGENTS.md,
            "Agent Guardrails"). Bridge-owned servers (built from
            `bridged_tools`) never carry credentials, so this only affects
            servers the caller passes directly via `mcp_servers`; route an
            authenticated server through `bridged_tools` instead.
    """
    servers: dict[str, Any] = {}
    for server in mcp_servers:
        if isinstance(server, MCPServerConfigHTTP) and server.headers:
            raise ValueError(
                f"MCP server {server.name!r} passed to `mcp_servers` carries "
                "HTTP headers (e.g. an Authorization token). Antigravity CLI "
                "persists this registry verbatim to "
                "$HOME/.gemini/config/mcp_config.json inside the sandbox, "
                "which the evaluated CLI -- and any of its tools -- can read, "
                "so a real credential placed here crosses the credential "
                "boundary (see AGENTS.md, 'Agent Guardrails'). Expose an "
                "authenticated server via `bridged_tools` instead, which "
                "keeps real credentials out of the sandbox entirely."
            )
        if isinstance(server, MCPServerConfigStdio) and server.env:
            raise ValueError(
                f"MCP server {server.name!r} passed to `mcp_servers` carries "
                "an `env` map that may hold real credentials. Antigravity CLI "
                "persists this registry verbatim to "
                "$HOME/.gemini/config/mcp_config.json inside the sandbox, "
                "which the evaluated CLI -- and any of its tools -- can read, "
                "so a real credential placed here crosses the credential "
                "boundary (see AGENTS.md, 'Agent Guardrails'). Expose an "
                "authenticated server via `bridged_tools` instead, which "
                "keeps real credentials out of the sandbox entirely."
            )
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        if isinstance(server, MCPServerConfigHTTP) and "url" in config:
            config["serverUrl"] = config.pop("url")
        if "cwd" in config and not isinstance(config["cwd"], str):
            config["cwd"] = str(config["cwd"])
        eager = sorted(eager_tools.get(server.name, ()))
        if eager:
            config["tools"] = {name: {"eager": True} for name in eager}
        servers[server.name] = config
    return json.dumps({"mcpServers": servers}, indent=2)


# Opaque base64 payloads the CLI prints on stdout -- Gemini thought signatures,
# emitted one per reasoning turn. They carry no diagnostic text, and a long run
# emits enough of them to fill any error budget, so they are replaced by a short
# placeholder before truncation rather than being allowed to evict the real
# message. 200 chars is well above any base64 token that might carry meaning and
# well below the ~2KB signatures.
_OPAQUE_BLOB = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_MAX_ERROR_LEN = 20000


def _clean_antigravity_error(stdout: str, stderr: str) -> str:
    """Trim the CLI's failure output down to something readable in a traceback.

    stderr is placed FIRST and stdout second: the CLI reports its actual failure
    reason on stderr (e.g. "Error: timeout waiting for response") while stdout
    carries the reasoning stream. Concatenating stdout first pushed every real
    error past the truncation limit, so a failed run surfaced as a wall of
    base64 with no cause in it.
    """

    def scrub(text: str) -> str:
        kept = [
            line for line in text.split("\n") if not line.strip().startswith("<think")
        ]
        return _OPAQUE_BLOB.sub(
            lambda match: f"<{len(match.group(0))}-char opaque payload>",
            "\n".join(kept),
        ).strip()

    sections = [
        f"{label}:\n{body}"
        for label, body in (("STDERR", scrub(stderr)), ("STDOUT", scrub(stdout)))
        if body
    ]
    cleaned = "\n\n".join(sections).strip()
    if len(cleaned) > _MAX_ERROR_LEN:
        cleaned = cleaned[:_MAX_ERROR_LEN] + "... (truncated)"
    return cleaned if cleaned else "Unknown error (no output)"


def _native_result_json(stdout: str) -> Mapping[str, Any]:
    """Return the CLI's JSON result, which is the whole of headless stdout.

    The `--output-format json` artifact is exactly one JSON object, carrying
    `conversation_id`, `status`, `response` and the run's usage totals.
    Anything else is a failure rather than something to search past: no
    object, a truncated one, or a second one after it all mean this attempt
    did not report its own result, and an earlier object cannot vouch for an
    attempt whose result never arrived. The CLI's stream-json dialect is a
    different envelope and is not read here.
    """
    try:
        parsed: Any = json.loads(stdout.strip())
    except json.JSONDecodeError as ex:
        raise RuntimeError(
            "antigravity cli did not print exactly one JSON result on stdout "
            f"({ex.msg}): {_clean_antigravity_error(stdout, '')}"
        ) from ex

    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"antigravity cli printed a JSON {type(parsed).__name__} rather "
            f"than a result object: {_clean_antigravity_error(stdout, '')}"
        )
    return parsed


def _verify_native_result(stdout: str, conversation_id: str | None) -> None:
    """Check the CLI's own result before this attempt is scored or returned.

    Only identity and status are read here. The envelope's aggregate
    `response` is deliberately never adopted: canonical output stays the
    bridge's real `ModelOutput`, messages, producer ids and usage.

    Args:
        stdout: Headless stdout, whose final JSON object is the CLI's result.
        conversation_id: The conversation this invocation bound, or `None` when
            it never saw the CLI open one -- itself a failure, not a wildcard.

    Raises:
        RuntimeError: There is no readable result, its status is not SUCCESS,
            this run bound no conversation, or the result names another one or
            none at all. None of these is covered by the process exit status --
            the CLI reports a failed conversation while exiting zero -- and
            each would otherwise be scored as an ordinary run, on a transcript
            nobody can attribute.
    """
    result = _native_result_json(stdout)

    status = result.get("status")
    if status != "SUCCESS":
        raise RuntimeError(
            f"antigravity cli reported status {status!r} for conversation "
            f"{result.get('conversation_id')!r}; expected SUCCESS"
        )

    # Comparing the two directly would let a run with no identity on either
    # side match itself: nothing bound, nothing named, and a success reported
    # for an attempt with no transcript behind it. Each side is required in its
    # own right, and only then compared.
    if conversation_id is None:
        raise RuntimeError(
            "antigravity cli reported success for conversation "
            f"{result.get('conversation_id')!r}, but this run bound no "
            "conversation at all: it never saw the CLI declare one, so there "
            "is nothing here to attribute this result to"
        )

    returned = result.get("conversation_id")
    if not isinstance(returned, str):
        raise RuntimeError(
            f"antigravity cli reported success with conversation id {returned!r} "
            f"({type(returned).__name__}) rather than a string, while this run "
            f"bound {conversation_id!r}; refusing to score a result whose own "
            "identity cannot be read"
        )

    if returned != conversation_id:
        raise RuntimeError(
            f"antigravity cli returned conversation {returned!r} but this run "
            f"bound {conversation_id!r}; refusing to score a conversation it "
            "did not run"
        )


async def _run_antigravity_cli_centaur(
    options: CentaurOptions,
    agy_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
    *,
    user: str | None,
    sandbox: str | None,
    cwd: str,
) -> None:
    instructions = (
        "Antigravity CLI:\n\n"
        " - You may also use the Antigravity CLI via the 'agy' command.\n"
        " - Use 'agy --continue' if you need to resume a previous session."
    )

    # Only the vars the alias needs: exporting HOME would break human_cli.
    centaur_env = {k: v for k, v in agent_env.items() if k != "HOME"}
    agent_env_vars = [f'export {k}="{v}"' for k, v in centaur_env.items()]
    alias_cmd = shlex.join(agy_cmd)
    alias_cmd = "alias agy='" + alias_cmd.replace("'", "'\\''") + "'"
    # The shell starts where this agent's own commands would have run. `agy`
    # resolves relative paths and writes its artifacts against the working
    # directory, and a login shell would otherwise start at the image's
    # default -- `/` for most images, which is not what the task set up.
    cd_cmd = f"cd {shlex.quote(cwd)}"
    bashrc = "\n".join(agent_env_vars + ["", cd_cmd, alias_cmd])

    await run_centaur(options, instructions, bashrc, state, user=user, sandbox=sandbox)
