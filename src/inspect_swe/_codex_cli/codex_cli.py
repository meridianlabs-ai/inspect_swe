import mimetypes
import shlex
from logging import getLogger
from pathlib import Path, PurePosixPath
from textwrap import dedent
from typing import Any, Literal, Sequence, cast

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
    ChatMessageSystem,
    ContentImage,
    GenerateFilter,
    Model,
    ModelName,
    get_model,
)
from inspect_ai.scorer import score
from inspect_ai.tool import (
    MCPServerConfig,
    MCPServerConfigHTTP,
    Skill,
    install_skills,
    read_skills,
)
from inspect_ai.util import SandboxEnvironment, checkpointer, store
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions
from typing_extensions import Unpack

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.centaur import CentaurOptions, CommandsFilter, run_centaur
from inspect_swe._util.mcp_ready import (
    DEFAULT_MCP_READY_TIMEOUT,
    wait_for_mcp_endpoints,
)
from inspect_swe._util.messages import build_user_prompt, collect_user_images
from inspect_swe._util.path import join_path
from inspect_swe._util.sandbox import resolve_agent_cwd, sandbox_exec
from inspect_swe._util.toml import to_toml
from inspect_swe._util.trace import trace

from .._util.agentbinary import ensure_agent_binary_installed
from ._events.consumer import CodexConsumer
from .agentbinary import (
    codex_binary_version,
    codex_cli_binary_source,
    codex_models_catalog,
)
from .config import (
    MCP_STARTUP_TIMEOUT_SEC,
    CodexApprovalPolicy,
    CodexAutoReview,
    CodexDeprecatedArgs,
    CodexSandboxMode,
    CodexWebSearch,
    check_codex_auto_review_version,
    codex_cli_config_overrides,
    codex_config_options,
    codex_mcp_servers_toml,
    codex_network_access_args,
    codex_sandbox_args,
    codex_sandbox_uses_bwrap,
    resolve_codex_approval_policy,
    resolve_codex_auto_review,
    resolve_codex_auto_review_model_aliases,
    resolve_codex_deprecated_args,
    resolve_codex_sandbox_mode,
    resolve_codex_web_search,
    validate_codex_bool_arg,
)
from .model_catalog import (
    is_latest_openai_model,
    is_openai_derived_api,
    openai_service_model_name,
    resolve_codex_model_slug,
)

logger = getLogger(__file__)


@agent
def codex_cli(
    name: str = "codex_cli",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    model_config: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    mcp_ready_timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    mcp_startup_timeout: int | None = MCP_STARTUP_TIMEOUT_SEC,
    web_search: CodexWebSearch = "live",
    goals: bool = True,
    auto_review: bool | CodexAutoReview = False,
    centaur: bool | CentaurOptions = False,
    commands_filter: CommandsFilter | None = None,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    transparent_proxy: bool = False,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    home_dir: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "latest"] | str = "auto",
    config_overrides: dict[str, str] | None = None,
    debug: bool | None = None,
    sandbox_mode: CodexSandboxMode = "danger-full-access",
    approval_policy: CodexApprovalPolicy = "never",
    network_access: bool = True,
    approve_static_mcp_tools: bool = False,
    **deprecated_args: Unpack[CodexDeprecatedArgs],
) -> Agent:
    """Codex CLI.

    Agent that uses OpenAI [Codex CLI](https://github.com/openai/codex) running in a sandbox.

    Image content in the input is written to files in the sandbox and attached
    to the prompt via `codex exec --image` (requires a codex version that
    supports the `--image` option).

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)
        system_prompt: Additional system prompt to append to default system prompt.
        model_config: Codex model slug used to select the system prompt and tool
            set. Defaults to `None`, which derives the slug from the real model so
            Codex's prompt/tooling aligns with what's actually running. Pass an
            explicit slug to override.
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent.
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
            Each BridgedToolsSpec creates an MCP server that makes the specified
            tools available to the agent running in the sandbox.
        mcp_ready_timeout: Seconds to wait for bridged MCP endpoints to serve
            tools before the agent launch errors.
        mcp_startup_timeout: Seconds Codex waits for bridged MCP server startup.
            Defaults to 300; pass `None` to use Codex's default.
        web_search: Web search mode. Use "live" for live web search, "cached" for cached web search, or "disabled" to disable web search. Defaults to "live".
        goals: Enable Codex goal tools (defaults to `True`).
        auto_review: Enable Codex automated approval review (guardian). When enabled,
            Codex runs with its own sandbox active (`workspace-write`) and `on-request`
            approvals; escalation requests (e.g. writes outside the workspace) are
            adjudicated by a guardian model rather than auto-approved. The
            `network_access` setting still applies, since that effective mode is
            `workspace-write`.
            Pass `CodexAutoReview` to customize the guardian policy and model.
            Requires Codex CLI >= 0.137.0. Defaults to `False`.
        centaur: Run in 'centaur' mode, which makes Codex CLI available to an Inspect `human_cli()` agent rather than running it unattended.
        commands_filter: In centaur mode only, filter or augment the human agent's
            command list (e.g. to add task-specific commands). Ignored outside centaur mode.
        attempts: Configure agent to make multiple attempts. When this is specified, the task will be scored when the agent stops calling tools. If the scoring is successful, execution will stop. Otherwise, the agent will be prompted to pick up where it left off for another attempt.
        model: Model name to use (defaults to main model for task).
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models.
            When a model name in the mapping is referenced, the corresponding Model/string is used.
        transparent_proxy: Bind the `auto_review` guardian slug
            (`codex-auto-review`) to the real target model even when
            `auto_review` doesn't configure its own `model`, and forward the
            client's generation parameters as authoritative instead of merging
            in the eval's active `GenerateConfig` (defaults to `False`). Codex
            hardcodes that guardian slug into every reviewer request
            regardless of configuration, so without this the request has no
            alias, falls through to the session model instead of the intended
            guardian, and the guardian's own generation parameters (e.g.
            `max_tokens`) are dropped. Required for `auto_review=True` with
            `approval_policy="on-request"` when no explicit
            `CodexAutoReview(model=...)` is set. The ordinary bridged request
            (Codex's own `--model` slug) is unaffected -- it keeps resolving
            to the real target model exactly as it does with `transparent_proxy=False`.
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        home_dir: Home directory to use for codex cli. If set, AGENTS.md, skills, and the MCP configuration will be written here.
        cwd: Working directory to run codex cli within.
        env: Environment variables to set for codex cli
        user: User to execute codex cli with.
        sandbox: Optional sandbox environment name.
        version: Version of codex cli to use. One of:
            - "auto": Use any available version of codex cli in the sandbox, otherwise download the latest version.
            - "sandbox": Use the version of codex cli in the sandbox (raises `RuntimeError` if codex is not available in the sandbox)
            - "latest": Download and use the very latest version of codex cli.
            - "x.x.x": Download and use a specific version of codex cli.
        config_overrides: Additional Codex CLI configuration overrides.
            Each key-value pair is passed as `-c key=value` to the CLI, except
            `approval_policy` and `sandbox_mode`, which are intercepted and
            validated (invalid values raise `ValueError`) so that command
            construction and the bridged-MCP configuration are derived from one
            effective value rather than silently disagreeing with the raw flag.
        debug: Trace all debug output.
        sandbox_mode: Codex's own sandbox policy for model-generated shell commands
            (`-s`/`--sandbox`), not related to the `sandbox` option above that selects
            an Inspect sandbox environment. Defaults to `"danger-full-access"`, which
            combined with `approval_policy="never"` (the default) reproduces the
            original `--dangerously-bypass-approvals-and-sandbox` behavior.

            The restricted modes run Codex's Linux sandbox, which on current codex
            releases requires a system `bwrap` (bubblewrap) binary in the sandbox
            image -- the codex release archive ships only the codex binary, so no
            bundled fallback exists, and codex panics at shell launch without it
            (this agent fails fast with an actionable error instead). The container
            runtime must also permit unprivileged user namespace creation; Docker's
            default seccomp profile does not. The mechanism is version-dependent:
            codex <= 0.77 used Landlock+seccomp (no bwrap), 0.98 had bwrap opt-in.

            Mode semantics for model-generated commands (the Codex parent process
            contacts MCP servers and the model proxy outside this sandbox):
            `"read-only"` blocks writes AND network, and with the default
            `approval_policy="never"` denies writes outright with no approval path
            (the agent cannot edit anything). `"workspace-write"` makes cwd, `/tmp`
            and `$TMPDIR` writable, but remounts top-level `.git` (and
            `.agents`/`.codex`) in each writable root read-only -- model-run
            `git commit`/`checkout`/`stash` fail (flip side: the agent cannot tamper
            with its own `.codex` CODEX_HOME).
        approval_policy: Codex's approval policy (`AskForApproval`). Defaults to
            `"never"`. Under headless `codex exec` (the default), codex itself
            overrides the runtime policy to `never` regardless of this setting
            (verified on codex 0.42.0 through 0.145.0), so a non-`"never"` value
            without an approvals reviewer raises `ValueError` here rather than
            silently breaking bridged tools. Prompting policies are effective in
            centaur mode, or headless with
            `config_overrides={"approvals_reviewer": ...}` (codex's auto-review
            escape hatch).
        network_access: Whether Codex's `"workspace-write"` sandbox may access the
            network for model-generated commands. Defaults to `True` -- note this
            inverts Codex's own default (`false`) to preserve continuity with the
            previous unconditional full-access behavior; ignored by other sandbox
            modes. Applies under `auto_review` too, whose effective mode is
            `workspace-write`: at `True` the sandbox grants network up front, so
            the guardian has no network escalation to adjudicate.
            `config_overrides={"sandbox_workspace_write.network_access":
            "false"}` also wins over this setting (later `-c` pairs take
            precedence in codex).
        approve_static_mcp_tools: Also pre-approve tool calls from static
            `mcp_servers` (not just Inspect-bridged servers) when the effective
            approval policy is `"never"`. Defaults to `False`: static servers keep
            Codex's per-server default gate. Note that under the restricted
            sandbox modes that default cancels un-annotated static tool calls
            headlessly (inspect's `MCPServerConfig` cannot express
            `default_tools_approval_mode`), so set this to `True` if you combine
            `sandbox_mode` with static `mcp_servers`. Raises if combined with
            `auto_review=True`: that macro's effective policy is `"on-request"`,
            never `"never"`, so the override this option applies would be
            silently unreachable.
        **deprecated_args: Deprecated compatibility arguments.
    """
    # resolve centaur
    if not isinstance(centaur, (bool, CentaurOptions)):
        # Agent kwargs arrive from task configs and `-S` args as arbitrary
        # values (bare `-S centaur=false` coerces via YAML, but quoted values,
        # config files, and programmatic callers don't), and the headless
        # guard below keys off `centaur is False` identity -- an unvalidated
        # non-bool silently skips it and produces a malformed headless run.
        raise ValueError(f"centaur must be a bool or CentaurOptions, got {centaur!r}.")
    if centaur is True:
        centaur = CentaurOptions()

    # resolve bridge model (preserve original `model` for prompt/tool alignment)
    bridge_model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # resolve deprecated arguments
    disallowed_tools = resolve_codex_deprecated_args(
        cast(dict[str, Any], deprecated_args)
    )
    effective_web_search = resolve_codex_web_search(web_search, disallowed_tools)
    network_access = validate_codex_bool_arg("network_access", network_access)
    approve_static_mcp_tools = validate_codex_bool_arg(
        "approve_static_mcp_tools", approve_static_mcp_tools
    )
    resolved_auto_review = resolve_codex_auto_review(auto_review)
    # Set when a prompting policy is allowed to stand only because an approvals
    # reviewer is configured through `config_overrides`. That escape hatch is
    # itself version-gated (see the check below), so the floor must be verified
    # on this path too.
    reviewer_hatch_needs_version_floor = False
    if resolved_auto_review is not None:
        # auto_review is a macro over the sandbox/approval controls: it forces
        # workspace-write + on-request with a guardian reviewer via final -c
        # overrides that must keep winning at the CLI level. Explicitly
        # combining it with these controls is contradictory -- fail loudly.
        if (
            sandbox_mode != "danger-full-access"
            or approval_policy != "never"
            or approve_static_mcp_tools
            or any(
                key in (config_overrides or {})
                for key in ("sandbox_mode", "approval_policy", "approvals_reviewer")
            )
        ):
            raise ValueError(
                "auto_review manages sandbox_mode, approval_policy and the "
                "approvals reviewer itself (workspace-write / on-request / "
                "guardian); do not combine it with explicit sandbox_mode, "
                "approval_policy, approve_static_mcp_tools, or those "
                "config_overrides keys. approve_static_mcp_tools has no "
                "effect under auto_review: the effective policy is "
                "'on-request', not 'never', so the static-server approval "
                "override this option applies is never reached."
            )
        effective_approval_policy: CodexApprovalPolicy = "on-request"
        effective_sandbox_mode: CodexSandboxMode = "workspace-write"
    else:
        effective_approval_policy = resolve_codex_approval_policy(
            approval_policy, config_overrides
        )
        effective_sandbox_mode = resolve_codex_sandbox_mode(
            sandbox_mode, config_overrides
        )

        # Headless `codex exec` hard-overrides the runtime approval policy to
        # `never` (harness override, precedence over `-c`; verified empirically
        # on 0.42.0-0.145.0), EXCEPT when an approvals reviewer is configured. A
        # prompting policy here would therefore not prompt -- but it WOULD
        # disable the bridged-MCP approval override below, cancelling every
        # bridged tool call. Fail fast instead of silently losing the agent's
        # tools.
        reviewer_hatch_needs_version_floor = (
            centaur is False
            and effective_approval_policy != "never"
            and (config_overrides or {}).get("approvals_reviewer") is not None
        )
        if (
            centaur is False
            and effective_approval_policy != "never"
            and (config_overrides or {}).get("approvals_reviewer") is None
        ):
            raise ValueError(
                f"approval_policy={effective_approval_policy!r} has no effect "
                "under headless `codex exec`: codex overrides the runtime policy "
                "to 'never' and bridged MCP tools would lose their approval "
                "override. Use approval_policy='never', run in centaur mode, "
                "enable auto_review, or configure an approvals reviewer via "
                "config_overrides={'approvals_reviewer': ...}."
            )

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "codex_cli_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        # Bridge ModelEventSink: the bridge hands us every ModelEvent instead of
        # emitting it to the transcript, and we attribute each to the correct
        # (sub-)agent span. The outer span (main-agent attribution + sub-agent
        # span parenting) is resolved at emission time so it tracks the
        # rotating checkpoint span. Reconstructs spans bridge-only (no Codex
        # --json parsing); see consumer.py.
        consumer = CodexConsumer()

        async with (
            checkpointer() as cp,
            sandbox_agent_bridge(
                state,
                model=bridge_model,
                model_aliases=resolve_codex_auto_review_model_aliases(
                    resolved_auto_review,
                    model_aliases,
                    default=get_model(model) if transparent_proxy else None,
                ),
                forward_generation_config=transparent_proxy,
                filter=filter,
                sandbox=sandbox,
                retry_refusals=retry_refusals,
                port=port,
                bridged_tools=bridged_tools,
                web_search=effective_web_search != "disabled",
                model_event_sink=consumer,
                checkpointer=cp,
            ) as bridge,
        ):
            if cp.attempt == "resume_for_scoring":
                return bridge.state

            # ensure codex is installed and get binary location
            codex_binary = await ensure_agent_binary_installed(
                codex_cli_binary_source(), version, user, sandbox_env(sandbox)
            )

            # resolve the installed codex version once (shared by the
            # auto_review gate and model alignment below)
            codex_version = await codex_binary_version(
                sandbox_env(sandbox), codex_binary, user
            )

            # auto_review requires on-request approval support (>= 0.137.0 for
            # headless exec); the floor is applied in centaur mode too so
            # behavior is consistent across modes. The bare `approvals_reviewer`
            # escape hatch rides the same codex relent, so it carries the same
            # floor -- without this, following the ValueError's own advice on an
            # older binary leaves the runtime policy at `never` while we track a
            # prompting policy, so the bridged-MCP approval override is omitted
            # and every bridged write tool call is cancelled silently.
            if resolved_auto_review is not None:
                check_codex_auto_review_version(codex_version)
            elif reviewer_hatch_needs_version_floor:
                check_codex_auto_review_version(
                    codex_version, feature="config_overrides['approvals_reviewer']"
                )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # Codex's restricted sandbox modes shell out to bubblewrap, and the
            # codex release archive ships no bundled bwrap (single-binary
            # tarball), so a system bwrap must exist in the sandbox image.
            # Without one codex PANICS at sandbox launch on every
            # model-generated shell command -- fail fast with an actionable
            # error instead of a mid-eval panic.
            #
            # Only preflight where bubblewrap is actually the mechanism: on
            # releases that launch it (see `codex_sandbox_uses_bwrap`) and on
            # Linux, since macOS sandboxes via Seatbelt. Under auto_review a
            # missing bwrap is not fatal -- each sandboxed command fails, a
            # capable model re-issues it with `require_escalated`, and the
            # guardian adjudicates -- so warn there rather than raising, and
            # keep that documented model-driven fallback working.
            if effective_sandbox_mode != "danger-full-access" and (
                codex_sandbox_uses_bwrap(codex_version)
            ):
                # an immutable property of the image: probe both facts in one
                # round-trip and reuse it across executions of the agent
                BWRAP_PROBE = "codex_cli_bwrap_probe"
                bwrap_probe = store().get(BWRAP_PROBE)
                if bwrap_probe is None:
                    bwrap_probe = await sandbox_exec(
                        sbox,
                        "uname -s; command -v bwrap || echo __MISSING__",
                        user=user,
                    )
                    store().set(BWRAP_PROBE, bwrap_probe)
                probe_lines = [ln for ln in bwrap_probe.splitlines() if ln.strip()]
                on_linux = bool(probe_lines) and probe_lines[0].strip() == "Linux"
                if on_linux and bwrap_probe.endswith("__MISSING__"):
                    missing = (
                        f"sandbox_mode={effective_sandbox_mode!r} requires a "
                        "bubblewrap (bwrap) binary in the sandbox image: codex "
                        "release binaries do not bundle one. Install the "
                        "'bubblewrap' package in the image. The container runtime "
                        "must also permit unprivileged user namespace creation "
                        "(Docker's default seccomp profile blocks it)."
                    )
                    if resolved_auto_review is not None:
                        logger.warning(
                            f"{missing} auto_review continues without an OS "
                            "sandbox: each sandboxed command fails and the "
                            "guardian adjudicates the model's escalation instead."
                        )
                    else:
                        raise RuntimeError(
                            f"{missing} Alternatively use "
                            "sandbox_mode='danger-full-access'."
                        )

            # resolve working directory (home dir if sandbox default is '/')
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # align Codex's `--model` slug to the real bridged model
            codex_model = await resolve_codex_model(model, model_config, codex_version)

            # determine CODEX_HOME (default to agent working dir)
            if home_dir is None:
                codex_home = join_path(agent_cwd, ".codex")
            else:
                # Resolve ~ and $VARS inside the sandbox
                codex_home = await sandbox_exec(
                    sbox, f'eval echo "{home_dir}"', user=user, cwd=agent_cwd
                )
            await sandbox_exec(sbox, cmd=f"mkdir -p {codex_home}", user=user)

            # location for agents_md
            def codex_agents_md() -> str:
                AGENTS_MD = "AGENTS.md"
                if home_dir is not None:
                    return join_path(codex_home, AGENTS_MD)
                else:
                    return join_path(agent_cwd, AGENTS_MD)

            # location for config_toml (either codex_home or cwd/.codex )
            async def codex_config_toml() -> str:
                CONFIG_TOML = "config.toml"
                if home_dir is not None:
                    return join_path(codex_home, CONFIG_TOML)
                else:
                    dir = join_path(agent_cwd, ".codex")
                    await sandbox_exec(sbox, cmd=f"mkdir -p {dir}", user=user)
                    return join_path(dir, CONFIG_TOML)

            # write system messages to AGENTS.md
            if system_messages:
                await sbox.write_file(codex_agents_md(), "\n\n".join(system_messages))

            # install skills
            if resolved_skills is not None:
                await install_skills(
                    resolved_skills, sbox, user, join_path(codex_home, "skills")
                )

            prompt, has_assistant_response = build_user_prompt(
                state.messages, handled_content=("image",)
            )

            # stage input images as files in the sandbox (empty if the input
            # has no images); attached to the prompt via `--image` below
            image_files = await _stage_prompt_images(
                collect_user_images(state.messages), sbox, user, codex_home
            )

            # build agent cmd
            cmd = [codex_binary]

            # headless
            if centaur is False:
                cmd.extend(["exec", "--color", "never", "--skip-git-repo-check"])

            # default cli args
            cmd.extend(
                [
                    # the real model is served via the bridge; this slug only
                    # selects Codex's system prompt + tool set (see codex_model above)
                    "--model",
                    codex_model,
                ]
            )
            # Sandbox/approval args from the effective values. Emitted ahead of
            # config_overrides, which wins over them (later -c pairs take
            # precedence). auto_review takes its mode and policy from its own
            # final -c overrides instead, so --sandbox/approval_policy args
            # here would only fight that precedence -- emit only the network
            # access args its still-workspace-write effective mode needs.
            cmd.extend(
                codex_network_access_args(network_access)
                if resolved_auto_review is not None
                else codex_sandbox_args(
                    effective_sandbox_mode,
                    effective_approval_policy,
                    network_access,
                )
            )

            # apply config overrides (approval_policy and sandbox_mode are
            # intercepted into the effective values above rather than passed raw)
            if config_overrides:
                for key, value in config_overrides.items():
                    if key not in ("approval_policy", "sandbox_mode"):
                        cmd.extend(["-c", f"{key}={value}"])

            # apply final Codex config overrides for explicit arguments
            for key, value in codex_cli_config_overrides(
                effective_web_search, goals, resolved_auto_review
            ).items():
                cmd.extend(["-c", f"{key}={value}"])

            # build toml config
            toml_config: dict[str, Any] = {}

            # disable codex analytics (both the chatgpt.com analytics-events
            # sink and the always-on Statsig OTel metrics to ab.chatgpt.com)
            toml_config["analytics"] = {"enabled": False}
            toml_config.update(
                codex_config_options(effective_web_search, goals, resolved_auto_review)
            )

            # Register static MCP servers. Bridged tools are supplied by the
            # evaluation author, so headless approval always skips their MCP gate;
            # static caller-configured servers keep Codex's per-server default
            # unless approve_static_mcp_tools opts them in (under the restricted
            # sandbox modes that default cancels un-annotated tool calls
            # headlessly -- see the approve_static_mcp_tools docstring).
            # Bridged servers are additionally marked required, so a failed MCP
            # startup fails the launch loudly instead of running the turn without
            # tools (see codex_mcp_server_config).
            bridged_server_names = {s.name for s in bridge.mcp_server_configs}
            toml_config.update(
                codex_mcp_servers_toml(
                    mcp_servers or [],
                    bridged_server_names,
                    effective_approval_policy,
                    force_approve=approve_static_mcp_tools,
                    bridged_startup_timeout=mcp_startup_timeout,
                )
            )
            toml_config.update(
                codex_mcp_servers_toml(
                    bridge.mcp_server_configs,
                    bridged_server_names,
                    effective_approval_policy,
                    force_approve=True,
                    bridged_startup_timeout=mcp_startup_timeout,
                )
            )

            # model provider (use a custom provider name so we can set
            # stream_idle_timeout_ms -- built-in providers can't be overridden)
            toml_config["preferred_auth_method"] = "apikey"
            toml_config["model_provider"] = "openai-proxy"
            toml_config["model_providers.openai-proxy"] = {
                "name": "OpenAI Proxy",
                "base_url": f"http://localhost:{bridge.port}/v1",
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
                "stream_idle_timeout_ms": 3_600_000,
            }

            # write toml config
            await sbox.write_file(await codex_config_toml(), to_toml(toml_config))

            # setup agent env
            agent_env = {
                "CODEX_HOME": codex_home,
                "OPENAI_API_KEY": "api-key",
                "OPENAI_BASE_URL": f"http://localhost:{bridge.port}/v1",
                "RUST_LOG": "warning",
            } | (env or {})

            # Compute bridged HTTP configs once at the outer scope so both the
            # centaur and non-centaur paths gate on the same set. Codex closes
            # the client-connect half of the first-turn race with
            # `required = true` on bridged servers (see codex_mcp_server_config),
            # but that only helps once the endpoint is answering; the pre-launch
            # gate covers the endpoint half in both centaur and non-centaur modes.
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
                await _run_codex_cli_centaur(
                    options=centaur,
                    codex_cmd=cmd,
                    image_files=image_files,
                    agent_env=agent_env,
                    state=state,
                    user=user,
                    commands_filter=commands_filter,
                )
            else:
                # execute the agent (track debug output)
                debug_output: list[str] = []
                agent_prompt = prompt
                # images accompany the original prompt only: retry attempts
                # send `incorrect_message` (text-only), and a checkpoint
                # resume replays a prompt codex already received, with the
                # images already embedded in its rollout
                agent_image_args = (
                    []
                    if cp.attempt == "resume"
                    else [arg for file in image_files for arg in ("--image", file)]
                )
                attempt_count = cp.track(
                    "codex_attempt_count", lambda: attempt_count, 0
                )
                while True:
                    # append prompt
                    agent_cmd = cmd.copy()
                    agent_cmd.append(agent_prompt)

                    # resume previous conversation
                    if (
                        has_assistant_response
                        or attempt_count > 0
                        or cp.attempt == "resume"
                    ):
                        agent_cmd.extend(["resume", "--last"])

                    # Image args go LAST: `--image` is multi-value, so a
                    # following positional (the prompt, or the `resume`
                    # subcommand) would be swallowed as another image path.
                    # Trailing, it binds to `exec` or `exec resume` alike.
                    agent_cmd.extend(agent_image_args)

                    # Retry-loop gate: fires ONLY when this loop is actually
                    # retrying (attempt_count > 0 or the checkpoint says
                    # resume), so the cold-start pre-centaur gate is not paid
                    # for twice on the first iteration.
                    is_retry = attempt_count > 0 or cp.attempt == "resume"
                    if _http_mcp_configs and is_retry:
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
                            cwd=agent_cwd, env=agent_env, user=user, concurrency=False
                        ),
                        stream=False,
                    )

                    # record output for debug
                    if debug:
                        debug_output.append(result.stdout)
                        debug_output.append(result.stderr)

                    # close any sub-agent spans left open by this attempt so the
                    # span tree stays balanced across restarts and on error
                    # (Codex doesn't carry sub-agent spans across resumes)
                    consumer.reset()

                    # raise for error
                    if not result.success:
                        raise RuntimeError(
                            f"Error executing codex cli agent {result.returncode}: {result.stdout}\n{result.stderr}"
                        )

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
                        agent_image_args = []

                # trace debug info
                if debug:
                    debug_output.insert(0, "Codex CLI Debug Output:")
                    trace("\n".join(debug_output))

        # return success
        return bridge.state

    return agent_with(execute, name=name, description=description)


# codex derives the request MIME type from the file extension
# (mime_guess::from_path), and stdlib mimetypes tables vary by Python
# version and platform (e.g. image/webp is absent before 3.11), so map
# common image types explicitly rather than trusting guess_extension.
_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


async def _stage_prompt_images(
    images: list[ContentImage],
    sandbox: SandboxEnvironment,
    user: str | None,
    codex_home: str,
) -> list[str]:
    """Write prompt images to files in the sandbox, returning their paths.

    `codex exec` cannot accept image content inline in its prompt argument
    (the prompt is plain argv text), so images ride along as files attached
    via `--image` args built from the returned paths.

    Filenames restart at image-0 on each execution, so a follow-up turn
    overwrites the previous turn's files. This is harmless: codex embeds the
    image bytes into its rollout history at launch (verified on 0.151.0 --
    a resumed turn's request still carries the original bytes).
    """
    if not images:
        return []

    # private inspect_ai API with no public equivalent; imported lazily so
    # a future rename breaks image staging with a clear error here rather
    # than failing `import inspect_swe` for every user
    from inspect_ai._util.images import file_as_data

    images_dir = join_path(codex_home, "images")
    await sandbox_exec(sandbox, f"mkdir -p {images_dir}", user=user)
    files: list[str] = []
    for idx, image in enumerate(images):
        image_bytes, mime_type = await file_as_data(image.image)
        extension = (
            _IMAGE_EXTENSIONS.get(mime_type)
            or mimetypes.guess_extension(mime_type, strict=False)
            or ".png"
        )
        image_file = join_path(images_dir, f"image-{idx}{extension}")
        await sandbox.write_file(image_file, image_bytes)
        files.append(image_file)
    return files


async def resolve_codex_model(
    model: str | None,
    model_config: str | None,
    codex_version: str | None,
) -> str:
    """Resolve the Codex `--model` slug aligned to the real bridged model.

    Derives the slug from the real model so Codex's system prompt and tool set
    match what's actually running; an explicit `model_config` overrides. Providers
    that are `OpenAIAPI`-derived (e.g. a pre-deployment stand-in registered under a
    custom provider name) are treated as OpenAI and resolved by their declared
    `service_model_name()` rather than the registry name — so a custom `otter`
    provider reporting `gpt-5.5` aligns to that catalog entry. "latest"/codename
    models (per the provider's `is_latest()`) align to the latest catalog profile
    rather than Codex's generic fallback.
    """
    resolved_model = get_model(model)
    real_model = ModelName(resolved_model)
    api = resolved_model.api
    real_api = "openai" if is_openai_derived_api(api) else real_model.api
    # an OpenAI-derived provider reports its true model identity via
    # service_model_name() (e.g. a custom 'otter' provider -> 'gpt-5.5'); align to
    # that, not the registry name ('otter'), which Codex wouldn't recognize.
    model_name = openai_service_model_name(api, real_model.name)
    codex_catalog = await codex_models_catalog(codex_version)
    resolution = resolve_codex_model_slug(
        model_name,
        api=real_api,
        catalog=codex_catalog,
        override=model_config,
        is_latest=is_latest_openai_model(api),
    )
    trace(
        f"Codex model alignment: real model '{real_model}' (as '{model_name}') "
        f"→ --model '{resolution.slug}' ({resolution.reason})"
    )
    return resolution.slug


async def _run_codex_cli_centaur(
    options: CentaurOptions,
    codex_cmd: list[str],
    image_files: list[str],
    agent_env: dict[str, str],
    state: AgentState,
    user: str | None = None,
    commands_filter: CommandsFilter | None = None,
) -> None:
    instructions = "Codex CLI:\n\n - You may also use Codex CLI via the 'codex' command.\n - Use 'codex resume' if you need to resume a previous codex session."

    if image_files:
        # Attach input images to the human's first codex invocation only,
        # via a self-disarming shell function (the marker file survives new
        # shells re-sourcing .bashrc). A permanent alias would re-attach the
        # images on every invocation -- codex applies root `--image` args to
        # `resume` too, duplicating images already embedded in the session
        # rollout. The flags are spliced right after the binary rather than
        # trailing: `--image` is multi-value, so tokens the human types after
        # the command (a prompt, or `resume`) would be swallowed as image
        # paths if the command ended with an `--image` flag.
        image_args = [arg for file in image_files for arg in ("--image", file)]
        marker = shlex.quote(str(PurePosixPath(image_files[0]).parent / ".attached"))
        first_cmd = shlex.join(codex_cmd[:1] + image_args + codex_cmd[1:])
        plain_cmd = shlex.join(codex_cmd)
        codex_cmd_def = dedent(f"""
            codex() {{
              if [ ! -e {marker} ]; then
                touch {marker}
                {first_cmd} "$@"
              else
                {plain_cmd} "$@"
              fi
            }}
        """).strip()
        instructions += (
            "\n - The task input includes image file(s), which the 'codex' "
            "command attaches to your first invocation automatically: "
            f"{', '.join(image_files)}\n"
            " - If that first invocation fails before the images are sent, "
            f"delete {marker} to re-attach them on your next invocation."
        )
    else:
        alias_cmd = shlex.join(codex_cmd)
        codex_cmd_def = "alias codex='" + alias_cmd.replace("'", "'\\''") + "'"

    # build .bashrc content
    agent_env_vars = [f'export {k}="{v}"' for k, v in agent_env.items()]
    bashrc = "\n".join(agent_env_vars + ["", codex_cmd_def])

    # run the human cli
    await run_centaur(
        options, instructions, bashrc, state, user=user, commands_filter=commands_filter
    )


async def _last_rollout(
    sandbox: SandboxEnvironment, codex_home: str, user: str | None
) -> str | None:
    try:
        rollout = await sandbox_exec(
            sandbox,
            f"find '{codex_home}/sessions' -type f -name 'rollout-*.jsonl' -exec ls -t -- {{}} + | head -n 1",
            user=user,
        )
        return rollout
    except RuntimeError as ex:
        logger.warning(f"Error attempting to read rollout file: {ex}")
        return None
