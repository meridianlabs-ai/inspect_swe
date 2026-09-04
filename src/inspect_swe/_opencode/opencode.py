import json
import shlex
from logging import getLogger
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
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.agentcontext import (
    ModelFilter,
    classify_filter,
    slug_map_classifier,
)
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.mcp_ready import (
    DEFAULT_MCP_READY_TIMEOUT,
    wait_for_mcp_endpoints,
)
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import resolve_agent_cwd
from inspect_swe._util.trace import trace

from .agentbinary import ensure_opencode_setup

logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent-context classification (config-injected model slugs)
# ---------------------------------------------------------------------------
#
# OpenCode has its own subagent system (Task-tool delegation to `general`,
# `explore`, `scout`, and any user-defined agent) plus internal utility
# calls (title generation, summarization, compaction) that all share the
# main thread's model unless configured otherwise. Probe P2 (agent-bridge-
# context plan, live-verified 2026-08-08) confirmed OpenCode's per-agent
# `model` config IS honored end to end: a project-level `opencode.json`
# defining a subagent's `model` produced bridged requests whose raw
# `current_bridge_request().model` carried that model's id -- distinct from
# the main thread's. Since `execute()` below already owns the *global*
# config OpenCode reads (`OPENCODE_CONFIG`), we inject the same overrides
# there directly rather than requiring a project-file dance.
#
# Caveats surfaced empirically (P2, plus this task's own live-verification
# run against the code below), baked into the constants/helpers that follow:
#
#   1. `small_model` alone does NOT redirect title generation (live, docker,
#      opencode-ai 1.18.15): with a `small_model` sentinel set, the title-gen
#      request still arrived under the *primary* slug. No error surfaced; it
#      is a silent no-op. The config schema also exposes `title`/`summary`/
#      `compaction` as named `AgentConfig` entries under `agent`
#      (https://opencode.ai/config.json), so routing them through the same
#      per-agent `model` override proven for `general` (caveat 2) is
#      plausible -- but UNVERIFIED, and not inert if wrong: on an install
#      where those names are NOT reserved, each entry would create a real,
#      prompt-less, mode-"all" agent the main agent could spawn, whose
#      requests would carry the small-model sentinel and classify "utility",
#      so `is_sub_agent()` would miss real delegation. Per-agent overrides
#      for the three utility agents (`title`, `summary`, `compaction`) are
#      therefore NOT injected; only `small_model` (OpenCode's documented
#      mechanism) is set, and utility traffic classifies "root" today --
#      honest under-attribution. Re-enabling needs a live run showing both
#      that `agent.title.model` etc. are honored and that those names are
#      reserved (not spawnable).
#   2. Built-in-subagent overridability: `general`, `explore`, and `scout`
#      are the three built-in subagents OpenCode ships
#      (https://opencode.ai/docs/agents/, opencode-ai 1.18.x). We set *only*
#      `model` on each so their built-in description/prompt/mode are left
#      alone. Per OpenCode's docs a config-defined agent with no `mode`
#      defaults to `"all"`, so on an install old enough to lack one of these
#      names the entry would create a spawnable prompt-less agent -- an edge
#      case, since `version="auto"` resolves to the latest release. Live-
#      verified (see #3): OpenCode reads `general`'s override and resolves
#      its `model`; it was the sentinel *value*'s catalog rejection, not the
#      override mechanism, that failed the first live run.
#   3. Sentinel catalog constraint (LIVE-VERIFIED, 2026-08-08, docker,
#      opencode-ai 1.18.15): the config *schema* types `AgentConfig.model`/
#      `small_model` as plain strings, but OpenCode's *runtime* resolves them
#      against a known catalog (models.dev). A synthetic id
#      (`anthropic/inspect-subagent`) on `agent.general.model` was REJECTED
#      and the failure is HARD: the `general` subagent's Task call returned
#      `"Model not found: anthropic/inspect-subagent."` and the main agent
#      improvised without delegating. The rejected request never reaches the
#      bridge, so nothing on our side can observe it. A rejected
#      `small_model`, by contrast, degrades silently to the primary slug.
#      So unlike `claude_code` (whose CLI performs no such validation, see
#      `_claude_code/model.py`), sentinels here must be real catalog ids, and
#      `_SENTINEL_MODELS` holds ONLY providers whose subagent candidate has
#      been live-verified as accepted (anthropic). Candidates drafted by the
#      same reasoning for other providers -- openai: `gpt-5-mini` /
#      `gpt-5-nano` / `gpt-4o-mini` (subagent) and `gpt-5-nano` /
#      `gpt-4o-mini` / `gpt-3.5-turbo` (small model); google:
#      `gemini-2.5-flash` / `gemini-2.5-flash-lite` / `gemini-2.0-flash`
#      (subagent) and `gemini-2.5-flash-lite` / `gemini-2.0-flash` /
#      `gemini-1.5-flash` (small model) -- were never exercised live and are
#      deliberately not in the table: an `opencode_model` under any provider
#      not in `_SENTINEL_MODELS` skips sentinel injection entirely (logged
#      once) and classifies all traffic "root", rather than risk hard-failing
#      built-in delegation that works fine without us. To re-enable a
#      provider, either live-verify its subagent candidate or validate
#      candidates at runtime against the installed OpenCode's catalog --
#      neither is done here.
#   4. Primary-collision guard: since sentinels are real catalog ids, a
#      caller's `opencode_model` can legitimately BE one (e.g.
#      `opencode_model="anthropic/claude-haiku-4-5-20251001"`) -- that would
#      put the same bare slug in both `slug_map_classifier`'s `root_slugs`
#      and `kind_by_slug`, and root wins (checked first), silently swallowing
#      subagent classification. `_select_sentinel` therefore skips any
#      candidate that collides with the primary (and, for the small-model
#      role, with the chosen subagent sentinel). The subagent role has a
#      SINGLE candidate -- the verified id -- because falling back to an
#      unverified alternate would trade under-attribution for the hard
#      failure in caveat 3 (as of 2026-09 models.dev lists no Claude 3.x ids
#      at all). On collision that role's override is omitted (logged): its
#      traffic carries the primary slug (OpenCode's behavior for an
#      unconfigured agent) and classifies "root" -- the same honest
#      degradation as an unverified provider, NOT "unknown" and not a
#      misclassification. The small-model role keeps alternates since its
#      rejection is silent. Should a caller of `build_opencode_filter` pass
#      a colliding sentinel anyway, the shared `slug_map_classifier` drops
#      the `kind_by_slug` entry (logged, never raised) so that slug
#      classifies "root" -- it IS the root slug on the wire.
#
# Regardless of provider, the OpenCode provider clients put only the bare
# model id (no `provider/` prefix) in the wire request's `model` field --
# confirmed against probe P2 traffic, where a project config's
# `anthropic/claude-haiku-4-5-20251001` subagent model arrived at
# `current_bridge_request().model` as `claude-haiku-4-5-20251001` -- so
# classification uses the bare forms, never the `provider/`-prefixed config
# values. No bridge `model_aliases` entries are needed for the sentinels:
# the bridge's fallback `model` already serves any unaliased slug
# (`resolve_inspect_model`).

_SENTINEL_MODELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # provider_id -> (subagent_candidates, small_model_candidates): ORDERED
    # preference lists of bare (no `provider/` prefix) REAL catalog ids;
    # `_select_sentinel` picks the first that doesn't collide with the
    # primary / already-chosen sentinel (caveat 4). ONLY providers whose
    # subagent candidate is live-verified belong here (caveat 3).
    "anthropic": (
        # single candidate: the id live-verified as an accepted
        # `agent.*.model` override -- a rejected alternate would hard-fail
        # delegation, so on collision the override is omitted instead
        ("claude-haiku-4-5-20251001",),
        # `small_model` rejection is silent, so alternates are safe here.
        # None of these appear in the 2026-09 models.dev catalog, and
        # `small_model` was already observed as a no-op for title-gen
        # (caveat 1) -- retained as OpenCode's documented mechanism.
        (
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ),
    ),
}

OPENCODE_BUILTIN_SUBAGENTS: tuple[str, ...] = ("general", "explore", "scout")
"""Built-in OpenCode subagents (opencode.ai/docs/agents/), routed to the
subagent sentinel so their traffic is slug-distinguishable from the main
thread even when invoked without a `model:` override of their own.
Live-verified (2026-08-08, `general`): the per-agent override IS honored."""

_warned_unverified_providers: set[str] = set()
"""Dedupe key for the "no live-verified sentinel models" warning below --
logged once per distinct provider id (mirrors `classify_filter`'s `warned`
set), not once per `opencode()` call/request."""


def _bare_model_id(model_ref: str) -> str:
    """Strip a `provider/model` config value down to the bare model id.

    Mirrors what OpenCode's provider clients actually place in the wire
    request's `model` field (see module docstring) -- the form
    `current_bridge_request().model` carries, and so the form the
    classifier and `model_aliases` keys below must match against.
    """
    return model_ref.split("/", 1)[1] if "/" in model_ref else model_ref


def _select_sentinel(
    candidates: tuple[str, ...], excluded: Sequence[str]
) -> str | None:
    """First candidate bare id not in `excluded`, or `None` if all collide."""
    for candidate in candidates:
        if candidate not in excluded:
            return candidate
    return None


def build_opencode_config_overrides(
    provider_id: str, opencode_model: str
) -> tuple[dict[str, Any], str | None, str | None]:
    """`agent`/`small_model` config fragment routing traffic to sentinel slugs.

    Returns `(config_fragment, subagent_sentinel, small_model_sentinel)`
    where `config_fragment` is merged into the generated OpenCode config and
    the two sentinels are the `provider/`-prefixed config values (matching
    `opencode_model`'s provider, so they still resolve through the
    overridden `baseURL`).

    When `provider_id` isn't in `_SENTINEL_MODELS` (i.e. its sentinel
    candidates are not live-verified — currently everything but anthropic),
    returns an empty fragment and `(None, None)`: no sentinel injection is
    attempted (see caveat 3) and a warning is logged once per provider id.

    Otherwise, picks each sentinel from its provider's ordered candidate
    list, skipping any candidate whose bare id collides with `opencode_model`
    (and, for the small-model role, with the already-chosen subagent
    sentinel too) — see caveat 4. Either pick may independently come back
    `None` (with a logged warning) if every candidate for that role
    collides; that role's override is simply omitted from the config rather
    than risking a slug that silently reclassifies as "root".

    Only `OPENCODE_BUILTIN_SUBAGENTS` receive per-agent overrides; the
    utility agents (`title`, `summary`, `compaction`) are deliberately not
    injected (caveat 1).
    """
    sentinel_candidates = _SENTINEL_MODELS.get(provider_id)
    if sentinel_candidates is None:
        if provider_id not in _warned_unverified_providers:
            _warned_unverified_providers.add(provider_id)
            logger.warning(
                f"opencode(): no live-verified catalog sentinel models for "
                f"provider {provider_id!r}; skipping sentinel injection, so "
                f"subagent/utility traffic will not be slug-distinguishable "
                f"from root for this opencode_model."
            )
        return {}, None, None

    subagent_candidates, small_model_candidates = sentinel_candidates
    primary_bare = _bare_model_id(opencode_model)

    subagent_id = _select_sentinel(subagent_candidates, (primary_bare,))
    if subagent_id is None:
        logger.warning(
            f"opencode(): every candidate subagent sentinel model for "
            f"provider {provider_id!r} collides with opencode_model "
            f"{opencode_model!r}; subagent traffic will not be "
            f"slug-distinguishable from root."
        )

    small_model_excluded = (
        (primary_bare, subagent_id) if subagent_id is not None else (primary_bare,)
    )
    small_model_id = _select_sentinel(small_model_candidates, small_model_excluded)
    if small_model_id is None:
        logger.warning(
            f"opencode(): every candidate small-model sentinel for provider "
            f"{provider_id!r} collides with opencode_model {opencode_model!r} "
            f"or the chosen subagent sentinel; utility traffic will not be "
            f"slug-distinguishable from root."
        )

    subagent_sentinel = (
        f"{provider_id}/{subagent_id}" if subagent_id is not None else None
    )
    small_model_sentinel = (
        f"{provider_id}/{small_model_id}" if small_model_id is not None else None
    )

    # built-in subagents only -- the utility agents (`title`, `summary`,
    # `compaction`) are intentionally not routed through per-agent overrides
    # (caveat 1); `small_model` below is the sole utility-routing mechanism
    agent_overrides: dict[str, Any] = {}
    if subagent_sentinel is not None:
        agent_overrides.update(
            {name: {"model": subagent_sentinel} for name in OPENCODE_BUILTIN_SUBAGENTS}
        )

    config_fragment: dict[str, Any] = {}
    if small_model_sentinel is not None:
        config_fragment["small_model"] = small_model_sentinel
    if agent_overrides:
        config_fragment["agent"] = agent_overrides

    return config_fragment, subagent_sentinel, small_model_sentinel


def build_opencode_filter(
    filter: GenerateFilter | None,
    opencode_model: str,
    subagent_sentinel: str | None,
    small_model_sentinel: str | None,
) -> ModelFilter:
    """OpenCode bridge filter: agent-context classification by requested slug.

    The root slug is `opencode_model` with its `provider/` prefix stripped
    (the form the bridge actually sees); the subagent/small-model sentinels
    (when not `None` — see `build_opencode_config_overrides`) are matched by
    their bare ids regardless of which provider prefix they were given.
    A sentinel that collides with the root slug (caveat 4) is resolved as
    root by `slug_map_classifier`, which drops the shadowed entry itself.
    """
    root_slug = _bare_model_id(opencode_model)
    kind_by_slug: dict[str, Literal["subagent", "utility"]] = {}
    if subagent_sentinel is not None:
        kind_by_slug[_bare_model_id(subagent_sentinel)] = "subagent"
    if small_model_sentinel is not None:
        kind_by_slug[_bare_model_id(small_model_sentinel)] = "utility"

    return classify_filter(filter, slug_map_classifier({root_slug}, kind_by_slug))


def build_opencode_config(
    provider_id: str,
    provider_base_url: str,
    agent_context_config: dict[str, Any],
    skills_enabled: bool,
    mcp_servers: Sequence[MCPServerConfig],
) -> dict[str, Any]:
    """Assemble the full OpenCode global config JSON (pure, no I/O).

    Extracted from `execute()` verbatim (no behavior change) so the
    assembled shape is unit-testable without a sandbox — in particular that
    `agent_context_config` (the `agent`/`small_model` sentinel overrides
    from `build_opencode_config_overrides`) actually lands in the written
    config. Without this, a future refactor of `execute()` could drop that
    spread and silently kill agent-context classification while every
    other test stayed green (nothing else observes the written config).
    """
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {"options": {"baseURL": provider_base_url}},
        },
        **agent_context_config,
    }
    if skills_enabled:
        config["permission"] = {"skill": {"*": "allow"}}
    if mcp_servers:
        config["mcp"] = resolve_mcp_servers(mcp_servers)
    return config


@agent
def opencode(
    name: str = "OpenCode",
    description: str = dedent("""
       Open-source autonomous coding agent for the terminal, capable
       of writing, testing, debugging, and iterating on code across
       multiple languages.
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
    opencode_model: str = "anthropic/claude-sonnet-4-5",
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
) -> Agent:
    """OpenCode agent.

    Agent that uses [OpenCode](https://github.com/anomalyco/opencode)
    running in a sandbox with Inspect model bridging.

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
        centaur: Run in 'centaur' mode, which makes OpenCode available to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models.
            When a model name in the mapping is referenced, the corresponding Model/string is used.
        opencode_model: OpenCode model identifier to pass to the CLI in the form
            `provider/model` (default: `"anthropic/claude-sonnet-4-5"`). The actual model
            calls still go through the Inspect bridge; this just selects which provider
            client OpenCode uses to format the request.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run opencode within
        env: Environment variables to set for opencode
        user: User to execute opencode with
        sandbox: Optional sandbox environment name
        version: Version of opencode to use. One of:
            - "auto": Use any available version in sandbox, otherwise download latest
            - "sandbox": Use sandbox version (raises RuntimeError if not available)
            - "stable"/"latest": Download and use the latest version
            - "x.x.x": Download and use a specific version
        debug: Trace all debug output.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # determine which provider client opencode will use, so we know which
    # provider entry's baseURL to override in the config (the bridge intercepts
    # the request regardless of which provider protocol opencode picks).
    provider_id = (
        opencode_model.split("/", 1)[0] if "/" in opencode_model else "anthropic"
    )

    # agent-context config overrides (subagent/small-model sentinel slugs —
    # see module docstring above)
    agent_context_config, subagent_sentinel, small_model_sentinel = (
        build_opencode_config_overrides(provider_id, opencode_model)
    )
    opencode_filter = build_opencode_filter(
        filter, opencode_model, subagent_sentinel, small_model_sentinel
    )

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "opencode_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        bridge_model = f"inspect/{model}" if model is not None else "inspect"

        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            # the sentinel slugs need no alias entries: the bridge's fallback
            # `model` serves any unaliased slug (`resolve_inspect_model`), and
            # resolves `inspect/<role>` names that `get_model(model)` would not
            model_aliases=model_aliases,
            filter=opencode_filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
            # granted unconditionally to preserve today's behaviour; a grant is
            # inert unless the CLI declares a native web tool
            web_search=True,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # resolve working directory (home dir if sandbox default is '/')
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # install opencode and its runtime dependencies in sandbox
            opencode_binary, dependency_bin_dirs = await ensure_opencode_setup(
                sbox, version, user
            )

            # combine static mcp configs with bridged tools' mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            # detect sandbox home directory
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"

            # write opencode config to redirect provider baseURL to the bridge
            # and (optionally) configure mcp servers.
            #
            # The bridge's model-proxy server registers OpenAI-compatible
            # routes (/v1/responses, /v1/chat/completions), the Anthropic
            # Messages route (/v1/messages), and Gemini routes
            # (/v1beta/models/*, /models/*). The AI SDK provider clients
            # append the API-relative path (e.g. "/messages",
            # "/chat/completions") to the configured baseURL, so we must
            # include "/v1" in the baseURL we hand to opencode.
            bridge_url = f"http://localhost:{bridge.port}"
            provider_base_url = f"{bridge_url}/v1"
            opencode_config = build_opencode_config(
                provider_id,
                provider_base_url,
                agent_context_config,
                resolved_skills is not None,
                all_mcp_servers,
            )

            opencode_config_dir = f"{sandbox_home}/.config/opencode"
            opencode_config_path = f"{opencode_config_dir}/opencode.json"
            await sbox.exec(["mkdir", "-p", opencode_config_dir], user=user)
            if resolved_skills is not None:
                await install_skills(
                    resolved_skills, sbox, user, f"{opencode_config_dir}/skills"
                )
            await sbox.write_file(opencode_config_path, json.dumps(opencode_config))

            # build system prompt (opencode run takes a single positional message
            # and has no separate --system-prompt flag, so we prepend)
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            prompt, has_assistant_response = build_user_prompt(state.messages)

            if system_messages:
                combined_system = "\n\n".join(system_messages)
                prompt = f"{combined_system}\n\n{prompt}"

            # base command
            cmd = [
                opencode_binary,
                "run",
                "--model",
                opencode_model,
                "--format",
                "json",
            ]

            # add auto-approve flag only for non-centaur mode
            if centaur is False:
                cmd.append("--dangerously-skip-permissions")

            # setup agent env (add dependencies to PATH so opencode can find them)
            path = ":".join(
                [*dependency_bin_dirs, "/usr/local/bin", "/usr/bin", "/bin"]
            )
            agent_env = {
                # belt-and-braces: set per-provider base URL env vars in addition
                # to the config file. Different opencode provider clients honor
                # different env conventions; the config file is authoritative
                # but env vars don't hurt. The bridge mounts API-specific routes
                # under /v1, so anthropic/openai callers that append "/messages"
                # or "/chat/completions" land on the right handler.
                "ANTHROPIC_BASE_URL": f"{bridge_url}/v1",
                "OPENAI_BASE_URL": f"{bridge_url}/v1",
                "ANTHROPIC_API_KEY": "sk-none",
                "OPENAI_API_KEY": "sk-none",
                "OPENCODE_CONFIG": opencode_config_path,
                "PATH": path,
                "HOME": sandbox_home,
            } | (env or {})

            # Compute bridged HTTP configs once at the outer scope so both the
            # centaur and non-centaur paths gate on the same set. OpenCode's
            # headless mode blocks the first turn on MCP connect, but the
            # endpoint has to be answering `tools/list` first -- this pre-launch
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
                await _run_opencode_centaur(
                    options=centaur,
                    opencode_cmd=cmd,
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

                    # add prompt as positional argument at the end
                    agent_cmd.append(agent_prompt)

                    # Retry-loop gate: fires ONLY when this loop is actually
                    # retrying (attempt_count > 0), so the cold-start
                    # pre-centaur gate is not paid for twice on the first
                    # iteration.
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
                        cli_error_msg = _clean_opencode_error(
                            result.stdout, result.stderr
                        )
                        raise RuntimeError(
                            f"Error executing opencode agent {result.returncode}: {cli_error_msg}"
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
                    debug_output.insert(0, "OpenCode Debug Output:")
                    trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def resolve_mcp_servers(
    mcp_servers: Sequence[MCPServerConfig],
) -> dict[str, dict[str, Any]]:
    """Build OpenCode `mcp` config block from MCP server configs.

    OpenCode expects entries keyed by server name with either:
      - {"type": "local", "command": [...], "environment": {...}}
      - {"type": "remote", "url": "...", "headers": {...}}
    """
    out: dict[str, dict[str, Any]] = {}
    for server in mcp_servers:
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        entry: dict[str, Any] = {"enabled": True}
        if isinstance(server, MCPServerConfigHTTP):
            entry["type"] = "remote"
            if "url" in config:
                entry["url"] = config.pop("url")
            if "headers" in config:
                entry["headers"] = config.pop("headers")
        else:
            entry["type"] = "local"
            # opencode expects the command as a single array including args
            command = config.pop("command", None)
            args = config.pop("args", None)
            if command is None:
                raise ValueError(f"Local MCP server {server.name!r} has no command")
            cmd_list = [command] if isinstance(command, str) else list(command)
            if args:
                cmd_list = cmd_list + list(args)
            entry["command"] = cmd_list
            env_block = config.pop("env", None)
            if env_block:
                entry["environment"] = env_block
        out[server.name] = entry
    return out


def _clean_opencode_error(stdout: str, stderr: str) -> str:
    """Trim OpenCode CLI output to a manageable size for error messages."""
    combined = f"{stdout}\n{stderr}".strip()
    max_len = 2000
    if len(combined) > max_len:
        combined = combined[:max_len] + "... (truncated)"
    return combined if combined else "Unknown error (no output)"


async def _run_opencode_centaur(
    options: CentaurOptions,
    opencode_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    instructions = (
        "OpenCode:\n\n"
        " - You may also use OpenCode via the 'opencode' command.\n"
        " - Use 'opencode run --continue' if you need to resume a previous opencode session."
    )

    # build .bashrc content - only export vars needed for the opencode alias,
    # not HOME which would break human_cli (PATH is needed for node)
    centaur_env = {k: v for k, v in agent_env.items() if k != "HOME"}
    agent_env_vars = [f'export {k}="{v}"' for k, v in centaur_env.items()]
    alias_cmd = shlex.join(opencode_cmd)
    alias_cmd = "alias opencode='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(agent_env_vars + ["", alias_cmd])

    await run_centaur(options, instructions, bashrc, state)
