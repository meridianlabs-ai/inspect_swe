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
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model, get_model
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
#   1. Slug collision: OpenCode's internal title-generator used its default
#      Anthropic small model, which happened to collide with the sentinel
#      probe-worker slug the run picked ad hoc. `small_model` is therefore
#      set *explicitly* to its own sentinel here so title/summary/
#      compaction traffic is intended to land under a slug distinct from
#      both the primary and the subagent sentinel, classified "utility" --
#      BUT this task's live-verification run found `small_model` alone does
#      NOT actually redirect title-generation: with a real catalog
#      `small_model` sentinel set, the title-gen request still arrived under
#      the *primary* slug (not even OpenCode's own hardcoded small-model
#      default, which is what it used with no override at all -- see probe
#      P2). No error surfaced either time; it's a silent no-op, not a
#      rejection. The config schema separately exposes `title`/`summary`/
#      `compaction` as their own named `AgentConfig` entries under `agent`
#      (`https://opencode.ai/config.json`) alongside `general`/`explore` --
#      the same per-agent `model` override mechanism this run DID prove
#      works for `general` (see caveat 2). So `OPENCODE_UTILITY_AGENTS`
#      routes `title`/`summary`/`compaction` through the small-model
#      sentinel via that mechanism too, belt-and-braces alongside
#      `small_model` -- plausible given the proven `general` precedent, but
#      UNVERIFIED (this task's 2-live-run budget was spent confirming the
#      subagent-sentinel fallback; a follow-up run should confirm this
#      before relying on "utility" classification for title/summary/
#      compaction traffic).
#   2. Built-in-subagent overridability was untested by P2 (it only proved
#      per-agent config works for a *custom* agent). `general`, `explore`,
#      and `scout` are the three built-in subagents OpenCode ships
#      (https://opencode.ai/docs/agents/, opencode-ai 1.18.x). We route all
#      three through the subagent sentinel, setting *only* `model` on each
#      so their built-in description/prompt/mode (if any) are left alone;
#      if a given install doesn't actually ship one of these names the
#      entry is simply an inert, never-invoked custom-agent definition (the
#      config schema permits arbitrary agent keys via `additionalProperties`
#      and every `AgentConfig` field is optional, so this can't fail config
#      parsing). Live-verified (see #3): OpenCode DOES read `general`'s
#      config override and attempt to resolve its `model` -- proving the
#      per-built-in override mechanism itself works -- it was the *sentinel
#      value's* catalog rejection (#3), not the override mechanism, that
#      failed the first live run.
#   3. Sentinel catalog constraint (LIVE-VERIFIED, 2026-08-08, docker,
#      opencode-ai 1.18.15): although the config *schema* places no catalog
#      constraint on `AgentConfig.model`/`small_model` (both typed plain
#      `string`, per https://opencode.ai/config.json), OpenCode's *runtime*
#      model resolution does validate against a known catalog. A first
#      attempt using a non-catalog synthetic id
#      (`anthropic/inspect-subagent`) was REJECTED: the `general` subagent's
#      Task-tool call failed outright with the tool result
#      `"Model not found: anthropic/inspect-subagent."` (the main agent then
#      improvised by running the shell command itself rather than via the
#      subagent) -- so unlike `claude_code` (whose CLI performs no such
#      validation, see `_claude_code/model.py`), OpenCode's synthetic-slug
#      approach does not transfer. `small_model` failed the same lookup but
#      degraded silently instead of surfacing an error -- title-generation
#      requests were observed carrying the *primary* slug, not the sentinel,
#      confirming the rejection without a visible failure. `_SENTINEL_MODELS`
#      below is the fallback this task's instructions anticipated for
#      exactly this outcome: real, distinct, same-provider catalog ids
#      aliased to the served model (same mechanism as the rejected
#      approach, just real identities) -- re-verified live after switching
#      to it (see PR description / task report for the resulting jsonl).
#      The anthropic pair is live-verified both ways (rejected as synthetic,
#      accepted as catalog names); openai/google pairs are UNVERIFIED
#      best-effort by the same reasoning (no probe/live run has exercised a
#      non-anthropic `opencode_model`) -- flagged as a concern pending
#      verification. An `opencode_model` under any other provider skips
#      sentinel injection entirely (logged once) rather than risk the same
#      "Model not found" failure mode with a guessed id.
#   4. Primary-collision guard (spec review, 2026-08-08): since sentinels
#      must be real catalog ids (caveat 3), a caller's `opencode_model` can
#      legitimately BE one of our fixed sentinel picks (e.g.
#      `opencode_model="anthropic/claude-haiku-4-5-20251001"`, our exact
#      subagent-sentinel default) -- that would put the same bare slug in
#      both `slug_map_classifier`'s `root_slugs` and `kind_by_slug`, and
#      root wins (checked first), silently swallowing all subagent/utility
#      classification. `_SENTINEL_MODELS` therefore holds an ORDERED
#      3-candidate preference list per role per provider, not a single id;
#      `build_opencode_config_overrides` picks the first candidate that
#      collides with neither the primary's bare id nor (for the small-model
#      role) the already-chosen subagent sentinel. If every candidate for a
#      role collides (degenerate: caller's `opencode_model` IS the whole
#      preference list), that role's override is omitted entirely and its
#      traffic falls back to the existing "unknown" classification rather
#      than a silently-wrong "root". Only the first (default) candidate per
#      role is live-verified (caveat 3); the alternates are real,
#      catalog-plausible model ids chosen by the same reasoning but not
#      independently live-verified as *sentinels* (they're genuine model
#      names, just unexercised in this specific role). `build_opencode_filter`
#      additionally asserts (by construction, not by trusting callers) that
#      `root_slugs` and `kind_by_slug` never share a key: any collision is
#      logged and the colliding `kind_by_slug` entry is dropped rather than
#      raised, so a direct/unusual call into these builders degrades to
#      "unknown" instead of misclassifying as "root".
#
# Regardless of provider, the OpenCode provider clients put only the bare
# model id (no `provider/` prefix) in the wire request's `model` field --
# confirmed against probe P2 traffic, where a project config's
# `anthropic/claude-haiku-4-5-20251001` subagent model arrived at
# `current_bridge_request().model` as `claude-haiku-4-5-20251001` -- so
# classification and the bridge's `model_aliases` keys both use the bare
# forms, never the `provider/`-prefixed config values.

_SENTINEL_MODELS: dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]] = {
    # provider_id -> (subagent_candidates, small_model_candidates), each an
    # ORDERED 3-candidate preference list of bare (no `provider/` prefix)
    # REAL catalog model ids (see caveat 4 above for why a list rather than
    # a single fixed id). `_select_sentinel` picks the first candidate that
    # doesn't collide with the primary/already-chosen sentinel. Only each
    # list's first (default) entry is live-verified for the anthropic
    # provider (see caveat 3); all other entries/providers are unverified.
    "anthropic": (
        (
            "claude-haiku-4-5-20251001",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
        ),
        (
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ),
    ),
    "openai": (
        ("gpt-5-mini", "gpt-5-nano", "gpt-4o-mini"),
        ("gpt-5-nano", "gpt-4o-mini", "gpt-3.5-turbo"),
    ),
    "google": (
        ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"),
        ("gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"),
    ),
}

OPENCODE_BUILTIN_SUBAGENTS: tuple[str, ...] = ("general", "explore", "scout")
"""Built-in OpenCode subagents (opencode.ai/docs/agents/), routed to the
subagent sentinel so their traffic is slug-distinguishable from the main
thread even when invoked without a `model:` override of their own.
Live-verified (2026-08-08, `general`): the per-agent override IS honored."""

OPENCODE_UTILITY_AGENTS: tuple[str, ...] = ("title", "summary", "compaction")
"""Built-in OpenCode utility agents (`AgentConfig` entries per
`https://opencode.ai/config.json`), routed to the small-model sentinel via
the same per-agent `model` override mechanism `OPENCODE_BUILTIN_SUBAGENTS`
uses -- belt-and-braces alongside `small_model` (see caveat 1 above; NOT
independently live-verified)."""


def _bare_model_id(model_ref: str) -> str:
    """Strip a `provider/model` config value down to the bare model id.

    Mirrors what OpenCode's provider clients actually place in the wire
    request's `model` field (see module docstring) -- the form
    `current_bridge_request().model` carries, and so the form the
    classifier and `model_aliases` keys below must match against.
    """
    return model_ref.split("/", 1)[1] if "/" in model_ref else model_ref


def _select_sentinel(
    candidates: tuple[str, str, str], excluded: Sequence[str]
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

    When `provider_id` isn't in `_SENTINEL_MODELS`, returns an empty
    fragment and `(None, None)` — no sentinel injection is attempted (see
    caveat 3), and a warning is logged once per provider id.

    Otherwise, picks each sentinel from its provider's ordered candidate
    list, skipping any candidate whose bare id collides with `opencode_model`
    (and, for the small-model role, with the already-chosen subagent
    sentinel too) — see caveat 4. Either pick may independently come back
    `None` (with a logged warning) if every candidate for that role
    collides; that role's override is simply omitted from the config rather
    than risking a slug that silently reclassifies as "root".
    """
    sentinel_candidates = _SENTINEL_MODELS.get(provider_id)
    if sentinel_candidates is None:
        logger.warning(
            f"opencode(): no known catalog sentinel models for provider "
            f"{provider_id!r}; subagent/utility traffic will not be "
            f"slug-distinguishable from root for this opencode_model."
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

    agent_overrides: dict[str, Any] = {}
    if subagent_sentinel is not None:
        agent_overrides.update(
            {name: {"model": subagent_sentinel} for name in OPENCODE_BUILTIN_SUBAGENTS}
        )
    if small_model_sentinel is not None:
        agent_overrides.update(
            {name: {"model": small_model_sentinel} for name in OPENCODE_UTILITY_AGENTS}
        )

    config_fragment: dict[str, Any] = {}
    if small_model_sentinel is not None:
        config_fragment["small_model"] = small_model_sentinel
    if agent_overrides:
        config_fragment["agent"] = agent_overrides

    return config_fragment, subagent_sentinel, small_model_sentinel


def build_opencode_model_aliases(
    served_model: Model,
    model_aliases: dict[str, str | Model] | None,
    subagent_sentinel: str | None,
    small_model_sentinel: str | None,
) -> dict[str, str | Model]:
    """Bridge `model_aliases` routing the sentinel slugs to `served_model`.

    Not strictly required for correct routing (the bridge's fallback model
    already serves any unaliased slug -- see `resolve_inspect_model`), but
    made explicit here so the aliasing is visible/introspectable and
    mirrors how `claude_code`'s `resolve_claude_code_models` handles its own
    synthetic subagent slug. Caller-supplied `model_aliases` take precedence
    over the sentinel entries. Either sentinel may be `None` (unrecognized
    provider — see `build_opencode_config_overrides`), in which case no
    alias entry is added for it.
    """
    sentinel_aliases: dict[str, str | Model] = {}
    if subagent_sentinel is not None:
        sentinel_aliases[_bare_model_id(subagent_sentinel)] = served_model
    if small_model_sentinel is not None:
        sentinel_aliases[_bare_model_id(small_model_sentinel)] = served_model
    return {**sentinel_aliases, **(model_aliases or {})}


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

    Invariant (caveat 4 above): `root_slugs` and `kind_by_slug` must never
    share a key — `slug_map_classifier` checks `root_slugs` first, so a
    shared key would silently classify that sentinel's traffic "root"
    instead of "subagent"/"utility". `build_opencode_config_overrides`
    already picks collision-free sentinels, so this shouldn't trigger via
    the normal `opencode()` call path; it's enforced here too (rather than
    trusted from the caller) since this function is itself importable and
    callable directly. A collision is logged and the colliding
    `kind_by_slug` entry is dropped — never raised — so the request falls
    back to the existing "unknown"/"root" classification instead of a
    silently wrong one.
    """
    root_slug = _bare_model_id(opencode_model)
    kind_by_slug: dict[str, Literal["subagent", "utility"]] = {}
    if subagent_sentinel is not None:
        kind_by_slug[_bare_model_id(subagent_sentinel)] = "subagent"
    if small_model_sentinel is not None:
        kind_by_slug[_bare_model_id(small_model_sentinel)] = "utility"

    if root_slug in kind_by_slug:
        logger.warning(
            f"opencode(): sentinel slug {root_slug!r} collides with the "
            f"root slug (opencode_model {opencode_model!r}); dropping its "
            f"subagent/utility classification, which would otherwise be "
            f"unreachable (slug_map_classifier checks root_slugs first)."
        )
        del kind_by_slug[root_slug]

    return classify_filter(filter, slug_map_classifier({root_slug}, kind_by_slug))


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

        # resolve model — must happen at execution time: get_model() resolves
        # the active model from the current eval/sample context
        served_model = get_model(model)
        bridge_model = f"inspect/{model}" if model is not None else "inspect"

        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases=build_opencode_model_aliases(
                served_model, model_aliases, subagent_sentinel, small_model_sentinel
            ),
            filter=opencode_filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
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
            opencode_config: dict[str, Any] = {
                "$schema": "https://opencode.ai/config.json",
                "provider": {
                    provider_id: {"options": {"baseURL": provider_base_url}},
                },
                **agent_context_config,
            }
            if resolved_skills is not None:
                opencode_config["permission"] = {"skill": {"*": "allow"}}
            if all_mcp_servers:
                opencode_config["mcp"] = resolve_mcp_servers(all_mcp_servers)

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

                    # run agent
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
