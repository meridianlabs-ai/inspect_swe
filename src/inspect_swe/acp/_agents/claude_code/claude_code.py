"""Claude Code agent via the ``claude-agent-acp`` ACP adapter."""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inspect_ai.agent import AgentState, SandboxAgentBridge, agent, sandbox_agent_bridge
from inspect_ai.model import GenerateFilter, Model, get_model
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import ExecRemoteProcess, ExecRemoteStreamingOptions, store
from inspect_ai.util import sandbox as sandbox_env
from typing_extensions import Unpack

from inspect_swe._claude_code.env import DISABLE_AUTO_MEMORY_ENV
from inspect_swe._claude_code.model import distinct_subagent_name
from inspect_swe._util.agentcontext import (
    ModelFilter,
    classify_filter,
    slug_map_classifier,
)
from inspect_swe._util.path import join_path
from inspect_swe._util.websearch import web_search_tool_disallowed
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

from .agentbinary import ensure_claude_code_acp_setup

logger = logging.getLogger(__name__)


# Claude Code has several client-side watchdogs that abort an in-flight
# request and re-send it when the bridged model call stalls. A bridge
# GenerateFilter that deliberately blocks the model proxy trips them, and
# the retry replays a stale request. Disable them by default; user ``env``
# still overrides.
_BRIDGE_SAFE_ENV: dict[str, str] = {
    # Bun fetch requestTimeout
    "API_FORCE_IDLE_TIMEOUT": "0",
    # SDK client timeout — no disable sentinel, so use a large value
    "API_TIMEOUT_MS": "100000000",
    # SSE event-idle watchdog (dead code as of CC 2.1.220 — the chunk-idle
    # byte watchdog below replaced it; kept for older CC versions)
    "CLAUDE_ENABLE_STREAM_WATCHDOG": "0",
    # SSE chunk-idle byte watchdog (~180 s default, remotely tunable via a
    # feature gate). Currently only arms on first-party base URLs, so it does
    # not fire behind the bridge's localhost ANTHROPIC_BASE_URL — disabled
    # explicitly rather than relying on that implementation detail
    "CLAUDE_ENABLE_BYTE_WATCHDOG": "0",
    # Idle watchdog on bridged MCP tool calls
    "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT": "0",
    # Block the first model call until MCP servers are connected; otherwise a
    # slow bridge handshake yields a first call with no bridged tools (only a
    # WaitForMcpServers placeholder) and the sample silently proceeds toolless.
    # Inverted polarity: unset and "1" both mean non-blocking; only an explicit
    # falsy token ("0"/"false"/"no"/"off") blocks
    "MCP_CONNECTION_NONBLOCKING": "0",
    # MCP server connect/init handshake (defaults 5 s / 30 s); on timeout the
    # server is silently dropped and its tools never appear. 300 s to cover
    # slow sandbox backends under many-concurrent-samples startup contention
    "MCP_CONNECT_TIMEOUT_MS": "300000",
    "MCP_TIMEOUT": "300000",
    # No telemetry or update checks from inside the sandbox
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    # Auto-memory targets future conversations a sandbox never has, and its
    # system-prompt section diverts the model from task-provided memory tools
    **DISABLE_AUTO_MEMORY_ENV,
}


@dataclass(frozen=True)
class ClaudeCodeAcpModels:
    """Presented per-role model names + bridge aliases for a Claude Code (ACP) run.

    ACP counterpart of `_claude_code.model.ClaudeCodeModels`: ``presented``
    and the role names are the values handed to Claude Code via
    ``ANTHROPIC_MODEL`` and the per-role env vars in `_start_agent`;
    ``aliases`` maps each of them to the `Model` the bridge should serve
    (the *derived* table -- `ACPAgent` layers the caller's ``model_map``
    override on top when building the agent's ``model_map``).
    Same invariant as the native path: ``subagent`` never equals
    ``presented``, ``opus``, ``sonnet``, or ``haiku``, so sub-agent traffic
    is distinguishable by requested slug alone.
    """

    presented: str
    opus: str
    sonnet: str
    haiku: str
    subagent: str
    aliases: dict[str, Model]


def resolve_claude_code_acp_models(
    model: str | Model,
    *,
    opus_model: str | Model | None = None,
    sonnet_model: str | Model | None = None,
    haiku_model: str | Model | None = None,
    subagent_model: str | Model | None = None,
    model_map: Mapping[str, str | Model] | None = None,
) -> ClaudeCodeAcpModels:
    """Resolve the model names Claude Code (ACP) presents and their bridge aliases.

    Kept separate from `_claude_code.model.resolve_claude_code_models`
    because this variant keys the bridge on ``canonical_name()`` (the native
    path uses bare ``.name``) and accepts `Model` instances whose bound
    config must survive (the native signature takes ``str`` and
    re-resolves). The subagent-distinctness invariant is shared, via
    `distinct_subagent_name`: an unset or colliding ``subagent_model`` is
    presented as ``"<presented>-subagent"`` (first free ``-N`` suffix if a
    role already claims that name) and aliased to the intended served
    model, so only the label differs.

    ``model_map`` is the caller's `ACPAgent` override (``ACPAgentParams``'
    ``model_map``), which `ACPAgent` applies on top of ``aliases`` *after*
    this function returns. It is consulted here for one thing: with
    ``subagent_model`` unset the synthetic slug is only a label for the
    primary's route, so when the caller re-maps the presented name the
    synthetic slug follows that override (resolved via ``get_model()``,
    exactly as the bridge resolves alias targets) rather than silently
    staying on the un-overridden served model while main-thread traffic
    moves. A caller mapping for the synthetic slug itself is left to
    `ACPAgent` to apply and wins; an explicit ``subagent_model`` is never
    redirected by a presented-name override.

    Calls ``get_model()``, so it needs an active eval/sample; `ACPAgent`
    already requires one at construction, which is where `_build_model_map`
    invokes this.
    """
    served = get_model(model)
    presented = served.canonical_name()
    aliases: dict[str, Model] = {presented: served}

    def role_name(role_model: str | Model | None) -> str:
        if role_model is None:
            return presented
        role = get_model(role_model)
        aliases[role.canonical_name()] = role
        return role.canonical_name()

    opus = role_name(opus_model)
    sonnet = role_name(sonnet_model)
    haiku = role_name(haiku_model)

    taken = {presented, opus, sonnet, haiku}
    if subagent_model is None:
        subagent = distinct_subagent_name(presented, taken)
        subagent_route = served
        # follow a caller override of the presented name (see docstring)
        if model_map and presented in model_map and subagent not in model_map:
            subagent_route = get_model(model_map[presented])
    else:
        subagent_route = get_model(subagent_model)
        subagent = subagent_route.canonical_name()
        if subagent in taken:
            subagent = distinct_subagent_name(presented, taken)
    aliases[subagent] = subagent_route

    return ClaudeCodeAcpModels(
        presented=presented,
        opus=opus,
        sonnet=sonnet,
        haiku=haiku,
        subagent=subagent,
        aliases=aliases,
    )


def build_claude_code_acp_filter(
    filter: GenerateFilter | None, models: ClaudeCodeAcpModels
) -> ModelFilter:
    """Claude Code (ACP) bridge filter: agent-context classification by requested slug.

    This ACP adapter (``claude-agent-acp``) has no JSONL/event stream for a
    `LiveConsumer` to parse, so there's no pending-subagent tracking or
    prompt substring matching available -- just the per-role names Claude
    Code was given (`_start_agent`), mirroring the structural (slug) half of
    `LiveConsumer.classify`'s truth table.

    ``opus``/``sonnet`` are configured *tiers* of main-thread traffic
    (Claude Code's own role swap), not delegation, so they are root.
    ``subagent`` is always distinct (see `resolve_claude_code_acp_models`)
    and classifies "subagent". ``haiku`` classifies "utility" only when it
    is its own slug: an unset or root-colliding ``haiku_model`` puts
    small-fast traffic on a root slug, where it is indistinguishable, so it
    is left to classify root rather than registered as a known collision.
    """
    root_slugs = {models.presented, models.opus, models.sonnet}
    kind_by_slug: dict[str, Literal["subagent", "utility"]] = {
        models.subagent: "subagent"
    }
    if models.haiku not in root_slugs:
        kind_by_slug[models.haiku] = "utility"
    return classify_filter(filter, slug_map_classifier(root_slugs, kind_by_slug))


class ClaudeCode(ACPAgent):
    """Claude Code agent via the ``claude-agent-acp`` ACP adapter.

    Subclasses :class:`ACPAgent` to provide Claude-specific setup
    (bridge, env vars, MCP config, skills).
    """

    def __init__(
        self,
        *,
        disallowed_tools: list[str] | None = None,
        skills: list[str | Path | Skill] | None = None,
        opus_model: str | Model | None = None,
        sonnet_model: str | Model | None = None,
        haiku_model: str | Model | None = None,
        subagent_model: str | Model | None = None,
        **kwargs: Unpack[ACPAgentParams],
    ) -> None:
        self._disallowed_tools = list(disallowed_tools or [])
        self._resolved_skills = read_skills(skills) if skills else None
        self._opus_model: str | Model | None = opus_model
        self._sonnet_model: str | Model | None = sonnet_model
        self._haiku_model: str | Model | None = haiku_model
        self._subagent_model: str | Model | None = subagent_model
        # ACPAgent applies this override to model_map after _build_model_map;
        # the resolver needs it so the default subagent follows the presented
        # name's override (see resolve_claude_code_acp_models)
        self._model_map_override = kwargs.get("model_map")
        super().__init__(**kwargs)

    def _resolve_models(self) -> ClaudeCodeAcpModels:
        return resolve_claude_code_acp_models(
            self.model,
            opus_model=self._opus_model,
            sonnet_model=self._sonnet_model,
            haiku_model=self._haiku_model,
            subagent_model=self._subagent_model,
            model_map=self._model_map_override,
        )

    def _build_model_map(self) -> dict[str, str | Model]:
        """Build model map from all presented CC model names (incl. the subagent alias)."""
        model_map = super()._build_model_map()
        model_map.update(self._resolve_models().aliases)
        return model_map

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, SandboxAgentBridge]]:
        sbox = sandbox_env(self.sandbox)
        models = self._resolve_models()

        # Use a unique port per agent invocation so re-running the agent in the
        # same sandbox doesn't collide with a stale model_proxy on 13131
        # (mirrors the non-ACP claude_code and the ACP codex/gemini agents).
        MODEL_PORT = "claude_code_acp_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=None,
            model_aliases=self.model_map,
            filter=build_claude_code_acp_filter(self.filter, models),
            retry_refusals=self.retry_refusals,
            bridged_tools=self.bridged_tools or None,
            web_search=not web_search_tool_disallowed(
                self._disallowed_tools, "WebSearch"
            ),
            port=port,
        ) as bridge:
            # Install node and claude-agent-acp in the sandbox.
            acp_binary, node_binary = await ensure_claude_code_acp_setup(
                sbox, self.user
            )
            node_dir = str(Path(node_binary).parent)

            # Presented (canonical) model names — the bridge resolves them via
            # model_aliases (self.model_map) to Model instances directly.
            agent_env = (
                _BRIDGE_SAFE_ENV
                | {
                    "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                    "ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-DOq5tyLPrk9M4hPE",
                    "ANTHROPIC_MODEL": models.presented,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": models.opus,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": models.sonnet,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": models.haiku,
                    "CLAUDE_CODE_SUBAGENT_MODEL": models.subagent,
                    "ANTHROPIC_SMALL_FAST_MODEL": models.haiku,
                    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                    "IS_SANDBOX": "1",
                    "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
                }
                | self.env
            )

            # System prompt via env (the ACP adapter will forward to CC)
            resolved_prompt = self._resolve_system_prompt(state)
            if resolved_prompt:
                agent_env["CLAUDE_CODE_APPEND_SYSTEM_PROMPT"] = resolved_prompt

            # Disallowed tools
            if self._disallowed_tools:
                agent_env["CLAUDE_CODE_DISALLOWED_TOOLS"] = ",".join(
                    self._disallowed_tools
                )

            # Install skills
            if self._resolved_skills:
                skills_dir = join_path(self.cwd, ".claude/skills")
                await install_skills(self._resolved_skills, sbox, self.user, skills_dir)

            # Start ACP adapter process
            logger.info("Starting claude-agent-acp adapter...")
            proc = await sbox.exec_remote(
                cmd=[acp_binary],
                options=ExecRemoteStreamingOptions(
                    stdin_open=True,
                    cwd=self.cwd,
                    env=agent_env,
                    user=self.user,
                ),
            )

            yield proc, bridge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@agent(name="Claude Code")
def interactive_claude_code(
    *,
    # Claude-specific
    disallowed_tools: list[str] | None = None,
    skills: list[str | Path | Skill] | None = None,
    opus_model: str | Model | None = None,
    sonnet_model: str | Model | None = None,
    haiku_model: str | Model | None = None,
    subagent_model: str | Model | None = None,
    # Forwarded to ACPAgent
    **kwargs: Unpack[ACPAgentParams],
) -> ACPAgent:
    """Claude Code agent via ACP.

    Uses the ``claude-agent-acp`` adapter in a sandbox.  Supports
    multi-turn sessions and mid-turn interrupts.

    Args:
        disallowed_tools: Tool names to disallow.
        skills: Additional skills to make available.
        opus_model: Model for opus calls.
        sonnet_model: Model for sonnet calls.
        haiku_model: Model for haiku / background calls.
        subagent_model: Model for subagents.
        **kwargs: See :class:`ACPAgentParams` for all base options.
    """
    return ClaudeCode(
        disallowed_tools=disallowed_tools,
        skills=skills,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        haiku_model=haiku_model,
        subagent_model=subagent_model,
        **kwargs,
    )
