"""Claude Code agent via the ``claude-code-acp`` ACP adapter."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from inspect_ai.agent import AgentState, SandboxAgentBridge, agent, sandbox_agent_bridge
from inspect_ai.model import Model, get_model
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import ExecRemoteProcess, ExecRemoteStreamingOptions
from inspect_ai.util import sandbox as sandbox_env
from typing_extensions import Unpack

from inspect_swe._util.path import join_path
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

from .agentbinary import ensure_claude_code_acp_setup

logger = logging.getLogger(__name__)


class ClaudeCode(ACPAgent):
    """Claude Code agent via the ``claude-code-acp`` ACP adapter.

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
        super().__init__(**kwargs)

    def _build_model_map(self) -> dict[str, str | Model]:
        """Build model map from all configured CC model names."""
        model_map = super()._build_model_map()
        for entry in (
            self._opus_model,
            self._sonnet_model,
            self._haiku_model,
            self._subagent_model,
        ):
            if entry is not None:
                model = get_model(entry)
                model_map[model.canonical_name()] = model
        return model_map

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, SandboxAgentBridge]]:
        sbox = sandbox_env(self.sandbox)
        default_model = get_model(self.model).canonical_name()

        async with sandbox_agent_bridge(
            state,
            model=None,
            model_aliases=self.model_map,
            filter=self.filter,
            retry_refusals=self.retry_refusals,
            bridged_tools=self.bridged_tools or None,
        ) as bridge:
            # Install node and claude-code-acp in the sandbox.
            acp_binary, node_binary = await ensure_claude_code_acp_setup(
                sbox, self.user
            )
            node_dir = str(Path(node_binary).parent)

            # Use canonical model names — the bridge resolves them via
            # model_aliases to Model instances directly.
            agent_env = {
                "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                "ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-DOq5tyLPrk9M4hPE",
                "ANTHROPIC_MODEL": default_model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": get_model(
                    self._opus_model
                ).canonical_name()
                if self._opus_model
                else default_model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": get_model(
                    self._sonnet_model
                ).canonical_name()
                if self._sonnet_model
                else default_model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": get_model(
                    self._haiku_model
                ).canonical_name()
                if self._haiku_model
                else default_model,
                "CLAUDE_CODE_SUBAGENT_MODEL": get_model(
                    self._subagent_model
                ).canonical_name()
                if self._subagent_model
                else default_model,
                "ANTHROPIC_SMALL_FAST_MODEL": get_model(
                    self._haiku_model
                ).canonical_name()
                if self._haiku_model
                else default_model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                "IS_SANDBOX": "1",
                "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
            } | self.env

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
            logger.info("Starting claude-code-acp adapter...")
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

    Uses the ``claude-code-acp`` adapter in a sandbox.  Supports
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
