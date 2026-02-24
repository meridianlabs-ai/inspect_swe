"""Codex CLI agent via the ``codex-acp`` ACP adapter."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from inspect_ai.agent import AgentState, agent, sandbox_agent_bridge
from inspect_ai.model import get_model
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import (
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    SandboxEnvironment,
    store,
)
from inspect_ai.util import sandbox as sandbox_env
from typing_extensions import Unpack

from inspect_swe._util.path import join_path
from inspect_swe._util.sandbox import sandbox_exec
from inspect_swe._util.toml import to_toml
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

from .agentbinary import ensure_codex_acp_setup

logger = logging.getLogger(__name__)


class CodexCli(ACPAgent):
    """Codex CLI agent via the ``codex-acp`` ACP adapter.

    Subclasses :class:`ACPAgent` to provide Codex-specific setup
    (bridge, env vars, AGENTS.md, skills).
    """

    def __init__(
        self,
        *,
        disallowed_tools: list[Literal["web_search"]] | None = None,
        skills: list[str | Path | Skill] | None = None,
        home_dir: str | None = None,
        config_overrides: dict[str, str] | None = None,
        **kwargs: Unpack[ACPAgentParams],
    ) -> None:
        self._disallowed_tools = list(disallowed_tools or [])
        self._resolved_skills = read_skills(skills) if skills else None
        self._home_dir = home_dir
        self._config_overrides = config_overrides or {}
        super().__init__(**kwargs)

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, object]]:
        sbox = sandbox_env(self.sandbox)
        default_model = get_model(self.model).canonical_name()

        # Use a unique port per sample to avoid conflicts with codex-core's
        # internal services (mirrors the non-ACP codex_cli approach).
        MODEL_PORT = "codex_acp_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=None,
            model_aliases=self.model_map,
            filter=self.filter,
            retry_refusals=self.retry_refusals,
            bridged_tools=self.bridged_tools or None,
            port=port,
        ) as bridge:
            # Install node and codex-acp in the sandbox.
            acp_binary, node_binary = await ensure_codex_acp_setup(
                sbox, self.user
            )
            node_dir = str(Path(node_binary).parent)

            # Resolve CODEX_HOME (mirrors the non-ACP codex_cli agent).
            if self._home_dir is None:
                working_dir = await sandbox_exec(
                    sbox, "pwd", user=self.user, cwd=self.cwd
                )
                codex_home = join_path(working_dir, ".codex")
            else:
                codex_home = await sandbox_exec(
                    sbox,
                    f'eval echo "{self._home_dir}"',
                    user=self.user,
                    cwd=self.cwd,
                )
            await sandbox_exec(sbox, cmd=f"mkdir -p {codex_home}", user=self.user)

            # Write system prompt to AGENTS.md (Codex convention).
            resolved_prompt = self._resolve_system_prompt(state)
            if resolved_prompt:
                agents_md_path = self._agents_md_path(codex_home)
                await sbox.write_file(agents_md_path, resolved_prompt)

            # Install skills.
            if self._resolved_skills:
                skills_dir = join_path(codex_home, "skills")
                await install_skills(
                    self._resolved_skills, sbox, self.user, skills_dir
                )

            # Write config.toml with model provider pointing at the bridge.
            # Use the canonical model name so the bridge can resolve it
            # via model_aliases (consistent with how claude-code-acp passes
            # ANTHROPIC_MODEL).
            bridge_url = f"http://127.0.0.1:{bridge.port}/v1"
            config_toml_path = await self._config_toml_path(sbox, codex_home)
            toml_config: dict[str, Any] = {
                "model": default_model,
                "preferred_auth_method": "apikey",
                "model_provider": "openai-proxy",
                "model_providers.openai-proxy": {
                    "name": "OpenAI Proxy",
                    "base_url": bridge_url,
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                },
            }
            toml_config.update(self._config_overrides)
            await sbox.write_file(config_toml_path, to_toml(toml_config))

            # Environment variables (same as the non-ACP codex agent).
            agent_env = {
                "CODEX_HOME": codex_home,
                "OPENAI_API_KEY": "api-key",
                "OPENAI_BASE_URL": bridge_url,
                "RUST_LOG": "warning",
                "NO_BROWSER": "1",
                "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
            } | self.env

            # Start ACP adapter process.
            logger.info("Starting codex-acp adapter...")
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

    def _agents_md_path(self, codex_home: str) -> str:
        """Determine where to write AGENTS.md."""
        if self._home_dir is not None:
            return join_path(codex_home, "AGENTS.md")
        elif self.cwd is not None:
            return join_path(self.cwd, "AGENTS.md")
        return "AGENTS.md"

    async def _config_toml_path(
        self,
        sbox: SandboxEnvironment,
        codex_home: str,
    ) -> str:
        """Determine where to write config.toml."""
        if self._home_dir is not None:
            return join_path(codex_home, "config.toml")
        directory = (
            ".codex" if self.cwd is None else join_path(self.cwd, ".codex")
        )
        await sandbox_exec(sbox, cmd=f"mkdir -p {directory}", user=self.user)
        return join_path(directory, "config.toml")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@agent(name="Codex CLI")
def interactive_codex_cli(
    *,
    # Codex-specific
    disallowed_tools: list[Literal["web_search"]] | None = None,
    skills: list[str | Path | Skill] | None = None,
    home_dir: str | None = None,
    config_overrides: dict[str, str] | None = None,
    # Forwarded to ACPAgent
    **kwargs: Unpack[ACPAgentParams],
) -> ACPAgent:
    """Codex CLI agent via ACP.

    Uses the ``codex-acp`` adapter in a sandbox.  Supports
    multi-turn sessions and mid-turn interrupts.

    Args:
        disallowed_tools: Tools to disable (currently only ``"web_search"``).
        skills: Additional skills to make available.
        home_dir: Override for ``CODEX_HOME`` directory in the sandbox.
        config_overrides: Extra Codex config.toml key-value pairs.
        **kwargs: See :class:`ACPAgentParams` for all base options.
    """
    return CodexCli(
        disallowed_tools=disallowed_tools,
        skills=skills,
        home_dir=home_dir,
        config_overrides=config_overrides,
        **kwargs,
    )
