"""Gemini CLI agent via native ``--experimental-acp`` support."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from inspect_ai.agent import AgentState, agent, sandbox_agent_bridge
from inspect_ai.model import Model, get_model
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import ExecRemoteProcess, ExecRemoteStreamingOptions, store
from inspect_ai.util import sandbox as sandbox_env
from typing_extensions import Unpack

from inspect_swe._gemini_cli.agentbinary import ensure_gemini_cli_setup
from inspect_swe._util.path import join_path
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

logger = logging.getLogger(__name__)


class GeminiCli(ACPAgent):
    """Gemini CLI agent via native ACP support.

    Subclasses :class:`ACPAgent` to provide Gemini-specific setup.
    Uses gemini's built-in ``--experimental-acp`` flag — no separate
    ACP adapter package needed.
    """

    def __init__(
        self,
        *,
        skills: list[str | Path | Skill] | None = None,
        version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
        **kwargs: Unpack[ACPAgentParams],
    ) -> None:
        self._resolved_skills = read_skills(skills) if skills else None
        self._version = version
        super().__init__(**kwargs)

    def _build_model_map(self) -> dict[str, str | Model]:
        """Add slash-free model name for Google API URL path compatibility."""
        model_map = super()._build_model_map()
        model = get_model(self.model)
        model_map[model.name] = model
        return model_map

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, object]]:
        sbox = sandbox_env(self.sandbox)
        model = get_model(self.model)

        # Use a unique port per sample (mirrors non-ACP gemini_cli approach).
        MODEL_PORT = "gemini_acp_model_port"
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
            # Install node and gemini CLI in the sandbox.
            gemini_binary, node_binary = await ensure_gemini_cli_setup(
                sbox, self._version, self.user
            )
            node_dir = str(Path(node_binary).parent)

            # Detect sandbox home directory.
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=self.user)
            sandbox_home = home_result.stdout.strip() or "/root"

            # Install skills.
            if self._resolved_skills:
                GEMINI_SKILLS = ".gemini/skills"
                skills_dir = (
                    join_path(self.cwd, GEMINI_SKILLS)
                    if self.cwd is not None
                    else GEMINI_SKILLS
                )
                await install_skills(
                    self._resolved_skills, sbox, self.user, skills_dir
                )

            # Environment variables (matching non-ACP gemini_cli agent).
            agent_env = {
                "GOOGLE_GEMINI_BASE_URL": f"http://127.0.0.1:{bridge.port}",
                "GEMINI_API_KEY": "api-key",
                "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
                "HOME": sandbox_home,
            } | self.env

            # MCP servers are passed via the ACP protocol in the base class
            # (conn.new_session(mcp_servers=...)). Gemini's --experimental-acp
            # mode natively supports this and merges them into its tool registry.

            # Start gemini in ACP mode.
            logger.info("Starting gemini CLI in ACP mode...")
            proc = await sbox.exec_remote(
                cmd=[
                    gemini_binary,
                    "--experimental-acp",
                    "--model",
                    model.name,
                ],
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


@agent(name="Gemini CLI")
def interactive_gemini_cli(
    *,
    # Gemini-specific
    skills: list[str | Path | Skill] | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    # Forwarded to ACPAgent
    **kwargs: Unpack[ACPAgentParams],
) -> ACPAgent:
    """Gemini CLI agent via ACP.

    Uses gemini's native ``--experimental-acp`` flag in a sandbox.
    Supports multi-turn sessions and mid-turn interrupts.

    Args:
        skills: Additional skills to make available.
        version: Version of gemini CLI to use. One of:
            ``"auto"``, ``"sandbox"``, ``"stable"``, ``"latest"``,
            or a specific semver version string.
        **kwargs: See :class:`ACPAgentParams` for all base options.
    """
    return GeminiCli(
        skills=skills,
        version=version,
        **kwargs,
    )
