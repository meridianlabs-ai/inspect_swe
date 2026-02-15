"""Claude Code agent via ACP (Agent Client Protocol).

The ``claude_code()`` function returns an ``Agent`` backed by the
``claude-code-acp`` adapter running in a sandbox, communicating over ACP
via ``exec_remote`` with ``stdin_open=True``.

Two modes:

* **Non-interactive** (``interactive=False``, default): sends one prompt
  built from ``state.messages``, waits for the response, and returns.
  Drop-in replacement for the old ``sbox.exec()`` path.

* **Interactive** (``interactive=True``): sets up the ACP lifecycle,
  exposes ``.conn`` and ``.session_id`` on the agent object, and waits
  for the caller to drive prompts via ``conn.prompt()`` / ``conn.cancel()``.
"""

import logging
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from textwrap import dedent
from typing import Any, Sequence

from inspect_ai.agent import (
    Agent,
    AgentState,
    BridgedToolsSpec,
    agent,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.util import SandboxEnvironment, sandbox as sandbox_env
from inspect_ai.util._sandbox.exec_remote import (
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
)
from pydantic_core import to_json

from inspect_swe._acp import ACPAgent
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.path import join_path

from .._util.messages import build_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClaudeCode agent class
# ---------------------------------------------------------------------------


class ClaudeCode(ACPAgent):
    """Claude Code agent via the ``claude-code-acp`` ACP adapter.

    Subclasses :class:`ACPAgent` to provide Claude-specific setup
    (bridge, binary installation, env vars, MCP config, skills).
    """

    def __init__(
        self,
        *,
        interactive: bool = False,
        model: str | None = None,
        filter: GenerateFilter | None = None,
        bridged_tools: Sequence[BridgedToolsSpec] | None = None,
        disallowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        mcp_servers: Sequence[MCPServerConfig] | None = None,
        skills: Sequence[str | Path | Skill] | None = None,
        retry_refusals: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        sandbox: str | None = None,
        opus_model: str | None = None,
        sonnet_model: str | None = None,
        haiku_model: str | None = None,
        subagent_model: str | None = None,
    ) -> None:
        super().__init__(interactive=interactive, cwd=cwd)
        self._model = f"inspect/{model}" if model is not None else "inspect"
        self._filter = filter
        self._bridged_tools = list(bridged_tools or [])
        self._disallowed_tools = list(disallowed_tools or [])
        self._system_prompt = system_prompt
        self._mcp_servers = list(mcp_servers or [])
        self._resolved_skills = read_skills(skills) if skills else None
        self._retry_refusals = retry_refusals
        self._env = env or {}
        self._user = user
        self._sandbox = sandbox
        self._opus_model = inspect_model(opus_model)
        self._sonnet_model = inspect_model(sonnet_model)
        self._haiku_model = inspect_model(haiku_model)
        self._subagent_model = inspect_model(subagent_model)

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, object]]:
        sbox = sandbox_env(self._sandbox)

        async with sandbox_agent_bridge(
            state,
            model=self._model,
            filter=self._filter,
            retry_refusals=self._retry_refusals,
            bridged_tools=self._bridged_tools or None,
        ) as bridge:
            # Ensure claude-code-acp is available in the sandbox.
            await _ensure_claude_code_acp_installed(sbox, self._user)

            # System prompt
            system_messages = [
                m.text
                for m in state.messages
                if isinstance(m, ChatMessageSystem)
            ]
            if self._system_prompt is not None:
                system_messages.append(self._system_prompt)

            # Agent environment variables
            agent_env = {
                "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                "ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-DOq5tyLPrk9M4hPE",
                "ANTHROPIC_MODEL": self._model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": self._opus_model or self._model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": self._sonnet_model or self._model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": self._haiku_model or self._model,
                "CLAUDE_CODE_SUBAGENT_MODEL": self._subagent_model or self._model,
                "ANTHROPIC_SMALL_FAST_MODEL": self._haiku_model or self._model,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                "IS_SANDBOX": "1",
            } | self._env

            # System prompt via env (the ACP adapter will forward to CC)
            if system_messages:
                agent_env["CLAUDE_CODE_APPEND_SYSTEM_PROMPT"] = "\n\n".join(
                    system_messages
                )

            # MCP servers (combine static configs with bridged tools)
            all_mcp_servers = self._mcp_servers + bridge.mcp_server_configs
            if all_mcp_servers:
                mcp_config_json = _build_mcp_config_json(all_mcp_servers)
                agent_env["CLAUDE_CODE_MCP_CONFIG"] = mcp_config_json

            # Disallowed tools
            if self._disallowed_tools:
                agent_env["CLAUDE_CODE_DISALLOWED_TOOLS"] = ",".join(
                    self._disallowed_tools
                )

            # Install skills
            if self._resolved_skills:
                skills_dir = (
                    join_path(self.cwd, ".claude/skills")
                    if self.cwd
                    else ".claude/skills"
                )
                await install_skills(
                    self._resolved_skills, sbox, self._user, skills_dir
                )

            # Start ACP adapter process
            logger.info("Starting claude-code-acp adapter...")
            proc = await sbox.exec_remote(
                cmd=["claude-code-acp"],
                options=ExecRemoteStreamingOptions(
                    stdin_open=True,
                    cwd=self.cwd,
                    env=agent_env,
                    user=self._user,
                ),
            )

            yield proc, bridge

    def _build_prompt(self, state: AgentState) -> str:
        """Build prompt from state messages using Claude Code's format."""
        prompt, _ = build_user_prompt(state.messages)
        return prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@agent
def claude_code(
    name: str = "Claude Code",
    description: str = dedent("""\
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    *,
    interactive: bool = False,
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    disallowed_tools: list[str] | None = None,
    model: str | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
    subagent_model: str | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
) -> Agent:
    """Claude Code agent.

    Agent that uses `Claude Code <https://docs.anthropic.com/en/docs/claude-code/overview>`_
    running in a sandbox via the ACP (Agent Client Protocol) adapter.

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)
        interactive: If True, the agent exposes ``.conn`` and ``.session_id``
            for the caller to drive prompts directly.  If False (default),
            sends one prompt from ``state.messages`` and returns.
        system_prompt: Additional system prompt to append to default system prompt.
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent.
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
            Each BridgedToolsSpec creates an MCP server that makes the specified
            tools available to the agent running in the sandbox.
        disallowed_tools: List of tool names to disallow entirely.
        model: Model name to use for Opus and Sonnet calls (defaults to main model for task).
        opus_model: The model to use for `opus`, or for `opusplan` when Plan Mode is active. Defaults to `model`.
        sonnet_model: The model to use for `sonnet`, or for `opusplan` when Plan Mode is not active. Defaults to `model`.
        haiku_model: The model to use for haiku, or [background functionality](https://code.claude.com/docs/en/costs#background-token-usage). Defaults to `model`.
        subagent_model: The model to use for [subagents](https://code.claude.com/docs/en/sub-agents). Defaults to `model`.
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Number of times to retry refusals.
        cwd: Working directory.
        env: Environment variables.
        user: User to execute as.
        sandbox: Sandbox environment name.
    """
    return ClaudeCode(
        interactive=interactive,
        model=model,
        filter=filter,
        bridged_tools=bridged_tools,
        disallowed_tools=disallowed_tools,
        system_prompt=system_prompt,
        mcp_servers=mcp_servers,
        skills=skills,
        retry_refusals=retry_refusals,
        cwd=cwd,
        env=env,
        user=user,
        sandbox=sandbox,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        haiku_model=haiku_model,
        subagent_model=subagent_model,
    )


# ---------------------------------------------------------------------------
# ACP adapter installation
# ---------------------------------------------------------------------------

_ACP_ADAPTER_PACKAGE = "@zed-industries/claude-code-acp"


async def _ensure_claude_code_acp_installed(
    sbox: SandboxEnvironment,
    user: str | None = None,
) -> None:
    """Ensure ``claude-code-acp`` is available in the sandbox.

    Checks if the binary is already on ``$PATH``.  If not, installs
    the ``@zed-industries/claude-code-acp`` npm package globally.
    """
    result = await sbox.exec(["bash", "-c", "which claude-code-acp"], user=user)
    if result.success:
        logger.info("claude-code-acp already installed: %s", result.stdout.strip())
        return

    logger.info("Installing %s in sandbox...", _ACP_ADAPTER_PACKAGE)
    install_result = await sbox.exec(
        ["bash", "-c", f"npm install -g {_ACP_ADAPTER_PACKAGE}"],
        user="root",
    )
    if not install_result.success:
        raise RuntimeError(
            f"Failed to install {_ACP_ADAPTER_PACKAGE}: "
            f"{install_result.stderr}"
        )
    logger.info("Installed %s successfully", _ACP_ADAPTER_PACKAGE)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _build_mcp_config_json(
    mcp_servers: Sequence[MCPServerConfig],
) -> str:
    """Build a JSON string for MCP server config."""
    mcp_servers_json: dict[str, dict[str, Any]] = {}
    for mcp_server in mcp_servers:
        mcp_servers_json[mcp_server.name] = mcp_server.model_dump(
            exclude={"name", "tools"}, exclude_none=True
        )
    return to_json(
        {"mcpServers": mcp_servers_json}, exclude_none=True
    ).decode()


def resolve_mcp_servers(
    mcp_servers: Sequence[MCPServerConfig],
) -> tuple[list[str], list[str]]:
    """Build CLI args and allowed tool patterns for MCP servers."""
    mcp_servers_json: dict[str, dict[str, Any]] = {}
    allowed_tools: list[str] = []
    for mcp_server in mcp_servers:
        mcp_servers_json[mcp_server.name] = mcp_server.model_dump(
            exclude={"name", "tools"}, exclude_none=True
        )
        if mcp_server.tools == "all":
            allowed_tools.append(f"mcp__{mcp_server.name}_*")
        elif isinstance(mcp_server.tools, list):
            allowed_tools.extend(
                [
                    f"mcp__{mcp_server.name}__{tool}"
                    for tool in mcp_server.tools
                ]
            )
        else:
            raise ValueError(
                f"Unexpected value for mcp server tools: {mcp_server.tools}"
            )

    mcp_config_cmds: list[str] = []
    if len(mcp_servers_json) > 0:
        mcp_config_cmds.append("--mcp-config")
        mcp_config_cmds.append(
            to_json(
                {"mcpServers": mcp_servers_json}, exclude_none=True
            ).decode()
        )

    return mcp_config_cmds, allowed_tools


def inspect_model(model: str | None) -> str | None:
    """Ensure that model name is prefaced with 'inspect/'."""
    if model is not None:
        if model != "inspect" and not model.startswith("inspect/"):
            return f"inspect/{model}"
    return model


async def run_claude_code_centaur(
    options: CentaurOptions,
    claude_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    """Run Claude Code in centaur (human-in-the-loop) mode."""
    instructions = (
        "Claude Code:\n\n"
        " - You may also use Claude Code via the 'claude' command.\n"
        " - Use 'claude --resume' if you need to resume a previous session."
    )

    agent_env_vars = [f'export {k}="{v}"' for k, v in agent_env.items()]
    claude_config = (
        """echo '{"hasCompletedOnboarding":true,"""
        """"bypassPermissionsModeAccepted":true}' > "$HOME"/.claude.json"""
    )
    path_config = [
        'mkdir -p "$HOME/.local/bin"',
        'export PATH="$HOME/.local/bin:$PATH"',
        f'ln -sf {claude_cmd[0]} "$HOME/.local/bin/claude"',
    ]
    alias_cmd = shlex.join(claude_cmd)
    alias_cmd = "alias claude='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(
        agent_env_vars + path_config + ["", claude_config, "", alias_cmd]
    )

    await run_centaur(options, instructions, bashrc, state)
