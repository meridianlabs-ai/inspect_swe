"""Claude Code agent via the ``claude-agent-acp`` ACP adapter."""

import logging
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.agent import AgentState, SandboxAgentBridge, agent, sandbox_agent_bridge
from inspect_ai.model import Model, get_model
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import (
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    SandboxEnvironment,
    store,
)
from inspect_ai.util import sandbox as sandbox_env
from typing_extensions import Unpack

from inspect_swe._claude_code.env import DISABLE_AUTO_MEMORY_ENV
from inspect_swe._util.path import join_path
from inspect_swe._util.sandbox import expand_sandbox_home, sandbox_exec
from inspect_swe._util.websearch import web_search_tool_disallowed
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

from .agentbinary import ensure_claude_code_acp_setup
from .transcript import TranscriptSpec, build_transcript, project_slug

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
        config_dir: str | None = None,
        resume_transcript: TranscriptSpec | None = None,
        resume_message_uuid: str | None = None,
        **kwargs: Unpack[ACPAgentParams],
    ) -> None:
        self._disallowed_tools = list(disallowed_tools or [])
        self._resolved_skills = read_skills(skills) if skills else None
        self._opus_model: str | Model | None = opus_model
        self._sonnet_model: str | Model | None = sonnet_model
        self._haiku_model: str | Model | None = haiku_model
        self._subagent_model: str | Model | None = subagent_model
        self._config_dir = config_dir
        # Resolved in _start_agent (needs the sandbox), read by
        # _resolve_resume_session to place the transcript.
        self._resolved_config_dir: str | None = None
        # A transcript carries its own session id, so combining it with either
        # of the base class's resume inputs is contradictory.
        if resume_transcript is not None and (
            kwargs.get("resume_session_id") is not None
            or kwargs.get("resume_messages") is not None
        ):
            raise ValueError(
                "`resume_transcript` already carries the session id and the content "
                "to materialize; pass it alone, not with `resume_session_id` or "
                "`resume_messages`."
            )
        self._resume_transcript = resume_transcript
        self._resume_message_uuid = resume_message_uuid
        if resume_transcript is not None:
            kwargs["resume_session_id"] = resume_transcript.session_id
        super().__init__(**kwargs)
        if resume_message_uuid is not None and (
            self.resume_session_id is None or self.resume_messages is not None
        ):
            raise ValueError(
                "`resume_message_uuid` truncates a conversation being resumed, so it "
                "needs `resume_session_id` or `resume_transcript` alongside it; "
                "`resume_messages` builds fresh row uuids that the caller cannot target."
            )

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
            filter=self.filter,
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

            # Use canonical model names — the bridge resolves them via
            # model_aliases to Model instances directly.
            agent_env = (
                _BRIDGE_SAFE_ENV
                | {
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
                    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                    "IS_SANDBOX": "1",
                    "PATH": f"{node_dir}:/usr/local/bin:/usr/bin:/bin",
                }
                | self.env
            )

            # Resolve CLAUDE_CONFIG_DIR — where Claude Code keeps its sessions,
            # and so where a resumed transcript has to be written. When resuming
            # we pin it explicitly even if the caller didn't override it: we
            # resolve $HOME through a shell exec, the adapter resolves it in its
            # own process env, and if those disagree the transcript lands where
            # session/load never looks (which reads as a silently fresh session).
            if self._config_dir is not None or self.is_resuming:
                self._resolved_config_dir = await self._resolve_config_dir(sbox)
                agent_env["CLAUDE_CONFIG_DIR"] = str(self._resolved_config_dir)

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

    async def _resolve_config_dir(self, sbox: SandboxEnvironment) -> str:
        """Resolve ``CLAUDE_CONFIG_DIR`` in the sandbox (default ``$HOME/.claude``)."""
        target = (
            self._config_dir
            if self._config_dir is not None
            else self.env.get("CLAUDE_CONFIG_DIR", "$HOME/.claude")
        )
        resolved = await expand_sandbox_home(sbox, target, user=self.user, cwd=self.cwd)
        await sandbox_exec(
            sbox,
            cmd=f"mkdir -p {shlex.quote(resolved)}",
            user=self.user,
            cwd=self.cwd,
        )
        return resolved

    async def _resolve_resume_session(self) -> str:
        """Write the session to resume into the sandbox and return its id.

        Claude Code resolves a resumed session by reading
        ``$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-id>.jsonl`` off disk,
        so the transcript must exist before the base class issues
        ``session/load``. ``_start_agent`` has already resolved
        ``CLAUDE_CONFIG_DIR`` by the time this runs.

        ``resume_session_id`` alone re-attaches to a transcript already on disk
        and writes nothing.
        """
        spec = self._resume_transcript
        if spec is None and self.resume_messages is not None:
            spec = build_transcript(
                cwd=self.cwd,
                items=self.resume_messages,
                model=get_model(self.model).canonical_name(),
            )
        if spec is None:
            assert self.resume_session_id is not None  # guarded by is_resuming
            return self.resume_session_id

        if self._resolved_config_dir is None:
            raise RuntimeError(
                "ClaudeCode._resolve_resume_session invoked before _start_agent "
                "resolved CLAUDE_CONFIG_DIR"
            )
        if spec.cwd != self.cwd:
            raise ValueError(
                f"Resume transcript was built for cwd {spec.cwd!r} but this agent runs "
                f"in {self.cwd!r}. Claude Code locates a session by its cwd, so the "
                f"two have to match — rebuild the transcript with cwd={self.cwd!r}."
            )
        sbox = sandbox_env(self.sandbox)
        # The SDK resolves symlinks before slugging the cwd, so place the file
        # under the resolved spelling rather than spec.relative_path (built from
        # the logical cwd on the host, which can't see the sandbox's fs).
        resolved_cwd = await sandbox_exec(
            sbox, f"realpath {shlex.quote(self.cwd)}", user=self.user, cwd=self.cwd
        )
        transcript_path = join_path(
            self._resolved_config_dir,
            f"projects/{project_slug(resolved_cwd)}/{spec.session_id}.jsonl",
        )
        self._warn_if_visible_to_agent(transcript_path, resolved_cwd)
        await sbox.write_file(transcript_path, spec.content)
        logger.info(
            "Wrote synthetic claude code transcript to %s (session_id=%s)",
            transcript_path,
            spec.session_id,
        )
        return spec.session_id

    def _warn_if_visible_to_agent(
        self, transcript_path: str, resolved_cwd: str
    ) -> None:
        """Warn when the planted transcript is readable below the agent cwd."""
        if PurePosixPath(transcript_path).is_relative_to(resolved_cwd):
            logger.warning(
                "Synthetic Claude Code transcript will be written to %s, inside "
                "the agent's working directory (%s), where the agent can read its "
                "own planted history. Pass config_dir=... (a path outside cwd, "
                "e.g. '/opt/claude-home') to move CLAUDE_CONFIG_DIR out of the "
                "agent's working directory.",
                transcript_path,
                resolved_cwd,
            )

    def _load_session_meta(self) -> dict[str, Any]:
        """Pass ``resumeSessionAt`` through to the Agent SDK when truncating.

        The adapter spreads ``_meta.claudeCode.options`` into the options it
        hands the Agent SDK, whose ``resumeSessionAt`` resumes a session only up
        to and including the row with that uuid.
        """
        if self._resume_message_uuid is None:
            return {}
        return {
            "claudeCode": {"options": {"resumeSessionAt": self._resume_message_uuid}}
        }


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
    config_dir: str | None = None,
    resume_transcript: TranscriptSpec | None = None,
    resume_message_uuid: str | None = None,
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
        config_dir: Override for ``CLAUDE_CONFIG_DIR`` in the sandbox, where
            Claude Code keeps its sessions (default ``$HOME/.claude``).
        resume_transcript: Resume from a prior session instead of starting
            fresh, with full control over the transcript. Build it with
            :func:`build_transcript` (its ``cwd`` must match this agent's);
            the transcript is written into the sandbox's ``CLAUDE_CONFIG_DIR``
            and loaded via ACP ``session/load``. For the common case pass
            ``resume_messages`` instead and the transcript is built for you.
        resume_message_uuid: Resume only up to and including the transcript row
            with this uuid — the branch-at-a-turn case. Combine with
            ``resume_session_id`` or ``resume_transcript``; uuids come from
            ``TranscriptSpec.item_uuids`` or ``ParsedTranscript.item_uuids``.
        **kwargs: See :class:`ACPAgentParams` for all base options, including
            ``resume_messages`` and ``resume_session_id``.
    """
    return ClaudeCode(
        disallowed_tools=disallowed_tools,
        skills=skills,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        haiku_model=haiku_model,
        subagent_model=subagent_model,
        config_dir=config_dir,
        resume_transcript=resume_transcript,
        resume_message_uuid=resume_message_uuid,
        **kwargs,
    )
