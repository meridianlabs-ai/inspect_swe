"""Codex CLI agent via the ``codex-acp`` ACP adapter."""

import logging
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any, cast

from inspect_ai.agent import AgentState, SandboxAgentBridge, agent, sandbox_agent_bridge
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

from inspect_swe._codex_cli.config import (
    CodexAutoReview,
    CodexDeprecatedArgs,
    CodexWebSearch,
    codex_config_options,
    resolve_codex_auto_review,
    resolve_codex_auto_review_model_aliases,
    resolve_codex_deprecated_args,
    resolve_codex_web_search,
)
from inspect_swe._util.path import join_path
from inspect_swe._util.sandbox import expand_sandbox_home, sandbox_exec
from inspect_swe._util.toml import to_toml
from inspect_swe.acp import ACPAgent
from inspect_swe.acp.agent import ACPAgentParams

from .agentbinary import ensure_codex_acp_setup
from .rollout import RolloutSpec, build_rollout

logger = logging.getLogger(__name__)


class CodexACPAgentParams(ACPAgentParams, CodexDeprecatedArgs, total=False):
    pass


class CodexCli(ACPAgent):
    """Codex CLI agent via the ``codex-acp`` ACP adapter.

    Subclasses :class:`ACPAgent` to provide Codex-specific setup
    (bridge, env vars, AGENTS.md, skills).
    """

    def __init__(
        self,
        *,
        web_search: CodexWebSearch = "live",
        goals: bool = True,
        auto_review: bool | CodexAutoReview = False,
        skills: list[str | Path | Skill] | None = None,
        home_dir: str | None = None,
        config_overrides: dict[str, str] | None = None,
        resume_rollout: RolloutSpec | None = None,
        **kwargs: Unpack[CodexACPAgentParams],
    ) -> None:
        deprecated_args = cast(dict[str, Any], kwargs)
        self._disallowed_tools = resolve_codex_deprecated_args(
            {"disallowed_tools": deprecated_args.pop("disallowed_tools", None)}
        )
        self._web_search = resolve_codex_web_search(web_search, self._disallowed_tools)
        self._goals = goals
        self._auto_review = resolve_codex_auto_review(auto_review)
        self._resolved_skills = read_skills(skills) if skills else None
        self._home_dir = home_dir
        self._config_overrides = config_overrides or {}
        # Resume: writing the synthetic rollout is deferred to
        # _resolve_resume_session (it needs the sandbox + resolved CODEX_HOME).
        # A rollout carries its own session id, so combining it with either of
        # the base class's resume inputs is contradictory — reject rather than
        # let one silently win.
        if resume_rollout is not None and (
            deprecated_args.get("resume_session_id") is not None
            or deprecated_args.get("resume_messages") is not None
        ):
            raise ValueError(
                "`resume_rollout` already carries the session id and the content to "
                "materialize; pass it alone, not with `resume_session_id` or "
                "`resume_messages`."
            )
        self._resume_rollout = resume_rollout
        self._codex_home: str | None = None
        if resume_rollout is not None:
            kwargs["resume_session_id"] = resume_rollout.session_id
        super().__init__(**cast(ACPAgentParams, kwargs))

    @asynccontextmanager
    async def _start_agent(
        self, state: AgentState
    ) -> AsyncIterator[tuple[ExecRemoteProcess, SandboxAgentBridge]]:
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
            model_aliases=resolve_codex_auto_review_model_aliases(
                self._auto_review,
                self.model_map,
                # the ACP bridge has no fallback model (model=None), so the
                # guardian slug must be bound explicitly; default guardian
                # requests to this agent's model
                default=get_model(self.model),
            ),
            filter=self.filter,
            retry_refusals=self.retry_refusals,
            bridged_tools=self.bridged_tools or None,
            web_search=self._web_search != "disabled",
            port=port,
        ) as bridge:
            # Install node and codex-acp in the sandbox.
            acp_binary, node_binary = await ensure_codex_acp_setup(sbox, self.user)
            node_dir = str(Path(node_binary).parent)

            # Resolve CODEX_HOME (mirrors the non-ACP codex_cli agent).
            codex_home = await self._resolve_codex_home(sbox)
            # Stash for _resolve_resume_session (base class calls it after this yields).
            self._codex_home = codex_home

            # Write system prompt to AGENTS.md (Codex convention).
            resolved_prompt = self._resolve_system_prompt(state)
            if resolved_prompt:
                agents_md_path = self._agents_md_path(codex_home)
                await sbox.write_file(agents_md_path, resolved_prompt)

            # Install skills.
            if self._resolved_skills:
                skills_dir = join_path(codex_home, "skills")
                await install_skills(self._resolved_skills, sbox, self.user, skills_dir)

            # Write config.toml with model provider pointing at the bridge.
            # Use the canonical model name so the bridge can resolve it
            # via model_aliases (consistent with how claude-agent-acp passes
            # ANTHROPIC_MODEL).
            bridge_url = f"http://127.0.0.1:{bridge.port}/v1"
            config_toml_path = await self._config_toml_path(sbox, codex_home)
            toml_config: dict[str, Any] = {
                "model": default_model,
                "preferred_auth_method": "apikey",
                # overridden below by codex_config_options() when auto_review is enabled
                "approval_policy": "never",
                "sandbox_mode": "danger-full-access",
                "model_provider": "openai-proxy",
                "model_providers.openai-proxy": {
                    "name": "OpenAI Proxy",
                    "base_url": bridge_url,
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                    "stream_idle_timeout_ms": 3_600_000,
                },
            }
            toml_config.update(self._config_overrides)
            toml_config.update(
                codex_config_options(self._web_search, self._goals, self._auto_review)
            )
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
            # Pin the resolved spelling so the adapter reads the same directory
            # that setup and synthetic-session materialization wrote to.
            agent_env["CODEX_HOME"] = codex_home

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

    async def _resolve_codex_home(self, sbox: SandboxEnvironment) -> str:
        """Resolve and create the effective ``CODEX_HOME`` in the sandbox."""
        target = (
            self._home_dir if self._home_dir is not None else self.env.get("CODEX_HOME")
        )
        if target is None:
            working_dir = await sandbox_exec(sbox, "pwd", user=self.user, cwd=self.cwd)
            codex_home = join_path(working_dir, ".codex")
        else:
            codex_home = await expand_sandbox_home(
                sbox, target, user=self.user, cwd=self.cwd
            )
        await sandbox_exec(
            sbox,
            cmd=f"mkdir -p {shlex.quote(codex_home)}",
            user=self.user,
            cwd=self.cwd,
        )
        return codex_home

    def _agents_md_path(self, codex_home: str) -> str:
        """Determine where to write AGENTS.md."""
        if self._home_dir is not None or "CODEX_HOME" in self.env:
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
        if self._home_dir is not None or "CODEX_HOME" in self.env:
            return join_path(codex_home, "config.toml")
        directory = ".codex" if self.cwd is None else join_path(self.cwd, ".codex")
        await sandbox_exec(
            sbox, cmd=f"mkdir -p {shlex.quote(directory)}", user=self.user
        )
        return join_path(directory, "config.toml")

    async def _resolve_resume_session(self) -> str:
        """Write the session to resume into the sandbox and return its id.

        Codex resolves a resumed session by reading
        ``$CODEX_HOME/sessions/.../rollout-<id>.jsonl`` off disk, so the rollout
        must exist before the base class issues ``session/load``. ``_start_agent``
        has already resolved and created ``CODEX_HOME`` by the time this runs.

        ``resume_session_id`` alone re-attaches to a rollout already on disk and
        writes nothing.
        """
        spec = self._resume_rollout
        if spec is None and self.resume_messages is not None:
            spec = build_rollout(
                cwd=self.cwd,
                prior=self.resume_messages,
                model=get_model(self.model).canonical_name(),
            )
        if spec is None:
            assert self.resume_session_id is not None  # guarded by is_resuming
            return self.resume_session_id

        if self._codex_home is None:
            raise RuntimeError(
                "CodexCli._resolve_resume_session invoked before _start_agent "
                "resolved CODEX_HOME"
            )
        if spec.cwd != self.cwd:
            raise ValueError(
                f"Resume rollout was built for cwd {spec.cwd!r} but this agent runs "
                f"in {self.cwd!r}. Codex records cwd in both session metadata and "
                f"turn context, so rebuild the rollout with cwd={self.cwd!r}."
            )
        self._warn_on_model_mismatch(spec.model)
        self._warn_if_visible_to_agent()
        sbox = sandbox_env(self.sandbox)
        rollout_path = join_path(self._codex_home, spec.relative_path)
        await sbox.write_file(rollout_path, spec.content)
        logger.info(
            "Wrote synthetic codex rollout to %s (session_id=%s)",
            rollout_path,
            spec.session_id,
        )
        return spec.session_id

    def _warn_on_model_mismatch(self, rollout_model: str) -> None:
        """Warn when the rollout records a different model than we will serve.

        Codex splices a ``<model_switch>`` banner into the resumed conversation
        when they differ.
        """
        resolved_model = get_model(self.model).canonical_name()
        # Compare provider-tolerantly: the rollout records a bare model name
        # while canonical_name() may be provider-qualified ("openai/gpt-5.5").
        if rollout_model not in (resolved_model, resolved_model.rsplit("/", 1)[-1]):
            logger.warning(
                "Resume rollout model (%s) differs from this agent's model (%s); "
                "codex will splice a <model_switch> banner into the resumed "
                "conversation. Build the rollout with model=%r to avoid it.",
                rollout_model,
                resolved_model,
                resolved_model,
            )

    def _warn_if_visible_to_agent(self) -> None:
        """Warn when the synthetic rollout lands inside the agent's own cwd.

        ``CODEX_HOME`` defaults to ``<cwd>/.codex``, which puts the planted
        conversation somewhere the agent can read it — an eval-awareness hazard
        specific to synthesizing a prior. Pass ``home_dir`` to move it out (and
        note that with ``cwd`` at the user's home, ``$HOME/.codex`` is still
        inside it — the rollout needs a directory outside the agent's cwd).
        """
        if self._codex_home is not None and PurePosixPath(
            self._codex_home
        ).is_relative_to(self.cwd):
            logger.warning(
                "Synthetic rollout will be written to %s, inside the agent's working "
                "directory (%s), where the agent can read its own planted history. "
                "Pass home_dir=... (a path outside cwd, e.g. '/opt/codex-home') to "
                "move CODEX_HOME out of the agent's working directory.",
                self._codex_home,
                self.cwd,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@agent(name="Codex CLI")
def interactive_codex_cli(
    *,
    # Codex-specific
    web_search: CodexWebSearch = "live",
    goals: bool = True,
    auto_review: bool | CodexAutoReview = False,
    skills: list[str | Path | Skill] | None = None,
    home_dir: str | None = None,
    config_overrides: dict[str, str] | None = None,
    resume_rollout: RolloutSpec | None = None,
    # Forwarded to ACPAgent
    **kwargs: Unpack[CodexACPAgentParams],
) -> ACPAgent:
    """Codex CLI agent via ACP.

    Uses the ``codex-acp`` adapter in a sandbox.  Supports
    multi-turn sessions and mid-turn interrupts.

    Args:
        web_search: Web search mode. Use ``"live"`` for live web search, ``"cached"`` for cached web search, or ``"disabled"`` to disable web search.
        goals: Enable Codex goal tools.
        auto_review: Enable Codex automated approval review (guardian): Codex runs
            with its own ``workspace-write`` sandbox and ``on-request`` approvals,
            with escalations adjudicated by a guardian model. Pass
            :class:`CodexAutoReview` to customize the guardian policy and model.
            Note: guardian adjudication depends on the codex-core embedded in the
            ``codex-acp`` adapter honoring ``approvals_reviewer`` (Codex CLI >=
            0.137.0 behavior); unlike `codex_cli()`, this cannot be version-checked
            here. If the embedded core instead surfaces an escalation as an ACP
            permission request, it is auto-approved.
        skills: Additional skills to make available.
        home_dir: Override for ``CODEX_HOME`` directory in the sandbox.
        config_overrides: Extra Codex config.toml key-value pairs.
        resume_rollout: Resume from a prior session instead of starting fresh,
            with full control over the rollout. Build it with
            :func:`build_rollout` (its ``cwd`` must match this agent's and its
            ``model`` should match this agent's model); the rollout is written
            into the sandbox's ``CODEX_HOME`` and loaded via ACP
            ``session/load``. For the common case pass ``resume_messages``
            instead and the rollout is built for you.
        **kwargs: See :class:`ACPAgentParams` for all base options, including
            ``resume_messages`` and ``resume_session_id``.
    """
    return CodexCli(
        web_search=web_search,
        goals=goals,
        auto_review=auto_review,
        skills=skills,
        home_dir=home_dir,
        config_overrides=config_overrides,
        resume_rollout=resume_rollout,
        **kwargs,
    )
