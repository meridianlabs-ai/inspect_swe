import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.agent._bridge.util import resolve_inspect_model
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model, get_model_info
from inspect_ai.scorer import score
from inspect_ai.tool import Skill, install_skills, read_skills
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.agentbinary import ensure_agent_binary_installed
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import resolve_agent_cwd
from inspect_swe._util.trace import trace

from .agentbinary import pi_binary_source
from .config import pi_models_json, pi_result_error


@agent
def pi(
    name: str = "Pi",
    description: str = "Terminal coding agent that reads and edits files and runs shell commands.",
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool = False,
    thinking: Literal[
        "off", "minimal", "low", "medium", "high", "xhigh", "max"
    ] = "off",
    max_context_size: int | None = None,
) -> Agent:
    """Pi coding agent running in a sandbox with Inspect model bridging.

    Pi's built-in tools run without approval prompts. Model calls use the
    Inspect bridge's OpenAI Chat Completions endpoint, not Pi's provider
    credentials. MCP servers and bridged tools are not supported.

    Args:
        name: Agent name for multi-agent systems.
        description: Agent description.
        system_prompt: Instructions to append to Pi's default system prompt.
        skills: Additional Inspect skills to install for Pi.
        centaur: Make Pi available to an Inspect human_cli agent.
        attempts: Number of scored attempts, or an AgentAttempts configuration.
        model: Inspect model name; defaults to the task's main model.
        model_aliases: Mapping of requested model names to Inspect models.
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Number of times to retry refusals.
        cwd: Working directory in the sandbox.
        env: Additional sandbox environment variables. Pi's config directory
            and offline settings are reserved and cannot be overridden.
        user: Sandbox user to run Pi as.
        sandbox: Optional sandbox environment name.
        version: "auto" uses an installed Pi or downloads the latest release;
            "sandbox" requires an installed Pi; "stable" and "latest" download
            the latest release; a version number pins a release.
        debug: Include CLI output in the debug trace.
        thinking: Pi thinking level. "off" sends no reasoning effort, leaving
            Inspect and provider defaults in effect. Other levels are forwarded
            through the bridge and require a model that supports reasoning.
        max_context_size: Context window for Pi's compaction. Defaults to Inspect
            model metadata; required when the model has no context metadata.
    """
    if thinking not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError(f"Unsupported Pi thinking level: {thinking}")
    if max_context_size is not None and max_context_size <= 0:
        raise ValueError("max_context_size must be positive")
    if centaur is True:
        centaur = CentaurOptions()
    resolved_skills = read_skills(skills=skills) if skills is not None else None
    attempts = (
        AgentAttempts(attempts=attempts) if isinstance(attempts, int) else attempts
    )
    if attempts.attempts < 1:
        raise ValueError("attempts must be at least 1")
    requested_model = model or "inspect"
    bridge_model = f"inspect/{model}" if model is not None else "inspect"
    session_key = f"pi_session_{uuid4().hex}"

    async def execute(state: AgentState) -> AgentState:
        prompt, has_assistant_response = build_user_prompt(messages=state.messages)
        sbox = sandbox_env(name=sandbox)
        agent_cwd = await resolve_agent_cwd(sandbox=sbox, user=user, cwd=cwd)
        resolved_model = resolve_inspect_model(
            model_name=requested_model,
            model_aliases=model_aliases,
            fallback_model=bridge_model,
        )
        model_info = get_model_info(model=resolved_model)
        context_size = max_context_size or (
            model_info.context_length if model_info else None
        )
        if context_size is None or context_size <= 0:
            raise ValueError(
                "Context length metadata is unavailable; pass max_context_size to pi()."
            )
        max_tokens = min(
            resolved_model.config.max_tokens
            or (model_info.output_tokens if model_info else None)
            or 16384,
            context_size,
        )

        # Keep each agent's config and session separate, including within one sample.
        session = store().get(session_key)
        if session is None or not has_assistant_response:
            result = await sbox.exec(
                cmd=["mktemp", "-d", "/tmp/inspect-pi-XXXXXXXX"], user=user
            )
            if not result.success:
                raise RuntimeError(
                    f"Unable to create Pi config directory: {result.stderr}"
                )
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)
            session = {
                "directory": result.stdout.strip(),
                "system_prompt": "\n\n".join(system_messages),
            }
            store().set(key=session_key, value=session)
        pi_dir = session["directory"]

        port = store().get("pi_model_port", 3200) + 1
        store().set(key="pi_model_port", value=port)
        async with sandbox_agent_bridge(
            state=state,
            model=bridge_model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            forward_generation_config=True,
            port=port,
        ) as bridge:
            pi_binary = await ensure_agent_binary_installed(
                source=pi_binary_source(), sandbox=sbox, version=version, user=user
            )
            await sbox.write_file(
                file=f"{pi_dir}/models.json",
                contents=pi_models_json(
                    port=bridge.port,
                    model=requested_model,
                    context_size=context_size,
                    max_tokens=max_tokens,
                    reasoning=thinking != "off",
                ),
            )
            await sbox.write_file(
                file=f"{pi_dir}/settings.json",
                contents=json.dumps(
                    {
                        "retry": {"enabled": False},
                        "compaction": {"reserveTokens": min(16384, context_size // 4)},
                    }
                ),
            )
            cmd = [
                pi_binary,
                "--provider",
                "inspect",
                "--model",
                requested_model,
                "--thinking",
                thinking,
                "--session",
                f"{pi_dir}/session.jsonl",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "--no-approve",
            ]
            # Pi rebuilds its system prompt on each launch. Reapply only the original
            # instructions, never the full Pi prompt captured by the bridge.
            if session["system_prompt"]:
                system_file = f"{pi_dir}/system.txt"
                await sbox.write_file(
                    file=system_file, contents=session["system_prompt"]
                )
                cmd.extend(["--append-system-prompt", system_file])
            if resolved_skills is not None:
                skills_dir = f"{pi_dir}/skills"
                await install_skills(
                    skills=resolved_skills, sandbox=sbox, user=user, dir=skills_dir
                )
                cmd.extend(["--skill", skills_dir])
            agent_env = (env or {}) | {
                "PI_CODING_AGENT_DIR": pi_dir,
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
            }
            if centaur:
                exports = [
                    f"export {key}={shlex.quote(s)}" for key, s in agent_env.items()
                ]
                alias = f"alias pi={shlex.quote(shlex.join(cmd))}"
                await run_centaur(
                    options=centaur,
                    instructions="Use 'pi' to run the Pi coding agent. Repeated calls resume the same session. Pi tools run without approval prompts.",
                    bashrc="\n".join([*exports, alias]),
                    state=state,
                )
            else:
                for attempt in range(attempts.attempts):
                    # Stdin avoids argv limits and Pi's @file argument expansion.
                    prompt_file = f"{pi_dir}/prompt.txt"
                    await sbox.write_file(file=prompt_file, contents=prompt)
                    result = await sbox.exec_remote(
                        cmd=[
                            "bash",
                            "-c",
                            'prompt=$1; shift; exec "$@" < "$prompt"',
                            "bash",
                            prompt_file,
                            *cmd,
                            "--print",
                            "--mode",
                            "json",
                        ],
                        options=ExecRemoteAwaitableOptions(
                            cwd=agent_cwd, env=agent_env, user=user, concurrency=False
                        ),
                        stream=False,
                    )
                    if debug:
                        trace(
                            message=f"Pi Debug Output:\n{result.stdout}\n{result.stderr}"
                        )
                    error = pi_result_error(stdout=result.stdout)
                    if not result.success or error:
                        detail = result.stderr or error or result.stdout or "No output"
                        raise RuntimeError(
                            f"Error executing Pi agent {result.returncode}: {detail[:2000]}"
                        )
                    if attempt + 1 >= attempts.attempts:
                        break
                    answer_scores = await score(conversation=bridge.state)
                    if attempts.score_value(answer_scores[0].value) == 1.0:
                        break
                    if callable(attempts.incorrect_message):
                        if not is_callable_coroutine(
                            func_or_cls=attempts.incorrect_message
                        ):
                            raise ValueError(
                                "The incorrect_message function must be async."
                            )
                        prompt = await attempts.incorrect_message(
                            bridge.state, answer_scores
                        )
                    else:
                        prompt = attempts.incorrect_message
        return bridge.state

    return agent_with(agent=execute, name=name, description=description)
