import shlex
from pathlib import PurePosixPath
from textwrap import dedent
from typing import Any, Literal

import yaml
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
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.path import join_path
from inspect_swe._util.sandbox import resolve_agent_cwd
from inspect_swe._util.trace import trace

from .agentbinary import ensure_minimax_code_setup


@agent
def minimax_code(
    name: str = "MiniMax Code",
    description: str = dedent("""
        MiniMax Code terminal coding agent capable of reading and editing code,
        running commands, and iterating on test results.
    """),
    system_prompt: str | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    max_context_size: int | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool = False,
) -> Agent:
    """MiniMax Code agent running in a sandbox with Inspect model bridging.

    MiniMax Code's ``mcode exec`` interface is used for unattended runs. Its
    OpenAI Responses-compatible provider is configured to target the Inspect
    bridge, so model requests and tool calls remain in the Inspect transcript.
    """
    if centaur and sandbox is not None:
        raise ValueError("Centaur mode requires the default sandbox; omit sandbox")
    if centaur is True:
        centaur = CentaurOptions()

    requested_model = model or "inspect"
    bridge_model = f"inspect/{model}" if model is not None else "inspect"
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        port = store().get("minimax_code_model_port", 3000) + 1
        store().set("minimax_code_model_port", port)
        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            web_search=True,
        ) as bridge:
            sbox = sandbox_env(sandbox)
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)
            minimax_binary, node_binary = await ensure_minimax_code_setup(
                sbox, version, user
            )
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"
            data_dir = join_path(sandbox_home, ".minimax-code-inspect")

            resolved_context_size = _resolve_max_context_size(
                model=model, model_aliases=model_aliases, override=max_context_size
            )
            await sbox.exec(["mkdir", "-p", data_dir], user=user)
            await sbox.write_file(
                join_path(data_dir, "config.yaml"),
                _config_yaml(
                    port=bridge.port,
                    model=requested_model,
                    max_context_size=resolved_context_size,
                ),
            )

            system_messages = [
                message.text
                for message in state.messages
                if isinstance(message, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)
            prompt, has_assistant_response = build_user_prompt(state.messages)
            if system_messages and not has_assistant_response:
                prompt = "\n\n".join(system_messages) + "\n\n" + prompt

            agent_env = {
                "MINIMAX_DATA_DIR": data_dir,
                "HOME": sandbox_home,
                "PATH": f"{PurePosixPath(node_binary).parent}:/usr/local/bin:/usr/bin:/bin",
                "MCODE_DISABLE_TELEMETRY": "1",
            } | (env or {})
            cmd = [minimax_binary, "exec", "--permission", "full", "--input", "-"]
            if centaur:
                await _run_minimax_code_centaur(
                    options=centaur,
                    minimax_cmd=[minimax_binary],
                    agent_env=agent_env,
                    cwd=agent_cwd,
                    state=state,
                    user=user,
                )
            else:
                agent_prompt = prompt
                attempt_count = 0
                while True:
                    agent_cmd = cmd.copy()
                    if has_assistant_response or attempt_count > 0:
                        agent_cmd.append("--continue")
                    result = await sbox.exec_remote(
                        cmd=agent_cmd,
                        options=ExecRemoteAwaitableOptions(
                            input=agent_prompt,
                            cwd=agent_cwd,
                            env=agent_env,
                            user=user,
                            concurrency=False,
                        ),
                        stream=False,
                    )
                    if debug:
                        trace(
                            "MiniMax Code Debug Output:\n"
                            + result.stdout
                            + "\n"
                            + result.stderr
                        )
                    if not result.success:
                        raise RuntimeError(
                            f"Error executing MiniMax Code agent {result.returncode}:\n"
                            f"stdout: {result.stdout}\nstderr: {result.stderr}"
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

        return bridge.state

    return agent_with(execute, name=name, description=description)


def _resolve_max_context_size(
    *,
    model: str | None,
    model_aliases: dict[str, str | Model] | None,
    override: int | None,
) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("max_context_size must be a positive integer")
        return override
    requested_model = model or "inspect"
    bridge_model = f"inspect/{model}" if model is not None else "inspect"
    resolved = resolve_inspect_model(requested_model, model_aliases, bridge_model)
    info = get_model_info(resolved)
    if info is None or info.context_length is None or info.context_length <= 0:
        raise ValueError(
            f"Context length metadata is unavailable for model {resolved.name!r}; "
            "pass max_context_size explicitly."
        )
    return info.context_length


def _config_yaml(*, port: int, model: str, max_context_size: int) -> str:
    document: dict[str, Any] = {
        "custom_provider": {
            "inspect": {
                "name": "Inspect bridge",
                "kind": "custom",
                "enabled": True,
                "api": "openai-responses",
                "options": {
                    "apiKey": "api-key",
                    "baseURL": f"http://127.0.0.1:{port}/v1",
                    "authMode": "api-key",
                },
                "models": {
                    model: {
                        "name": model,
                        "tool_call": True,
                        "reasoning": True,
                        "limit": {"context": max_context_size},
                    }
                },
            }
        },
        "defaultModel": f"custom_provider:inspect/{model}",
    }
    return yaml.safe_dump(document, sort_keys=False)


async def _run_minimax_code_centaur(
    options: CentaurOptions,
    minimax_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
    cwd: str,
    user: str | None = None,
) -> None:
    instructions = (
        "MiniMax Code:\n\n - You may also use MiniMax Code via the 'mcode' command."
    )
    centaur_env = {key: value for key, value in agent_env.items() if key != "HOME"}
    agent_env_vars = [
        f"export {key}={shlex.quote(value)}" for key, value in centaur_env.items()
    ]
    alias = "alias mcode=" + shlex.quote(shlex.join(minimax_cmd))
    await run_centaur(
        options,
        instructions,
        "\n".join(agent_env_vars + [f"cd -- {shlex.quote(cwd)}", alias]),
        state,
        user=user,
    )
