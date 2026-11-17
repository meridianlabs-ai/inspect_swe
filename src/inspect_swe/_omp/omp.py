import json
from hashlib import sha256
from typing import Any
from uuid import uuid4

import anyio
from inspect_ai.agent import Agent, AgentState, agent, agent_with, sandbox_agent_bridge
from inspect_ai.model import ChatMessage, ChatMessageSystem, GenerateFilter
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import (
    ExecCompleted,
    ExecRemoteStreamingOptions,
    ExecStderr,
    ExecStdout,
    SandboxEnvironmentLimits,
)
from inspect_ai.util._subprocess import CircularByteBuffer

from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import resolve_agent_cwd
from inspect_swe._util.trace import trace

from .agentbinary import ensure_omp_setup
from .config import omp_models_config, omp_settings


@agent
def omp(
    name: str = "Oh My Pi",
    description: str = "Sandboxed Oh My Pi coding agent with Inspect model bridging.",
    system_prompt: str | None = None,
    model: str | None = None,
    filter: GenerateFilter | None = None,
    cwd: str | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: str = "18.6.1",
    context_window: int = 32768,
    max_tokens: int = 8192,
    debug: bool = False,
) -> Agent:
    """Run Oh My Pi in a sandbox with model requests routed through Inspect.

    The adapter uses a private OMP home for each agent instance and sample.
    Repeated calls within that sample resume its session. Provider credentials
    remain on the host; the agent receives only a dummy bridge key.

    Args:
        name: Agent name for multi-agent systems.
        description: Agent description.
        system_prompt: Additional instructions appended to OMP's system prompt.
        model: Inspect model to use, or the task's main model when omitted.
        filter: Filter for intercepting bridged model requests.
        cwd: Sandbox working directory, defaulting to its working or home directory.
        user: Sandbox user running OMP and its tools.
        sandbox: Optional sandbox environment name.
        version: OMP release version, or "auto", "sandbox", "stable", or "latest".
            The default pins the supported release rather than updating silently.
        context_window: Context capacity advertised to OMP. Set this to the
            serving model's actual context window.
        max_tokens: Output capacity advertised to OMP. Actual generation settings
            remain controlled by Inspect's GenerateConfig.
        debug: Include OMP process output in the debug trace.

    Returns:
        Agent using OMP's sandbox-local coding tools and Inspect model accounting.
    """
    if context_window <= 0 or max_tokens <= 0 or max_tokens >= context_window:
        raise ValueError("Require 0 < max_tokens < context_window for OMP.")

    instance_key = f"omp_home_{uuid4().hex}"
    bridge_model = f"inspect/{model}" if model is not None else "inspect"

    async def execute(state: AgentState) -> AgentState:
        if any(
            message.role == "user"
            and not isinstance(message.content, str)
            and any(content.type != "text" for content in message.content)
            for message in state.messages
        ):
            raise ValueError("The OMP adapter accepts text-only input.")
        prompt, has_history = build_user_prompt(state.messages)
        home: str | None = store().get(instance_key)
        if has_history:
            message_count: int = store().get(f"{instance_key}_message_count", 0)
            history: str | None = store().get(f"{instance_key}_history")
            if (
                home is None
                or history is None
                or _history_digest(state.messages[:message_count]) != history
                or any(
                    message.role != "user" for message in state.messages[message_count:]
                )
            ):
                raise ValueError(
                    "OMP requires the matching OMP session and conversation to resume; "
                    "imported or modified assistant history is not supported."
                )

        sbox = sandbox_env(sandbox)
        agent_cwd = await resolve_agent_cwd(sbox, user, cwd)
        binary = await ensure_omp_setup(sbox, version, user)
        if home is None or not has_history:
            result = await sbox.exec(
                ["mktemp", "-d", "/tmp/inspect-omp-XXXXXXXXXX"], user=user
            )
            if not result.success:
                raise RuntimeError(f"Unable to create OMP home: {result.stderr}")
            home = result.stdout.strip()
            store().set(instance_key, home)
            instructions = [
                message.text
                for message in state.messages
                if isinstance(message, ChatMessageSystem)
            ]
            if system_prompt is not None:
                instructions.append(system_prompt)
            store().set(f"{instance_key}_instructions", "\n\n".join(instructions))

        port = store().get("omp_model_port", 3000) + 1
        store().set("omp_model_port", port)
        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases={"inspect": model} if model is not None else None,
            filter=filter,
            sandbox=sandbox,
            port=port,
        ) as bridge:
            agent_dir = f"{home}/.omp/agent"
            settings_path = f"{home}/config.yml"
            setup = await sbox.exec(["mkdir", "-p", agent_dir], user=user)
            if not setup.success:
                raise RuntimeError(
                    f"Unable to create OMP configuration: {setup.stderr}"
                )
            for path, value in (
                (
                    f"{agent_dir}/models.yml",
                    omp_models_config(bridge.port, context_window, max_tokens),
                ),
                (settings_path, omp_settings()),
            ):
                result = await sbox.exec(
                    ["sh", "-c", 'cat > "$1"', "sh", path],
                    input=json.dumps(value),
                    user=user,
                )
                if not result.success:
                    raise RuntimeError(
                        f"Unable to write OMP configuration: {result.stderr}"
                    )

            cmd = [
                "env",
                "-i",
                f"HOME={home}",
                "PATH=/usr/local/bin:/usr/bin:/bin",
                f"PI_CODING_AGENT_DIR={agent_dir}",
                "OTEL_SDK_DISABLED=true",
                binary,
                "--print",
                "--mode",
                "json",
                "--model",
                "inspect/inspect",
                "--smol",
                "inspect/inspect",
                "--slow",
                "inspect/inspect",
                "--plan",
                "inspect/inspect",
                "--profile",
                "default",
                "--tools",
                "read,write,edit,bash,grep,glob,task,todo,wait",
                "--config",
                settings_path,
                "--session",
                f"{home}/session.jsonl",
                "--no-title",
                "--no-extensions",
                "--no-skills",
                "--no-rules",
                "--no-lsp",
                "--no-pty",
                "--no-prewalk",
            ]
            appended_instructions: str = store().get(f"{instance_key}_instructions", "")
            if appended_instructions:
                cmd.extend(["--append-system-prompt", appended_instructions])

            stdout_buffer = CircularByteBuffer(
                SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
            )
            stderr_buffer = CircularByteBuffer(
                SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
            )
            # The first model request can arrive before the start RPC returns.
            # Acquire the handle before allowing cancellation so cleanup owns it.
            with anyio.CancelScope(shield=True):
                process = await sbox.exec_remote(
                    cmd=cmd,
                    options=ExecRemoteStreamingOptions(
                        input=prompt, cwd=agent_cwd, user=user, concurrency=False
                    ),
                    stream=True,
                )
            returncode: int | None = None
            try:
                async for event in process:
                    if isinstance(event, ExecStdout):
                        stdout_buffer.write(event.data.encode("utf-8"))
                    elif isinstance(event, ExecStderr):
                        stderr_buffer.write(event.data.encode("utf-8"))
                    elif isinstance(event, ExecCompleted):
                        returncode = event.exit_code
            finally:
                with anyio.CancelScope(shield=True):
                    await process.kill()
            stdout = stdout_buffer.getvalue().decode("utf-8", errors="replace")
            stderr = stderr_buffer.getvalue().decode("utf-8", errors="replace")
            if debug:
                trace(f"OMP stdout:\n{stdout}\nOMP stderr:\n{stderr}")
            if returncode is None:
                raise RuntimeError("OMP terminated without an exit status.")
            if returncode != 0:
                raise RuntimeError(
                    f"OMP exited with status {returncode}: {_omp_error(stdout, stderr)}"
                )
        store().set(f"{instance_key}_message_count", len(bridge.state.messages))
        store().set(f"{instance_key}_history", _history_digest(bridge.state.messages))
        return bridge.state

    return agent_with(execute, name=name, description=description)


def _history_digest(messages: list[ChatMessage]) -> str:
    """Bind a saved session to its conversation without duplicating it in the store."""
    digest = sha256()
    for message in messages:
        digest.update(message.model_dump_json().encode("utf-8"))
    return digest.hexdigest()


def _omp_error(stdout: str, stderr: str) -> str:
    """Extract structured model failures from OMP's JSON event stream."""
    error = ""
    for line in stdout.splitlines():
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("errorMessage"), str):
            error = message["errorMessage"]
    return (error or stderr.strip() or stdout.strip())[-8000:]
