import asyncio
import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import detect_sandbox_platform
from inspect_swe._util.trace import trace

from .agentbinary import ensure_antigravity_cli_setup, provision_antigravity_home
from .interceptor import ensure_mitmproxy, start_interceptor


@agent
def antigravity_cli(
    name: str = "Antigravity CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    binary_source: str | None = None,
    token_source: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = "root",
    sandbox: str | None = None,
    print_timeout: str = "876000h",
) -> Agent:
    """Create an Antigravity CLI agent using the Inspect model bridge."""
    model = f"inspect/{model}" if model is not None else "inspect"
    resolved_skills = read_skills(skills) if skills is not None else None
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        bridge_port_key = "antigravity_bridge_port"
        bridge_port = store().get(bridge_port_key, 3000) + 1
        store().set(bridge_port_key, bridge_port)
        intercept_port = bridge_port + 5000

        async with sandbox_agent_bridge(
            state,
            model=model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=bridge_port,
            bridged_tools=bridged_tools,
        ) as bridge:
            sbox = sandbox_env(sandbox)
            platform = await detect_sandbox_platform(sbox)
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"

            if resolved_skills is not None:
                await install_skills(
                    resolved_skills, sbox, user, f"{sandbox_home}/.gemini/skills"
                )

            agy = await ensure_antigravity_cli_setup(
                sbox, binary_source=binary_source, user=user
            )
            antigravity_home = await provision_antigravity_home(
                sbox,
                sandbox_home=sandbox_home,
                token_source=token_source,
                user=user,
            )
            await ensure_mitmproxy(sbox, platform, user)
            proxy_proc, ca_cert_path, monitor_task = await start_interceptor(
                sbox,
                listen_port=intercept_port,
                bridge_port=bridge.port,
                confdir=f"{sandbox_home}/.mitmproxy-antigravity",
                user=user,
            )

            try:
                all_mcp_servers = list(mcp_servers or [])
                await sbox.write_file(
                    f"{antigravity_home}/mcp_config.json",
                    resolve_mcp_servers_antigravity(all_mcp_servers),
                )

                system_messages = [
                    message.text
                    for message in state.messages
                    if isinstance(message, ChatMessageSystem)
                ]
                if system_prompt is not None:
                    system_messages.append(system_prompt)

                prompt, _ = build_user_prompt(state.messages)
                if system_messages:
                    combined_system = "\n\n".join(system_messages)
                    prompt = f"{combined_system}\n\n{prompt}"

                agent_env = build_antigravity_agent_env(
                    sandbox_home=sandbox_home,
                    ca_cert_path=ca_cert_path,
                    intercept_port=intercept_port,
                    env=env,
                )
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    result = await sbox.exec_remote(
                        cmd=[
                            "bash",
                            "-c",
                            'exec 0</dev/null; "$@"',
                            "bash",
                            agy,
                            "-p",
                            agent_prompt,
                            "--dangerously-skip-permissions",
                            # agy's -p print mode self-terminates the ENTIRE run
                            # after --print-timeout (agy default 5m0s) — a total
                            # wall-clock cap on the whole trajectory, not a per-call
                            # timeout, and unique to agy (other inspect_swe CLIs have
                            # no such cap). agent-c trajectories routinely run 200+
                            # tool calls / 30m+, so any fixed cap eventually kills a
                            # legit run mid-trajectory. --print-timeout=0 is NOT
                            # "disabled" (tested: it times out immediately), so this
                            # defaults absurdly high (~100y) to never bind — Inspect's
                            # --time-limit/--working-limit govern duration instead.
                            f"--print-timeout={print_timeout}",
                        ],
                        options=ExecRemoteAwaitableOptions(
                            cwd=cwd,
                            env=agent_env,
                            concurrency=False,
                        ),
                        stream=False,
                    )
                    debug_output.append(result.stdout)
                    debug_output.append(result.stderr)

                    if not result.success:
                        # Surface agy's real failure — its stderr and exit code.
                        # agy dumps its entire thinking transcript to stdout even
                        # on failure, which buries the actual error; keep it out of
                        # the exception and put the full output in the trace instead.
                        debug_output.insert(0, "Antigravity CLI Debug Output (failed):")
                        trace("\n".join(debug_output))
                        detail = result.stderr.strip() or (
                            "no stderr (agy may have hit --print-timeout); "
                            "full output in trace"
                        )
                        raise RuntimeError(
                            f"Antigravity CLI exited {result.returncode}: {detail}"
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

                debug_output.insert(0, "Antigravity CLI Debug Output:")
                trace("\n".join(debug_output))
            finally:
                monitor_task.cancel()
                await asyncio.shield(proxy_proc.kill())

        return bridge.state

    return agent_with(execute, name=name, description=description)


def build_antigravity_agent_env(
    *,
    sandbox_home: str,
    ca_cert_path: str,
    intercept_port: int,
    env: dict[str, str] | None,
) -> dict[str, str]:
    """Build the Antigravity CLI environment for proxy-mediated model calls."""
    return {
        "HTTPS_PROXY": f"http://127.0.0.1:{intercept_port}",
        "HTTP_PROXY": f"http://127.0.0.1:{intercept_port}",
        "SSL_CERT_FILE": ca_cert_path,
        "HOME": sandbox_home,
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    } | (env or {})


def resolve_mcp_servers_antigravity(mcp_servers: Sequence[MCPServerConfig]) -> str:
    """Build Antigravity CLI MCP configuration from Inspect server configs."""
    mcp_servers_config: dict[str, Any] = {}
    for server in mcp_servers:
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        if isinstance(server, MCPServerConfigHTTP) and "url" in config:
            config["serverUrl"] = config.pop("url")
        if "cwd" in config and not isinstance(config["cwd"], str):
            config["cwd"] = str(config["cwd"])
        mcp_servers_config[server.name] = config
    return json.dumps({"mcpServers": mcp_servers_config}, indent=2)
