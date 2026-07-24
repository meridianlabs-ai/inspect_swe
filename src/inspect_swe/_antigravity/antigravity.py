from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

from inspect_ai.agent import (
    Agent,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model
from inspect_ai.tool import MCPServerConfig
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.trace import trace

from .agentbinary import ensure_antigravity_sdk


class RunnerPayload(TypedDict):
    """Host-safe request payload written to the sandbox runner.

    Defined here (not imported from sdk_runner) so importing the host agent never
    pulls in google.antigravity, which lives only in the sandbox.
    """

    prompt: str
    system_instructions: str
    bridge_base_url: str
    mcp_name: Literal["taiga-mcp"]
    mcp_url: str
    app_data_dir: str
    save_dir: str


_MODEL_USER: Final = "model"
_MODEL_HOME: Final = "/home/model"
_RUNNER_DIRECTORY: Final = f"{_MODEL_HOME}/.antigravity"
_RUNNER_PATH: Final = f"{_RUNNER_DIRECTORY}/runner.py"
_CONFIG_PATH: Final = f"{_RUNNER_DIRECTORY}/request.json"
_BRIDGE_PORT_KEY: Final = "antigravity_bridge_port"
# localharness wants GEMINI_API_KEY present when it builds its Gemini client, but
# on the native path all inference is routed to the loopback bridge via the endpoint
# base_url, so the value is never used for auth. A DUMMY keeps the real host
# credential out of the sandbox (only-dummy-creds-in-sandbox invariant).
_SANDBOX_DUMMY_API_KEY: Final = "inspect-bridge-unused"


@dataclass(frozen=True, slots=True)
class SDKExecutionSpec:
    command: list[str]
    cwd: str
    env: dict[str, str]
    user: str


def sdk_execution_spec(
    *, python: str, runner_path: str, config_path: str
) -> SDKExecutionSpec:
    """Describe the unprivileged process that runs the SDK and localharness."""
    env = {
        "HOME": _MODEL_HOME,
        "NO_PROXY": "127.0.0.1,localhost",
        "PYTHONNOUSERSITE": "1",
        "no_proxy": "127.0.0.1,localhost",
        "GEMINI_API_KEY": _SANDBOX_DUMMY_API_KEY,
    }
    return SDKExecutionSpec(
        command=[python, runner_path, "--config", config_path],
        cwd=_MODEL_HOME,
        env=env,
        user=_MODEL_USER,
    )


def _taiga_mcp_config(configs: Sequence[MCPServerConfig]) -> MCPServerConfigHTTP:
    matching_configs = [config for config in configs if config.name == "taiga-mcp"]
    match matching_configs:
        case [MCPServerConfigHTTP(url=url) as config] if url:
            return config
        case []:
            raise RuntimeError(
                "The antigravity agent requires one taiga-mcp bridge server."
            )
        case [MCPServerConfigHTTP()]:
            raise RuntimeError(
                "The taiga-mcp bridge server must provide a nonempty URL."
            )
        case _:
            raise RuntimeError(
                "The antigravity agent requires exactly one HTTP taiga-mcp server."
            )


@agent
def antigravity(
    name: str = "Antigravity",
    description: str = "Sandboxed Google Antigravity SDK coding agent.",
    system_prompt: str | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    user: str | None = _MODEL_USER,
    cwd: str | None = _MODEL_HOME,
    sandbox: str | None = None,
) -> Agent:
    """Google Antigravity SDK agent.

    Runs Google's Antigravity SDK (``google-antigravity``, which bundles the
    ``localharness`` engine) headless inside an Inspect sandbox, with model calls
    routed through the sandbox agent bridge. The SDK's native connection speaks the
    Gemini generateContent wire directly to the bridge via a ``GeminiAPIEndpoint``
    ``base_url`` override (no OpenAI translation), mirroring ``gemini_cli``.

    Args:
        name: Agent name.
        description: Agent description.
        system_prompt: Additional system prompt to append.
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
        model: Model name to use for the inspect bridge (defaults to the task model).
        model_aliases: Optional mapping of model names to Model instances/strings.
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        user: Sandbox user to run the SDK as.
        cwd: Working directory for the SDK.
        sandbox: Optional sandbox environment name.
    """
    if user != _MODEL_USER:
        raise ValueError("antigravity only supports the 'model' sandbox user.")
    if cwd != _MODEL_HOME:
        raise ValueError("antigravity only supports /home/model as its cwd.")

    bridge_model = f"inspect/{model}" if model else "inspect"

    async def execute(state: AgentState) -> AgentState:
        bridge_port = store().get(_BRIDGE_PORT_KEY, 3000) + 1
        store().set(_BRIDGE_PORT_KEY, bridge_port)

        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=bridge_port,
            bridged_tools=bridged_tools,
        ) as bridge:
            sbox = sandbox_env(sandbox)

            # ensure google-antigravity is present (skip if baked into the image)
            python = await ensure_antigravity_sdk(sbox, user)

            mcp_server = _taiga_mcp_config(bridge.mcp_server_configs)
            prompt, _ = build_user_prompt(state.messages)
            system_messages = [
                message.text
                for message in state.messages
                if isinstance(message, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            payload: RunnerPayload = {
                "prompt": prompt,
                "system_instructions": "\n\n".join(system_messages),
                "bridge_base_url": f"http://127.0.0.1:{bridge_port}",
                "mcp_name": "taiga-mcp",
                "mcp_url": mcp_server.url,
                "app_data_dir": _RUNNER_DIRECTORY,
                "save_dir": f"{_RUNNER_DIRECTORY}/session",
            }

            directory_result = await sbox.exec(
                [
                    "bash",
                    "-c",
                    f"install -d -o {user} -g {user} -m 0700 {_RUNNER_DIRECTORY}",
                ],
                user="root",
            )
            if not directory_result.success:
                detail = directory_result.stderr.strip()
                raise RuntimeError(
                    f"Failed to create antigravity runner directory: {detail}"
                )

            await sbox.write_file(
                _RUNNER_PATH, Path(__file__).with_name("sdk_runner.py").read_bytes()
            )
            runner_permissions = await sbox.exec(
                [
                    "bash",
                    "-c",
                    f"chown root:root {_RUNNER_PATH} && chmod 0644 {_RUNNER_PATH}",
                ],
                user="root",
            )
            if not runner_permissions.success:
                raise RuntimeError(
                    f"Failed to secure antigravity runner: {runner_permissions.stderr.strip()}"
                )

            await sbox.write_file(
                _CONFIG_PATH,
                json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
            config_permissions = await sbox.exec(
                [
                    "bash",
                    "-c",
                    f"chown {user}:{user} {_CONFIG_PATH} && chmod 0600 {_CONFIG_PATH}",
                ],
                user="root",
            )
            if not config_permissions.success:
                raise RuntimeError(
                    f"Failed to secure antigravity request: {config_permissions.stderr.strip()}"
                )

            spec = sdk_execution_spec(
                python=python,
                runner_path=_RUNNER_PATH,
                config_path=_CONFIG_PATH,
            )
            result = await sbox.exec_remote(
                cmd=spec.command,
                options=ExecRemoteAwaitableOptions(
                    concurrency=False,
                    cwd=spec.cwd,
                    env=spec.env,
                    user=spec.user,
                ),
                stream=False,
            )
            trace(
                "\n".join(
                    (
                        "Antigravity SDK runner output:",
                        "stdout:",
                        result.stdout,
                        "stderr:",
                        result.stderr,
                    )
                )
            )
            if not result.success:
                detail = result.stderr.strip() or "no stderr; full output in trace"
                raise RuntimeError(
                    f"Antigravity SDK exited {result.returncode}: {detail}"
                )

        return bridge.state

    return agent_with(execute, name=name, description=description)
