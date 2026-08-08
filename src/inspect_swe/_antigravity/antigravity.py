from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

from inspect_ai.agent import (
    Agent,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.tool import MCPServerConfig, ToolChoice, ToolInfo
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util.agentcontext import classify_filter, static_root_classifier
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.path import join_path
from inspect_swe._util.sandbox import resolve_agent_cwd
from inspect_swe._util.trace import trace

from .agentbinary import ensure_antigravity_sdk

logger = getLogger(__file__)


class McpServerEntry(TypedDict):
    name: str
    url: str
    headers: dict[str, str] | None
    tools: list[str] | None


class RunnerPayload(TypedDict):
    """Host-safe request payload written to the sandbox runner.

    Defined here (not imported from sdk_runner) so importing the host agent never
    pulls in google.antigravity, which lives only in the sandbox; sdk_runner's
    pydantic model is the validating counterpart, and load_payload is
    host-testable to keep the two shapes from drifting.
    """

    prompt: str
    system_instructions: str
    bridge_base_url: str
    endpoint_model: str
    api_key: str
    mcp_servers: list[McpServerEntry]
    app_data_dir: str
    save_dir: str
    conversation_id: str | None


# The harness's own data directory (app_data_dir): localharness writes offloaded
# tool results here, and view_file is confined to it so the model can read them
# back. Nothing WE write may live here -- see _STATE_DIRECTORY_NAME.
_DATA_DIRECTORY_NAME: Final = ".antigravity"
# Our state directory, deliberately a SIBLING of the harness data directory and
# outside view_file's readable tree. request.json serializes the MCP server
# configs including their `headers` (an authenticated server's Authorization
# token), so a model that could read it could exfiltrate that credential. The
# SDK's session store lives here too: what the compiled harness persists under
# save_dir is not inspectable, and it may serialize the same configuration.
_STATE_DIRECTORY_NAME: Final = ".antigravity-state"
_RUNNER_FILE: Final = "runner.py"
_CONFIG_FILE: Final = "request.json"
_BRIDGE_PORT_KEY: Final = "antigravity_bridge_port"
_CONVERSATION_ID_KEY: Final = "antigravity_conversation_id"
# The SDK client name the bridge sees on every request; model_aliases keys
# resolve against this name.
_DEFAULT_ENDPOINT_MODEL: Final = "gemini-3.6-flash"
# localharness wants GEMINI_API_KEY present when it builds its Gemini client, but
# on the native path all inference is routed to the loopback bridge via the endpoint
# base_url, so the value is never used for auth. A DUMMY keeps the real host
# credential out of the sandbox (only-dummy-creds-in-sandbox invariant). The same
# value rides the endpoint config in the payload, so there is exactly one constant.
_SANDBOX_DUMMY_API_KEY: Final = "inspect-bridge-unused"


# localharness advertises engine builtin tools (list_resources, read_resource,
# manage_task, schedule) in EVERY generateContent request. They are not members
# of the SDK's BuiltinTools enum, and no CapabilitiesConfig / McpServerConfig /
# policy knob removes them from the DECLARATION (policy.deny_all() gates execution
# only). To keep the agent hermetic -- only the environment's tools, reached via
# the call_mcp_tool dispatcher -- strip the model's tool declaration to this
# allow-list before it reaches the model (the analog of the claude_code/codex CLIs'
# --disallowed-tools), via Inspect's GenerateFilter hook. Agent-scoped; no change
# to the shared bridge. Note this supplements capability delegation rather than
# replacing it: the near-empty ``enabled_tools`` already delegates the nameable
# builtins to the harness config; the filter covers only the residual engine
# tools the SDK's closed enum cannot express.
#
# view_file is allow-listed alongside call_mcp_tool because localharness
# offloads any large MCP tool result to a file and returns the model only a
# file:// pointer ("The output was large and was saved to: ..."). Without a
# read-back path the offloaded payload is unrecoverable, so tool observations
# above localharness's internal cap (~4KB) never reach the model -- which on
# injection tasks silently suppresses the very content under test. view_file is
# that read-back path; the runner confines it to the agent's own app_data_dir.
_ALLOWED_TOOL_NAMES: Final = frozenset({"call_mcp_tool", "view_file"})
# Engine tools known to ride the declaration on the pinned SDK; anything outside
# allowlist + this set means the SDK's tool surface drifted, which should be a
# visible event rather than a silent drop.
_KNOWN_ENGINE_TOOL_NAMES: Final = frozenset(
    {"list_resources", "read_resource", "manage_task", "schedule"}
)


_ModelGenerateFilter: TypeAlias = Callable[
    [Model, list[ChatMessage], list[ToolInfo], ToolChoice | None, GenerateConfig],
    Awaitable[ModelOutput | GenerateInput | None],
]


def _confine_declared_tools(
    user_filter: _ModelGenerateFilter | None,
) -> _ModelGenerateFilter:
    """Wrap any user filter so the model is only ever offered the allow-listed tools."""
    warned_names: set[str] = set()

    async def _filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        eff = (messages, tools, tool_choice, config)
        if user_filter is not None:
            result = await user_filter(model, messages, tools, tool_choice, config)
            if isinstance(result, ModelOutput):
                return result
            if result is not None:
                eff = (result.input, result.tools, result.tool_choice, result.config)
        eff_messages, eff_tools, eff_choice, eff_config = eff
        # SDK drift detection: a declared tool outside allowlist + known engine
        # tools means localharness changed its tool surface; the filter still
        # drops it (fail closed), but loudly enough to attribute downstream
        # denial loops or missing tools to the SDK version.
        unexpected = {
            tool.name
            for tool in eff_tools
            if tool.name not in _ALLOWED_TOOL_NAMES
            and tool.name not in _KNOWN_ENGINE_TOOL_NAMES
        } - warned_names
        if unexpected:
            warned_names.update(unexpected)
            logger.warning(
                "antigravity: dropping unrecognized declared tool(s) "
                f"{sorted(unexpected)}; the SDK's tool surface has drifted from "
                "the pinned version this agent was validated against."
            )
        confined = [tool for tool in eff_tools if tool.name in _ALLOWED_TOOL_NAMES]
        if user_filter is None and len(confined) == len(eff_tools):
            return None
        return GenerateInput(
            input=eff_messages,
            tools=confined,
            tool_choice=eff_choice,
            config=eff_config,
        )

    return _filter


def build_antigravity_filter(
    filter: _ModelGenerateFilter | None,
) -> _ModelGenerateFilter:
    """Antigravity bridge filter: tool confinement + static root agent context."""
    return classify_filter(_confine_declared_tools(filter), static_root_classifier)


@dataclass(frozen=True, slots=True)
class SDKExecutionSpec:
    command: list[str]
    cwd: str
    env: dict[str, str]
    user: str | None


def sdk_execution_spec(
    *,
    python: str,
    runner_path: str,
    config_path: str,
    cwd: str,
    home: str,
    user: str | None,
    api_key: str = _SANDBOX_DUMMY_API_KEY,
) -> SDKExecutionSpec:
    """Describe the unprivileged process that runs the SDK and localharness.

    ``cwd`` is the agent's working directory (typically the evaluated
    workspace); ``home`` is the sandbox user's home directory, where SDK state
    lives. They are distinct on purpose: pointing HOME at the workspace would
    present the evaluated repository as the user's home.
    """
    env = {
        "HOME": home,
        "NO_PROXY": "127.0.0.1,localhost",
        "PYTHONNOUSERSITE": "1",
        "no_proxy": "127.0.0.1,localhost",
        "GEMINI_API_KEY": api_key,
    }
    return SDKExecutionSpec(
        command=[python, runner_path, "--config", config_path],
        cwd=cwd,
        env=env,
        user=user,
    )


def _mcp_server_entries(configs: Sequence[MCPServerConfig]) -> list[McpServerEntry]:
    """Convert configured MCP servers to runner payload entries.

    localharness reaches MCP servers over streamable HTTP, so every configured
    server (static or bridged) must be an HTTP config with a URL. The SDK has
    no SSE client (``types.McpStreamableHttpServer`` is its only HTTP server),
    so ``type="sse"`` configs are rejected rather than silently reconstructed
    as streamable HTTP. Inspect's per-server ``tools`` allowlist and ``headers``
    ride the entry so the runner can enforce and forward them (a ``tools`` list
    dropped here would silently expose every tool on the server).
    """
    entries: list[McpServerEntry] = []
    for config in configs:
        if not isinstance(config, MCPServerConfigHTTP) or not config.url:
            raise ValueError(
                f"antigravity requires HTTP MCP server configs with a URL; "
                f"server {config.name!r} is {type(config).__name__}."
            )
        if config.type == "sse":
            raise ValueError(
                f"antigravity does not support SSE MCP servers (the SDK speaks "
                f"streamable HTTP only); server {config.name!r} declares "
                f'type="sse".'
            )
        entries.append(
            {
                "name": config.name,
                "url": config.url,
                "headers": config.headers,
                "tools": list(config.tools) if isinstance(config.tools, list) else None,
            }
        )
    return entries


@agent
def antigravity(
    name: str = "Antigravity",
    description: str = "Sandboxed Google Antigravity SDK coding agent.",
    system_prompt: str | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: _ModelGenerateFilter | None = None,
    retry_refusals: int | None = None,
    user: str | None = None,
    cwd: str | None = None,
    sandbox: str | None = None,
    version: str = "0.1.7",
    endpoint_model: str = _DEFAULT_ENDPOINT_MODEL,
    debug: bool | None = None,
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
        mcp_servers: MCP servers to make available to the agent (HTTP configs;
            localharness reaches servers over streamable HTTP).
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP.
        model: Model name to use for the inspect bridge (defaults to the task model).
        model_aliases: Optional mapping of model names to Model instances/strings.
            Alias keys resolve against the client-sent model name, which for this
            agent is ``endpoint_model``.
        filter: Filter for intercepting bridged model requests. Note this agent's
            filter is typed Model-first (the non-deprecated GenerateFilter form);
            the deprecated str-first form is not accepted, unlike sibling agents,
            because the confinement wrapper dispatches on the outermost filter's
            first-parameter annotation.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        user: Sandbox user to run the SDK as (defaults to the sandbox default user).
        cwd: Working directory for the SDK (defaults to the user's home directory).
        sandbox: Optional sandbox environment name.
        version: Version of the google-antigravity SDK to provision when it is not
            already present in the sandbox image.
        endpoint_model: Model name the SDK client presents to the bridge endpoint.
        debug: Trace the full runner output.
    """
    bridge_model = f"inspect/{model}" if model else "inspect"

    async def execute(state: AgentState) -> AgentState:
        bridge_port = store().get(_BRIDGE_PORT_KEY, 3000) + 1
        store().set(_BRIDGE_PORT_KEY, bridge_port)

        async with sandbox_agent_bridge(
            state,
            model=bridge_model,
            model_aliases=model_aliases,
            filter=build_antigravity_filter(filter),
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=bridge_port,
            bridged_tools=bridged_tools,
        ) as bridge:
            sbox = sandbox_env(sandbox)

            # ensure google-antigravity is present (skip if baked into the image)
            python = await ensure_antigravity_sdk(sbox, user, version=version)

            # resolve working directory (home dir if sandbox default is '/')
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # SDK state (runner, request, session store, HOME) lives in the
            # sandbox user's home, not the working directory: with an explicit
            # cwd the working directory is typically the evaluated workspace,
            # and writing .antigravity/ there (or presenting it as HOME) would
            # leak agent state into the repository under evaluation. Same
            # detection as gemini_cli.
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"
            data_dir = join_path(sandbox_home, _DATA_DIRECTORY_NAME)
            state_dir = join_path(sandbox_home, _STATE_DIRECTORY_NAME)
            runner_path = join_path(state_dir, _RUNNER_FILE)
            config_path = join_path(state_dir, _CONFIG_FILE)

            server_entries = _mcp_server_entries(
                [*(mcp_servers or []), *bridge.mcp_server_configs]
            )

            prompt, has_assistant_response = build_user_prompt(state.messages)
            system_messages = [
                message.text
                for message in state.messages
                if isinstance(message, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            # Resume a saved SDK conversation on re-invocation (handoff back,
            # operator follow-up): the runner reports the SDK conversation id and
            # we thread it into the next run's config. Without an id to resume,
            # a re-invocation would silently drop all prior context
            # (build_user_prompt only includes messages after the last assistant
            # turn), so fail loudly instead.
            conversation_id: str | None = store().get(_CONVERSATION_ID_KEY, None)
            if has_assistant_response and conversation_id is None:
                raise RuntimeError(
                    "antigravity cannot resume a conversation it did not start: "
                    "the transcript already has assistant turns but no saved SDK "
                    "conversation id exists for this sample."
                )

            payload: RunnerPayload = {
                "prompt": prompt,
                "system_instructions": "\n\n".join(system_messages),
                "bridge_base_url": f"http://127.0.0.1:{bridge.port}",
                "endpoint_model": endpoint_model,
                "api_key": _SANDBOX_DUMMY_API_KEY,
                "mcp_servers": server_entries,
                "app_data_dir": data_dir,
                "save_dir": join_path(state_dir, "session"),
                "conversation_id": conversation_id,
            }

            # Both directories run AS the agent user, so file ownership
            # hardening would be ineffective (unlink rights come from the
            # directory) and the agent process is model-controlled anyway; none
            # is attempted. What matters is the split: view_file is confined to
            # data_dir, and everything we write goes to state_dir.
            for directory, label in ((data_dir, "data"), (state_dir, "state")):
                mkdir = await sbox.exec(
                    ["mkdir", "-p", "-m", "0700", directory], user=user
                )
                if not mkdir.success:
                    raise RuntimeError(
                        f"Failed to create antigravity {label} directory: "
                        f"{mkdir.stderr.strip()}"
                    )
            await sbox.write_file(
                runner_path, Path(__file__).with_name("sdk_runner.py").read_bytes()
            )
            await sbox.write_file(
                config_path, json.dumps(payload, sort_keys=True).encode("utf-8")
            )

            spec = sdk_execution_spec(
                python=python,
                runner_path=runner_path,
                config_path=config_path,
                cwd=agent_cwd,
                home=sandbox_home,
                user=user,
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
            if debug:
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
                detail = result.stderr.strip() or (
                    "no stderr; run with debug=True for full output"
                )
                raise RuntimeError(
                    f"Antigravity SDK exited {result.returncode}: {detail}"
                )

            # Report view_file path-argument drift. The runner cannot warn
            # usefully itself: its stderr only reaches the host with debug=True
            # or on failure, and drift makes view_file deny-all on an otherwise
            # successful run -- offloaded results unreadable, silently.
            drifted_arg_keys = _reported_view_file_arg_drift(result.stdout)
            if drifted_arg_keys:
                logger.warning(
                    "antigravity: view_file was called without the expected "
                    f"path argument; the harness sent {drifted_arg_keys} "
                    "instead. Offloaded tool results are being denied, so large "
                    "tool output is not reaching the model -- the SDK's tool "
                    "argument surface has drifted from the pinned version."
                )

            # persist the SDK conversation id for resume on re-invocation
            reported_id = _reported_conversation_id(result.stdout)
            if reported_id is not None:
                store().set(_CONVERSATION_ID_KEY, reported_id)
            else:
                logger.warning(
                    "antigravity: runner output carried no conversation id; a "
                    "re-invocation of this agent will not be able to resume."
                )

        return bridge.state

    return agent_with(execute, name=name, description=description)


def _reported_conversation_id(stdout: str) -> str | None:
    """Extract the SDK conversation id from the runner's JSON result line."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        conversation_id = parsed.get("conversation_id")
        if isinstance(conversation_id, str) and conversation_id:
            return conversation_id
        return None
    return None


def _runner_result_line(stdout: str) -> dict[str, object] | None:
    """Parse the runner's JSON result line (the last JSON object it printed)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def _reported_view_file_arg_drift(stdout: str) -> list[str]:
    """Argument keys seen when view_file was called without its path argument."""
    parsed = _runner_result_line(stdout)
    if parsed is None:
        return []
    drifted = parsed.get("view_file_unexpected_arg_keys")
    if not isinstance(drifted, list):
        return []
    return [str(key) for key in drifted]
