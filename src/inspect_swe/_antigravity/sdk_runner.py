from __future__ import annotations

import argparse
import json
import posixpath
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

import anyio
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from google.antigravity import LocalAgentConfig


def _confine_to_dir(directory: str) -> Callable[[dict[str, Any]], bool]:
    """Policy predicate limiting view_file to paths inside ``directory``.

    view_file exists ONLY to read back localharness's offloaded tool results,
    which live under the agent's own app_data_dir. Paths are lexically
    normalized first so ``..`` traversal cannot escape the prefix, and the
    prefix match is segment-aware so a sibling like ``<dir>-evil`` does not
    match. Host-testable: no SDK import.
    """
    confined = posixpath.normpath(directory)

    def _within(args: dict[str, Any]) -> bool:
        raw = args.get("AbsolutePath")
        if not isinstance(raw, str) or not raw:
            return False
        resolved = posixpath.normpath(raw)
        return resolved == confined or resolved.startswith(confined + "/")

    return _within


class McpServerEntry(TypedDict):
    name: str
    url: str
    headers: dict[str, str] | None
    tools: list[str] | None


class RunnerPayload(TypedDict):
    prompt: str
    system_instructions: str
    bridge_base_url: str
    endpoint_model: str
    api_key: str
    mcp_servers: list[McpServerEntry]
    app_data_dir: str
    save_dir: str
    conversation_id: str | None


class _McpServerModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: str
    url: str
    headers: dict[str, str] | None
    tools: list[str] | None


class _RunnerPayloadModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    prompt: str
    system_instructions: str
    bridge_base_url: str
    endpoint_model: str
    api_key: str
    mcp_servers: list[_McpServerModel]
    app_data_dir: str
    save_dir: str
    conversation_id: str | None = None


def load_payload(config_path: Path) -> RunnerPayload:
    """Parse the host-created request into the runner's trusted payload.

    The google.antigravity import lives inside build_config()/run() (not module
    scope) so this loader stays importable -- and therefore testable -- on the
    host, where the SDK is intentionally not installed.
    """
    parsed = _RunnerPayloadModel.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    return {
        "prompt": parsed.prompt,
        "system_instructions": parsed.system_instructions,
        "bridge_base_url": parsed.bridge_base_url,
        "endpoint_model": parsed.endpoint_model,
        "api_key": parsed.api_key,
        "mcp_servers": [
            {
                "name": s.name,
                "url": s.url,
                "headers": s.headers,
                "tools": list(s.tools) if s.tools is not None else None,
            }
            for s in parsed.mcp_servers
        ],
        "app_data_dir": parsed.app_data_dir,
        "save_dir": parsed.save_dir,
        "conversation_id": parsed.conversation_id,
    }


def build_config(payload: RunnerPayload) -> "LocalAgentConfig":
    """Create the confined native-Gemini SDK configuration for one bridged sample.

    The SDK's native localharness backend speaks the Gemini generateContent wire
    directly to the Inspect loopback bridge (``base_url``), so there is no OpenAI
    translation seam. This mirrors inspect_swe's gemini_cli, which points a Google
    agent at the bridge via GOOGLE_GEMINI_BASE_URL plus a placeholder api key.
    localharness dispatches configured MCP tools through its ``call_mcp_tool``
    wrapper (with ServerName/ToolName in the args), so the policy must allow that
    dispatcher for every configured server. Inspect's per-server ``tools``
    allowlist is enforced at three layers, all fail-closed: ``enabled_tools`` on
    the server config (limits what the SDK declares), per-tool APPROVE policies
    (``policy.allow(server, mcp_tools=...)`` instead of the server-wide
    wildcard), and the dispatcher predicate (rejects ``call_mcp_tool`` calls
    naming a tool outside the allowlist). Tool RESULTS come back as
    functionResponse parts inside model-role turns, which the host bridge
    converter re-roles into tool messages (requires inspect_ai >= 0.3.250).

    VIEW_FILE is the one builtin the model keeps: localharness offloads any
    large MCP tool result to a file and returns only a file:// pointer, so
    without a read-back path every tool observation above its internal cap
    (~4KB) is silently lost. A policy predicate confines view_file to the
    agent's own app_data_dir, where localharness writes offloaded results; it
    cannot read task files, the workspace, or grading ground truth elsewhere
    in the sandbox.
    """
    from google.antigravity import LocalAgentConfig, types
    from google.antigravity.hooks import policy

    mcp_servers = [
        types.McpStreamableHttpServer(
            name=server["name"],
            url=server["url"],
            headers=server["headers"],
            enabled_tools=server["tools"],
        )
        for server in payload["mcp_servers"]
    ]
    allowed_tools: dict[str, frozenset[str] | None] = {
        server["name"]: (
            frozenset(server["tools"]) if server["tools"] is not None else None
        )
        for server in payload["mcp_servers"]
    }

    def _targets_configured_server(args: dict[str, Any]) -> bool:
        server_name = args.get("ServerName")
        if server_name not in allowed_tools:
            return False
        server_allowed = allowed_tools[server_name]
        return server_allowed is None or args.get("ToolName") in server_allowed

    # Setting base_url makes GeminiAPIEndpoint.validate_endpoint() short-circuit
    # the real-key requirement; the placeholder api_key rides the localharness
    # proto but the bridge never checks it (only a dummy value enters the
    # sandbox -- the same value the runner env carries as GEMINI_API_KEY).
    bridge_model = types.ModelTarget(
        name=payload["endpoint_model"],
        types=[types.ModelType.TEXT, types.ModelType.IMAGE],
        endpoint=types.GeminiAPIEndpoint(
            base_url=payload["bridge_base_url"],
            api_key=payload["api_key"],
        ),
    )
    server_policies = [
        allowed
        for entry, server in zip(payload["mcp_servers"], mcp_servers, strict=False)
        for allowed in policy.allow(server, mcp_tools=entry["tools"])
    ]
    return LocalAgentConfig(
        models=[bridge_model],
        system_instructions=payload["system_instructions"],
        capabilities=types.CapabilitiesConfig(
            # VIEW_FILE is the read-back path for offloaded MCP tool results
            # (see the docstring); everything else stays delegated/denied.
            enabled_tools=[types.BuiltinTools.VIEW_FILE],
            enable_subagents=False,
        ),
        mcp_servers=mcp_servers,
        policies=[
            policy.deny_all(),
            policy.allow("call_mcp_tool", when=_targets_configured_server),
            policy.allow(
                types.BuiltinTools.VIEW_FILE.value,
                when=_confine_to_dir(payload["app_data_dir"]),
            ),
            *server_policies,
        ],
        workspaces=[],
        app_data_dir=payload["app_data_dir"],
        save_dir=payload["save_dir"],
        conversation_id=payload["conversation_id"],
    )


async def run(payload: RunnerPayload) -> None:
    """Run one SDK Agent turn (resuming a saved conversation when given one)."""
    from google.antigravity import Agent

    async with Agent(build_config(payload)) as sdk_agent:
        response = await sdk_agent.chat(payload["prompt"])
        final_text = await response.text()
        print(
            json.dumps(
                {
                    "conversation_id": getattr(sdk_agent, "conversation_id", None),
                    "final_text": final_text,
                    "steps": len(sdk_agent.conversation.history),
                    "turn_count": sdk_agent.conversation.turn_count,
                },
                sort_keys=True,
            )
        )


def main() -> None:
    """Load one host-created request and execute it with AnyIO."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    anyio.run(run, load_payload(Path(arguments.config)))


if __name__ == "__main__":
    main()
