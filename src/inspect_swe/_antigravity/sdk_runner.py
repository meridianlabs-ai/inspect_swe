from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

import anyio
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from google.antigravity import LocalAgentConfig


class McpServerEntry(TypedDict):
    name: str
    url: str


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
        "mcp_servers": [{"name": s.name, "url": s.url} for s in parsed.mcp_servers],
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
    dispatcher for every configured server. Tool RESULTS come back as
    functionResponse parts inside model-role turns, which the host bridge
    converter re-roles into tool messages (requires inspect_ai >= 0.3.250).
    """
    from google.antigravity import LocalAgentConfig, types
    from google.antigravity.hooks import policy

    mcp_servers = [
        types.McpStreamableHttpServer(name=server["name"], url=server["url"])
        for server in payload["mcp_servers"]
    ]
    configured_names = {server["name"] for server in payload["mcp_servers"]}

    def _targets_configured_server(args: dict[str, Any]) -> bool:
        return args.get("ServerName") in configured_names

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
        allowed for server in mcp_servers for allowed in policy.allow(server)
    ]
    return LocalAgentConfig(
        models=[bridge_model],
        system_instructions=payload["system_instructions"],
        capabilities=types.CapabilitiesConfig(
            enabled_tools=[],
            enable_subagents=False,
        ),
        mcp_servers=mcp_servers,
        policies=[
            policy.deny_all(),
            policy.allow("call_mcp_tool", when=_targets_configured_server),
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
