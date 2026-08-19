"""Web search availability for CLI agents.

CLI agents implement their web tools by issuing a *nested* model request carrying
the provider's native `web_search` / `web_fetch` tool, so the tool reaches the
model over the agent bridge rather than from the sandbox. Configuring the CLI
alone is therefore not sufficient to withhold it — an agent can address the model
proxy directly. Each agent passes its effective setting to
`sandbox_agent_bridge(web_search=...)`, which is where the capability is actually
granted or withheld.
"""

from typing import Sequence


def web_search_tool_disallowed(
    disallowed_tools: Sequence[str] | None, tool_name: str
) -> bool:
    """Whether a CLI tool name appears in a `disallowed_tools` list.

    Accepts both the bare name and the scoped `Tool(...)` form the CLIs allow.
    """
    return any(
        entry == tool_name or entry.startswith(f"{tool_name}(")
        for entry in (disallowed_tools or [])
    )
