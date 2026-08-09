from typing import Sequence

from inspect_ai.tool import WebSearchProviders


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


def web_search_grant(enabled: bool) -> WebSearchProviders | None:
    """Bridge grant for an agent's built-in web search tools.

    CLI agents implement their web tools by issuing a *nested* model request
    carrying the provider's native `web_search` / `web_fetch` tool, so the tool
    reaches the model over the bridge rather than from the sandbox. Configuring
    the CLI alone is therefore not sufficient to withhold it — an agent can
    address the model proxy directly — which is why the grant is made here.

    An empty `WebSearchProviders` resolves to the same internal-provider set the
    bridge applies by default, so a grant does not pin a provider list.
    """
    return WebSearchProviders() if enabled else None
