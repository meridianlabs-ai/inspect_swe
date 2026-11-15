from collections.abc import Sequence

from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import Tool, ToolDef

DEFAULT_BRIDGED_TOOLS_NAME = "tools"
"""MCP server name for plain `Tool`s passed in `bridged_tools`."""


def resolve_bridged_tools(
    bridged_tools: Sequence[Tool | BridgedToolsSpec] | None,
) -> list[BridgedToolsSpec] | None:
    """Resolve `bridged_tools` into a list of `BridgedToolsSpec`.

    Specs pass through unchanged. Plain `Tool`s are grouped (in order) into
    one additional spec named `DEFAULT_BRIDGED_TOOLS_NAME`, with the default
    `require_proposal=True`.

    Raises:
        TypeError: An item is neither a `BridgedToolsSpec` nor a `Tool`.
        ValueError: Two plain `Tool`s share a name, or a plain `Tool` is
            passed alongside a spec that already uses the default server name.
    """
    if not bridged_tools:
        return None

    specs: list[BridgedToolsSpec] = []
    tools: list[Tool] = []
    for item in bridged_tools:
        if isinstance(item, BridgedToolsSpec):
            specs.append(item)
        elif callable(item):
            tools.append(item)
        else:
            raise TypeError(
                "bridged_tools items must be Tool or BridgedToolsSpec, "
                f"not {type(item).__name__}"
            )

    if tools:
        names = [ToolDef(tool).name for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Plain Tools passed in bridged_tools share a name: "
                f"{', '.join(duplicates)}. Each needs a unique name."
            )
        if any(spec.name == DEFAULT_BRIDGED_TOOLS_NAME for spec in specs):
            raise ValueError(
                f"Plain Tools passed in bridged_tools are served as MCP server "
                f"'{DEFAULT_BRIDGED_TOOLS_NAME}', which a BridgedToolsSpec "
                "already uses. Rename that spec or add the tools to it."
            )
        specs.append(BridgedToolsSpec(name=DEFAULT_BRIDGED_TOOLS_NAME, tools=tools))

    return specs
