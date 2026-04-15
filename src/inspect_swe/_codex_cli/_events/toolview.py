from typing import Any

from inspect_ai.tool import ToolCallContent


def tool_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    """Build a compact view for common Codex tool calls."""
    for key in ("cmd", "command"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return ToolCallContent(
                title=tool,
                format="markdown",
                content=f"```bash\n{value}\n```",
            )
    return None
