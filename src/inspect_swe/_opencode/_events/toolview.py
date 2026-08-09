from typing import Any

from inspect_ai.tool import ToolCallContent


def tool_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    if tool == "task":
        subagent_type = str(arguments.get("subagent_type", ""))
        content = "### {{description}}\n\n{{prompt}}"
        return ToolCallContent(
            title=f"Task: {subagent_type}" if subagent_type else "Task",
            format="markdown",
            content=content,
        )
    return None
