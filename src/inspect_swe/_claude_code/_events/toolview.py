from posixpath import splitext
from typing import Any, Callable

from inspect_ai.tool import ToolCallContent

_CODE_FENCE_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".css": "css",
    ".html": "html",
    ".sql": "sql",
    ".rs": "rust",
    ".go": "go",
    ".r": "r",
    ".R": "r",
    ".md": "markdown",
    ".qmd": "markdown",
}


def tool_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    tool_view = tool_views.get(tool)
    if tool_view:
        return tool_view(arguments)
    else:
        return None


def write_tool_view(arguments: dict[str, Any]) -> ToolCallContent:
    file_path = str(arguments.get("file_path", "") or "")
    end_body = "\n" if not str(arguments.get("content", "")).endswith("\n") else ""
    _, ext = splitext(file_path)
    ext_lower = ext.lower()
    if ext_lower in _CODE_FENCE_LANGUAGES:
        lang = _CODE_FENCE_LANGUAGES[ext_lower]
    else:
        lang = ""
    body = "``````" + lang + "\n{{content}}" + end_body + "``````"
    content = f"`file_path: {file_path}`\n\n{body}\n"
    return ToolCallContent(title="Write", format="markdown", content=content)


def exit_plan_mode_tool_view(arguments: dict[str, Any]) -> ToolCallContent:
    content = "``````markdown\n" + "{{plan}}" + "\n``````"
    return ToolCallContent(title="ExitPlanMode", format="markdown", content=content)


def task_tool_view(arguments: dict[str, Any]) -> ToolCallContent | None:
    subagent_type = str(arguments.get("subagent_type", ""))
    content = "### {{description}}\n\n" + "``````markdown\n" + "{{prompt}}" + "\n``````"
    return ToolCallContent(
        title=subagent_type or "Task", format="markdown", content=content
    )


tool_views: dict[str, Callable[[dict[str, Any]], ToolCallContent | None]] = {
    "Write": write_tool_view,
    "ExitPlanMode": exit_plan_mode_tool_view,
    "Task": task_tool_view,
}
