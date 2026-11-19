"""Custom `ToolCallContent` views for dsh's built-in tools.

inspect_ai's `tool_call_view` only renders tools registered as `ToolDef`s; dsh
runs its own (`bash`, `write`, `subagent`, ...), so the viewer would show their
arguments as raw JSON. Applied in `DshConsumer.on_complete` before the event is
rendered. `{{arg}}` placeholders are filled by the viewer from the call's
arguments (same mechanism as `_codex_cli/_events/toolview.py`).
"""

from typing import Any, Callable

from inspect_ai.tool import ToolCallContent


def tool_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    view = _tool_views.get(tool)
    return view(tool, arguments) if view else None


def _bash_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    if "command" not in arguments:
        return None
    content = "``````bash\n{{command}}\n``````\n"
    return ToolCallContent(title=tool, format="markdown", content=content)


def _delegation_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    if "prompt" not in arguments:
        return None
    name = str(arguments.get("name") or "")
    title = f"{tool}: {name}" if name else tool
    return ToolCallContent(title=title, format="markdown", content="{{prompt}}")


def _send_message_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    if not arguments.get("message"):
        return None
    agent = str(arguments.get("agent") or "")
    title = f"{tool}: {agent}" if agent else tool
    return ToolCallContent(title=title, format="markdown", content="{{message}}")


def _write_view(tool: str, arguments: dict[str, Any]) -> ToolCallContent | None:
    if "content" not in arguments:
        return None
    path = str(arguments.get("path") or "")
    title = f"{tool}: {path}" if path else tool
    content = "``````\n{{content}}\n``````\n"
    return ToolCallContent(title=title, format="markdown", content=content)


# only tools whose payload reads poorly as a function call
_tool_views: dict[str, Callable[[str, dict[str, Any]], ToolCallContent | None]] = {
    "bash": _bash_view,
    "subagent": _delegation_view,
    "subagent_fork": _delegation_view,
    "send_message": _send_message_view,
    "write": _write_view,
}
