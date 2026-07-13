"""Custom `ToolCallContent` views for Codex's built-in tools.

inspect_ai's built-in `tool_call_view` only renders tools registered as
`ToolDef`s; Codex's built-in tools (`exec_command`, `spawn_agent`, …) aren't, so
we supply our own views. Applied in `CodexConsumer.on_complete` before the event
is rendered. `{{arg}}` placeholders are filled by the viewer from the tool call's
arguments (same mechanism as `_claude_code/_events/toolview.py`).
"""

from typing import Any, Callable, Mapping

from inspect_ai.tool import ToolCallContent

from .detection import WAIT_AGENT


def tool_view(
    tool: str,
    arguments: dict[str, Any],
    nicknames: Mapping[str, str] | None = None,
) -> ToolCallContent | None:
    # wait_agent renders agent ids → nicknames, so it needs the consumer's
    # thread_id → nickname map (the others are a pure function of arguments).
    if tool == WAIT_AGENT:
        return _wait_agent_view(arguments, nicknames or {})
    view = _tool_views.get(tool)
    return view(arguments) if view else None


def _exec_command_view(arguments: dict[str, Any]) -> ToolCallContent | None:
    if "cmd" not in arguments:
        return None
    content = "``````bash\n{{cmd}}\n``````\n"
    return ToolCallContent(title="exec_command", format="markdown", content=content)


def _spawn_agent_view(arguments: dict[str, Any]) -> ToolCallContent | None:
    if "message" not in arguments:
        return None
    agent_type = str(arguments.get("agent_type") or "")
    title = f"spawn_agent: {agent_type}" if agent_type else "spawn_agent"
    return ToolCallContent(title=title, format="markdown", content="{{message}}")


def _apply_patch_view(arguments: dict[str, Any]) -> ToolCallContent | None:
    key = "input" if "input" in arguments else "patch" if "patch" in arguments else None
    if key is None:
        return None
    content = "``````diff\n{{" + key + "}}\n``````\n"
    return ToolCallContent(title="apply_patch", format="markdown", content=content)


def _web_search_view(arguments: dict[str, Any]) -> ToolCallContent | None:
    if "query" not in arguments:
        return None
    return ToolCallContent(title="web_search", format="markdown", content="{{query}}")


def _wait_agent_view(
    arguments: dict[str, Any],
    nicknames: Mapping[str, str],
) -> ToolCallContent | None:
    # `targets` is a list of sub-agent thread ids; render each as
    # `nickname — id` (matching the agent-card format), with the id in code
    # font and the nickname omitted when unknown.
    targets = arguments.get("targets")
    if not isinstance(targets, list) or not targets:
        return None
    lines = "\n".join(f"- {_target_label(t, nicknames)}" for t in targets)
    timeout_ms = arguments.get("timeout_ms")
    suffix = (
        f"\n\n_timeout: {int(timeout_ms) // 1000}s_"
        if isinstance(timeout_ms, (int, float))
        else ""
    )
    return ToolCallContent(
        title="wait_agent", format="markdown", content=f"{lines}{suffix}"
    )


def _target_label(target: Any, nicknames: Mapping[str, str]) -> str:
    if not isinstance(target, str):
        return f"`{target}`"
    nickname = nicknames.get(target)
    return f"{nickname} — `{target}`" if nickname else f"`{target}`"


def _send_input_view(arguments: dict[str, Any]) -> ToolCallContent | None:
    # message sent to a running sub-agent; render the free-text payload (the
    # `target` thread-id is left to the default function-call rendering when no
    # message is present).
    if not arguments.get("message"):
        return None
    return ToolCallContent(title="send_input", format="markdown", content="{{message}}")


# Views are supplied only for tools whose payload renders poorly as a
# syntax-highlighted Python function call. `wait_agent` is handled above (it
# needs the nickname map); `close_agent` / `resume_agent` (single id) are left to
# inspect_ai's default rendering, which already reads cleanly.
_tool_views: dict[str, Callable[[dict[str, Any]], ToolCallContent | None]] = {
    "exec_command": _exec_command_view,
    "spawn_agent": _spawn_agent_view,
    "apply_patch": _apply_patch_view,
    "web_search": _web_search_view,
    "send_input": _send_input_view,
}
