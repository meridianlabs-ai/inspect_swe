"""Request shapes of DeepSeek Harness (dsh), as the agent bridge hands them over.

Copied from a real run (dsh 0.2.2rc2, `--profile headless`, 2026-11-19,
openrouter/deepseek/deepseek-v4.1-flash-1017; the full recording is
`tests/data/deepseek_harness/delegation-0.2.2rc2.json`). dsh sends each turn's
blocks as separate text blocks of one user message, and the bridge keeps them
separate.
"""

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError

SYSTEM_PROMPT = (
    "You are an AI agent powered by DeepSeek Harness.\n\n"
    "You are a coding agent powered by the deepseek/deepseek-v4.1-flash-1017 model."
)

RETURN_GUIDANCE = (
    "Your final answer is delivered automatically if your parent is still "
    "online. Include the result and relevant evidence in that answer; the "
    "parent does not receive your full transcript, tool output, or reasoning. "
    "If your last action is a message to your parent, the completion notice "
    "omits your final answer; include the result and relevant evidence in "
    "that message."
)

_SESSION = (
    'Current working directory: "/root".\n\n'
    "Current DSH file policy: danger-full-access. The DSH file sandbox does "
    "not restrict file modifications by available operations.\n\n"
    "Approval prompts are disabled in this session: actions that require "
    "approval are rejected automatically — do not request sandbox escalation "
    "(do not set `sandbox_permissions`).\n\n"
)
_MESSAGING = (
    "\n\nTo message other agents, use send_message({ agent: "
    '"<agent-name>", message: "..." }).\n'
    "Use list_agents({}) to find your collaborator agents and their names."
)
_FIRST = "The following information describes your current session:"
_LATER = (
    "The following information describes your current session. It replaces "
    "earlier versions of this context, including any retained in "
    "conversation summaries."
)

# opening paragraph and the line the classifier also requires; the real
# instruction goes on with the checkpoint's section list
COMPACTION_INSTRUCTION = (
    "You are now acting as a compaction engine for this AI coding assistant. "
    "Condense the conversation ABOVE into a structured checkpoint that lets "
    "another model resume the work with no loss of essential context.\n\n"
    "Output EXACTLY the Markdown structure below: keep every section, in "
    'order. Use terse bullets, not prose paragraphs. Write "(none)" for an '
    "empty section — never drop a section.\n\n"
    "## Primary Request and Intent\n- [the user's original and evolving goals]"
)


CHECKPOINT_PREAMBLE = (
    "This is an automatically generated checkpoint condensing an earlier span "
    "of the conversation to free up context. Treat the captured context as "
    "established background and build on it without restating it. Continue "
    "the task directly from the messages that follow, without acknowledging "
    "this checkpoint.\n\n<compacted-summary>"
)


def checkpoint(summary: str) -> ChatMessageUser:
    """What replaces the condensed span of a compacted conversation."""
    return ChatMessageUser(
        content=blocks(CHECKPOINT_PREAMBLE, summary, "</compacted-summary>")
    )


def root_reminder(*, replaces: bool = False) -> str:
    """The root agent's session reminder."""
    return (
        f"<system-reminder>\n{_LATER if replaces else _FIRST}\n\n{_SESSION}"
        f'You are the root agent. Your name is "lead".{_MESSAGING}\n'
        "</system-reminder>"
    )


def child_reminder(name: str, parent: str = "lead", *, replaces: bool = False) -> str:
    """A delegated agent's session reminder."""
    return (
        f"<system-reminder>\n{_LATER if replaces else _FIRST}\n\n{_SESSION}"
        f'You are a subagent delegated by agent "{parent}".\n'
        f'Your name is "{name}".\n'
        f'The root agent of your collaboration tree is "lead".{_MESSAGING}\n\n'
        "Delegation depth for `subagent`: current 1, maximum 1, remaining "
        "levels 0. Do not call `subagent` at this depth; carry out the work "
        "directly.\n\n"
        "You are a delegated subagent: your permission scope was fixed when "
        "you were started and cannot be widened from inside this session — "
        "operations that require approval are rejected automatically. When "
        "the task needs access beyond that scope, do not retry the denied "
        "operation; state the limitation in your reply so the delegating "
        "agent can handle it.\n"
        "</system-reminder>"
    )


def system() -> ChatMessageSystem:
    return ChatMessageSystem(content=SYSTEM_PROMPT)


def blocks(*texts: str) -> list[Content]:
    return [ContentText(text=text) for text in texts]


def root_task(prompt: str) -> ChatMessageUser:
    """The root agent's opening turn: the task, then its session reminder."""
    return ChatMessageUser(content=blocks(prompt, root_reminder()))


def child_task(prompt: str, name: str, parent: str = "lead") -> ChatMessageUser:
    """A fresh child's opening turn: prompt, return guidance, session reminder."""
    return ChatMessageUser(
        content=blocks(prompt, RETURN_GUIDANCE, child_reminder(name, parent))
    )


def fork_task(prompt: str, name: str, parent: str = "lead") -> ChatMessageUser:
    """A forked child's own turn, which follows its parent's completed turns."""
    return ChatMessageUser(
        content=blocks(
            prompt, RETURN_GUIDANCE, child_reminder(name, parent, replaces=True)
        )
    )


def delegation_call(
    call_id: str,
    prompt: str,
    name: str | None = None,
    tool: str = "subagent",
    description: str = "Run a command",
) -> ToolCall:
    arguments = {"description": description, "prompt": prompt}
    if name is not None:
        arguments["name"] = name
    return ToolCall(id=call_id, function=tool, arguments=arguments)


def started(call_id: str, name: str, tool: str = "subagent") -> ChatMessageTool:
    """The result of an accepted start: a launch acknowledgement only."""
    return ChatMessageTool(
        content=f"started subagent {name}", tool_call_id=call_id, function=tool
    )


def start_rejected(
    call_id: str, message: str, tool: str = "subagent"
) -> ChatMessageTool:
    return ChatMessageTool(
        content=message,
        tool_call_id=call_id,
        function=tool,
        error=ToolCallError(type="unknown", message=message),
    )


def notice(name: str, *closing: str) -> ChatMessageUser:
    """The settlement notice a child's parent receives when the child's run ends."""
    return ChatMessageUser(
        content=blocks(
            f"Background subagent {name} finished and will do no further work "
            "unless you send it more.",
            *(
                ["Its closing message:", *closing]
                if closing
                else ["It left no closing message."]
            ),
        )
    )


def agent_message(sender: str, text: str) -> ChatMessageUser:
    """A `send_message` from another agent, as its recipient receives it."""
    return ChatMessageUser(content=blocks(f"Agent {sender} sent a message: ", text))


def assistant(
    text: str = "", tool_calls: list[ToolCall] | None = None
) -> ChatMessageAssistant:
    return ChatMessageAssistant(content=text, tool_calls=tool_calls)


def bash_call(call_id: str, command: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="bash",
        arguments={"description": "Run it", "command": command},
    )


def tool_result(call_id: str, text: str, function: str = "bash") -> ChatMessageTool:
    return ChatMessageTool(content=text, tool_call_id=call_id, function=function)


def conversation(*messages: ChatMessage) -> list[ChatMessage]:
    """A request's input: dsh's system prompt, then the agent's history."""
    return [system(), *messages]
