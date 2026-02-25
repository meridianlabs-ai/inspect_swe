"""Unit tests for trajectory parsing functions.

Tests the exit-stripping + _fix_dangling_tool_calls resume logic
for both Anthropic and OpenAI trajectory patterns.
"""

from inspect_swe._mini_swe_agent.resumable_agent import _fix_dangling_tool_calls

VALID_FORMAT = "mini-swe-agent-1.1"


def strip_exit_and_fix(messages: list[dict]) -> list[dict]:
    """Replicate the resume logic: strip exit messages, then fix dangling tool calls."""
    msgs = [dict(m) for m in messages]
    while msgs and msgs[-1].get("role") == "exit":
        msgs.pop()
    _fix_dangling_tool_calls(msgs)
    return msgs


def test_anthropic_dangling_tool_call_preserves_alternation() -> None:
    """When InterruptAgentFlow catches exit during tool execution, the tool
    result ends up in the exit message. After stripping exit, the assistant
    has dangling tool_calls. The fix strips tool_calls but keeps text,
    so appending a new user prompt doesn't create consecutive user messages."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What color is the sky?"},
        {
            "role": "assistant",
            "content": "The sky is blue.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo COMPLETE"}',
                    },
                }
            ],
        },
        {"role": "exit", "content": "blue"},
    ]
    result = strip_exit_and_fix(messages)

    assert result[-1]["role"] == "assistant"
    assert result[-1]["content"] == "The sky is blue."
    assert "tool_calls" not in result[-1]

    # Appending a new user prompt should NOT create consecutive users
    result.append({"role": "user", "content": "Next task"})
    roles = [m["role"] for m in result]
    assert roles == ["system", "user", "assistant", "user"]


def test_openai_dangling_tool_call_preserves_alternation() -> None:
    """Same scenario with OpenAI-style tool call IDs."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": "2",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo COMPLETE"}',
                    },
                }
            ],
        },
        {"role": "exit", "content": "2"},
    ]
    result = strip_exit_and_fix(messages)

    assert result[-1]["role"] == "assistant"
    assert result[-1]["content"] == "2"
    assert "tool_calls" not in result[-1]


def test_dangling_tool_call_no_content_gets_placeholder() -> None:
    """When the assistant had no text content, a placeholder is added."""
    messages = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "tool_calls": [{"id": "tc_1"}]},
        {"role": "exit", "content": "done"},
    ]
    result = strip_exit_and_fix(messages)

    assert result[-1]["role"] == "assistant"
    assert result[-1]["content"] == "Task completed."


def test_answered_tool_calls_not_stripped() -> None:
    """Tool calls with matching results stay the same."""
    messages = [
        {"role": "assistant", "content": "checking", "tool_calls": [{"id": "tc_1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "tc_1"},
        {"role": "assistant", "content": "done"},
        {"role": "exit", "content": "done"},
    ]
    result = strip_exit_and_fix(messages)

    assert len(result) == 3
    assert "tool_calls" in result[0]
