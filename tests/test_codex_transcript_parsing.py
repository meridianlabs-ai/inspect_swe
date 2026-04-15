import asyncio
import json
from pathlib import Path

from inspect_ai.event import ModelEvent, SpanBeginEvent, SpanEndEvent, ToolEvent
from inspect_ai.model._chat_message import (
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_swe._codex_cli._events import (
    codex_cli_events,
    extract_session_metadata,
    is_subagent_session,
    parse_events,
    process_parsed_events,
)


def _collect(async_iter):
    async def collect():
        return [event async for event in async_iter]

    return asyncio.run(collect())


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def test_codex_cli_events_reconstructs_model_turns_and_tool_results() -> None:
    raw_events = [
        {
            "timestamp": "2026-04-13T14:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "session-1", "model_provider": "openai"},
        },
        {
            "timestamp": "2026-04-13T14:00:00.001Z",
            "type": "turn_context",
            "payload": {"model": "gpt-5.4"},
        },
        {
            "timestamp": "2026-04-13T14:00:00.002Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Follow repo rules."}],
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.003Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect the repo"}],
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.004Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 11,
                        "cached_input_tokens": 2,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 1,
                        "total_tokens": 19,
                    }
                },
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.005Z",
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"text": "Search the tree."}],
                "content": None,
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.006Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Checking files."}],
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.007Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_1",
                "arguments": '{"cmd":"pwd"}',
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.008Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_2",
                "arguments": '{"cmd":"ls"}',
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.009Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "/repo\n",
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.010Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "src\n",
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.011Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 20,
                        "cached_input_tokens": 0,
                        "output_tokens": 8,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 28,
                    }
                },
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.012Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        },
    ]

    events = _collect(codex_cli_events(raw_events))

    assert [type(event) for event in events] == [
        ModelEvent,
        SpanBeginEvent,
        ToolEvent,
        SpanEndEvent,
        SpanBeginEvent,
        ToolEvent,
        SpanEndEvent,
        ModelEvent,
    ]

    first_model = events[0]
    assert isinstance(first_model, ModelEvent)
    assert first_model.model == "openai/gpt-5.4"
    assert isinstance(first_model.input[0], ChatMessageSystem)
    assert isinstance(first_model.input[1], ChatMessageUser)
    assert first_model.output.message.content[0].reasoning == "Search the tree."
    assert first_model.output.message.content[1].text == "Checking files."
    assert [tool.function for tool in first_model.output.message.tool_calls or []] == [
        "exec_command",
        "exec_command",
    ]
    assert first_model.output.usage.input_tokens == 11
    assert first_model.output.usage.input_tokens_cache_read == 2
    assert first_model.output.usage.reasoning_tokens == 1

    first_tool = events[2]
    assert isinstance(first_tool, ToolEvent)
    assert first_tool.function == "exec_command"
    assert first_tool.arguments == {"cmd": "pwd"}
    assert first_tool.result == "/repo\n"

    second_tool = events[5]
    assert isinstance(second_tool, ToolEvent)
    assert second_tool.arguments == {"cmd": "ls"}
    assert second_tool.result == "src\n"

    second_model = events[-1]
    assert isinstance(second_model, ModelEvent)
    assert second_model.output.message.content == "Done."
    assert second_model.output.usage.total_tokens == 28
    assert isinstance(second_model.input[-2], ChatMessageTool)
    assert second_model.input[-2].tool_call_id == "call_1"
    assert second_model.input[-1].tool_call_id == "call_2"


def test_process_parsed_events_flushes_dangling_tool_calls() -> None:
    raw_events = [
        {
            "timestamp": "2026-04-13T14:00:00.000Z",
            "type": "turn_context",
            "payload": {"model": "gpt-5.4"},
        },
        {
            "timestamp": "2026-04-13T14:00:00.001Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Run something"}],
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.002Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_pending",
                "arguments": '{"cmd":"echo hi"}',
            },
        },
    ]

    parsed = parse_events(raw_events)
    events = _collect(process_parsed_events(parsed))

    assert isinstance(events[0], ModelEvent)
    assert isinstance(events[1], SpanBeginEvent)
    assert isinstance(events[2], ToolEvent)
    assert isinstance(events[3], SpanEndEvent)
    assert events[2].id == "call_pending"
    assert events[2].result == ""


def test_codex_cli_events_nest_subagent_rollout_under_spawn_agent(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex"
    parent_file = (
        codex_home
        / "sessions"
        / "2026"
        / "04"
        / "13"
        / "rollout-2026-04-13T14-00-00-parent-thread.jsonl"
    )
    child_file = (
        codex_home
        / "sessions"
        / "2026"
        / "04"
        / "13"
        / "rollout-2026-04-13T14-00-01-child-thread.jsonl"
    )

    parent_events = [
        {
            "timestamp": "2026-04-13T14:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "parent-thread", "model_provider": "openai"},
        },
        {
            "timestamp": "2026-04-13T14:00:00.001Z",
            "type": "turn_context",
            "payload": {"model": "gpt-5.4"},
        },
        {
            "timestamp": "2026-04-13T14:00:00.002Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Run a child task"}],
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.003Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "spawn_agent",
                "call_id": "call_spawn_1",
                "arguments": json.dumps(
                    {
                        "agent_type": "worker",
                        "model": "gpt-5.4-mini",
                        "reasoning_effort": "low",
                        "message": "Run exactly one task.",
                    }
                ),
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.004Z",
            "type": "event_msg",
            "payload": {
                "type": "collab_agent_spawn_end",
                "call_id": "call_spawn_1",
                "sender_thread_id": "parent-thread",
                "new_thread_id": "child-thread",
                "new_agent_nickname": "Sagan",
                "new_agent_role": "worker",
                "prompt": "Run exactly one task.",
                "model": "gpt-5.4-mini",
                "reasoning_effort": "low",
                "status": "pending_init",
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.005Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_spawn_1",
                "output": {"agent_id": "child-thread", "nickname": "Sagan"},
            },
        },
        {
            "timestamp": "2026-04-13T14:00:00.006Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final",
                "content": [{"type": "output_text", "text": "Parent finished."}],
            },
        },
    ]
    child_events = [
        {
            "timestamp": "2026-04-13T14:00:01.000Z",
            "type": "session_meta",
            "payload": {
                "id": "child-thread",
                "forked_from_id": "parent-thread",
                "agent_nickname": "Sagan",
                "agent_role": "worker",
                "model_provider": "openai",
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": "parent-thread",
                            "depth": 1,
                            "agent_nickname": "Sagan",
                            "agent_role": "worker",
                        }
                    }
                },
            },
        },
        {
            "timestamp": "2026-04-13T14:00:01.001Z",
            "type": "turn_context",
            "payload": {"model": "gpt-5.4-mini"},
        },
        {
            "timestamp": "2026-04-13T14:00:01.002Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Child task"}],
            },
        },
        {
            "timestamp": "2026-04-13T14:00:01.003Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final",
                "content": [{"type": "output_text", "text": "Child finished."}],
            },
        },
    ]

    _write_jsonl(parent_file, parent_events)
    _write_jsonl(child_file, child_events)

    events = _collect(codex_cli_events(parent_events, session_file=parent_file))

    assert [type(event).__name__ for event in events] == [
        "ModelEvent",
        "SpanBeginEvent",
        "ToolEvent",
        "ModelEvent",
        "SpanEndEvent",
        "ModelEvent",
    ]

    parent_spawn = events[1]
    assert isinstance(parent_spawn, SpanBeginEvent)
    assert parent_spawn.id == "agent-call_spawn_1"
    assert parent_spawn.type == "agent"
    assert parent_spawn.name == "Sagan"
    assert parent_spawn.metadata == {
        "thread_id": "child-thread",
        "prompt": "Run exactly one task.",
        "role": "worker",
        "model": "gpt-5.4-mini",
        "reasoning_effort": "low",
    }

    spawn_tool = events[2]
    assert isinstance(spawn_tool, ToolEvent)
    assert spawn_tool.function == "spawn_agent"
    assert spawn_tool.span_id == "agent-call_spawn_1"
    assert spawn_tool.agent_span_id == "agent-call_spawn_1"

    child_model = events[3]
    assert isinstance(child_model, ModelEvent)
    assert child_model.model == "openai/gpt-5.4-mini"
    assert child_model.span_id == "agent-call_spawn_1"
    assert isinstance(child_model.input[0], ChatMessageUser)
    assert child_model.output.message.content == "Child finished."

    parent_final = events[-1]
    assert isinstance(parent_final, ModelEvent)
    assert parent_final.output.message.content == "Parent finished."
    assert parent_final.input[-1].tool_call_id == "call_spawn_1"


def test_codex_subagent_session_metadata_identifies_parent_thread() -> None:
    parsed = parse_events(
        [
            {
                "timestamp": "2026-04-13T14:00:01.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "child-thread",
                    "forked_from_id": "parent-thread",
                    "agent_nickname": "Sagan",
                    "agent_role": "worker",
                    "model_provider": "openai",
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "parent-thread",
                                "depth": 1,
                                "agent_nickname": "Sagan",
                                "agent_role": "worker",
                            }
                        }
                    },
                },
            }
        ]
    )

    assert extract_session_metadata(parsed) == {
        "session_id": "child-thread",
        "model_provider": "openai",
        "parent_thread_id": "parent-thread",
        "agent_nickname": "Sagan",
        "agent_role": "worker",
    }
    assert is_subagent_session(parsed) is True
