"""Unit tests for the pure helpers in claude_code.py.

Covers the module-level functions the ACP live-mode refactor relies on:
`_build_agent_cmd` (argv assembly + the resume/system-prompt rule),
`_user_text` (channel-drain → prompt text), `_is_turn_boundary` (top-level
assistant-turn detection), and the operator-delivery gate
(`_operator_delivery_gate` + `_top_level_tool_use_ids` / `_tool_result_ids`,
which decide when a queued operator message may be delivered). The live
orchestration itself (`consume` / `run_prompt`) is integration-level and
covered by the manual ACP smoke test.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCallError
from inspect_swe._claude_code._events.stream import (
    ExitEvent,
    JsonlEvent,
    JsonlParseError,
    StderrEvent,
)
from inspect_swe._claude_code.claude_code import (
    _build_agent_cmd,
    _is_turn_boundary,
    _operator_delivery_gate,
    _tool_result_ids,
    _top_level_tool_use_ids,
    _user_text,
)

# --- _build_agent_cmd ---------------------------------------------------


def _cmd(
    resume: bool,
    messages: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
) -> list[str]:
    return _build_agent_cmd(
        claude_binary="claude",
        session_id="sess-1",
        cmd=["--model", "m", "--print"],
        messages=messages or [],
        system_prompt=system_prompt,
        resume=resume,
        prompt="do the thing",
    )


def test_build_agent_cmd_fresh_session_uses_session_id() -> None:
    argv = _cmd(resume=False)
    assert argv[0] == "claude"
    assert argv[1:3] == ["--session-id", "sess-1"]
    assert "--resume" not in argv
    # prompt is the final positional after "--"
    assert argv[-2:] == ["--", "do the thing"]
    # base cmd args are carried through
    assert "--model" in argv and "--print" in argv


def test_build_agent_cmd_resume_uses_resume_flag() -> None:
    argv = _cmd(resume=True)
    assert argv[1:3] == ["--resume", "sess-1"]
    assert "--session-id" not in argv


def test_build_agent_cmd_fresh_session_appends_system_prompt() -> None:
    argv = _cmd(
        resume=False,
        messages=[ChatMessageSystem(content="sys-from-messages")],
        system_prompt="extra-sys",
    )
    assert "--append-system-prompt" in argv
    idx = argv.index("--append-system-prompt")
    assert argv[idx + 1] == "sys-from-messages\n\nextra-sys"


def test_build_agent_cmd_resume_omits_system_prompt() -> None:
    # Even with system content present, resume must NOT re-send it (#64).
    argv = _cmd(
        resume=True,
        messages=[ChatMessageSystem(content="sys-from-messages")],
        system_prompt="extra-sys",
    )
    assert "--append-system-prompt" not in argv


def test_build_agent_cmd_fresh_session_no_system_when_none() -> None:
    argv = _cmd(resume=False, messages=[], system_prompt=None)
    assert "--append-system-prompt" not in argv


# --- _user_text ---------------------------------------------------------


def test_user_text_none_on_empty() -> None:
    assert _user_text([]) is None


def test_user_text_none_when_only_repair_tool_messages() -> None:
    repair = ChatMessageTool(
        tool_call_id="t1",
        content="cancelled",
        error=ToolCallError(type="cancelled", message="cancelled"),
    )
    assert _user_text([repair]) is None


def test_user_text_single_user_message() -> None:
    assert _user_text([ChatMessageUser(content="hello")]) == "hello"


def test_user_text_joins_multiple_user_messages() -> None:
    msgs = [ChatMessageUser(content="a"), ChatMessageUser(content="b")]
    assert _user_text(msgs) == "a\n\nb"


def test_user_text_filters_repair_keeps_user() -> None:
    # after_cancel returns [*repair_tool_msgs, *coalesced_user]; we keep only user.
    repair = ChatMessageTool(
        tool_call_id="t1",
        content="cancelled",
        error=ToolCallError(type="cancelled", message="cancelled"),
    )
    msgs: list[ChatMessage] = [repair, ChatMessageUser(content="redirect")]
    assert _user_text(msgs) == "redirect"


# --- _is_turn_boundary --------------------------------------------------


def _jsonl(raw: dict[str, Any]) -> JsonlEvent:
    return JsonlEvent(raw=raw, line="{}")


def test_is_turn_boundary_top_level_assistant_no_key() -> None:
    assert _is_turn_boundary(_jsonl({"type": "assistant"})) is True


def test_is_turn_boundary_top_level_assistant_null_parent() -> None:
    assert (
        _is_turn_boundary(_jsonl({"type": "assistant", "parent_tool_use_id": None}))
        is True
    )


def test_is_turn_boundary_false_for_subagent_assistant() -> None:
    # sub-agent assistant events carry the parent Task's tool_use id
    assert (
        _is_turn_boundary(
            _jsonl({"type": "assistant", "parent_tool_use_id": "toolu_123"})
        )
        is False
    )


def test_is_turn_boundary_false_for_user_tool_result() -> None:
    assert _is_turn_boundary(_jsonl({"type": "user"})) is False


def test_is_turn_boundary_false_for_system_and_result() -> None:
    assert _is_turn_boundary(_jsonl({"type": "system"})) is False
    assert _is_turn_boundary(_jsonl({"type": "result"})) is False


def test_is_turn_boundary_false_for_non_jsonl_events() -> None:
    assert _is_turn_boundary(ExitEvent(code=0)) is False
    assert _is_turn_boundary(StderrEvent(data="x")) is False
    assert _is_turn_boundary(JsonlParseError(line="x")) is False


# --- tool-id helpers + _operator_delivery_gate ----------------------------


def _assistant(*tool_ids: str, parent: str | None = None) -> JsonlEvent:
    raw: dict[str, Any] = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "thinking"},
                *(
                    {"type": "tool_use", "id": tid, "name": "Bash", "input": {}}
                    for tid in tool_ids
                ),
            ]
        },
    }
    if parent is not None:
        raw["parent_tool_use_id"] = parent
    return _jsonl(raw)


def _tool_results(*tool_use_ids: str, parent: str | None = None) -> JsonlEvent:
    raw: dict[str, Any] = {
        "type": "user",
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": tid} for tid in tool_use_ids
            ]
        },
    }
    if parent is not None:
        raw["parent_tool_use_id"] = parent
    return _jsonl(raw)


def test_top_level_tool_use_ids_collects_parallel_calls() -> None:
    assert _top_level_tool_use_ids(_assistant("t1", "t2")) == {"t1", "t2"}


def test_top_level_tool_use_ids_empty_for_text_only_turn() -> None:
    assert _top_level_tool_use_ids(_assistant()) == set()


def test_top_level_tool_use_ids_excludes_subagent() -> None:
    assert _top_level_tool_use_ids(_assistant("s1", parent="toolu_task")) == set()


def test_top_level_tool_use_ids_empty_for_non_assistant() -> None:
    assert _top_level_tool_use_ids(_tool_results("t1")) == set()
    assert _top_level_tool_use_ids(ExitEvent(code=0)) == set()


def test_tool_result_ids_collects_top_level_results() -> None:
    assert _tool_result_ids(_tool_results("t1", "t2")) == {"t1", "t2"}


def test_tool_result_ids_excludes_subagent_results() -> None:
    assert _tool_result_ids(_tool_results("s1", parent="toolu_task")) == set()


def test_tool_result_ids_empty_for_non_user() -> None:
    assert _tool_result_ids(_assistant("t1")) == set()


def test_gate_assistant_event_is_a_boundary_seam() -> None:
    # A top-level assistant event is itself a turn boundary (seam #2), so it is
    # a valid delivery point regardless of whether it carries tool calls.
    is_delivery_seam = _operator_delivery_gate()
    assert is_delivery_seam(_assistant("t1", "t2")) is True


def test_gate_partial_tool_completion_is_not_a_seam_then_full_is() -> None:
    is_delivery_seam = _operator_delivery_gate()
    is_delivery_seam(_assistant("t1", "t2"))  # boundary; registers t1, t2
    # one result back, one still outstanding -> NOT a (tool-completion) seam
    assert is_delivery_seam(_tool_results("t1")) is False
    # last result back -> all the turn's tools resolved -> early seam, BEFORE
    # the next generation starts
    assert is_delivery_seam(_tool_results("t2")) is True


def test_gate_parallel_results_complete_in_one_event() -> None:
    is_delivery_seam = _operator_delivery_gate()
    is_delivery_seam(_assistant("t1", "t2"))  # boundary; registers t1, t2
    assert is_delivery_seam(_tool_results("t1", "t2")) is True


def test_gate_text_only_turn_is_a_boundary_seam() -> None:
    # No tool calls: the outstanding set never populates; delivery rides on the
    # top-level assistant boundary (the backstop).
    is_delivery_seam = _operator_delivery_gate()
    assert is_delivery_seam(_assistant()) is True


def test_gate_waits_for_subagent_task_to_finish() -> None:
    is_delivery_seam = _operator_delivery_gate()
    # the top-level assistant spawning the Task is itself a boundary seam, and
    # registers the Task's tool_use id as outstanding
    assert is_delivery_seam(_assistant("toolu_task")) is True
    # the sub-agent's own events are NOT seams and do NOT clear the Task
    assert is_delivery_seam(_assistant("s1", parent="toolu_task")) is False
    assert is_delivery_seam(_tool_results("s1", parent="toolu_task")) is False
    # the Task's own top-level result lands only once the sub-agent is done ->
    # tool-completion seam
    assert is_delivery_seam(_tool_results("toolu_task")) is True


def test_gate_stray_tool_result_with_no_outstanding_is_not_seam() -> None:
    is_delivery_seam = _operator_delivery_gate()
    assert is_delivery_seam(_tool_results("t1")) is False
