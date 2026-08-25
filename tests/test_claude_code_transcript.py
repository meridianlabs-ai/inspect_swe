"""Tests for synthetic Claude Code transcripts (build / parse / ChatMessage).

The on-disk layout is reverse-engineered from real Claude Code transcripts: one
row per content block, chained by ``parentUuid``, ``message`` holding a raw
Anthropic message. These tests pin that layout plus the round-trips.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError
from inspect_swe.acp._agents.claude_code.transcript import (
    AssistantText,
    ParsedTranscript,
    RawBlock,
    Thinking,
    ToolResult,
    ToolUse,
    TranscriptItem,
    UserText,
    build_transcript,
    items_from_messages,
    messages_from_items,
    parse_transcript,
    project_slug,
)

_TS = datetime(2026, 6, 11, 12, 30, 0, tzinfo=timezone.utc)


def _rows(content: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in content.split("\n") if line.strip()]


def test_project_slug_matches_sdk_scheme() -> None:
    # SDK: cwd.replace(/[^a-zA-Z0-9]/g, "-")
    assert project_slug("/home/user/my.repo") == "-home-user-my-repo"


def test_project_slug_rejects_paths_needing_the_sdk_hash() -> None:
    # Past 200 chars the SDK appends a hash of the path we can't reproduce, so a
    # transcript written under a truncated slug would never be found.
    with pytest.raises(ValueError, match="too long"):
        project_slug("/" + "a" * 250)


def test_relative_path_and_session_id() -> None:
    spec = build_transcript(
        cwd="/home/user",
        items=[UserText(text="go")],
        model="claude-opus-5",
        session_id="11111111-2222-3333-4444-555555555555",
        timestamp=_TS,
    )
    assert spec.relative_path == (
        "projects/-home-user/11111111-2222-3333-4444-555555555555.jsonl"
    )
    assert spec.cwd == "/home/user"


def test_rows_are_one_per_block_and_chained() -> None:
    spec = build_transcript(
        cwd="/w",
        items=[
            UserText(text="go"),
            Thinking(thinking="hmm", signature="sig"),
            AssistantText(text="running it"),
            ToolUse(id="toolu_1", name="Bash", input={"command": "ls"}),
            ToolResult(tool_use_id="toolu_1", content="a.py"),
        ],
        model="claude-opus-5",
        timestamp=_TS,
    )
    rows = _rows(spec.content)
    assert [r["type"] for r in rows] == [
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
    ]
    # parentUuid chain: first row is a root, each later row points at the prior
    assert rows[0]["parentUuid"] is None
    assert [r["parentUuid"] for r in rows[1:]] == [r["uuid"] for r in rows[:-1]]
    assert [r["uuid"] for r in rows] == spec.item_uuids
    # blocks of one assistant turn share a message id; the user rows end it
    assistant_ids = {r["message"]["id"] for r in rows if r["type"] == "assistant"}
    assert len(assistant_ids) == 1
    # a typed user turn is written with plain-string content, as CC does
    assert rows[0]["message"] == {"role": "user", "content": "go"}
    assert rows[1]["message"]["content"] == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"}
    ]
    assert rows[3]["message"]["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "Bash",
            "input": {"command": "ls"},
        }
    ]
    assert rows[4]["message"]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "a.py",
            "is_error": False,
        }
    ]


def test_every_row_carries_session_metadata() -> None:
    spec = build_transcript(
        cwd="/w",
        items=[UserText(text="go"), AssistantText(text="ok")],
        model="claude-opus-5",
        git_branch="main",
        timestamp=_TS,
    )
    for row in _rows(spec.content):
        assert row["cwd"] == "/w"
        assert row["sessionId"] == spec.session_id
        assert row["gitBranch"] == "main"
        assert row["isSidechain"] is False
        assert row["userType"] == "external"
        assert row["timestamp"] == "2026-06-11T12:30:00.000Z"


def test_thinking_without_signature_omits_the_key() -> None:
    # A fabricated signature is worse than none: Anthropic validates it.
    spec = build_transcript(
        cwd="/w",
        items=[Thinking(thinking="hmm")],
        model="claude-opus-5",
        timestamp=_TS,
    )
    block = _rows(spec.content)[0]["message"]["content"][0]
    assert "signature" not in block


def test_items_round_trip_through_build_and_parse() -> None:
    items: list[TranscriptItem] = [
        UserText(text="go"),
        Thinking(thinking="hmm", signature="sig"),
        AssistantText(text="running it"),
        ToolUse(id="toolu_1", name="Bash", input={"command": "ls"}),
        ToolResult(tool_use_id="toolu_1", content="a.py", is_error=False),
        ToolResult(tool_use_id="toolu_2", content="boom", is_error=True),
        RawBlock(role="assistant", block={"type": "server_tool_use", "id": "srv_1"}),
    ]
    spec = build_transcript(cwd="/w", items=items, model="claude-opus-5", timestamp=_TS)
    parsed = parse_transcript(spec.content)
    assert parsed.items == items
    assert parsed.item_uuids == spec.item_uuids
    assert parsed.session_id == spec.session_id
    assert parsed.cwd == "/w"
    assert parsed.skipped_rows == 0


def test_truncate_and_rebuild() -> None:
    items: list[TranscriptItem] = [
        UserText(text="go"),
        AssistantText(text="step one"),
        UserText(text="keep going"),
        AssistantText(text="step two"),
    ]
    original = build_transcript(
        cwd="/w", items=items, model="claude-opus-5", timestamp=_TS
    )
    parsed = parse_transcript(original.content)
    rebuilt = build_transcript(
        cwd=parsed.cwd, items=parsed.items[:2], model="claude-opus-5", timestamp=_TS
    )
    assert parse_transcript(rebuilt.content).items == items[:2]
    # a rebuilt transcript is a new session, not an edit of the original
    assert rebuilt.session_id != original.session_id


def test_parse_skips_claude_code_state_rows() -> None:
    rows = [
        {"type": "mode", "mode": "normal", "sessionId": "s1"},
        {
            "type": "user",
            "message": {"role": "user", "content": "go"},
            "uuid": "u1",
            "sessionId": "s1",
            "cwd": "/w",
            "version": "2.1.220",
        },
        {"type": "file-history-snapshot", "messageId": "u1"},
        {"type": "summary", "summary": "a title"},
    ]
    parsed = parse_transcript("".join(json.dumps(r) + "\n" for r in rows))
    assert parsed.items == [UserText(text="go")]
    assert parsed.item_uuids == ["u1"]
    assert parsed.skipped_rows == 3
    assert parsed.session_id == "s1"


def test_parse_multi_block_row_maps_blocks_to_one_uuid() -> None:
    # Real transcripts are one block per row, but the API allows several; every
    # block of a row resumes at the same point, so they share its uuid.
    row = {
        "type": "assistant",
        "uuid": "u1",
        "sessionId": "s1",
        "cwd": "/w",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
            ],
        },
    }
    parsed = parse_transcript(json.dumps(row) + "\n")
    assert parsed.items == [
        AssistantText(text="one"),
        ToolUse(id="t1", name="Bash", input={}),
    ]
    assert parsed.item_uuids == ["u1", "u1"]


def test_parse_preserves_unmodelled_blocks_verbatim() -> None:
    row = {
        "type": "assistant",
        "uuid": "u1",
        "cwd": "/w",
        "message": {
            "role": "assistant",
            "content": [{"type": "web_search_tool_result", "content": [{"x": 1}]}],
        },
    }
    parsed = parse_transcript(json.dumps(row) + "\n")
    assert parsed.items == [
        RawBlock(
            role="assistant",
            block={"type": "web_search_tool_result", "content": [{"x": 1}]},
        )
    ]


def test_write_to_host_fs(tmp_path: Path) -> None:
    spec = build_transcript(
        cwd="/w", items=[UserText(text="go")], model="claude-opus-5", timestamp=_TS
    )
    path = spec.write_to(tmp_path)
    assert path == tmp_path / spec.relative_path
    assert parse_transcript(path.read_text()).items == [UserText(text="go")]


# ---------------------------------------------------------------------------
# ChatMessage conversion
# ---------------------------------------------------------------------------


def test_build_transcript_accepts_messages() -> None:
    spec = build_transcript(
        cwd="/w",
        items=[
            ChatMessageUser(content="add a test"),
            ChatMessageAssistant(
                content="on it",
                tool_calls=[
                    ToolCall(id="toolu_1", function="Bash", arguments={"command": "ls"})
                ],
            ),
            ChatMessageTool(content="a.py", tool_call_id="toolu_1", function="Bash"),
        ],
        model="claude-opus-5",
        timestamp=_TS,
    )
    assert parse_transcript(spec.content).items == [
        UserText(text="add a test"),
        AssistantText(text="on it"),
        ToolUse(id="toolu_1", name="Bash", input={"command": "ls"}),
        ToolResult(tool_use_id="toolu_1", content="a.py"),
    ]


def test_build_transcript_rejects_mixed_items() -> None:
    with pytest.raises(ValueError, match="not a mix"):
        build_transcript(
            cwd="/w",
            items=[ChatMessageUser(content="hi"), UserText(text="hi")],
            model="claude-opus-5",
        )


def test_thinking_is_dropped_from_messages_by_default() -> None:
    # An unsigned (or foreign-provider) thinking block fails the next turn once
    # Anthropic sees it replayed, so the safe default is to leave it out.
    messages = [
        ChatMessageAssistant(
            content=[
                ContentReasoning(reasoning="hmm", signature="sig"),
                ContentText(text="done"),
            ]
        )
    ]
    assert items_from_messages(messages) == [AssistantText(text="done")]
    assert items_from_messages(messages, include_thinking=True) == [
        Thinking(thinking="hmm", signature="sig"),
        AssistantText(text="done"),
    ]


def test_system_messages_raise_instead_of_disappearing() -> None:
    with pytest.raises(ValueError, match="system messages"):
        items_from_messages([ChatMessageSystem(content="be terse")])


def test_tool_error_maps_to_is_error() -> None:
    items = items_from_messages(
        [
            ChatMessageTool(
                content="",
                tool_call_id="toolu_1",
                function="Bash",
                error=ToolCallError("unknown", "command not found"),
            )
        ]
    )
    assert items == [
        ToolResult(tool_use_id="toolu_1", content="command not found", is_error=True)
    ]


def test_non_text_tool_content_raises_instead_of_becoming_empty() -> None:
    with pytest.raises(ValueError, match="non-text tool content"):
        items_from_messages(
            [
                ChatMessageTool(
                    content=[ContentImage(image="data:image/png;base64,AAA")],
                    tool_call_id="toolu_1",
                    function="ViewImage",
                )
            ]
        )


@pytest.mark.parametrize(
    "message",
    [
        ChatMessageUser(content=[ContentImage(image="data:image/png;base64,AAA")]),
        ChatMessageAssistant(content=[ContentImage(image="data:image/png;base64,AAA")]),
    ],
)
def test_inspect_image_content_raises_instead_of_emitting_invalid_block(
    message: ChatMessageUser | ChatMessageAssistant,
) -> None:
    with pytest.raises(ValueError, match="not a valid Anthropic transcript block"):
        items_from_messages([message])


def test_messages_from_items_groups_assistant_items() -> None:
    messages = messages_from_items(
        [
            UserText(text="go"),
            Thinking(thinking="hmm", signature="sig"),
            AssistantText(text="running it"),
            ToolUse(id="toolu_1", name="Bash", input={"command": "ls"}),
            ToolResult(tool_use_id="toolu_1", content="a.py"),
        ]
    )
    assert [m.role for m in messages] == ["user", "assistant", "tool"]
    assistant = messages[1]
    assert isinstance(assistant, ChatMessageAssistant)
    assert [type(c).__name__ for c in assistant.content] == [
        "ContentReasoning",
        "ContentText",
    ]
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].arguments == {"command": "ls"}
    tool_message = messages[2]
    assert isinstance(tool_message, ChatMessageTool)
    assert tool_message.function == "Bash"


def test_messages_from_items_marks_tool_errors() -> None:
    messages = messages_from_items(
        [ToolResult(tool_use_id="toolu_1", content="boom", is_error=True)]
    )
    tool_message = messages[0]
    assert isinstance(tool_message, ChatMessageTool)
    assert tool_message.error is not None
    assert tool_message.error.message == "boom"


def test_parsed_transcript_as_messages() -> None:
    parsed = ParsedTranscript(
        session_id="s1",
        cwd="/w",
        version="2.1.220",
        items=[UserText(text="go"), AssistantText(text="done")],
        item_uuids=["u1", "u2"],
        skipped_rows=0,
    )
    assert [(m.role, m.text) for m in parsed.as_messages()] == [
        ("user", "go"),
        ("assistant", "done"),
    ]
