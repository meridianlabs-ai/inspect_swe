"""Tests for the synthetic codex rollout serializer (no live codex needed)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_swe.acp._agents.codex_cli.rollout import (
    AssistantText,
    CustomToolCall,
    CustomToolCallOutput,
    DeveloperText,
    FunctionCall,
    FunctionCallOutput,
    PriorItem,
    Reasoning,
    UserText,
    build_rollout,
    parse_rollout,
    synthesize_rollout,
)

_TS = datetime(2026, 6, 11, 12, 30, 0, tzinfo=UTC)


def _rows(content: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def test_header_rows_and_session_id_consistency() -> None:
    spec = build_rollout(cwd="/work", prior=[UserText(text="hi")], timestamp=_TS)
    rows = _rows(spec.content)

    assert rows[0]["type"] == "session_meta"
    assert rows[0]["payload"]["id"] == spec.session_id  # id matches the spec
    assert rows[0]["payload"]["base_instructions"]["text"]  # default present
    assert rows[1]["type"] == "turn_context"
    assert rows[1]["payload"]["model"] == "gpt-5.5"
    assert rows[2]["type"] == "response_item"
    # the session id is embedded in the on-disk path codex will look up
    assert spec.session_id in spec.relative_path
    assert spec.relative_path.startswith("sessions/2026/06/11/rollout-")


def test_message_items_role_and_content_type() -> None:
    spec = build_rollout(
        cwd="/w",
        prior=[
            UserText(text="u"),
            AssistantText(text="a"),
            DeveloperText(text="d"),
        ],
        timestamp=_TS,
    )
    payloads = [r["payload"] for r in _rows(spec.content)[2:]]
    assert payloads[0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "u"}],
    }
    # assistant turns use output_text; user/developer use input_text
    assert payloads[1]["role"] == "assistant"
    assert payloads[1]["content"][0]["type"] == "output_text"
    assert payloads[2]["role"] == "developer"
    assert payloads[2]["content"][0]["type"] == "input_text"


def test_tool_call_items() -> None:
    spec = build_rollout(
        cwd="/w",
        prior=[
            FunctionCall(name="exec", arguments='{"cmd":"ls"}', call_id="c1"),
            FunctionCallOutput(call_id="c1", output="file.txt"),
            CustomToolCall(name="apply_patch", input="*** Begin Patch", call_id="c2"),
            CustomToolCallOutput(call_id="c2", output="ok"),
        ],
        timestamp=_TS,
    )
    p = [r["payload"] for r in _rows(spec.content)[2:]]
    assert p[0] == {
        "type": "function_call",
        "name": "exec",
        "arguments": '{"cmd":"ls"}',
        "call_id": "c1",
    }
    assert p[1] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "file.txt",
    }
    assert p[2] == {
        "type": "custom_tool_call",
        "status": "completed",
        "call_id": "c2",
        "name": "apply_patch",
        "input": "*** Begin Patch",
    }
    assert p[3] == {"type": "custom_tool_call_output", "call_id": "c2", "output": "ok"}


def test_reasoning_variants() -> None:
    spec = build_rollout(
        cwd="/w",
        prior=[
            Reasoning(text="thinking"),
            Reasoning(text="hidden", redacted=True, encrypted_content="ENC"),
            Reasoning(summary="brief"),
        ],
        timestamp=_TS,
    )
    p = [r["payload"] for r in _rows(spec.content)[2:]]
    # plaintext present -> content carries reasoning_text, summary empty
    assert p[0]["content"] == [{"type": "reasoning_text", "text": "thinking"}]
    assert p[0]["summary"] == []
    assert "encrypted_content" not in p[0]
    # redacted -> no plaintext content, but the signature is preserved
    assert p[1]["content"] is None
    assert p[1]["encrypted_content"] == "ENC"
    # summary-only -> summary list populated, content None
    assert p[2]["summary"] == [{"type": "summary_text", "text": "brief"}]
    assert p[2]["content"] is None


def test_synthesize_rollout_roundtrips_on_host(tmp_path: Path) -> None:
    path, session_id = synthesize_rollout(
        cwd="/w", prior=[UserText(text="hi")], codex_home=tmp_path, timestamp=_TS
    )
    assert path.exists()
    assert path.is_relative_to(tmp_path / "sessions")
    rows = _rows(path.read_text())
    assert rows[0]["payload"]["id"] == session_id


def test_parse_rollout_roundtrips_items_and_metadata() -> None:
    prior: list[PriorItem] = [
        UserText(text="u"),
        AssistantText(text="a"),
        DeveloperText(text="d"),
        FunctionCall(name="exec", arguments='{"cmd":"ls"}', call_id="c1"),
        FunctionCallOutput(call_id="c1", output="out"),
        CustomToolCall(name="apply_patch", input="patch", call_id="c2"),
        CustomToolCallOutput(call_id="c2", output="ok"),
        Reasoning(text="thinking"),
        Reasoning(summary="brief"),
    ]
    spec = build_rollout(
        cwd="/proj", prior=prior, model="gpt-5.4", base_instructions="BI", timestamp=_TS
    )
    parsed = parse_rollout(spec.content)
    assert parsed.prior == prior  # lossless for these variants
    assert parsed.session_id == spec.session_id
    assert parsed.cwd == "/proj"
    assert parsed.model == "gpt-5.4"
    assert parsed.base_instructions == "BI"


def test_parse_rollout_redacted_reasoning_drops_plaintext() -> None:
    # codex withholds redacted reasoning plaintext on write; parse reflects that
    # (text dropped, redacted inferred from no-content + signature present).
    spec = build_rollout(
        cwd="/w",
        prior=[Reasoning(text="secret", redacted=True, encrypted_content="ENC")],
        timestamp=_TS,
    )
    [item] = parse_rollout(spec.content).prior
    assert isinstance(item, Reasoning)
    assert item.text == ""
    assert item.redacted is True
    assert item.encrypted_content == "ENC"


def test_parse_truncate_rebuild_resumes_from_a_node() -> None:
    # "resume from a specific part": parse a saved rollout, slice, rebuild.
    full: list[PriorItem] = [
        UserText(text="q1"),
        AssistantText(text="a1"),
        UserText(text="q2"),
    ]
    saved = build_rollout(cwd="/w", prior=full, timestamp=_TS).content
    parsed = parse_rollout(saved)
    truncated = build_rollout(
        cwd=parsed.cwd, prior=parsed.prior[:1], model=parsed.model, timestamp=_TS
    )
    assert parse_rollout(truncated.content).prior == [UserText(text="q1")]
