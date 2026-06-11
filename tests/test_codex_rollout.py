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
    RawResponseItem,
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
    spec = build_rollout(
        cwd="/work", prior=[UserText(text="hi")], model="gpt-5.5", timestamp=_TS
    )
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
        model="gpt-5.5",
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
        model="gpt-5.5",
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
            # signature-only: no plaintext (text=""), encrypted signature kept —
            # this is how codex persists reasoning whose plaintext was withheld
            Reasoning(text="", encrypted_content="ENC"),
            Reasoning(summary="brief"),
        ],
        model="gpt-5.5",
        timestamp=_TS,
    )
    p = [r["payload"] for r in _rows(spec.content)[2:]]
    # plaintext present -> content carries reasoning_text, summary empty
    assert p[0]["content"] == [{"type": "reasoning_text", "text": "thinking"}]
    assert p[0]["summary"] == []
    assert "encrypted_content" not in p[0]
    # no plaintext -> content null, but the signature is preserved
    assert p[1]["content"] is None
    assert p[1]["encrypted_content"] == "ENC"
    # summary-only -> summary list populated, content None
    assert p[2]["summary"] == [{"type": "summary_text", "text": "brief"}]
    assert p[2]["content"] is None


def test_synthesize_rollout_roundtrips_on_host(tmp_path: Path) -> None:
    path, session_id = synthesize_rollout(
        cwd="/w",
        prior=[UserText(text="hi")],
        codex_home=tmp_path,
        model="gpt-5.5",
        timestamp=_TS,
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
        cwd="/proj",
        prior=prior,
        model="gpt-5.4",
        base_instructions="BI",
        cli_version="9.9.9",
        model_provider="azure",
        timestamp=_TS,
    )
    parsed = parse_rollout(spec.content)
    assert parsed.prior == prior  # lossless for these variants
    assert parsed.session_id == spec.session_id
    assert parsed.cwd == "/proj"
    assert parsed.model == "gpt-5.4"
    assert parsed.base_instructions == "BI"
    assert parsed.cli_version == "9.9.9"
    assert parsed.model_provider == "azure"


def test_parse_rollout_signature_only_reasoning() -> None:
    # codex withholds reasoning plaintext on write (signature in
    # encrypted_content); parse reflects that — text empty, signature kept.
    spec = build_rollout(
        cwd="/w",
        prior=[Reasoning(text="", encrypted_content="ENC")],
        model="gpt-5.5",
        timestamp=_TS,
    )
    [item] = parse_rollout(spec.content).prior
    assert isinstance(item, Reasoning)
    assert item.text == ""
    assert item.encrypted_content == "ENC"


def _hand_rollout(*response_payloads: dict[str, Any]) -> str:
    """A hand-written rollout mimicking real codex output (NOT from build_rollout)."""
    rows: list[dict[str, Any]] = [
        {
            "type": "session_meta",
            "payload": {
                "id": "S1",
                "cwd": "/w",
                "model_provider": "openai",
                "cli_version": "0.130.0",
                "base_instructions": {"text": "BI"},
            },
        },
        {"type": "turn_context", "payload": {"model": "gpt-5.4"}},
        # real rollouts interleave non-response_item rows; parse must skip them
        {"type": "event_msg", "payload": {"type": "token_count", "n": 5}},
    ]
    rows += [{"type": "response_item", "payload": p} for p in response_payloads]
    return "".join(json.dumps(r) + "\n" for r in rows)


def test_parse_ignores_non_response_item_rows_and_reads_meta() -> None:
    content = _hand_rollout(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }
    )
    parsed = parse_rollout(content)
    assert parsed.prior == [UserText(text="hi")]  # event_msg row skipped
    assert parsed.session_id == "S1"
    assert parsed.model == "gpt-5.4"
    assert parsed.base_instructions == "BI"


def test_parse_preserves_unmodelled_row_types_as_raw() -> None:
    # web_search_call appears in real codex sessions; must not crash the parse
    web = {"type": "web_search_call", "id": "ws1", "action": {"query": "x"}}
    parsed = parse_rollout(_hand_rollout(web))
    [item] = parsed.prior
    assert isinstance(item, RawResponseItem)
    assert item.payload == web
    # and it round-trips verbatim through a rebuild
    rebuilt = build_rollout(
        cwd="/w", prior=parsed.prior, model="gpt-5.4", timestamp=_TS
    )
    assert parse_rollout(rebuilt.content).prior == [RawResponseItem(payload=web)]


def test_parse_unknown_message_role_preserved_as_raw() -> None:
    payload = {
        "type": "message",
        "role": "tool",
        "content": [{"type": "input_text", "text": "x"}],
    }
    [item] = parse_rollout(_hand_rollout(payload)).prior
    assert isinstance(item, RawResponseItem)


def test_parse_handles_list_valued_tool_output() -> None:
    # codex writes a list of content blocks for image/structured tool output
    out = [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]
    [item] = parse_rollout(
        _hand_rollout({"type": "function_call_output", "call_id": "c1", "output": out})
    ).prior
    assert isinstance(item, FunctionCallOutput)
    assert item.output == out


def test_parse_dict_valued_tool_output_preserved_as_raw() -> None:
    # codex-rs can serialize a tool output as a {content, success} object; the
    # str|list output field rejects it, so it must degrade to RawResponseItem
    # (verbatim) rather than crash the whole parse.
    payload = {
        "type": "function_call_output",
        "call_id": "c1",
        "output": {"content": "stdout", "success": True},
    }
    [item] = parse_rollout(_hand_rollout(payload)).prior
    assert isinstance(item, RawResponseItem)
    assert item.payload == payload
    # round-trips verbatim
    rebuilt = build_rollout(cwd="/w", prior=[item], model="gpt-5.4", timestamp=_TS)
    assert parse_rollout(rebuilt.content).prior == [item]


def test_parse_modelled_type_with_missing_key_preserved_as_raw() -> None:
    # a function_call row missing 'arguments' (schema drift) must not KeyError-
    # abort the parse; it degrades to a verbatim RawResponseItem.
    payload = {"type": "function_call", "name": "exec", "call_id": "c1"}
    [item] = parse_rollout(_hand_rollout(payload)).prior
    assert isinstance(item, RawResponseItem)
    assert item.payload == payload


def test_parse_handles_unicode_line_separators_in_content() -> None:
    # real codex rollouts embed U+2028 / U+2029 inside JSON string values
    # (written with ensure_ascii=False); str.splitlines() would split there and
    # corrupt the row, so parse must split on "\n" only.
    meta = {
        "type": "session_meta",
        "payload": {"id": "S1", "cwd": "/w", "base_instructions": {"text": "BI"}},
    }
    turn = {"type": "turn_context", "payload": {"model": "gpt-5.4"}}
    msg = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "a b c"}],
        },
    }
    content = "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in (meta, turn, msg)
    )
    parsed = parse_rollout(content)
    assert parsed.prior == [AssistantText(text="a b c")]


def test_parse_concatenates_multiblock_message_content() -> None:
    payload = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "a"},
            {"type": "output_text", "text": "b"},
        ],
    }
    [item] = parse_rollout(_hand_rollout(payload)).prior
    assert isinstance(item, AssistantText)
    assert item.text == "ab"


def test_build_parse_empty_prior() -> None:
    spec = build_rollout(cwd="/w", prior=[], model="gpt-5.4", timestamp=_TS)
    assert parse_rollout(spec.content).prior == []


def test_truncate_rebuild_preserves_raw_items() -> None:
    # raw items survive parse -> slice -> rebuild so a truncated prefix stays faithful
    web = {"type": "web_search_call", "id": "ws1"}
    content = _hand_rollout(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "q"}],
        },
        web,
    )
    parsed = parse_rollout(content)
    assert len(parsed.prior) == 2
    rebuilt = build_rollout(
        cwd=parsed.cwd, prior=parsed.prior, model=parsed.model, timestamp=_TS
    )
    assert parse_rollout(rebuilt.content).prior == parsed.prior


def test_parse_truncate_rebuild_resumes_from_a_node() -> None:
    # "resume from a specific part": parse a saved rollout, slice, rebuild.
    full: list[PriorItem] = [
        UserText(text="q1"),
        AssistantText(text="a1"),
        UserText(text="q2"),
    ]
    saved = build_rollout(cwd="/w", prior=full, model="gpt-5.5", timestamp=_TS).content
    parsed = parse_rollout(saved)
    truncated = build_rollout(
        cwd=parsed.cwd, prior=parsed.prior[:1], model=parsed.model, timestamp=_TS
    )
    assert parse_rollout(truncated.content).prior == [UserText(text="q1")]
