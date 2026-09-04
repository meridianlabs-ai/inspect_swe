"""Tests for the ChatMessage front door on codex rollouts.

Covers ``build_rollout(prior=<messages>)``, ``prior_from_messages`` /
``messages_from_prior`` round-tripping, and the documented lossy edges
(non-JSON custom tool input, unmodelled rows, images).
"""

import ast
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

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
from inspect_swe.acp._agents.codex_cli.rollout import (
    _REASONING_ENCRYPTED_CONTENT,
    AssistantText,
    CustomToolCall,
    DeveloperText,
    FunctionCall,
    FunctionCallOutput,
    RawResponseItem,
    Reasoning,
    UserText,
    build_rollout,
    messages_from_prior,
    parse_rollout,
    prior_from_messages,
)

_TS = datetime(2026, 6, 11, 12, 30, 0, tzinfo=timezone.utc)
_ENCRYPTED_KEY = _REASONING_ENCRYPTED_CONTENT


def test_reasoning_encrypted_key_matches_inspect() -> None:
    spec = find_spec("inspect_ai.model._openai_responses")
    assert spec is not None and spec.origin is not None
    module = ast.parse(Path(spec.origin).read_text())
    values = [
        statement.value
        for statement in module.body
        if isinstance(statement, ast.Assign)
        if any(
            isinstance(target, ast.Name) and target.id == "REASONING_ENCRYPTED_CONTENT"
            for target in statement.targets
        )
    ]

    assert len(values) == 1
    assert _REASONING_ENCRYPTED_CONTENT == ast.literal_eval(values[0])


def test_build_rollout_accepts_messages() -> None:
    spec = build_rollout(
        cwd="/w",
        prior=[
            ChatMessageSystem(content="be terse"),
            ChatMessageUser(content="add a test"),
            ChatMessageAssistant(content="on it"),
        ],
        model="gpt-5.5",
        timestamp=_TS,
    )
    parsed = parse_rollout(spec.content)
    assert parsed.prior == [
        DeveloperText(text="be terse"),
        UserText(text="add a test"),
        AssistantText(text="on it"),
    ]


def test_build_rollout_rejects_mixed_prior() -> None:
    with pytest.raises(ValueError, match="not a mix"):
        build_rollout(
            cwd="/w",
            prior=[ChatMessageUser(content="hi"), UserText(text="hi")],
            model="gpt-5.5",
        )


def test_tool_call_and_output_conversion() -> None:
    prior = prior_from_messages(
        [
            ChatMessageAssistant(
                content="checking",
                tool_calls=[
                    ToolCall(id="call_1", function="bash", arguments={"cmd": "ls"})
                ],
            ),
            ChatMessageTool(content="a.py\n", tool_call_id="call_1", function="bash"),
        ]
    )
    assert prior == [
        AssistantText(text="checking"),
        FunctionCall(name="bash", arguments='{"cmd": "ls"}', call_id="call_1"),
        FunctionCallOutput(call_id="call_1", output="a.py\n"),
    ]


def test_tool_error_text_becomes_the_output() -> None:
    # codex has no error flag on a call output, so the error text has to land in
    # the output or it's lost.
    prior = prior_from_messages(
        [
            ChatMessageTool(
                content="",
                tool_call_id="call_1",
                function="bash",
                error=ToolCallError("unknown", "command not found"),
            )
        ]
    )
    assert prior == [FunctionCallOutput(call_id="call_1", output="command not found")]


def test_non_text_tool_content_raises_instead_of_becoming_empty() -> None:
    with pytest.raises(ValueError, match="non-text tool content"):
        prior_from_messages(
            [
                ChatMessageTool(
                    content=[ContentImage(image="data:image/png;base64,AAA")],
                    tool_call_id="call_1",
                    function="view_image",
                )
            ]
        )


def test_empty_assistant_text_alongside_tool_call_is_dropped() -> None:
    # Inspect emits an empty text block next to tool calls; codex has no row for
    # it, and an empty message row is what makes a resumed session look corrupt.
    prior = prior_from_messages(
        [
            ChatMessageAssistant(
                content=[ContentText(text="")],
                tool_calls=[ToolCall(id="c1", function="bash", arguments={})],
            )
        ]
    )
    assert prior == [FunctionCall(name="bash", arguments="{}", call_id="c1")]


def test_reasoning_plaintext_and_ciphertext_survive() -> None:
    # The reason for not reusing Inspect's own Responses converter: it zeroes
    # reasoning content (the API rejects it on input items), but a rollout on
    # disk is history and should keep the plaintext.
    prior = prior_from_messages(
        [
            ChatMessageAssistant(
                content=[
                    ContentReasoning(
                        reasoning="think think",
                        summary="a summary",
                        internal={_ENCRYPTED_KEY: "cipher"},
                    ),
                    ContentText(text="done"),
                ]
            )
        ]
    )
    assert prior == [
        Reasoning(text="think think", summary="a summary", encrypted_content="cipher"),
        AssistantText(text="done"),
    ]


def test_redacted_reasoning_maps_to_ciphertext_only() -> None:
    prior = prior_from_messages(
        [
            ChatMessageAssistant(
                content=[ContentReasoning(reasoning="cipher", redacted=True)]
            )
        ]
    )
    assert prior == [Reasoning(text="", encrypted_content="cipher")]


def test_reasoning_round_trips_through_messages() -> None:
    original: list[Reasoning] = [
        Reasoning(text="plain", summary="sum", encrypted_content="cipher"),
        Reasoning(text="", encrypted_content="cipher-only"),
        Reasoning(text="no signature"),
    ]
    assert prior_from_messages(messages_from_prior(original)) == original


def test_image_content_becomes_a_raw_row() -> None:
    prior = prior_from_messages(
        [
            ChatMessageUser(
                content=[
                    ContentText(text="look at this"),
                    ContentImage(image="data:image/png;base64,AAA"),
                ]
            )
        ]
    )
    assert prior[0] == UserText(text="look at this")
    raw = prior[1]
    assert isinstance(raw, RawResponseItem)
    assert raw.payload["content"][0]["type"] == "input_image"


def test_unsupported_content_raises() -> None:
    from inspect_ai.model import ContentAudio

    with pytest.raises(ValueError, match="no codex rollout equivalent"):
        prior_from_messages(
            [ChatMessageUser(content=[ContentAudio(audio="a.mp3", format="mp3")])]
        )


def test_messages_from_prior_groups_assistant_items() -> None:
    messages = messages_from_prior(
        [
            UserText(text="go"),
            Reasoning(text="hmm"),
            AssistantText(text="running it"),
            FunctionCall(name="bash", arguments='{"cmd": "ls"}', call_id="c1"),
            FunctionCallOutput(call_id="c1", output="a.py"),
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
    assert assistant.tool_calls[0].arguments == {"cmd": "ls"}
    tool_message = messages[2]
    assert isinstance(tool_message, ChatMessageTool)
    assert tool_message.function == "bash"  # named from the matching call


def test_custom_tool_call_input_is_wrapped() -> None:
    # apply_patch takes free-form text, not JSON args, so it can only be
    # expressed as a ToolCall by wrapping it — the fidelity gap that keeps the
    # typed items around.
    messages = messages_from_prior(
        [CustomToolCall(name="apply_patch", input="*** Begin Patch", call_id="c1")]
    )
    assistant = messages[0]
    assert isinstance(assistant, ChatMessageAssistant)
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].arguments == {"input": "*** Begin Patch"}


def test_non_json_arguments_become_a_parse_error() -> None:
    messages = messages_from_prior(
        [FunctionCall(name="bash", arguments="not json", call_id="c1")]
    )
    assistant = messages[0]
    assert isinstance(assistant, ChatMessageAssistant)
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].arguments == {}
    assert assistant.tool_calls[0].parse_error is not None


def test_raw_rows_are_dropped_by_messages_from_prior() -> None:
    messages = messages_from_prior(
        [
            UserText(text="go"),
            RawResponseItem(payload={"type": "web_search_call", "id": "ws_1"}),
            AssistantText(text="done"),
        ]
    )
    assert [m.role for m in messages] == ["user", "assistant"]


def test_parsed_rollout_as_messages() -> None:
    spec = build_rollout(
        cwd="/w",
        prior=[UserText(text="go"), AssistantText(text="done")],
        model="gpt-5.5",
        timestamp=_TS,
    )
    messages = parse_rollout(spec.content).as_messages()
    assert [(m.role, m.text) for m in messages] == [
        ("user", "go"),
        ("assistant", "done"),
    ]
