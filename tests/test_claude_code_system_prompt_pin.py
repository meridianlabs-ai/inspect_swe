"""Tests for pinning Claude Code's system prompt across --resume launches."""

import logging
import uuid
from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentText,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe._claude_code.systemprompt import pin_system_prompt_filter

pytestmark = pytest.mark.anyio

# A vanilla Claude Code 2.1.257 sends `system` as three blocks (captured
# against a mock endpoint): a billing header, an identity line, and the
# instructions block ending with the gitStatus section. inspect_ai >= 0.3.262
# hoists each block into its own ChatMessageSystem; older versions flatten
# them into one. On --resume the identity line flips, --append-system-prompt
# text disappears, and the git status is regenerated; the first user message
# (system reminders + the task prompt) is replayed byte-identical.
BILLING = "x-anthropic-billing-header: cc_version=2.1.257.1f2; cc_entry=cli"
IDENTITY_FIRST = (
    "You are Claude Code, Anthropic's official CLI for Claude, running within "
    "the Claude Agent SDK."
)
IDENTITY_RESUMED = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
INSTRUCTIONS = "\nYou are an interactive agent that helps users...\n\nMore.\n\n"
APPENDED = "You are an ace researcher.\n\n"
GIT_STATUS_FIRST = (
    "gitStatus: ...\n\nStatus:\n?? a.txt\n\nRecent commits:\naaa111 initial"
)
GIT_STATUS_RESUMED = "gitStatus: ...\n\nStatus:\n?? b.txt\n\nRecent commits:\nbbb222 agent\naaa111 initial"
SYSTEM_FIRST = [BILLING, IDENTITY_FIRST, INSTRUCTIONS + APPENDED + GIT_STATUS_FIRST]
SYSTEM_RESUMED = [BILLING, IDENTITY_RESUMED, INSTRUCTIONS + GIT_STATUS_RESUMED]

ROOT_PROMPT = (
    "<system-reminder>\nSessionStart hook...\n</system-reminder>\ndo the thing"
)
SUBAGENT_PROMPT = (
    "<system-reminder>\nThe following skills...\n</system-reminder>\nlist files"
)
SUBAGENT_SYSTEM = [
    "x-anthropic-billing-header: cc_version=2.1.257.1f2; cc_entry=agent",
    IDENTITY_RESUMED,
    "\nYou are an agent for Claude Code...\n\n" + GIT_STATUS_RESUMED,
]


def request(
    system: list[str], prompt: str = ROOT_PROMPT, turns: int = 0
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [ChatMessageSystem(content=s) for s in system]
    messages.append(ChatMessageUser(content=prompt))
    for i in range(turns):
        messages.append(ChatMessageAssistant(content=f"reply {i}"))
        messages.append(ChatMessageUser(content=f"follow-up {i}"))
    return messages


def flattened(system: list[str], prompt: str = ROOT_PROMPT) -> list[ChatMessage]:
    """Layout produced by inspect_ai < 0.3.262 (system blocks joined)."""
    return [
        ChatMessageSystem(content="\n\n".join(system)),
        ChatMessageUser(content=prompt),
    ]


def filter_args(
    messages: list[ChatMessage],
) -> tuple[Model, list[ChatMessage], list[ToolInfo], None, GenerateConfig]:
    return (get_model("mockllm/model"), messages, [], None, GenerateConfig())


def system_texts(messages: list[ChatMessage]) -> list[str]:
    return [m.text for m in messages if isinstance(m, ChatMessageSystem)]


def make_filter(user_filter: Any = None) -> Any:
    session = str(uuid.uuid4())
    return pin_system_prompt_filter(lambda: session, user_filter)


async def test_first_request_passes_through() -> None:
    wrapped = make_filter()
    messages = request(SYSTEM_FIRST)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_FIRST


async def test_resumed_request_gets_first_system_prompt_in_place() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None

    messages = request(SYSTEM_RESUMED, turns=1)
    originals = [m for m in messages if isinstance(m, ChatMessageSystem)]
    # rewritten in place and None returned, so the pinned text reaches the
    # model, the ModelEvent, and bridge.state.messages alike
    assert await wrapped(*filter_args(messages)) is None
    assert [m for m in messages if isinstance(m, ChatMessageSystem)] == originals
    assert system_texts(messages) == SYSTEM_FIRST
    # non-system messages untouched
    assert [m.text for m in messages if not isinstance(m, ChatMessageSystem)] == [
        ROOT_PROMPT,
        "reply 0",
        "follow-up 0",
    ]


async def test_unchanged_system_prompt_left_alone() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SYSTEM_FIRST, turns=2)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_FIRST


async def test_flattened_system_message_is_pinned() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(flattened(SYSTEM_FIRST))) is None
    messages = flattened(SYSTEM_RESUMED)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == system_texts(flattened(SYSTEM_FIRST))


async def test_other_conversations_are_not_touched() -> None:
    # a sub-agent conversation has its own first user message, so its own
    # (shorter) system prompt is left as Claude Code rendered it
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SUBAGENT_SYSTEM, prompt=SUBAGENT_PROMPT)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SUBAGENT_SYSTEM
    # and it does not disturb the root baseline
    root = request(SYSTEM_RESUMED, turns=1)
    assert await wrapped(*filter_args(root)) is None
    assert system_texts(root) == SYSTEM_FIRST


async def test_system_message_count_mismatch_fails_open() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SYSTEM_RESUMED[1:], turns=1)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_RESUMED[1:]


async def test_request_without_user_message_is_ignored() -> None:
    wrapped = make_filter()
    orphan: list[ChatMessage] = [ChatMessageSystem(content=s) for s in SYSTEM_RESUMED]
    # neither claims the baseline...
    assert await wrapped(*filter_args(orphan)) is None
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SYSTEM_RESUMED, turns=1)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_FIRST
    # ...nor gets pinned
    assert await wrapped(*filter_args(orphan)) is None
    assert system_texts(orphan) == SYSTEM_RESUMED


async def test_request_without_system_messages_is_ignored() -> None:
    wrapped = make_filter()
    bare: list[ChatMessage] = [ChatMessageUser(content=ROOT_PROMPT)]
    assert await wrapped(*filter_args(bare)) is None
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SYSTEM_RESUMED, turns=1)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_FIRST


async def test_list_content_system_message_fails_open() -> None:
    # the bridge only produces str-content system messages; a list-content
    # message would be collapsed by a str rewrite, so the request is skipped
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    parts = [ContentText(text="partA"), ContentText(text=SYSTEM_RESUMED[2])]
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=SYSTEM_RESUMED[0]),
        ChatMessageSystem(content=SYSTEM_RESUMED[1]),
        ChatMessageSystem(content=list(parts)),
        ChatMessageUser(content=ROOT_PROMPT),
    ]
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages)[:2] == SYSTEM_RESUMED[:2]
    assert messages[2].content == parts


async def test_sessions_do_not_share_baselines() -> None:
    first = make_filter()
    second = make_filter()
    assert await first(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SYSTEM_RESUMED, turns=1)
    assert await second(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_RESUMED


async def test_session_id_is_read_per_request() -> None:
    # a session id restored from a checkpoint after the wrapper was built
    # must key the pin, so a new id starts a new baseline
    ids = iter([str(uuid.uuid4()), str(uuid.uuid4())])
    wrapped: Any = pin_system_prompt_filter(lambda: next(ids), None)
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SYSTEM_RESUMED, turns=1)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_RESUMED


async def test_pinning_is_idempotent_on_retries() -> None:
    # the bridge re-runs the filter on refusal retries with the same list
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages = request(SYSTEM_RESUMED, turns=1)
    for _ in range(3):
        assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_FIRST


async def test_pinning_error_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    def broken_session_id() -> str:
        raise RuntimeError("no session")

    wrapped: Any = pin_system_prompt_filter(broken_session_id, None)
    messages = request(SYSTEM_RESUMED)
    with caplog.at_level(logging.WARNING):
        assert await wrapped(*filter_args(messages)) is None
        assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_RESUMED
    warned = [r for r in caplog.records if "system prompt" in r.getMessage()]
    assert len(warned) == 1


async def test_user_filter_sees_pinned_messages() -> None:
    seen: dict[str, Any] = {}

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        seen["texts"] = system_texts(messages)
        return None

    wrapped = make_filter(user_filter)
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    assert await wrapped(*filter_args(request(SYSTEM_RESUMED, turns=1))) is None
    assert seen["texts"] == SYSTEM_FIRST


async def test_user_filter_result_is_returned() -> None:
    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        return GenerateInput(
            input=messages[-1:], tools=tools, tool_choice=tool_choice, config=config
        )

    wrapped = make_filter(user_filter)
    await wrapped(*filter_args(request(SYSTEM_FIRST)))
    result = await wrapped(*filter_args(request(SYSTEM_RESUMED, turns=1)))
    assert isinstance(result, GenerateInput)
    assert len(result.input) == 1


async def test_legacy_str_filter_gets_model_name_and_deprecation_warning() -> None:
    seen: dict[str, Any] = {}

    async def legacy_filter(
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        seen["model"] = model
        seen["texts"] = system_texts(messages)
        return None

    # the bridge warns about str-first filters; wrapping hides the user filter
    # from the bridge, so the wrapper must replicate the warning
    with pytest.warns(DeprecationWarning, match="str"):
        wrapped = make_filter(legacy_filter)
    model, *rest = filter_args(request(SYSTEM_FIRST))
    await wrapped(model, *rest)
    assert seen["model"] == model.name
    assert await wrapped(*filter_args(request(SYSTEM_RESUMED, turns=1))) is None
    assert seen["texts"] == SYSTEM_FIRST


def test_claude_code_wraps_filter_at_construction() -> None:
    from inspect_swe import claude_code

    async def legacy_filter(
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        return None

    # the wrapper is built once per agent instance (not per execute()), so the
    # legacy-filter warning surfaces when the agent is constructed
    with pytest.warns(DeprecationWarning, match="str"):
        claude_code(filter=legacy_filter)


async def test_only_leading_system_messages_are_pinned() -> None:
    # Claude Code (2.1.236 with Opus 4.8) injects role="system" messages into
    # the conversation history -- a skills reminder after the first prompt and
    # a per-turn "<total_tokens>N tokens left</total_tokens>" budget marker --
    # which the bridge maps to ChatMessageSystem in place. They are replayed
    # history, not the system prompt: they must be left alone and their
    # growing count must not prevent pinning the leading system prompt.
    wrapped = make_filter()
    first = request(SYSTEM_FIRST)
    first.append(
        ChatMessageSystem(
            content="skills...\n\n<total_tokens>15000000 tokens left</total_tokens>"
        )
    )
    assert await wrapped(*filter_args(first)) is None

    history: list[ChatMessage] = [
        ChatMessageSystem(
            content="skills...\n\n<total_tokens>15000000 tokens left</total_tokens>"
        ),
        ChatMessageAssistant(content="reply 0"),
        ChatMessageUser(content="tool result 0"),
        ChatMessageSystem(content="<total_tokens>14976413 tokens left</total_tokens>"),
        ChatMessageAssistant(content="reply 1"),
        ChatMessageUser(content="tool result 1"),
        ChatMessageSystem(content="<total_tokens>14976190 tokens left</total_tokens>"),
    ]
    messages = request(SYSTEM_RESUMED) + history
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages)[: len(SYSTEM_RESUMED)] == SYSTEM_FIRST
    assert system_texts(messages)[len(SYSTEM_RESUMED) :] == [
        m.text for m in history if isinstance(m, ChatMessageSystem)
    ]


async def test_leading_run_followed_by_assistant_is_ignored() -> None:
    # the anchor is the message right after the leading system run; a request
    # that continues with an assistant message has no anchor to compare
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    messages: list[ChatMessage] = [
        *(ChatMessageSystem(content=s) for s in SYSTEM_RESUMED),
        ChatMessageAssistant(content="continue"),
        ChatMessageUser(content=ROOT_PROMPT),
    ]
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == SYSTEM_RESUMED


async def test_anchor_compares_content_not_text() -> None:
    # `.text` drops non-text parts and joins text parts, so two different
    # first user messages could compare equal; the anchor must not
    from inspect_ai.model import ContentImage

    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    with_image: list[ChatMessage] = [
        *(ChatMessageSystem(content=s) for s in SYSTEM_RESUMED),
        ChatMessageUser(
            content=[
                ContentImage(image="data:image/png;base64,AAAA"),
                ContentText(text=ROOT_PROMPT),
            ]
        ),
    ]
    assert await wrapped(*filter_args(with_image)) is None
    assert system_texts(with_image) == SYSTEM_RESUMED


async def test_layout_mismatch_with_matching_anchor_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # same conversation but a different number of leading system messages:
    # nothing sensible to write back, so pass through -- and say so once,
    # since this is the one mismatch that indicates the pin has gone inert
    wrapped = make_filter()
    assert await wrapped(*filter_args(request(SYSTEM_FIRST))) is None
    grown = request(SYSTEM_RESUMED + ["extra block"], turns=1)
    with caplog.at_level(logging.WARNING):
        assert await wrapped(*filter_args(grown)) is None
        assert await wrapped(*filter_args(grown)) is None
    assert system_texts(grown) == SYSTEM_RESUMED + ["extra block"]
    warned = [r for r in caplog.records if "system prompt" in r.getMessage()]
    assert len(warned) == 1
