"""Tests for pinning Claude Code's git status section across --resume launches."""

import uuid
from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe._claude_code.gitstatus import (
    GIT_STATUS_SENTINEL,
    pin_git_status_filter,
    split_git_status,
)

pytestmark = pytest.mark.anyio

IDENTITY = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
INSTRUCTIONS = "Long built-in instructions...\n\nMore instructions.\n\n"
SUBAGENT_INSTRUCTIONS = "You are an agent. Notes:\n- do the task\n\n"

SECTION_START = (
    f"{GIT_STATUS_SENTINEL} Note that this status is a snapshot in time, and "
    "will not update during the conversation.\n\nCurrent branch: main\n\n"
    "Status:\n(clean)\n\nRecent commits:\naaa111 initial commit"
)
SECTION_RESUMED = (
    f"{GIT_STATUS_SENTINEL} Note that this status is a snapshot in time, and "
    "will not update during the conversation.\n\nCurrent branch: main\n\n"
    "Status:\nM a.txt\n\nRecent commits:\nbbb222 agent commit\naaa111 initial commit"
)


def root_messages(section: str) -> list[ChatMessage]:
    return [
        ChatMessageSystem(content=IDENTITY),
        ChatMessageSystem(content=INSTRUCTIONS + section),
        ChatMessageUser(content="do the thing"),
    ]


def subagent_messages(section: str) -> list[ChatMessage]:
    return [
        ChatMessageSystem(content=IDENTITY),
        ChatMessageSystem(content=SUBAGENT_INSTRUCTIONS + section),
        ChatMessageUser(content="subagent task"),
    ]


def filter_args(
    messages: list[ChatMessage],
) -> tuple[Model, list[ChatMessage], list[ToolInfo], None, GenerateConfig]:
    return (get_model("mockllm/model"), messages, [], None, GenerateConfig())


def system_texts(messages: list[ChatMessage]) -> list[str]:
    return [m.text for m in messages if isinstance(m, ChatMessageSystem)]


def make_filter(user_filter: Any = None) -> Any:
    session = str(uuid.uuid4())
    return pin_git_status_filter(lambda: session, user_filter)


def test_split_git_status_present() -> None:
    text = INSTRUCTIONS + SECTION_START
    split = split_git_status(text)
    assert split == (INSTRUCTIONS, SECTION_START)


def test_split_git_status_absent() -> None:
    assert split_git_status(INSTRUCTIONS) is None


async def test_first_request_passes_through_and_records_baseline() -> None:
    wrapped = make_filter()
    result = await wrapped(*filter_args(root_messages(SECTION_START)))
    assert result is None


async def test_resumed_request_gets_pinned_section() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None

    result = await wrapped(*filter_args(root_messages(SECTION_RESUMED)))
    assert isinstance(result, GenerateInput)
    assert system_texts(result.input) == [IDENTITY, INSTRUCTIONS + SECTION_START]


async def test_unchanged_section_passes_through() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None


async def test_subagent_request_not_pinned() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None
    # sub-agent conversation: different system prompt shape, current status
    assert await wrapped(*filter_args(subagent_messages(SECTION_RESUMED))) is None
    # and it does not disturb the root baseline
    result = await wrapped(*filter_args(root_messages(SECTION_RESUMED)))
    assert isinstance(result, GenerateInput)
    assert system_texts(result.input)[1] == INSTRUCTIONS + SECTION_START


async def test_no_git_status_section_passes_through() -> None:
    wrapped = make_filter()
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=INSTRUCTIONS),
        ChatMessageUser(content="hi"),
    ]
    assert await wrapped(*filter_args(messages)) is None
    assert await wrapped(*filter_args(messages)) is None


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
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None

    result = await wrapped(*filter_args(root_messages(SECTION_RESUMED)))
    assert isinstance(result, GenerateInput)
    assert seen["texts"][1] == INSTRUCTIONS + SECTION_START


async def test_user_filter_result_takes_precedence() -> None:
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
    await wrapped(*filter_args(root_messages(SECTION_START)))
    result = await wrapped(*filter_args(root_messages(SECTION_RESUMED)))
    assert isinstance(result, GenerateInput)
    assert len(result.input) == 1


async def test_legacy_str_filter_dispatch() -> None:
    seen: dict[str, Any] = {}

    async def legacy_filter(
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        seen["model"] = model
        return None

    wrapped = make_filter(legacy_filter)
    await wrapped(*filter_args(root_messages(SECTION_START)))
    assert isinstance(seen["model"], str)
