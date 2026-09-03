"""Tests for pinning Claude Code's git status section across --resume launches."""

import logging
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

# A vanilla Claude Code 2.1.257 sends `system` as three blocks (captured
# against a mock endpoint): a billing header, an identity line, and the
# instructions block whose tail is the gitStatus section (local settings can
# add blocks in between). inspect_ai >= 0.3.262 hoists each block into its own
# ChatMessageSystem; older versions flatten them into one.
BILLING = "x-anthropic-billing-header: cc_version=2.1.257.1f2; cc_entry=cli"
IDENTITY = "You are a Claude agent, built on Anthropic's Claude Agent SDK."
IDENTITY_APPENDED = (
    "You are Claude Code, Anthropic's official CLI for Claude, running within "
    "the Claude Agent SDK."
)
INSTRUCTIONS = "\nYou are an interactive agent that helps users...\n\nMore.\n\n"
APPENDED = "You are an ace researcher.\n\n"
SUBAGENT_BILLING = "x-anthropic-billing-header: cc_version=2.1.257.1f2; cc_entry=agent"
SUBAGENT_INSTRUCTIONS = "\nYou are an agent. Notes:\n- do the task\n\n"

SECTION_START = (
    f"{GIT_STATUS_SENTINEL} Note that this status is a snapshot in time, and "
    "will not update during the conversation.\n\nCurrent branch: main\n\n"
    "Status:\n?? a.txt\n\nRecent commits:\naaa111 initial commit"
)
SECTION_RESUMED = (
    f"{GIT_STATUS_SENTINEL} Note that this status is a snapshot in time, and "
    "will not update during the conversation.\n\nCurrent branch: main\n\n"
    "Status:\n?? b.txt\n\nRecent commits:\nbbb222 agent commit\naaa111 initial commit"
)


def root_messages(
    section: str, *, identity: str = IDENTITY, appended: str = ""
) -> list[ChatMessage]:
    return [
        ChatMessageSystem(content=BILLING),
        ChatMessageSystem(content=identity),
        ChatMessageSystem(content=INSTRUCTIONS + appended + section),
        ChatMessageUser(content="do the thing"),
    ]


def flattened_root_messages(section: str) -> list[ChatMessage]:
    """Layout produced by inspect_ai < 0.3.262 (system blocks joined)."""
    return [
        ChatMessageSystem(
            content="\n\n".join([BILLING, IDENTITY, INSTRUCTIONS + section])
        ),
        ChatMessageUser(content="do the thing"),
    ]


def subagent_messages(section: str) -> list[ChatMessage]:
    return [
        ChatMessageSystem(content=SUBAGENT_BILLING),
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
    assert split_git_status(text) == (INSTRUCTIONS, SECTION_START)


def test_split_git_status_absent() -> None:
    assert split_git_status(INSTRUCTIONS) is None


async def test_first_request_passes_through() -> None:
    wrapped = make_filter()
    messages = root_messages(SECTION_START)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == [BILLING, IDENTITY, INSTRUCTIONS + SECTION_START]


async def test_resumed_request_is_pinned_in_place() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None

    messages = root_messages(SECTION_RESUMED)
    original = messages[2]
    # pinning rewrites the message in place and returns None so that the
    # pinned text reaches the model, the ModelEvent, and bridge.state.messages
    assert await wrapped(*filter_args(messages)) is None
    assert messages[2] is original
    assert system_texts(messages) == [BILLING, IDENTITY, INSTRUCTIONS + SECTION_START]


async def test_unchanged_section_left_alone() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None
    messages = root_messages(SECTION_START)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages)[2] == INSTRUCTIONS + SECTION_START


async def test_flattened_system_message_is_pinned() -> None:
    wrapped = make_filter()
    assert await wrapped(*filter_args(flattened_root_messages(SECTION_START))) is None
    messages = flattened_root_messages(SECTION_RESUMED)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == system_texts(
        flattened_root_messages(SECTION_START)
    )


async def test_subagent_request_is_pinned_with_own_prefix() -> None:
    # a single Claude Code process renders git status once and shares it with
    # every sub-agent it spawns, so sub-agent conversations in a resumed
    # process get the session's first-seen section too -- under their own prefix
    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None
    messages = subagent_messages(SECTION_RESUMED)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == [
        SUBAGENT_BILLING,
        IDENTITY,
        SUBAGENT_INSTRUCTIONS + SECTION_START,
    ]


async def test_changed_prefix_still_pinned() -> None:
    # with system_prompt set, the first launch passes --append-system-prompt
    # and resumed launches do not: the identity line flips and the appended
    # text disappears. The git status section must still be pinned.
    wrapped = make_filter()
    first = root_messages(SECTION_START, identity=IDENTITY_APPENDED, appended=APPENDED)
    assert await wrapped(*filter_args(first)) is None
    messages = root_messages(SECTION_RESUMED)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == [BILLING, IDENTITY, INSTRUCTIONS + SECTION_START]


async def test_no_git_status_section_is_ignored() -> None:
    wrapped = make_filter()
    without: list[ChatMessage] = [
        ChatMessageSystem(content=INSTRUCTIONS),
        ChatMessageUser(content="hi"),
    ]
    assert await wrapped(*filter_args(without)) is None
    assert system_texts(without) == [INSTRUCTIONS]
    # a sentinel-free request does not claim the baseline: the first request
    # that carries the section does
    assert await wrapped(*filter_args(root_messages(SECTION_RESUMED))) is None
    messages = root_messages(SECTION_START)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages)[2] == INSTRUCTIONS + SECTION_RESUMED


async def test_sessions_do_not_share_baselines() -> None:
    first = make_filter()
    second = make_filter()
    assert await first(*filter_args(root_messages(SECTION_START))) is None
    messages = root_messages(SECTION_RESUMED)
    assert await second(*filter_args(messages)) is None
    assert system_texts(messages)[2] == INSTRUCTIONS + SECTION_RESUMED


async def test_pinning_error_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    def broken_session_id() -> str:
        raise RuntimeError("no session")

    wrapped: Any = pin_git_status_filter(broken_session_id, None)
    messages = root_messages(SECTION_RESUMED)
    with caplog.at_level(logging.WARNING):
        assert await wrapped(*filter_args(messages)) is None
        assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages)[2] == INSTRUCTIONS + SECTION_RESUMED
    warnings_logged = [r for r in caplog.records if "git status" in r.getMessage()]
    assert len(warnings_logged) == 1


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
    assert await wrapped(*filter_args(root_messages(SECTION_RESUMED))) is None
    assert seen["texts"][2] == INSTRUCTIONS + SECTION_START


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
    await wrapped(*filter_args(root_messages(SECTION_START)))
    result = await wrapped(*filter_args(root_messages(SECTION_RESUMED)))
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
    model, *rest = filter_args(root_messages(SECTION_START))
    await wrapped(model, *rest)
    assert seen["model"] == model.name
    assert await wrapped(*filter_args(root_messages(SECTION_RESUMED))) is None
    assert seen["texts"][2] == INSTRUCTIONS + SECTION_START


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


async def test_pinning_is_idempotent_on_retries() -> None:
    # the bridge re-runs the filter on refusal retries with the same list
    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None
    messages = root_messages(SECTION_RESUMED)
    for _ in range(3):
        assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages) == [BILLING, IDENTITY, INSTRUCTIONS + SECTION_START]


async def test_session_id_is_read_per_request() -> None:
    # a session id restored from a checkpoint after the wrapper was built
    # must key the pin, so a new id starts a new baseline
    ids = iter([str(uuid.uuid4()), str(uuid.uuid4())])
    wrapped: Any = pin_git_status_filter(lambda: next(ids), None)
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None
    messages = root_messages(SECTION_RESUMED)
    assert await wrapped(*filter_args(messages)) is None
    assert system_texts(messages)[2] == INSTRUCTIONS + SECTION_RESUMED


async def test_list_content_system_message_is_left_alone() -> None:
    # the bridge only produces str-content system messages; a list-content
    # message would be collapsed by a str rewrite, so it is skipped instead
    from inspect_ai.model import ContentText

    wrapped = make_filter()
    assert await wrapped(*filter_args(root_messages(SECTION_START))) is None
    parts = [
        ContentText(text="partA"),
        ContentText(text=INSTRUCTIONS + SECTION_RESUMED),
    ]
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=list(parts)),
        ChatMessageUser(content="hi"),
    ]
    assert await wrapped(*filter_args(messages)) is None
    assert messages[0].content == parts
