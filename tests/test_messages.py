import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
)
from inspect_swe._util.messages import build_user_prompt, collect_user_images

IMAGE_1 = ContentImage(image="data:image/png;base64,aW1hZ2Ux")
IMAGE_2 = ContentImage(image="data:image/png;base64,aW1hZ2Uy")


def test_build_user_prompt_initial_turn() -> None:
    messages: list[ChatMessage] = [
        ChatMessageSystem(content="be helpful"),
        ChatMessageUser(content="first"),
        ChatMessageUser(content="second"),
    ]
    prompt, has_assistant_response = build_user_prompt(messages)
    assert prompt == "first\n\nsecond"
    assert has_assistant_response is False


def test_build_user_prompt_after_assistant() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(content="first"),
        ChatMessageAssistant(content="answer"),
        ChatMessageUser(content="follow up"),
    ]
    prompt, has_assistant_response = build_user_prompt(messages)
    assert prompt == "follow up"
    assert has_assistant_response is True


def test_build_user_prompt_rejects_trailing_assistant() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(content="first"),
        ChatMessageAssistant(content="answer"),
    ]
    with pytest.raises(ValueError):
        build_user_prompt(messages)


def test_collect_user_images_initial_turn() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(content=[ContentText(text="look at these"), IMAGE_1, IMAGE_2]),
    ]
    assert collect_user_images(messages) == [IMAGE_1, IMAGE_2]


def test_collect_user_images_str_content() -> None:
    messages: list[ChatMessage] = [ChatMessageUser(content="no images here")]
    assert collect_user_images(messages) == []


def test_collect_user_images_scoped_to_current_turn() -> None:
    # images before the last assistant response belong to an already-delivered
    # turn and must not be re-collected
    messages: list[ChatMessage] = [
        ChatMessageUser(content=[ContentText(text="first"), IMAGE_1]),
        ChatMessageAssistant(content="answer"),
        ChatMessageUser(content=[ContentText(text="second"), IMAGE_2]),
    ]
    assert collect_user_images(messages) == [IMAGE_2]
