from logging import getLogger
from typing import Collection, NamedTuple

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    ContentImage,
)

logger = getLogger(__name__)


class UserTurn(NamedTuple):
    messages: list[ChatMessageUser]
    has_assistant_response: bool


def user_turn(messages: list[ChatMessage]) -> UserTurn:
    """User messages that form the next prompt to the agent.

    These are the user messages after the last assistant response (or all
    user messages if there is no assistant response yet).
    """
    if messages and isinstance(messages[-1], ChatMessageAssistant):
        raise ValueError("Messages input ends with an assistant messages.")

    last_assistant_idx = next(
        (
            i
            for i, m in reversed(list(enumerate(messages)))
            if isinstance(m, ChatMessageAssistant)
        ),
        None,
    )

    has_assistant_response = last_assistant_idx is not None
    start_idx = (last_assistant_idx + 1) if last_assistant_idx is not None else 0

    return UserTurn(
        [m for m in messages[start_idx:] if isinstance(m, ChatMessageUser)],
        has_assistant_response,
    )


def build_user_prompt(
    messages: list[ChatMessage], handled_content: Collection[str] = ()
) -> tuple[str, bool]:
    """Prompt text for the next agent turn.

    The prompt is the text of the user messages after the last assistant
    response. Non-text content in those messages cannot ride along in the
    prompt and is dropped with a warning — pass `handled_content` to name
    content types (e.g. "image") the calling agent delivers by other means.
    """
    turn = user_turn(messages)
    dropped = {
        content.type
        for m in turn.messages
        if isinstance(m.content, list)
        for content in m.content
        if content.type != "text" and content.type not in handled_content
    }
    if dropped:
        logger.warning(
            f"Input contains {', '.join(sorted(dropped))} content, which this "
            "agent does not support; it was dropped from the prompt."
        )
    prompt = "\n\n".join(m.text for m in turn.messages)
    return prompt, turn.has_assistant_response


def collect_user_images(messages: list[ChatMessage]) -> list[ContentImage]:
    """Image content from the user messages that form the next prompt.

    Scoped identically to `build_user_prompt` (i.e. the user messages after
    the last assistant response).
    """
    return [
        content
        for m in user_turn(messages).messages
        if isinstance(m.content, list)
        for content in m.content
        if isinstance(content, ContentImage)
    ]
