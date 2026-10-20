import re

from inspect_ai._util.constants import NO_CONTENT
from inspect_ai.event import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    Content,
    ContentReasoning,
    ContentText,
)

# The bridge's Anthropic API surface can only replay reasoning as a `thinking`
# block when it originated from Inspect's Anthropic provider. Reasoning from
# any other provider (DeepSeek, OpenRouter, ...) reaches the agent as a plain
# text block `<think>{reasoning}</think>` (ContentReasoning.text), with its
# signature dropped, and an otherwise empty text reply as NO_CONTENT. dsh
# stores both as ordinary assistant text and sends them back that way, so
# without repair the model would see its earlier reasoning as visible output
# and lose provider replay metadata (e.g. OpenRouter reasoning_details) that a
# native dsh run on DeepSeek's API keeps as thinking blocks.
_THINK_TEXT_RE = re.compile(r"\A<think>(.*)</think>\Z", re.DOTALL)


class ReasoningRestorer:
    """Restore reasoning that the bridge flattened into assistant text.

    Originals are looked up (by reasoning text) in the sample transcript's
    model events, so the restored blocks carry their signatures. Not
    thread-safe; use one instance per agent execution.
    """

    def __init__(self) -> None:
        self._originals: dict[str, ContentReasoning] = {}
        self._scanned_events = 0

    def restore(self, messages: list[ChatMessage]) -> bool:
        """Restore reasoning in place; returns whether anything changed."""
        if not any(_has_flattened_content(m) for m in messages):
            return False
        self._scan_transcript()
        changed = False
        for index, message in enumerate(messages):
            if not _has_flattened_content(message):
                continue
            assert isinstance(message, ChatMessageAssistant)
            assert isinstance(message.content, list)
            content: list[Content] = []
            for item in message.content:
                if isinstance(item, ContentText):
                    match = _THINK_TEXT_RE.match(item.text)
                    if match is not None:
                        reasoning = match.group(1)
                        content.append(
                            self._originals.get(reasoning)
                            or ContentReasoning(reasoning=reasoning)
                        )
                        continue
                    if item.text == NO_CONTENT:
                        continue
                content.append(item)
            # a reply that was only NO_CONTENT keeps it (providers reject an
            # assistant message with neither content nor tool calls)
            if not content and not message.tool_calls:
                continue
            messages[index] = message.model_copy(update={"content": content})
            changed = True
        return changed

    def _scan_transcript(self) -> None:
        events = transcript().events
        for event in events[self._scanned_events :]:
            if not isinstance(event, ModelEvent):
                continue
            content = event.output.message.content
            if isinstance(content, str):
                continue
            for item in content:
                if isinstance(item, ContentReasoning):
                    self._originals.setdefault(item.reasoning, item)
        self._scanned_events = len(events)


def _has_flattened_content(message: ChatMessage) -> bool:
    return (
        isinstance(message, ChatMessageAssistant)
        and isinstance(message.content, list)
        and any(
            isinstance(item, ContentText)
            and (_THINK_TEXT_RE.match(item.text) is not None or item.text == NO_CONTENT)
            for item in message.content
        )
    )
