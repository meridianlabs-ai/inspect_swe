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
# text block holding ContentReasoning.text (`<think>{reasoning}</think>`, or
# the summary for a redacted block) with its signature dropped, and an
# otherwise empty text reply as NO_CONTENT. dsh stores both as ordinary
# assistant text and sends them back that way, so without repair the model
# would see its earlier reasoning as visible output and lose provider replay
# metadata (e.g. OpenRouter reasoning_details) that a native dsh run on
# DeepSeek's API keeps as thinking blocks.


class ReasoningRestorer:
    """Restore reasoning that the bridge flattened into assistant text.

    Originals are looked up in the sample transcript's model events by their
    exact flattened text, so restored blocks carry their signatures. A text
    that maps to more than one original (e.g. several redacted blocks without
    summaries all flatten to `<think></think>`), or to none, is left as is
    rather than replayed with the wrong (or no) provider metadata. Not
    thread-safe; use one instance per agent execution.
    """

    def __init__(self) -> None:
        # flattened text -> original block, or None when ambiguous
        self._originals: dict[str, ContentReasoning | None] = {}
        self._scanned_events = 0

    def restore(self, messages: list[ChatMessage]) -> bool:
        """Restore reasoning in place; returns whether anything changed."""
        if not any(_has_candidate_content(m) for m in messages):
            return False
        self._scan_transcript()
        changed = False
        for index, message in enumerate(messages):
            if not _has_candidate_content(message):
                continue
            assert isinstance(message, ChatMessageAssistant)
            assert isinstance(message.content, list)
            content: list[Content] = []
            restored_any = False
            for item in message.content:
                if isinstance(item, ContentText):
                    original = self._originals.get(item.text)
                    if original is not None:
                        content.append(original)
                        restored_any = True
                        continue
                content.append(item)
            # drop the bridge's NO_CONTENT placeholder beside restored
            # reasoning (a reply that is only NO_CONTENT keeps it, as providers
            # reject an assistant message with neither content nor tool calls)
            if restored_any:
                without_placeholder = [
                    c
                    for c in content
                    if not (isinstance(c, ContentText) and c.text == NO_CONTENT)
                ]
                if without_placeholder or message.tool_calls:
                    content = without_placeholder
            if content != message.content:
                messages[index] = message.model_copy(update={"content": content})
                changed = True
        return changed

    def _scan_transcript(self) -> None:
        events = transcript().events
        index = self._scanned_events
        while index < len(events):
            event = events[index]
            if isinstance(event, ModelEvent):
                # a pending call has no output yet: resume scanning here next
                # time rather than skipping it (dsh runs subagents in parallel)
                if event.pending:
                    break
                content = event.output.message.content
                if not isinstance(content, str):
                    for item in content:
                        if isinstance(item, ContentReasoning):
                            self._add_original(item)
            index += 1
        self._scanned_events = index

    def _add_original(self, item: ContentReasoning) -> None:
        key = item.text
        existing = self._originals.get(key, item)
        if key not in self._originals:
            self._originals[key] = item
        elif existing is not None and existing != item:
            self._originals[key] = None


def _has_candidate_content(message: ChatMessage) -> bool:
    return (
        isinstance(message, ChatMessageAssistant)
        and isinstance(message.content, list)
        and any(
            isinstance(item, ContentText) and item.text.startswith("<think")
            for item in message.content
        )
    )
