"""Content/usage extraction helpers for Codex rollout conversion."""

import json
from datetime import datetime, timezone
from logging import getLogger
from typing import Any

from inspect_ai.event import Event, ModelEvent
from inspect_ai.model import (
    Content,
    ContentImage,
    ContentReasoning,
    ContentText,
)
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._model_output import ModelUsage

from .rollout_models import ResponseReasoning

logger = getLogger(__name__)


def parse_timestamp(timestamp_str: str | None) -> datetime | None:
    """Parse an RFC3339 timestamp string to a timezone-aware datetime."""
    if not timestamp_str:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, AttributeError):
        return None


# ── pseudo-user message classification ───────────────────────────────────

# Special context blocks codex injects as role="user" response items. See
# the *_OPEN_TAG constants in codex-rs/protocol/src/protocol.rs.
_CONTEXT_OPEN_TAGS = (
    "<user_instructions>",
    "<environment_context>",
    "<environments_instructions>",
    "<apps_instructions>",
    "<skills_instructions>",
    "<plugins_instructions>",
    "<recommended_plugins>",
    "<tools>",
    "<collaboration_mode>",
    "<multi_agent_mode>",
    "<realtime_conversation>",
    "<context_window>",
    "<context_window_guidance>",
    "<turn_context>",
)

_CONTEXT_TEXT_PREFIXES = ("# AGENTS.md instructions",)

# Genuine user text bundled with context blocks is prefixed with this marker.
USER_MESSAGE_BEGIN = "## My request for Codex:"


def is_context_message(text: str) -> bool:
    """Whether a role=user message is an injected context block, not user speech.

    Messages that contain the ``USER_MESSAGE_BEGIN`` marker carry genuine
    user text after the context prefix, so they are user speech.
    """
    stripped = text.lstrip()
    if USER_MESSAGE_BEGIN in text:
        return False
    return stripped.startswith((*_CONTEXT_OPEN_TAGS, *_CONTEXT_TEXT_PREFIXES))


# ── content conversion ───────────────────────────────────────────────────


def content_items_to_content(
    content: list[dict[str, Any]] | str,
) -> str | list[Content]:
    """Convert Responses-API content items to inspect content.

    Text-only content collapses to a plain string (matching the Claude Code
    converter's behaviour); images preserve interleaving in a content list.
    """
    if isinstance(content, str):
        return content

    blocks: list[Content] = []
    has_image = False
    for item in content:
        item_type = item.get("type")
        if item_type in ("input_text", "output_text"):
            text = item.get("text")
            if isinstance(text, str) and text:
                blocks.append(ContentText(text=text))
        elif item_type == "input_image":
            image_url = item.get("image_url")
            if isinstance(image_url, str) and image_url:
                blocks.append(ContentImage(image=image_url))
                has_image = True
            else:
                logger.warning("Skipping input_image with no image_url")
        # input_audio and unknown item types are dropped

    if not has_image:
        return "\n".join(
            block.text for block in blocks if isinstance(block, ContentText)
        )
    return blocks


def reasoning_to_content(item: ResponseReasoning) -> ContentReasoning | None:
    """Convert a reasoning item to ContentReasoning.

    Plaintext reasoning (``content``) is rare; usually only summaries plus an
    opaque ``encrypted_content`` are available, in which case the summary text
    is used and the block is marked redacted (the raw chain-of-thought is not
    recoverable). The encrypted payload itself is not preserved.
    """
    summary_text = "\n\n".join(
        text
        for block in item.summary
        if isinstance(text := block.get("text"), str) and text
    )
    plaintext = ""
    if item.content:
        plaintext = "\n\n".join(
            text
            for block in item.content
            if block.get("type") in ("reasoning_text", "text")
            and isinstance(text := block.get("text"), str)
            and text
        )

    if plaintext:
        return ContentReasoning(
            reasoning=plaintext, summary=summary_text or None, redacted=False
        )
    elif summary_text:
        return ContentReasoning(
            reasoning=summary_text,
            summary=summary_text,
            redacted=item.encrypted_content is not None,
        )
    elif item.encrypted_content is not None:
        return ContentReasoning(reasoning="", redacted=True)
    else:
        return None


# ── tool output conversion ───────────────────────────────────────────────


def output_to_result(
    output: Any,
) -> tuple[str | list[Content], int | None]:
    """Convert a function_call_output payload to a tool result.

    Handles the three wire forms: plain string, array of content items, and
    the legacy JSON-encoded ``{"output": ..., "metadata": {"exit_code": ...}}``
    string written by old codex versions. Returns (result, exit_code) where
    exit_code is only known for the legacy form.
    """
    if isinstance(output, str):
        # Legacy form: JSON-encoded object with output + metadata
        if output.startswith("{"):
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "output" in parsed:
                metadata = parsed.get("metadata")
                exit_code = (
                    metadata.get("exit_code") if isinstance(metadata, dict) else None
                )
                return str(parsed["output"]), (
                    exit_code if isinstance(exit_code, int) else None
                )
        return output, None
    elif isinstance(output, list):
        return content_items_to_content(output), None
    elif isinstance(output, dict):
        # FunctionCallOutputBody-style {"content": ...} wrapper
        content = output.get("content")
        if isinstance(content, (str, list)):
            return output_to_result(content)
        return json.dumps(output), None
    elif output is None:
        return "", None
    else:
        return str(output), None


def parse_arguments(arguments: str) -> dict[str, Any]:
    """Parse a function_call arguments JSON string, tolerating bad JSON."""
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"arguments": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"arguments": parsed}


# ── usage ────────────────────────────────────────────────────────────────


def usage_from_token_info(info: dict[str, Any]) -> ModelUsage | None:
    """Build ModelUsage from a token_count event's last_token_usage.

    Codex reports input_tokens inclusive of cached tokens (Responses API
    semantics); inspect's ModelUsage.input_tokens excludes them.
    """
    last = info.get("last_token_usage")
    if not isinstance(last, dict):
        return None

    def _tokens(key: str) -> int:
        value = last.get(key)
        return value if isinstance(value, int) else 0

    input_tokens = _tokens("input_tokens")
    cached = _tokens("cached_input_tokens")
    cache_write = _tokens("cache_write_input_tokens")
    output_tokens = _tokens("output_tokens")
    reasoning = _tokens("reasoning_output_tokens")
    total = _tokens("total_tokens") or (input_tokens + output_tokens)

    if total == 0 and input_tokens == 0 and output_tokens == 0:
        return None

    return ModelUsage(
        input_tokens=max(0, input_tokens - cached),
        output_tokens=output_tokens,
        total_tokens=total,
        input_tokens_cache_read=cached if cached else None,
        input_tokens_cache_write=cache_write if cache_write else None,
        reasoning_tokens=reasoning if reasoning else None,
    )


def total_tokens_from_token_info(info: dict[str, Any]) -> int | None:
    """The cumulative total token count from a token_count event."""
    total = info.get("total_token_usage")
    if isinstance(total, dict):
        value = total.get("total_tokens")
        if isinstance(value, int):
            return value
    return None


def sum_scout_tokens(events: list[Event]) -> int:
    """Sum total tokens across all ModelEvents (including nested agents)."""
    total = 0
    for event in events:
        if isinstance(event, ModelEvent):
            if event.output and event.output.usage:
                total += event.output.usage.total_tokens
    return total


# ── replacement history conversion ───────────────────────────────────────


def history_to_messages(items: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert a compacted item's replacement_history to chat messages.

    Only message items are converted (function calls etc. may appear in
    replacement history but have no natural chat-message form without their
    outputs).
    """
    messages: list[ChatMessage] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        role = item.get("role")
        raw_content = item.get("content")
        content: list[dict[str, Any]] | str
        if isinstance(raw_content, (list, str)):
            content = raw_content
        else:
            continue
        converted = content_items_to_content(content)
        if not converted:
            continue
        if role == "assistant":
            messages.append(ChatMessageAssistant(content=converted))
        elif role == "developer" or role == "system":
            messages.append(ChatMessageSystem(content=converted))
        else:
            messages.append(ChatMessageUser(content=converted))
    return messages
