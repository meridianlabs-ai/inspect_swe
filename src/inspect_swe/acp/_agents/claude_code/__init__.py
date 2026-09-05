"""ACP-based Claude Code agent."""

from .claude_code import ClaudeCode, interactive_claude_code
from .transcript import (
    AssistantText,
    ParsedTranscript,
    RawBlock,
    Thinking,
    ToolResult,
    ToolUse,
    TranscriptItem,
    TranscriptSpec,
    UserText,
    build_transcript,
    items_from_messages,
    messages_from_items,
    parse_transcript,
    project_slug,
)

__all__ = [
    "ClaudeCode",
    "interactive_claude_code",
    # Transcript API — build/parse claude code session transcripts for resume.
    "build_transcript",
    "parse_transcript",
    "project_slug",
    "TranscriptSpec",
    "ParsedTranscript",
    "TranscriptItem",
    # ChatMessage conversion (build_transcript takes messages directly; these
    # are for callers that want the items, or messages back out of a parse).
    "items_from_messages",
    "messages_from_items",
    # Transcript item types (construct a synthetic prior by hand).
    "UserText",
    "AssistantText",
    "Thinking",
    "ToolUse",
    "ToolResult",
    "RawBlock",
]
