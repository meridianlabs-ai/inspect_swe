"""ACP-based Codex CLI agent."""

from .codex_cli import CodexCli, interactive_codex_cli
from .rollout import (
    AssistantText,
    CustomToolCall,
    CustomToolCallOutput,
    DeveloperText,
    FunctionCall,
    FunctionCallOutput,
    ParsedRollout,
    PriorItem,
    RawResponseItem,
    Reasoning,
    RolloutSpec,
    UserText,
    build_rollout,
    parse_rollout,
    synthesize_rollout,
)

__all__ = [
    "CodexCli",
    "interactive_codex_cli",
    # Rollout API — build/parse codex session rollouts for resume.
    "build_rollout",
    "parse_rollout",
    "synthesize_rollout",
    "RolloutSpec",
    "ParsedRollout",
    "PriorItem",
    # Prior-item types (construct a synthetic prior by hand).
    "UserText",
    "AssistantText",
    "DeveloperText",
    "FunctionCall",
    "FunctionCallOutput",
    "Reasoning",
    "CustomToolCall",
    "CustomToolCallOutput",
    "RawResponseItem",
]
