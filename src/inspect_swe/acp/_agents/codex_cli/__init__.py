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
    messages_from_prior,
    parse_rollout,
    prior_from_messages,
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
    # ChatMessage conversion (build_rollout takes messages directly; these are
    # for callers that want the items, or messages back out of a parsed rollout).
    "prior_from_messages",
    "messages_from_prior",
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
