"""Model identity and bridge-alias resolution for the Claude Code agent.

Claude Code's ``--model`` (and the ``ANTHROPIC_*`` model env vars) are purely
cosmetic: they select the identity Claude Code presents to itself (its
"You are powered by the model ..." prompt) and any model-gated client behavior.
The real model is reached through the bridge. This module bundles that
resolution so ``claude_code()`` stays readable.
"""

from copy import copy
from dataclasses import dataclass
from typing import Literal

from inspect_ai.model import GenerateConfig, Model, get_model

ClaudeCodeEffort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class ClaudeCodeModels:
    """Resolved presented identities + bridge routing for a Claude Code run.

    ``presented`` and the per-role names (``opus``/``sonnet``/``haiku``/
    ``subagent``) are the *displayed* ids handed to Claude Code via ``--model``
    and the ``ANTHROPIC_*`` env vars — cosmetic only. ``aliases`` maps each
    presented name to its served ``Model`` so the bridge routes it to the real
    model; ``bridge_model`` is the sentinel fallback for any id the inner agent
    emits that isn't one of those names.
    """

    presented: str
    opus: str
    sonnet: str
    haiku: str
    subagent: str
    aliases: dict[str, str | Model]
    bridge_model: str


def resolve_claude_code_models(
    model: str | None,
    model_config: str | None,
    *,
    effort: ClaudeCodeEffort | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
    subagent_model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
) -> ClaudeCodeModels:
    """Resolve Claude Code's presented model identities and bridge aliases.

    The presented identity defaults to the real served model's name (override
    with ``model_config``); Claude Code renders the genuine name/cutoff for
    recognized Anthropic ids and shows anything else verbatim. Each
    opus/sonnet/haiku/subagent role inherits the primary presented name unless it
    is set, in which case it gets its own name *and* its own alias so it actually
    routes to its intended model (the bridge sentinel fallback would otherwise
    collapse them onto the main model). Caller-supplied ``model_aliases`` take
    precedence over the names we derive.

    ``effort`` is set on every served model this function resolves (the primary
    served model and any role that gets its own alias) by merging
    ``GenerateConfig(reasoning_effort=effort)`` onto a copy of each model's
    bound config, so it governs the model's own default whether or not the
    bridge forwards the inner agent's request-level generation config (the
    bridge drops those by default; see ``sandbox_agent_bridge``'s
    ``forward_generation_config``). It is not applied to caller-supplied
    ``model_aliases``, whose config is the caller's to control.

    Note: must be called at execution time — ``get_model()`` resolves the active
    model from the current eval/sample context.
    """

    def served(model_name: str | None) -> Model:
        resolved = get_model(model_name)
        if effort is not None:
            resolved = copy(resolved)
            resolved.config = resolved.config.merge(
                GenerateConfig(reasoning_effort=effort)
            )
        return resolved

    served_model = served(model)
    presented = model_config if model_config is not None else served_model.name
    aliases: dict[str, str | Model] = {presented: served_model}

    def role_name(role_model: str | None) -> str:
        # an unset role inherits the primary presented name (routing via its
        # alias); a set role registers its own name + alias so it routes to its
        # own model
        if role_model is None:
            return presented
        role = served(role_model)
        aliases[role.name] = role
        return role.name

    opus = role_name(opus_model)
    sonnet = role_name(sonnet_model)
    haiku = role_name(haiku_model)
    subagent = role_name(subagent_model)

    # caller-supplied aliases take precedence over the names we derived
    if model_aliases:
        aliases.update(model_aliases)

    # bridge sentinel — unchanged routing for any id the inner agent emits that
    # isn't one of the presented names above
    bridge_model = f"inspect/{model}" if model is not None else "inspect"

    return ClaudeCodeModels(
        presented=presented,
        opus=opus,
        sonnet=sonnet,
        haiku=haiku,
        subagent=subagent,
        aliases=aliases,
        bridge_model=bridge_model,
    )
