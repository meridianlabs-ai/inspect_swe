"""Tests for `slug_map_classifier` and its gemini-cli, claude_code (ACP), and codex_cli (ACP) wiring.

The three ACP-adapter variants (gemini's `--experimental-acp`,
`claude-agent-acp`, `codex-acp`) have no JSONL/event-stream consumer to
drive attribution, so each wires `slug_map_classifier` directly rather than
reusing a `LiveConsumer`/`CodexConsumer` instance -- see `build_gemini_filter`
(shared with the non-ACP gemini variant), `build_claude_code_acp_filter`, and
`build_codex_acp_filter` respectively. Constructing the real `ACPAgent`
subclasses requires an active sample (`ACPAgent.__init__` calls
`sample_active()`), so these tests exercise the extracted `build_*_filter`
functions directly instead -- wiring is exercised structurally (by reading
`_start_agent`) rather than through a live ACP session.
"""

import logging

import pytest
from inspect_ai.agent import AgentBridgeContext, current_agent_bridge_context
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_swe._codex_cli.config import GUARDIAN_MODEL_SLUG
from inspect_swe._gemini_cli.gemini_cli import build_gemini_filter
from inspect_swe._gemini_cli.models import (
    GEMINI_UTILITY_MODEL_KINDS,
    GEMINI_UTILITY_MODEL_SLUGS,
)
from inspect_swe._util.agentcontext import (
    AgentContextClassifier,
    ModelFilter,
    slug_map_classifier,
)
from inspect_swe.acp._agents.claude_code.claude_code import (
    build_claude_code_acp_filter,
    resolve_claude_code_acp_models,
)
from inspect_swe.acp._agents.codex_cli.codex_cli import build_codex_acp_filter

# ---------------------------------------------------------------------------
# slug_map_classifier unit tests
# ---------------------------------------------------------------------------


def _classify(
    classifier: AgentContextClassifier, slug: str | None
) -> AgentBridgeContext:
    with bridged_request_scope(slug):
        return classifier(
            get_model("mockllm/model"), [ChatMessageUser(content="hi")], []
        )


def test_root_slug_is_root() -> None:
    classifier = slug_map_classifier({"root-slug"}, {})
    assert _classify(classifier, "root-slug") == AgentBridgeContext("root")


def test_each_mapped_kind_is_honored() -> None:
    classifier = slug_map_classifier(
        {"root-slug"}, {"sub-slug": "subagent", "util-slug": "utility"}
    )
    assert _classify(classifier, "sub-slug") == AgentBridgeContext("subagent")
    assert _classify(classifier, "util-slug") == AgentBridgeContext("utility")


def test_unrecognized_slug_is_unknown() -> None:
    classifier = slug_map_classifier({"root-slug"}, {"util-slug": "utility"})
    assert _classify(classifier, "some-other-slug") == AgentBridgeContext("unknown")


def test_no_bridged_request_info_is_unknown() -> None:
    classifier = slug_map_classifier({"root-slug"}, {"util-slug": "utility"})
    assert _classify(classifier, None) == AgentBridgeContext("unknown")


def test_root_takes_priority_over_kind_by_slug() -> None:
    """A slug present in both maps resolves as root (root_slugs checked first)."""
    classifier = slug_map_classifier({"shared-slug"}, {"shared-slug": "utility"})
    assert _classify(classifier, "shared-slug") == AgentBridgeContext("root")


# ---------------------------------------------------------------------------
# gemini-cli internal utility model knowledge
# ---------------------------------------------------------------------------


def test_gemini_utility_model_kinds_all_map_to_utility() -> None:
    assert set(GEMINI_UTILITY_MODEL_KINDS) == GEMINI_UTILITY_MODEL_SLUGS
    assert all(kind == "utility" for kind in GEMINI_UTILITY_MODEL_KINDS.values())


# ---------------------------------------------------------------------------
# wiring: non-ACP gemini_cli()
# ---------------------------------------------------------------------------


async def _invoke(
    wrapped: ModelFilter, slug: str, messages: list[ChatMessage] | None = None
) -> AgentBridgeContext | None:
    with bridged_request_scope(slug):
        await wrapped(
            get_model("mockllm/model"),
            messages if messages is not None else [ChatMessageUser(content="hi")],
            [],
            None,
            GenerateConfig(),
        )
        return current_agent_bridge_context()


async def test_gemini_filter_stamps_root_for_presented_slug() -> None:
    wrapped = build_gemini_filter(None, "gemini-2.5-pro")
    assert await _invoke(wrapped, "gemini-2.5-pro") == AgentBridgeContext("root")


async def test_gemini_filter_stamps_utility_for_internal_slug() -> None:
    wrapped = build_gemini_filter(None, "gemini-2.5-pro")
    assert await _invoke(wrapped, "gemini-3-flash-preview") == AgentBridgeContext(
        "utility"
    )


async def test_gemini_filter_stamps_unknown_for_unrecognized_slug() -> None:
    wrapped = build_gemini_filter(None, "gemini-2.5-pro")
    assert await _invoke(wrapped, "some-unrecognized-slug") == AgentBridgeContext(
        "unknown"
    )


async def test_gemini_filter_uses_configured_gemini_model_as_root() -> None:
    """The root slug tracks whatever `gemini_model` was configured, not a fixed default."""
    wrapped = build_gemini_filter(None, "gemini-2.5-flash")
    assert await _invoke(wrapped, "gemini-2.5-flash") == AgentBridgeContext("root")


# ---------------------------------------------------------------------------
# wiring: ACP claude_code (interactive_claude_code)
#
# The ACP variant presents canonical names (mockllm's canonical_name() is the
# bare id, so "mockllm/model" presents as "model"). The slugs asserted below
# are exactly the values _start_agent exports as ANTHROPIC_MODEL /
# CLAUDE_CODE_SUBAGENT_MODEL / ANTHROPIC_SMALL_FAST_MODEL / ..., i.e. what
# arrives at current_bridge_request().model.
# ---------------------------------------------------------------------------


def _canonical(model: str) -> str:
    return get_model(model).canonical_name()


async def test_claude_code_acp_all_distinct_roles_classify_by_slug() -> None:
    models = resolve_claude_code_acp_models(
        "mockllm/model",
        opus_model="mockllm/opus",
        sonnet_model="mockllm/sonnet",
        haiku_model="mockllm/haiku",
        subagent_model="mockllm/subagent",
    )
    # explicit, distinct subagent keeps its own name and alias
    assert models.subagent == _canonical("mockllm/subagent")
    assert set(models.aliases) == {"model", "opus", "sonnet", "haiku", "subagent"}
    wrapped = build_claude_code_acp_filter(None, models)
    assert await _invoke(wrapped, models.presented) == AgentBridgeContext("root")
    # opus/sonnet are main-thread tiers (Claude Code's own role swap), not delegation
    assert await _invoke(wrapped, models.opus) == AgentBridgeContext("root")
    assert await _invoke(wrapped, models.sonnet) == AgentBridgeContext("root")
    assert await _invoke(wrapped, models.haiku) == AgentBridgeContext("utility")
    assert await _invoke(wrapped, models.subagent) == AgentBridgeContext("subagent")
    assert await _invoke(wrapped, "some-unrecognized-slug") == AgentBridgeContext(
        "unknown"
    )


async def test_claude_code_acp_unset_subagent_gets_synthetic_distinct_slug() -> None:
    """Unset subagent_model: presented as '<presented>-subagent', routed to the primary.

    Same invariant as the native path (`resolve_claude_code_models`): the
    subagent slug never collides with any other role, so sub-agent traffic
    is attributable even in the default configuration.
    """
    models = resolve_claude_code_acp_models("mockllm/model")
    assert models.presented == "model"
    assert models.opus == models.sonnet == models.haiku == "model"
    assert models.subagent == "model-subagent"
    assert models.aliases[models.subagent] is models.aliases[models.presented]
    wrapped = build_claude_code_acp_filter(None, models)
    assert await _invoke(wrapped, models.presented) == AgentBridgeContext("root")
    assert await _invoke(wrapped, models.subagent) == AgentBridgeContext("subagent")


async def test_claude_code_acp_subagent_equals_haiku_stays_distinct() -> None:
    """subagent_model == haiku_model ("cheap model for background AND subagents").

    Without distinct slugs the two kinds share one wire slug and the later
    `kind_by_slug` insert wins, stamping sub-agent traffic "utility". The
    subagent role gets the synthetic slug, aliased to the caller's chosen
    (haiku) model; haiku keeps its own slug and still classifies utility.
    """
    models = resolve_claude_code_acp_models(
        "mockllm/model", haiku_model="mockllm/haiku", subagent_model="mockllm/haiku"
    )
    assert models.haiku == "haiku"
    assert models.subagent == "model-subagent"
    assert models.aliases[models.subagent].canonical_name() == "haiku"
    assert models.aliases[models.haiku].canonical_name() == "haiku"
    wrapped = build_claude_code_acp_filter(None, models)
    assert await _invoke(wrapped, models.subagent) == AgentBridgeContext("subagent")
    assert await _invoke(wrapped, models.haiku) == AgentBridgeContext("utility")


async def test_claude_code_acp_subagent_equals_default_stays_distinct() -> None:
    """Explicit subagent_model resolving to the primary: root must not absorb it.

    Without the synthetic slug, `root_slugs` wins in `slug_map_classifier`
    and every sub-agent request stamps "root" (so an `is_root_agent()`-gated
    filter would steer sub-agents).
    """
    models = resolve_claude_code_acp_models(
        "mockllm/model", subagent_model="mockllm/model"
    )
    assert models.subagent == "model-subagent"
    wrapped = build_claude_code_acp_filter(None, models)
    assert await _invoke(wrapped, models.presented) == AgentBridgeContext("root")
    assert await _invoke(wrapped, models.subagent) == AgentBridgeContext("subagent")


async def test_claude_code_acp_haiku_equals_default_is_root_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """haiku_model == default_model is wire-indistinguishable: root, not utility.

    Nothing to rename here (haiku is genuinely the primary slug on the wire),
    and it's a legitimate config, so the classifier must neither stamp
    "utility" nor emit `slug_map_classifier`'s collision warning.
    """
    models = resolve_claude_code_acp_models(
        "mockllm/model", haiku_model="mockllm/model"
    )
    assert models.haiku == models.presented
    with caplog.at_level(logging.WARNING, logger="inspect_swe._util.agentcontext"):
        wrapped = build_claude_code_acp_filter(None, models)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert await _invoke(wrapped, models.presented) == AgentBridgeContext("root")


def test_claude_code_acp_synthetic_subagent_slug_avoids_role_collision() -> None:
    """A role that already presents as '<presented>-subagent' keeps it; subagent takes the next free suffix."""
    models = resolve_claude_code_acp_models(
        "mockllm/model",
        opus_model="mockllm/model-subagent",
        sonnet_model="mockllm/model-subagent-2",
    )
    assert models.opus == "model-subagent"
    assert models.sonnet == "model-subagent-2"
    assert models.subagent == "model-subagent-3"
    assert models.aliases[models.opus].canonical_name() == "model-subagent"
    assert models.aliases[models.subagent] is models.aliases[models.presented]


def test_claude_code_acp_model_instance_preserved_in_aliases() -> None:
    """A caller-supplied Model instance (with its bound config) is aliased as-is, not re-resolved."""
    subagent = get_model("mockllm/subagent")
    models = resolve_claude_code_acp_models("mockllm/model", subagent_model=subagent)
    assert models.aliases[models.subagent] is subagent


# ---------------------------------------------------------------------------
# wiring: ACP codex_cli (interactive_codex_cli)
# ---------------------------------------------------------------------------


async def test_codex_acp_filter_stamps_root_for_default_model() -> None:
    wrapped = build_codex_acp_filter(None, "openai/codex")
    assert await _invoke(wrapped, "openai/codex") == AgentBridgeContext("root")


async def test_codex_acp_filter_stamps_utility_for_guardian_slug() -> None:
    wrapped = build_codex_acp_filter(None, "openai/codex")
    assert await _invoke(wrapped, GUARDIAN_MODEL_SLUG) == AgentBridgeContext("utility")


async def test_codex_acp_filter_stamps_unknown_for_unrecognized_slug() -> None:
    wrapped = build_codex_acp_filter(None, "openai/codex")
    assert await _invoke(wrapped, "some-unrecognized-slug") == AgentBridgeContext(
        "unknown"
    )


async def test_gemini_filter_presented_utility_slug_is_root_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Presenting a model that is also a utility slug is documented under-attribution.

    `gemini-3-pro-preview` is both a legitimate `gemini_model` and the slug
    `loop-detection-double-check` resolves to. Requests under it classify
    "root", the other utility slugs still classify "utility", and building
    the filter must not log a collision warning on every sample.
    """
    with caplog.at_level(logging.WARNING, logger="inspect_swe._util.agentcontext"):
        wrapped = build_gemini_filter(None, "gemini-3-pro-preview")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert await _invoke(wrapped, "gemini-3-pro-preview") == AgentBridgeContext("root")
    assert await _invoke(wrapped, "gemini-3-flash-preview") == AgentBridgeContext(
        "utility"
    )
