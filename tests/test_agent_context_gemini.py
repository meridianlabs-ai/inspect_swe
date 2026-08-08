"""Tests for `slug_map_classifier` and its gemini-cli, claude_code (ACP), and codex_cli (ACP) wiring.

The three ACP-adapter variants (gemini's `--experimental-acp`,
`claude-agent-acp`, `codex-acp`) have no JSONL/event-stream consumer to
drive attribution, so each wires `slug_map_classifier` directly rather than
reusing a `LiveConsumer`/`CodexConsumer` instance -- see `build_gemini_acp_filter`,
`build_claude_code_acp_filter`, and `build_codex_acp_filter` respectively.
Constructing the real `ACPAgent` subclasses requires an active sample
(`ACPAgent.__init__` calls `sample_active()`), so these tests exercise the
extracted `build_*_filter` functions directly instead -- wiring is exercised
structurally (by reading `_start_agent`) rather than through a live ACP
session.
"""

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
)
from inspect_swe.acp._agents.codex_cli.codex_cli import build_codex_acp_filter
from inspect_swe.acp._agents.gemini_cli.gemini_cli import build_gemini_acp_filter

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
# wiring: ACP gemini_cli (interactive_gemini_cli)
# ---------------------------------------------------------------------------


async def test_gemini_acp_filter_stamps_root_for_primary_slug() -> None:
    wrapped = build_gemini_acp_filter(None, "gemini-3.1-pro-preview")
    assert await _invoke(wrapped, "gemini-3.1-pro-preview") == AgentBridgeContext(
        "root"
    )


async def test_gemini_acp_filter_stamps_utility_for_internal_slug() -> None:
    wrapped = build_gemini_acp_filter(None, "gemini-3.1-pro-preview")
    assert await _invoke(wrapped, "gemini-3.1-flash-lite") == AgentBridgeContext(
        "utility"
    )


async def test_gemini_acp_filter_stamps_unknown_for_unrecognized_slug() -> None:
    wrapped = build_gemini_acp_filter(None, "gemini-3.1-pro-preview")
    assert await _invoke(wrapped, "some-unrecognized-slug") == AgentBridgeContext(
        "unknown"
    )


# ---------------------------------------------------------------------------
# wiring: ACP claude_code (interactive_claude_code)
# ---------------------------------------------------------------------------


async def test_claude_code_acp_filter_stamps_root_for_default_model() -> None:
    wrapped = build_claude_code_acp_filter(None, "anthropic/claude", None, None)
    assert await _invoke(wrapped, "anthropic/claude") == AgentBridgeContext("root")


async def test_claude_code_acp_filter_stamps_subagent_for_subagent_model() -> None:
    wrapped = build_claude_code_acp_filter(
        None, "anthropic/claude", "mockllm/subagent", None
    )
    subagent_slug = get_model("mockllm/subagent").canonical_name()
    assert await _invoke(wrapped, subagent_slug) == AgentBridgeContext("subagent")


async def test_claude_code_acp_filter_stamps_utility_for_haiku_model() -> None:
    wrapped = build_claude_code_acp_filter(
        None, "anthropic/claude", None, "mockllm/haiku"
    )
    haiku_slug = get_model("mockllm/haiku").canonical_name()
    assert await _invoke(wrapped, haiku_slug) == AgentBridgeContext("utility")


async def test_claude_code_acp_filter_unconfigured_subagent_is_not_distinguishable() -> (
    None
):
    """An unconfigured subagent role collides with `default_model` -- root wins.

    Mirrors the non-ACP invariant (`resolve_claude_code_models`): without an
    explicit `subagent_model`, there's no slug to distinguish subagent
    traffic from root traffic, so `kind_by_slug` doesn't carry an entry for
    it and the call classifies as root instead of subagent.
    """
    wrapped = build_claude_code_acp_filter(None, "anthropic/claude", None, None)
    assert await _invoke(wrapped, "anthropic/claude") != AgentBridgeContext("subagent")


async def test_claude_code_acp_filter_stamps_unknown_for_unrecognized_slug() -> None:
    wrapped = build_claude_code_acp_filter(
        None, "anthropic/claude", "mockllm/subagent", "mockllm/haiku"
    )
    assert await _invoke(wrapped, "some-unrecognized-slug") == AgentBridgeContext(
        "unknown"
    )


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
