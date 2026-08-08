"""Tests for OpenCode's agent-context classification via config-injected slugs.

OpenCode has no live consumer/event stream to drive attribution (unlike
codex_cli/claude_code), so — per probe P2 (agent-bridge-context plan,
live-verified 2026-08-08, per-agent `model` config IS honored) —
`opencode()` injects sentinel model slugs into the generated global
config (`build_opencode_config_overrides`) and classifies purely by the
requested slug (`build_opencode_filter`, backed by the shared
`slug_map_classifier`). See the module docstring in
`inspect_swe._opencode.opencode` for the full sentinel-choice rationale,
including this task's own live-verification run: a first attempt using a
non-catalog synthetic sentinel (`anthropic/inspect-subagent`) was REJECTED
by OpenCode's runtime ("Model not found") even though the config *schema*
places no catalog constraint on the field, so the shipped sentinels
(`_SENTINEL_MODELS`) are real, distinct, same-provider catalog ids — and a
follow-up spec review (caveat 4): since sentinels are fixed real model ids,
a caller's `opencode_model` can legitimately collide with one, so
`_SENTINEL_MODELS` holds an ORDERED per-role candidate list and
`build_opencode_config_overrides` selects around collisions with the
primary (and, for the small-model role, the chosen subagent sentinel too).
"""

from typing import Any

from inspect_ai.agent import AgentBridgeContext, current_agent_bridge_context
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_swe._opencode.opencode import (
    _SENTINEL_MODELS,
    OPENCODE_BUILTIN_SUBAGENTS,
    OPENCODE_UTILITY_AGENTS,
    _bare_model_id,
    _select_sentinel,
    build_opencode_config,
    build_opencode_config_overrides,
    build_opencode_filter,
    build_opencode_model_aliases,
)
from inspect_swe._util.agentcontext import ModelFilter

_ANTHROPIC_PRIMARY = "anthropic/claude-sonnet-4-5"

# ---------------------------------------------------------------------------
# _bare_model_id
# ---------------------------------------------------------------------------


def test_bare_model_id_strips_provider_prefix() -> None:
    assert _bare_model_id("anthropic/claude-sonnet-4-5") == "claude-sonnet-4-5"


def test_bare_model_id_passes_through_unprefixed() -> None:
    assert _bare_model_id("claude-sonnet-4-5") == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# _select_sentinel
# ---------------------------------------------------------------------------


def test_select_sentinel_picks_first_non_excluded_candidate() -> None:
    assert _select_sentinel(("a", "b", "c"), ("a",)) == "b"


def test_select_sentinel_returns_none_when_all_excluded() -> None:
    assert _select_sentinel(("a", "b", "c"), ("a", "b", "c")) is None


def test_select_sentinel_no_exclusions_picks_first() -> None:
    assert _select_sentinel(("a", "b", "c"), ()) == "a"


# ---------------------------------------------------------------------------
# build_opencode_config_overrides — config-generation (no collision)
# ---------------------------------------------------------------------------


def test_config_overrides_set_small_model_to_distinct_sentinel() -> None:
    config, _subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "anthropic", _ANTHROPIC_PRIMARY
    )
    assert small_model_sentinel is not None
    assert config["small_model"] == small_model_sentinel
    assert small_model_sentinel.startswith("anthropic/")


def test_config_overrides_route_builtin_subagents_to_subagent_sentinel() -> None:
    config, subagent_sentinel, _small_model_sentinel = build_opencode_config_overrides(
        "anthropic", _ANTHROPIC_PRIMARY
    )
    assert subagent_sentinel is not None
    for name in OPENCODE_BUILTIN_SUBAGENTS:
        # only `model` is set on built-ins — no prompt/description/mode
        assert config["agent"][name] == {"model": subagent_sentinel}


def test_config_overrides_route_utility_agents_to_small_model_sentinel() -> None:
    config, _subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "anthropic", _ANTHROPIC_PRIMARY
    )
    assert small_model_sentinel is not None
    for name in OPENCODE_UTILITY_AGENTS:
        assert config["agent"][name] == {"model": small_model_sentinel}


def test_config_overrides_agent_map_covers_exactly_builtins_and_utility_agents() -> (
    None
):
    config, _subagent_sentinel, _small_model_sentinel = build_opencode_config_overrides(
        "anthropic", _ANTHROPIC_PRIMARY
    )
    assert set(config["agent"]) == set(OPENCODE_BUILTIN_SUBAGENTS) | set(
        OPENCODE_UTILITY_AGENTS
    )


def test_config_overrides_use_configured_provider_id() -> None:
    """Sentinels are prefixed with whichever provider `opencode_model` names.

    So the sentinel requests still resolve through the provider entry whose
    baseURL was overridden to the bridge.
    """
    config, subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "openai", "openai/gpt-5"
    )
    assert subagent_sentinel is not None and small_model_sentinel is not None
    assert subagent_sentinel.startswith("openai/")
    assert small_model_sentinel.startswith("openai/")
    assert config["agent"]["general"]["model"].startswith("openai/")


def test_config_overrides_subagent_and_small_sentinels_are_distinct() -> None:
    _config, subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "anthropic", _ANTHROPIC_PRIMARY
    )
    assert subagent_sentinel != small_model_sentinel


def test_config_overrides_default_picks_are_real_catalog_ids_not_synthetic() -> None:
    """Regression guard for the live-verified rejection.

    A first attempt used a non-catalog synthetic id
    (`anthropic/inspect-subagent`) and OpenCode's runtime rejected it
    ("Model not found") even though nothing in the config schema forbids
    it. `_SENTINEL_MODELS`' default (first) candidates must never regress
    to a fabricated id — this is the pair actually live-verified.
    """
    _config, subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "anthropic", _ANTHROPIC_PRIMARY
    )
    assert subagent_sentinel == "anthropic/claude-haiku-4-5-20251001"
    assert small_model_sentinel == "anthropic/claude-3-5-haiku-20241022"


def test_config_overrides_unrecognized_provider_skips_injection() -> None:
    """No sentinel table entry -> no injection attempt (avoids guessing a

    catalog id and risking the same "Model not found" failure mode).
    """
    config, subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "some-unrecognized-provider", "some-unrecognized-provider/some-model"
    )
    assert config == {}
    assert subagent_sentinel is None
    assert small_model_sentinel is None


# ---------------------------------------------------------------------------
# build_opencode_config_overrides — primary-collision guard (caveat 4)
# ---------------------------------------------------------------------------


def test_config_overrides_primary_equals_default_subagent_sentinel_falls_back() -> None:
    """`opencode_model` == the default subagent sentinel -> pick the alternate.

    The exact scenario spec review flagged: a caller configures
    `opencode_model="anthropic/claude-haiku-4-5-20251001"` (our default
    subagent sentinel). The picked sentinel must differ from the primary,
    and the config must reflect the alternate, not the (colliding) default.
    """
    config, subagent_sentinel, _small_model_sentinel = build_opencode_config_overrides(
        "anthropic", "anthropic/claude-haiku-4-5-20251001"
    )
    assert subagent_sentinel is not None
    assert subagent_sentinel != "anthropic/claude-haiku-4-5-20251001"
    assert subagent_sentinel == "anthropic/claude-3-5-haiku-20241022"
    assert config["agent"]["general"] == {"model": subagent_sentinel}


def test_config_overrides_primary_equals_default_small_model_sentinel_falls_back() -> (
    None
):
    """Mirror case for the small-model role."""
    config, _subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "anthropic", "anthropic/claude-3-5-haiku-20241022"
    )
    assert small_model_sentinel is not None
    assert small_model_sentinel != "anthropic/claude-3-5-haiku-20241022"
    assert config["small_model"] == small_model_sentinel
    assert config["agent"]["title"] == {"model": small_model_sentinel}


def test_config_overrides_primary_collides_with_all_subagent_candidates_omits_override(
    monkeypatch: Any,
) -> None:
    """Degenerate case: primary IS every subagent candidate -> omit entirely.

    Not reachable with the real (distinct) candidate lists via a single
    `opencode_model` value, so this drives the full
    `build_opencode_config_overrides` pipeline with a monkeypatched
    candidate table where every subagent candidate collides with the
    primary -- confirming the override is omitted (not just that
    `_select_sentinel` returns `None` in isolation) and that `agent`/
    `small_model` never reference a colliding id.
    """
    primary = "anthropic/claude-sonnet-4-5"
    primary_bare = "claude-sonnet-4-5"
    monkeypatch.setitem(
        _SENTINEL_MODELS,
        "anthropic",
        (
            (primary_bare, primary_bare, primary_bare),  # every candidate collides
            (
                "claude-3-5-haiku-20241022",
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229",
            ),
        ),
    )

    config, subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "anthropic", primary
    )
    assert subagent_sentinel is None
    assert "general" not in config.get("agent", {})
    assert "explore" not in config.get("agent", {})
    assert "scout" not in config.get("agent", {})
    # small-model role is unaffected by the subagent role's degeneracy
    assert small_model_sentinel is not None
    assert small_model_sentinel != primary


def test_config_overrides_small_model_avoids_chosen_subagent_sentinel() -> None:
    """Small-model selection also excludes whatever subagent sentinel was chosen.

    Force the subagent pick onto the small-model role's *default* candidate
    by colliding the primary with the subagent role's default, then confirm
    the small-model pick still avoids the (now-shared-candidate-list)
    collision with whichever subagent sentinel ended up chosen.
    """
    config, subagent_sentinel, small_model_sentinel = build_opencode_config_overrides(
        "anthropic", "anthropic/claude-haiku-4-5-20251001"
    )
    assert subagent_sentinel is not None and small_model_sentinel is not None
    assert small_model_sentinel != subagent_sentinel
    assert config["agent"]["general"]["model"] != config["agent"]["title"]["model"]


# ---------------------------------------------------------------------------
# build_opencode_model_aliases
# ---------------------------------------------------------------------------


def test_model_aliases_route_both_sentinels_to_served_model() -> None:
    served = get_model("mockllm/model")
    aliases = build_opencode_model_aliases(
        served, None, "anthropic/inspect-subagent", "anthropic/inspect-small"
    )
    assert aliases["inspect-subagent"] is served
    assert aliases["inspect-small"] is served


def test_model_aliases_caller_supplied_take_precedence() -> None:
    served = get_model("mockllm/model")
    override = get_model("mockllm/other")
    aliases = build_opencode_model_aliases(
        served,
        {"inspect-subagent": override},
        "anthropic/inspect-subagent",
        "anthropic/inspect-small",
    )
    assert aliases["inspect-subagent"] is override
    # untouched sentinel still routes to the served model
    assert aliases["inspect-small"] is served


def test_model_aliases_preserve_unrelated_caller_entries() -> None:
    served = get_model("mockllm/model")
    other = get_model("mockllm/other")
    aliases = build_opencode_model_aliases(
        served,
        {"some-other-slug": other},
        "anthropic/inspect-subagent",
        "anthropic/inspect-small",
    )
    assert aliases["some-other-slug"] is other
    assert aliases["inspect-subagent"] is served


def test_model_aliases_skip_none_sentinels() -> None:
    """Unrecognized-provider case: no sentinel entries are added at all."""
    served = get_model("mockllm/model")
    aliases = build_opencode_model_aliases(served, None, None, None)
    assert aliases == {}


# ---------------------------------------------------------------------------
# build_opencode_config — full-dict snapshot
# ---------------------------------------------------------------------------
#
# Guards the actual config written to the sandbox (execute() calls this
# function verbatim -- see opencode.py). Without a test observing the
# ASSEMBLED dict, a future refactor could drop the `agent_context_config`
# spread and silently kill agent-context classification while every other
# test (which only exercises the builder functions in isolation) stayed
# green.


def test_build_opencode_config_full_snapshot() -> None:
    agent_context_config, _subagent_sentinel, _small_model_sentinel = (
        build_opencode_config_overrides("anthropic", _ANTHROPIC_PRIMARY)
    )
    mcp_servers = [
        MCPServerConfigHTTP(
            type="http", name="test-server", url="http://example.com/mcp"
        )
    ]

    config = build_opencode_config(
        "anthropic",
        "http://localhost:3001/v1",
        agent_context_config,
        True,  # skills_enabled
        mcp_servers,
    )

    assert config == {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "anthropic": {"options": {"baseURL": "http://localhost:3001/v1"}},
        },
        "small_model": "anthropic/claude-3-5-haiku-20241022",
        "agent": {
            "general": {"model": "anthropic/claude-haiku-4-5-20251001"},
            "explore": {"model": "anthropic/claude-haiku-4-5-20251001"},
            "scout": {"model": "anthropic/claude-haiku-4-5-20251001"},
            "title": {"model": "anthropic/claude-3-5-haiku-20241022"},
            "summary": {"model": "anthropic/claude-3-5-haiku-20241022"},
            "compaction": {"model": "anthropic/claude-3-5-haiku-20241022"},
        },
        "permission": {"skill": {"*": "allow"}},
        "mcp": {
            "test-server": {
                "enabled": True,
                "type": "remote",
                "url": "http://example.com/mcp",
            }
        },
    }


def test_build_opencode_config_without_skills_or_mcp_omits_those_keys() -> None:
    config = build_opencode_config(
        "anthropic", "http://localhost:3001/v1", {}, False, []
    )
    assert config == {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "anthropic": {"options": {"baseURL": "http://localhost:3001/v1"}},
        },
    }
    assert "permission" not in config
    assert "mcp" not in config
    assert "agent" not in config
    assert "small_model" not in config


# ---------------------------------------------------------------------------
# build_opencode_filter — classifier wiring
# ---------------------------------------------------------------------------

_SUBAGENT_SENTINEL = "anthropic/claude-haiku-4-5-20251001"
_SMALL_MODEL_SENTINEL = "anthropic/claude-3-5-haiku-20241022"


async def _invoke(wrapped: ModelFilter, slug: str | None) -> AgentBridgeContext | None:
    with bridged_request_scope(slug):
        await wrapped(
            get_model("mockllm/model"),
            [ChatMessageUser(content="hi")],
            [],
            None,
            GenerateConfig(),
        )
        return current_agent_bridge_context()


def _filter(
    user_filter: Any = None,
    opencode_model: str = _ANTHROPIC_PRIMARY,
    subagent_sentinel: str | None = _SUBAGENT_SENTINEL,
    small_model_sentinel: str | None = _SMALL_MODEL_SENTINEL,
) -> ModelFilter:
    return build_opencode_filter(
        user_filter, opencode_model, subagent_sentinel, small_model_sentinel
    )


async def test_opencode_filter_stamps_root_for_primary_slug() -> None:
    wrapped = _filter()
    assert await _invoke(wrapped, "claude-sonnet-4-5") == AgentBridgeContext("root")


async def test_opencode_filter_stamps_subagent_for_sentinel_slug() -> None:
    wrapped = _filter()
    assert await _invoke(wrapped, "claude-haiku-4-5-20251001") == AgentBridgeContext(
        "subagent"
    )


async def test_opencode_filter_stamps_utility_for_small_model_sentinel_slug() -> None:
    wrapped = _filter()
    assert await _invoke(wrapped, "claude-3-5-haiku-20241022") == AgentBridgeContext(
        "utility"
    )


async def test_opencode_filter_stamps_unknown_for_unrecognized_slug() -> None:
    wrapped = _filter()
    assert await _invoke(wrapped, "some-unrecognized-slug") == AgentBridgeContext(
        "unknown"
    )


async def test_opencode_filter_stamps_unknown_with_no_bridge_request_info() -> None:
    wrapped = _filter()
    assert await _invoke(wrapped, None) == AgentBridgeContext("unknown")


async def test_opencode_filter_root_slug_matches_bare_id_not_provider_prefixed() -> (
    None
):
    """Root slug must be checked bare — OpenCode never sends the provider prefix."""
    wrapped = _filter()
    assert await _invoke(wrapped, "anthropic/claude-sonnet-4-5") != AgentBridgeContext(
        "root"
    )


async def test_opencode_filter_tracks_configured_opencode_model() -> None:
    """The root slug tracks whatever `opencode_model` was configured, not a fixed default."""
    wrapped = _filter(
        opencode_model="openai/gpt-5", subagent_sentinel=None, small_model_sentinel=None
    )
    assert await _invoke(wrapped, "gpt-5") == AgentBridgeContext("root")


async def test_opencode_filter_delegates_to_user_filter() -> None:
    seen: dict[str, Any] = {}

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> None:
        seen["ctx"] = current_agent_bridge_context()
        return None

    wrapped = _filter(user_filter)
    result = await _invoke(wrapped, "claude-haiku-4-5-20251001")
    assert result == AgentBridgeContext("subagent")
    assert seen["ctx"] == AgentBridgeContext("subagent")


async def test_opencode_filter_without_sentinels_falls_back_to_root_or_unknown() -> (
    None
):
    """Unrecognized-provider case: no subagent/utility classification is possible.

    `kind_by_slug` is empty, so anything other than the primary slug is
    "unknown" (never misclassified as subagent/utility).
    """
    wrapped = build_opencode_filter(None, "some-provider/some-model", None, None)
    assert await _invoke(wrapped, "some-model") == AgentBridgeContext("root")
    assert await _invoke(wrapped, "some-other-model") == AgentBridgeContext("unknown")


# ---------------------------------------------------------------------------
# build_opencode_filter — root_slugs/kind_by_slug disjointness guard (caveat 4)
# ---------------------------------------------------------------------------


async def test_opencode_filter_drops_sentinel_colliding_with_root_instead_of_misclassifying() -> (
    None
):
    """Defense-in-depth: a directly-constructed collision degrades safely.

    `build_opencode_config_overrides` never hands `build_opencode_filter` a
    sentinel equal to the primary (it selects around exactly this), but
    `build_opencode_filter` is independently importable/callable, so it
    enforces the invariant itself: if a caller passes a subagent sentinel
    equal to `opencode_model`'s bare id anyway, the request must still
    classify "root" (matching the actual wire slug) rather than "subagent" --
    root_slugs is checked first by design, so this is what "the colliding
    kind_by_slug entry is dropped" cashes out to.
    """
    wrapped = build_opencode_filter(
        None,
        _ANTHROPIC_PRIMARY,
        subagent_sentinel=_ANTHROPIC_PRIMARY,  # collides with opencode_model
        small_model_sentinel=_SMALL_MODEL_SENTINEL,
    )
    assert await _invoke(wrapped, "claude-sonnet-4-5") == AgentBridgeContext("root")
    # the small-model sentinel, uninvolved in the collision, still works
    assert await _invoke(wrapped, "claude-3-5-haiku-20241022") == AgentBridgeContext(
        "utility"
    )


async def test_opencode_filter_drops_small_model_sentinel_colliding_with_root() -> None:
    """Mirror case for the small-model sentinel."""
    wrapped = build_opencode_filter(
        None,
        _ANTHROPIC_PRIMARY,
        subagent_sentinel=_SUBAGENT_SENTINEL,
        small_model_sentinel=_ANTHROPIC_PRIMARY,  # collides with opencode_model
    )
    assert await _invoke(wrapped, "claude-sonnet-4-5") == AgentBridgeContext("root")
    assert await _invoke(wrapped, "claude-haiku-4-5-20251001") == AgentBridgeContext(
        "subagent"
    )
