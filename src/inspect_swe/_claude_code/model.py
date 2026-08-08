"""Model identity and bridge-alias resolution for the Claude Code agent.

Claude Code's ``--model`` (and the ``ANTHROPIC_*`` model env vars) are purely
cosmetic: they select the identity Claude Code presents to itself (its
"You are powered by the model ..." prompt) and any model-gated client behavior.
The real model is reached through the bridge. This module bundles that
resolution so ``claude_code()`` stays readable.
"""

from dataclasses import dataclass

from inspect_ai.model import Model, get_model


@dataclass(frozen=True)
class ClaudeCodeModels:
    """Resolved presented identities + bridge routing for a Claude Code run.

    ``presented`` and the per-role names (``opus``/``sonnet``/``haiku``/
    ``subagent``) are the *displayed* ids handed to Claude Code via ``--model``
    and the ``ANTHROPIC_*`` env vars — cosmetic only. ``aliases`` maps each
    presented name to its served ``Model`` so the bridge routes it to the real
    model; ``bridge_model`` is the sentinel fallback for any id the inner agent
    emits that isn't one of those names.

    ``subagent`` is always distinct from ``presented``, ``opus``, ``sonnet``,
    and ``haiku`` (never inherits any of them, even when the caller left
    ``subagent_model`` unset) — see the invariant enforced in
    ``resolve_claude_code_models``. This lets the bridge tell subagent
    traffic apart from main-thread and small-fast/utility traffic by
    requested slug alone.
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
    opus/sonnet/haiku role inherits the primary presented name unless it is
    set, in which case it gets its own name *and* its own alias so it
    actually routes to its intended model (the bridge sentinel fallback would
    otherwise collapse them onto the main model).

    The subagent role is different: it never inherits ``presented``,
    ``opus``, ``sonnet``, or ``haiku`` as-is. An unset ``subagent_model``
    still routes to the same served model as the primary, but is
    *presented* as ``"<presented>-subagent"`` so that ``models.subagent``
    never collides with any of the other role names (Claude Code requests
    subagent traffic using this slug, and small-fast/opus/sonnet traffic
    using their own, so a collision would make one role's traffic
    indistinguishable from another's at the bridge). Caller-supplied
    ``model_aliases`` take precedence over the names we derive.

    Note: must be called at execution time — ``get_model()`` resolves the active
    model from the current eval/sample context.
    """
    served_model = get_model(model)
    presented = model_config if model_config is not None else served_model.name
    aliases: dict[str, str | Model] = {presented: served_model}

    def role_name(role_model: str | None) -> str:
        # an unset role inherits the primary presented name (routing via its
        # alias); a set role registers its own name + alias so it routes to its
        # own model
        if role_model is None:
            return presented
        role = get_model(role_model)
        aliases[role.name] = role
        return role.name

    opus = role_name(opus_model)
    sonnet = role_name(sonnet_model)
    haiku = role_name(haiku_model)

    # Subagent traffic is structurally distinguishable at the wire (probe P1
    # of the agent-bridge-context plan, live-verified against CC 2.1.220):
    # every Task-tool subagent request — including custom agents with their
    # own `model:` front-matter — arrives carrying CLAUDE_CODE_SUBAGENT_MODEL's
    # value as its raw requested slug, while main-thread requests carry
    # ANTHROPIC_MODEL's value (`presented`), small-fast/utility traffic
    # (when distinguishable at all) carries ANTHROPIC_SMALL_FAST_MODEL's
    # value (`haiku`), and opus/sonnet swaps carry their own role's value.
    # That only works as a classifier if `subagent` doesn't collide with
    # ANY of those other slugs — a collision with `haiku` in particular
    # would make the classifier's subagent branch shadow its utility
    # branch for every small-fast request (see the LiveConsumer.classify
    # truth table). So `models.subagent` not in
    # `{presented, opus, sonnet, haiku}` is enforced here as an invariant
    # rather than left to fall out of whatever the caller configured:
    #
    # - an unset `subagent_model` would otherwise inherit `presented`
    #   verbatim (same as `role_name`'s None branch) — give it a synthetic
    #   "<presented>-subagent" name instead, aliased to the SAME served
    #   model, so routing is byte-for-byte unchanged and only the presented
    #   label differs.
    # - an explicit `subagent_model` that happens to *resolve* to the same
    #   name as `presented`, `opus`, `sonnet`, or `haiku` hits the identical
    #   collision (the degenerate case: the caller deliberately points
    #   subagents at one of the other roles — e.g. the natural "cheap model
    #   for background AND subagents" config of setting `subagent_model`
    #   equal to `haiku_model`) — apply the same synthetic suffix, aliased
    #   to the caller's resolved subagent model rather than `served_model`
    #   (respecting the caller's explicit choice even though the two
    #   currently coincide).
    #
    # Either way `aliases[presented]` (set above to `served_model`) is left
    # untouched — only a new alias key is added. opus/sonnet/haiku must
    # already be resolved above so all four names are known here.
    if subagent_model is None:
        subagent_name = presented
        subagent_route: Model = served_model
    else:
        subagent_route = get_model(subagent_model)
        subagent_name = subagent_route.name

    if subagent_name in {presented, opus, sonnet, haiku}:
        subagent = f"{presented}-subagent"
        aliases[subagent] = subagent_route
    else:
        subagent = subagent_name
        aliases[subagent] = subagent_route

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
