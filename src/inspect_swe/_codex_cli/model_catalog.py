"""Map the real bridged model onto a Codex ``--model`` slug.

inspect_swe runs Codex CLI against a bridge that serves the *real* Inspect
model, while Codex independently selects its system prompt and tool set from a
model *catalog* keyed by the ``--model`` slug (longest-prefix match; an unknown
slug falls back to a generic prompt with ``apply_patch`` disabled). To keep
Codex's prompt/tooling aligned with what's actually running, we translate the
real model into the most appropriate catalog slug.

This module is pure (no IO) so the mapping policy can be unit-tested in
isolation; the catalog dict is fetched/cached separately in ``agentbinary.py``.
The resolution returns the chosen slug plus a human-readable ``reason`` (the
bound capabilities and why), which the caller trace-logs.
"""

import re
from dataclasses import dataclass
from typing import Any

# Leading ``gpt-<major>[.<minor>]`` of a model name (ignores any snapshot date or
# ``-codex``/``-mini`` suffix, which trail the version).
_GPT_VERSION_RE = re.compile(r"^gpt-(\d+)(?:\.(\d+))?")

# OpenAI's public Responses API supports ``tool_search`` only on gpt-5.4 and
# later (earlier models return ``400: tool_search not supported with <model>``).
# The bridge always serves the real model via that public API, so this — not
# Codex's catalog, whose ``supports_search_tool`` reflects Codex's own backend —
# is the operative boundary. Every catalog slug bundles ``tool_search`` with
# ``apply_patch`` and ``tool_search`` cannot be disabled independently, so a model
# below this boundary gets Codex's generic fallback (neither tool).
_MIN_TOOL_SEARCH_GPT = (5, 4)


@dataclass(frozen=True)
class CodexModelResolution:
    """The resolved Codex ``--model`` slug plus why it was chosen.

    ``reason`` names the capabilities Codex will bind for ``slug`` and the policy
    branch that selected it; it is intended for trace logging.
    """

    slug: str
    reason: str


def _catalog_models(catalog: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the list of model entries (each with a string ``slug``)."""
    if not catalog:
        return []
    models = catalog.get("models")
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict) and isinstance(m.get("slug"), str)]


def latest_openai_slug(catalog: dict[str, Any] | None) -> str | None:
    """Return the "latest" general coding slug in the catalog.

    "Latest" is the entry with the highest priority (lowest ``priority`` value),
    excluding ``-mini`` variants when any non-mini entry exists. Returns ``None``
    if the catalog has no usable entries.
    """
    models = _catalog_models(catalog)
    if not models:
        return None
    preferred = [m for m in models if not m["slug"].endswith("-mini")] or models
    preferred.sort(key=lambda m: m.get("priority", 1_000_000))
    return str(preferred[0]["slug"])


def _matches_catalog(model_name: str, models: list[dict[str, Any]]) -> bool:
    """Whether Codex would resolve ``model_name`` to a catalog entry.

    Mirrors Codex's longest-prefix matching: a catalog slug is a prefix of the
    requested model name.
    """
    return any(model_name.startswith(m["slug"]) for m in models)


def _gpt_version(name: str) -> tuple[int, int] | None:
    """``(major, minor)`` for a ``gpt-N[.M]`` name, else ``None``.

    Returns ``None`` for non-``gpt`` names (e.g. ``o3``, codenames), so callers
    can treat those separately. ``minor`` defaults to 0 (``gpt-5`` → ``(5, 0)``).
    """
    m = _GPT_VERSION_RE.match(name)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2) or 0))


def _supports_tool_search(model_name: str) -> bool:
    """Whether the real OpenAI model supports ``tool_search`` on the public API.

    True for gpt-5.4+ (including future majors); False for earlier gpt versions
    (incl. sub-5.4 ``-codex`` variants) and non-``gpt`` models (o-series), which
    Codex must serve with its generic fallback. See ``_MIN_TOOL_SEARCH_GPT``.
    """
    version = _gpt_version(model_name)
    return version is not None and version >= _MIN_TOOL_SEARCH_GPT


def _slug_capabilities(slug: str, models: list[dict[str, Any]]) -> str:
    """Human-readable capabilities Codex binds for ``slug`` (for the trace)."""
    entry = next((m for m in models if slug.startswith(m["slug"])), None)
    if entry is None:
        return "generic fallback (no apply_patch, no tool_search)"
    apply_patch = entry.get("apply_patch_tool_type") or "none"
    tool_search = "yes" if entry.get("supports_search_tool") else "no"
    return f"apply_patch={apply_patch}, tool_search={tool_search}"


def resolve_codex_model_slug(
    model_name: str,
    *,
    api: str | None,
    catalog: dict[str, Any] | None,
    override: str | None,
    known_to_inspect: bool,
) -> CodexModelResolution:
    """Resolve the Codex ``--model`` slug for the real bridged model.

    Policy:

    - ``override`` (an explicit ``model_config``) is returned verbatim.
    - An OpenAI model whose name matches a catalog entry returns the model name
      (Codex resolves it to the native prompt + tools).
    - An OpenAI model *not* in the catalog returns the latest catalog slug —
      getting the full coding prompt + ``apply_patch`` + ``tool_search`` — when it
      supports ``tool_search`` (gpt-5.4+, see ``_supports_tool_search``) or is a
      name Inspect doesn't recognize at all (likely a pre-deployment frontier
      model). Otherwise (an established model that predates ``tool_search``, such
      as ``gpt-5`` or ``gpt-5.1-codex``) it passes through to Codex's generic
      fallback, which omits ``tool_search`` (and ``apply_patch``) — every catalog
      slug bundles the two and such models reject ``tool_search``.
    - A non-OpenAI model returns the model name as-is; Codex won't recognize it
      and falls back to its generic prompt (``apply_patch`` disabled), matching
      stock Codex behavior.

    Args:
        model_name: The real model's name without provider prefix (e.g. ``gpt-5``).
        api: The real model's provider/api (e.g. ``openai``).
        catalog: The version-matched Codex model catalog, or ``None`` if
            unavailable (in which case we defer to Codex's own bundled catalog).
        override: Explicit ``model_config`` value, or ``None`` to derive.
        known_to_inspect: Whether Inspect's model database recognizes the model
            (``get_model_info(...) is not None``). A recognized model is an
            established release; an unrecognized one is likely pre-deployment.

    Returns:
        A ``CodexModelResolution`` with the chosen ``slug`` and a ``reason``
        describing the bound capabilities and the policy branch taken.
    """
    if override is not None:
        return CodexModelResolution(
            slug=override,
            reason=f"explicit model_config override '{override}'",
        )

    if api != "openai":
        return CodexModelResolution(
            slug=model_name,
            reason=(
                f"non-openai model '{model_name}' → Codex generic prompt "
                "(no apply_patch)"
            ),
        )

    models = _catalog_models(catalog)

    if _matches_catalog(model_name, models):
        return CodexModelResolution(
            slug=model_name,
            reason=(
                f"openai '{model_name}' matches Codex catalog → native prompt/tools "
                f"({_slug_capabilities(model_name, models)})"
            ),
        )

    latest = latest_openai_slug(catalog)
    if latest is None:
        # catalog unavailable: defer to Codex's own bundled catalog, which decides
        # the prompt/tools for this slug.
        return CodexModelResolution(
            slug=model_name,
            reason=(
                f"Codex catalog unavailable → deferring '{model_name}' to Codex's "
                "bundled catalog"
            ),
        )

    if _supports_tool_search(model_name):
        return CodexModelResolution(
            slug=latest,
            reason=(
                f"openai '{model_name}' absent from catalog but supports tool_search "
                f"(gpt-5.4+) → aliased to latest '{latest}' "
                f"({_slug_capabilities(latest, models)})"
            ),
        )

    if not known_to_inspect:
        return CodexModelResolution(
            slug=latest,
            reason=(
                f"openai '{model_name}' unrecognized by Inspect (likely "
                f"pre-deployment) → aliased to latest '{latest}' "
                f"({_slug_capabilities(latest, models)})"
            ),
        )

    return CodexModelResolution(
        slug=model_name,
        reason=(
            f"openai '{model_name}' predates tool_search support (gpt-5.4+) → "
            "Codex generic prompt (no apply_patch, no tool_search)"
        ),
    )
