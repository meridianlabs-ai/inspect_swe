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

# Sentinel ``--model`` slug used to force Codex's generic fallback (no apply_patch,
# no tool_search) for a real model that IS present in the catalog but predates the
# public-API tool_search boundary. Passing the real slug would make Codex bind the
# catalog's ``supports_search_tool`` (true for sub-5.4 entries like ``gpt-5.2``),
# emitting a ``tool_search`` the public API rejects. An unknown slug → Codex's
# ``model_info_from_slug`` fallback (warn, not error; apply_patch/tool_search off).
# Chosen so no catalog slug is a prefix of it (Codex matches by longest prefix).
_GENERIC_FALLBACK_SLUG = "inspect-generic"


@dataclass(frozen=True)
class CodexModelResolution:
    """The resolved Codex ``--model`` slug plus why it was chosen.

    ``reason`` names the capabilities Codex will bind for ``slug`` and the policy
    branch that selected it; it is intended for trace logging.
    """

    slug: str
    reason: str


def is_openai_derived_api(model_api: object) -> bool:
    """Whether a model's ``ModelAPI`` is (a subclass of) Inspect's ``OpenAIAPI``.

    This identifies models that are *really* OpenAI — including custom providers
    (e.g. a pre-deployment stand-in) that subclass ``OpenAIAPI`` under a different
    registry name. It deliberately does NOT match ``OpenAICompatibleAPI`` (the
    sibling base used by Ollama/OpenRouter/etc. to serve non-OpenAI models over
    the OpenAI wire format).

    Matched by class name across the MRO so we don't couple to the private
    ``inspect_ai.model._providers.openai`` import path.
    """
    return any(cls.__name__ == "OpenAIAPI" for cls in type(model_api).__mro__)


def is_latest_openai_model(model_api: object) -> bool:
    """Whether an OpenAI-derived provider flags this as a "latest"/codename model.

    Delegates to ``OpenAIAPI.is_latest()``, which treats a name matching none of
    OpenAI's known conventions (not gpt/o-series/codex/deep-research, excluding
    azure/bedrock) as the current frontier — i.e. a pre-deployment or codename
    model (e.g. ``otter``, ``foo-bar-22``). Inherited by custom subclasses, so it
    covers both the real OpenAI provider pointed at a pre-release name and a
    custom provider stand-in.

    Returns False for non-OpenAI-derived providers (and is robust if the method
    is ever absent). Unlike inspecting the model database (``get_model_info``),
    this is immune to ``set_model_info``: describing a pre-deployment model's
    context window doesn't make it look like an established release.
    """
    if not is_openai_derived_api(model_api):
        return False
    is_latest = getattr(model_api, "is_latest", None)
    return bool(is_latest()) if callable(is_latest) else False


def openai_service_model_name(model_api: object, fallback: str) -> str:
    """The model identity an OpenAI-derived provider declares, else ``fallback``.

    OpenAI-derived providers report their true model via ``service_model_name()``
    (e.g. a custom ``otter`` provider that returns ``gpt-5.5``); align to that
    rather than the registry name (``otter``), which Codex wouldn't recognize.
    Returns ``fallback`` for providers that aren't OpenAI-derived or lack the
    method. This mirrors the identity ``is_latest_openai_model`` reads internally.
    """
    if not is_openai_derived_api(model_api):
        return fallback
    service_model_name = getattr(model_api, "service_model_name", None)
    return service_model_name() if callable(service_model_name) else fallback


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


def _matched_entry(
    model_name: str, models: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The catalog entry Codex would resolve ``model_name`` to, or ``None``.

    Mirrors Codex's longest-prefix matching: a catalog slug must be a prefix of the
    requested model name; the longest such slug wins.
    """
    matches = [m for m in models if model_name.startswith(m["slug"])]
    if not matches:
        return None
    return max(matches, key=lambda m: len(m["slug"]))


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
    entry = _matched_entry(slug, models)
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
    is_latest: bool,
) -> CodexModelResolution:
    """Resolve the Codex ``--model`` slug for the real bridged model.

    Policy:

    - ``override`` (an explicit ``model_config``) is returned verbatim.
    - An OpenAI model whose name matches a catalog entry returns the model name
      (Codex resolves it to the native prompt + tools) — *unless* that entry would
      bind ``tool_search`` while the real model predates the public-API boundary
      (gpt-5.4+), in which case it returns ``_GENERIC_FALLBACK_SLUG`` to force
      Codex's generic prompt (no ``apply_patch``, no ``tool_search``) and avoid a
      ``tool_search`` 400. (Catalog ``supports_search_tool`` reflects Codex's own
      backend, not the public Responses API the bridge uses.)
    - An OpenAI model *not* in the catalog returns the latest catalog slug —
      getting the full coding prompt + ``apply_patch`` + ``tool_search`` — when it
      supports ``tool_search`` (gpt-5.4+, see ``_supports_tool_search``) or is a
      "latest"/codename model (``is_latest``, a pre-deployment frontier model).
      Otherwise (an established model that predates ``tool_search``, such as
      ``gpt-5``, ``gpt-5.1-codex``, or ``o3``) it passes through to Codex's
      generic fallback, which omits ``tool_search`` (and ``apply_patch``).
    - A non-OpenAI model returns the model name as-is; Codex won't recognize it
      and falls back to its generic prompt (``apply_patch`` disabled), matching
      stock Codex behavior.

    Args:
        model_name: The real model's name without provider prefix (e.g. ``gpt-5``).
        api: The real model's provider/api (e.g. ``openai``).
        catalog: The version-matched Codex model catalog, or ``None`` if
            unavailable (in which case we defer to Codex's own bundled catalog).
        override: Explicit ``model_config`` value, or ``None`` to derive.
        is_latest: Whether the provider flags this as a "latest"/codename model
            (see ``is_latest_openai_model``) — a pre-deployment frontier model, so
            an otherwise-unrecognized name aliases to latest instead of degrading
            to the generic fallback. This signal is immune to ``set_model_info``
            (unlike model-database membership).

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
    # tool_search is exposed by every catalog slug; on the public Responses API
    # (which the bridge uses) only gpt-5.4+ — or a codename treated as the current
    # frontier — accept it. A model below this boundary must never be given a
    # tool_search-bearing slug, or the request 400s.
    tool_search_ok = _supports_tool_search(model_name) or is_latest

    matched = _matched_entry(model_name, models)
    if matched is not None:
        if matched.get("supports_search_tool") and not tool_search_ok:
            # catalog would bind tool_search, but this real model predates the
            # public-API boundary → force generic fallback to avoid a 400.
            return CodexModelResolution(
                slug=_GENERIC_FALLBACK_SLUG,
                reason=(
                    f"openai '{model_name}' is in the Codex catalog but predates "
                    "tool_search support (gpt-5.4+) → forcing generic prompt via "
                    f"'{_GENERIC_FALLBACK_SLUG}' (no apply_patch, no tool_search) to "
                    "avoid a tool_search 400"
                ),
            )
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

    if tool_search_ok:
        if _supports_tool_search(model_name):
            why = "supports tool_search (gpt-5.4+)"
        else:
            why = "is a latest/codename model (likely pre-deployment)"
        return CodexModelResolution(
            slug=latest,
            reason=(
                f"openai '{model_name}' absent from catalog but {why} → aliased to "
                f"latest '{latest}' ({_slug_capabilities(latest, models)})"
            ),
        )

    return CodexModelResolution(
        slug=model_name,
        reason=(
            f"openai '{model_name}' predates tool_search support (gpt-5.4+) → "
            "Codex generic prompt (no apply_patch, no tool_search)"
        ),
    )
