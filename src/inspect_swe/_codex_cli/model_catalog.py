"""Map the real bridged model onto a Codex ``--model`` slug.

inspect_swe runs Codex CLI against a bridge that serves the *real* Inspect
model, while Codex independently selects its system prompt and tool set from a
model *catalog* keyed by the ``--model`` slug (longest-prefix match; an unknown
slug falls back to a generic prompt with ``apply_patch`` disabled). To keep
Codex's prompt/tooling aligned with what's actually running, we translate the
real model into the most appropriate catalog slug.

This module is pure (no IO) so the mapping policy can be unit-tested in
isolation; the catalog dict is fetched/cached separately in ``agentbinary.py``.
"""

from typing import Any


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


def resolve_codex_model_slug(
    model_name: str,
    *,
    api: str | None,
    catalog: dict[str, Any] | None,
    override: str | None,
) -> str:
    """Resolve the Codex ``--model`` slug for the real bridged model.

    Policy:

    - ``override`` (an explicit ``model_config``) is returned verbatim.
    - An OpenAI model whose name matches a catalog entry returns the model name
      (Codex resolves it to the native prompt + tools).
    - An OpenAI model *not* in the catalog (e.g. a pre-deployment/unreleased
      model) returns the latest catalog slug, so testing still gets the full
      coding prompt + ``apply_patch``.
    - A non-OpenAI model returns the model name as-is; Codex won't recognize it
      and falls back to its generic prompt (``apply_patch`` disabled), matching
      stock Codex behavior.

    Args:
        model_name: The real model's name without provider prefix (e.g. ``gpt-5``).
        api: The real model's provider/api (e.g. ``openai``).
        catalog: The version-matched Codex model catalog, or ``None`` if
            unavailable (in which case we defer to Codex's own bundled catalog).
        override: Explicit ``model_config`` value, or ``None`` to derive.
    """
    if override is not None:
        return override

    if api == "openai":
        models = _catalog_models(catalog)
        if not _matches_catalog(model_name, models):
            latest = latest_openai_slug(catalog)
            if latest is not None:
                return latest

    return model_name
