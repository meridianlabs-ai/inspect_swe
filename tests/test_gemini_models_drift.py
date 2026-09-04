"""Live drift check for `inspect_swe._gemini_cli.models` against gemini-cli upstream.

`gemini_cli(version="auto")` installs whatever gemini-cli is latest, but the
slug knowledge in `inspect_swe._gemini_cli.models` is a snapshot of one
release's `defaultModelConfigs`. This fetches `defaultModelConfigs.ts` from
upstream main and checks two things against it:

* `GEMINI_UTILITY_MODEL_SLUGS`: each internal utility alias is resolved to
  its concrete model slug the way gemini-cli's `modelConfigService` does, and
  the test fails naming the alias/slug pair if any lands outside the snapshot.
* `GEMINI_PRIMARY_ALIAS_TARGETS`: for each primary-model alias (`auto`,
  `flash`, ...) the `modelIdResolutions` entry's `default` plus every
  `contexts[].target` must be a subset of the snapshot's set, and every
  non-concrete key in that table must have an entry; the test fails naming
  the alias and the missing target(s).

Skips when offline; github-action-gated so an upstream hiccup can't block
unrelated CI.
"""

import re
from dataclasses import dataclass

import anyio
import pytest
from inspect_swe._gemini_cli.models import (
    GEMINI_PRIMARY_ALIAS_TARGETS,
    GEMINI_UTILITY_MODEL_SLUGS,
)

from tests.conftest import skip_if_github_action

_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/google-gemini/gemini-cli/main/"
    "packages/core/src/config/defaultModelConfigs.ts"
)

# Every alias in `DEFAULT_MODEL_CONFIGS.aliases` that gemini-cli's internal
# machinery (not the chat loop) sends requests under. Excludes the
# `chat-compression-*` family, which resolves to the presented model's own
# family and is intentionally absent from GEMINI_UTILITY_MODEL_SLUGS (see the
# models.py docstring).
_UTILITY_ALIASES = (
    "loop-detection",
    "loop-detection-double-check",
    "web-search",
    "web-fetch",
    "web-fetch-fallback",
    "llm-edit-fixer",
    "next-speaker-checker",
    "context-snapshotter",
    "agent-history-provider-summarizer",
    "classifier",
    "prompt-completion",
    "fast-ack-helper",
    "edit-corrector",
    "summarizer-default",
    "summarizer-shell",
)

# Keys in `modelIdResolutions` that are concrete model ids (routing entries
# for themselves) rather than primary aliases.
_CONCRETE_PREFIXES = ("gemini-", "gemma-")


@dataclass(frozen=True)
class _Alias:
    extends: str | None
    model: str | None


@dataclass(frozen=True)
class _Route:
    """One `modelIdResolutions` entry: its `default` and every reachable target."""

    default: str
    targets: frozenset[str]
    """`default` plus every `contexts[].target`."""


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", src, flags=re.MULTILINE)


def _balanced_body(src: str, open_idx: int) -> str:
    """Return the text inside the `{` at `open_idx` up to its matching `}`."""
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_idx + 1 : i]
    raise ValueError("unbalanced braces")


def _section(src: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*:\s*\{{", src)
    assert match, f"no `{name}` section in upstream defaultModelConfigs.ts"
    return _balanced_body(src, match.end() - 1)


_KEY = re.compile(r"(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][\w.-]*))\s*:\s*\{")


def _top_level_entries(body: str) -> dict[str, str]:
    """Split an object literal body into its immediate `key: {...}` entries."""
    entries: dict[str, str] = {}
    pos = 0
    while (match := _KEY.search(body, pos)) is not None:
        key = match.group(1) or match.group(2) or match.group(3)
        inner = _balanced_body(body, match.end() - 1)
        entries[key] = inner
        pos = match.end() + len(inner) + 1
    return entries


def _first_str(body: str, field: str) -> str | None:
    match = re.search(rf"\b{field}\s*:\s*['\"]([^'\"]+)['\"]", body)
    return match.group(1) if match else None


def _all_str(body: str, field: str) -> list[str]:
    return re.findall(rf"\b{field}\s*:\s*['\"]([^'\"]+)['\"]", body)


def _parse(src: str) -> tuple[dict[str, _Alias], dict[str, _Route]]:
    src = _strip_comments(src)
    aliases = {
        name: _Alias(_first_str(body, "extends"), _first_str(body, "model"))
        for name, body in _top_level_entries(_section(src, "aliases")).items()
    }
    routing = {
        name: _Route(default, frozenset({default, *_all_str(body, "target")}))
        for name, body in _top_level_entries(
            _section(src, "modelIdResolutions")
        ).items()
        if (default := _first_str(body, "default")) is not None
    }
    return aliases, routing


def _resolve(
    name: str,
    aliases: dict[str, _Alias],
    routing: dict[str, _Route],
    seen: tuple[str, ...] = (),
) -> str | None:
    """Follow `model` -> `extends` -> routing-table default to a concrete slug.

    A name that resolves back to itself (the concrete-slug aliases such as
    `gemini-3-flash-preview`, whose `model` is their own key) is the concrete
    slug. A chain that dead-ends on an alias with neither `model` nor
    `extends` (`base`) is unresolved -> None.
    """
    if name in seen:
        return name
    seen = (*seen, name)
    if (alias := aliases.get(name)) is not None:
        if alias.model is not None:
            return _resolve(alias.model, aliases, routing, seen)
        if alias.extends is not None:
            return _resolve(alias.extends, aliases, routing, seen)
        return None
    if (route := routing.get(name)) is not None:
        return _resolve(route.default, aliases, routing, seen)
    return name


# Fixture-shaped excerpt mirroring upstream's structure, so the parser itself
# is covered offline (the live tests below only tell us about drift).
_SAMPLE_TS = """
/** license */
export const DEFAULT_MODEL_CONFIGS: ModelConfigServiceConfig = {
  aliases: {
    base: { modelConfig: { generateContentConfig: { temperature: 0 } } },
    // chat alias whose model is its own key
    'gemini-3-flash-preview': {
      extends: 'chat-base-3',
      modelConfig: { model: 'gemini-3-flash-preview' },
    },
    'gemini-3.1-flash-lite': {
      extends: 'chat-base-3',
      modelConfig: { model: 'gemini-3.1-flash-lite' },
    },
    'gemini-3-flash-base': {
      extends: 'base',
      modelConfig: { model: 'gemini-3-flash-preview' },
    },
    classifier: {
      extends: 'base',
      modelConfig: { model: 'flash-lite', generateContentConfig: {} },
    },
    'loop-detection': { extends: 'gemini-3-flash-base', modelConfig: {} },
    'loop-detection-double-check': {
      extends: 'base',
      modelConfig: { model: 'gemini-3-pro-preview' },
    },
  },
  overrides: [{ match: { model: 'chat-base', isRetry: true }, modelConfig: {} }],
  modelIdResolutions: {
    'flash-lite': { default: 'gemini-3.1-flash-lite' },
    'gemini-3-pro-preview': {
      default: 'gemini-3-pro-preview',
      contexts: [{ condition: { hasAccessToPreview: false }, target: 'x' }],
    },
    // primary alias with a default and several conditional targets
    flash: {
      default: 'gemini-3-flash-preview',
      contexts: [
        { condition: { useGemini3_5Flash: true }, target: 'gemini-3.5-flash' },
        {
          condition: { hasAccessToPreview: false },
          target: 'gemini-2.5-flash',
        },
      ],
    },
  },
  // a sibling table with the same key shape that must NOT be picked up
  classifierIdResolutions: {
    flash: {
      default: 'gemini-3-flash-preview',
      contexts: [{ condition: { requestedModels: ['a'] }, target: 'not-me' }],
    },
  },
};
"""


def test_alias_resolution_follows_extends_model_and_routing_table() -> None:
    aliases, routing = _parse(_SAMPLE_TS)
    assert set(aliases) == {
        "base",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-base",
        "classifier",
        "loop-detection",
        "loop-detection-double-check",
    }
    assert {name: route.default for name, route in routing.items()} == {
        "flash-lite": "gemini-3.1-flash-lite",
        "gemini-3-pro-preview": "gemini-3-pro-preview",
        "flash": "gemini-3-flash-preview",
    }
    # extends -> base alias -> model that is its own chat alias
    assert _resolve("loop-detection", aliases, routing) == "gemini-3-flash-preview"
    # model is a routing-table key -> default -> its own chat alias
    assert _resolve("classifier", aliases, routing) == "gemini-3.1-flash-lite"
    # model is a concrete slug present only in the routing table
    assert _resolve("loop-detection-double-check", aliases, routing) == (
        "gemini-3-pro-preview"
    )
    # dead end: no model, no extends
    assert _resolve("base", aliases, routing) is None


def test_routing_table_parser_collects_default_and_every_context_target() -> None:
    _, routing = _parse(_SAMPLE_TS)
    # default plus every contexts[].target, as a set
    assert routing["flash"].targets == {
        "gemini-3-flash-preview",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    }
    # no contexts -> just the default
    assert routing["flash-lite"].targets == {"gemini-3.1-flash-lite"}
    # a concrete id's own entry is parsed too (callers filter by key prefix)
    assert routing["gemini-3-pro-preview"].targets == {"gemini-3-pro-preview", "x"}
    # the sibling `classifierIdResolutions` table is not merged in
    assert "not-me" not in routing["flash"].targets


def _fetch_upstream() -> str:
    from inspect_swe._util.download import download_text_file

    try:
        return anyio.run(download_text_file, _UPSTREAM_URL)
    except Exception as ex:
        pytest.skip(f"live gemini-cli source unavailable: {ex}")


@skip_if_github_action
def test_live_gemini_utility_aliases_resolve_into_known_slugs() -> None:
    aliases, routing = _parse(_fetch_upstream())
    missing = [alias for alias in _UTILITY_ALIASES if alias not in aliases]
    assert not missing, (
        f"upstream defaultModelConfigs.ts no longer defines utility aliases "
        f"{missing}; update _UTILITY_ALIASES and re-check "
        f"GEMINI_UTILITY_MODEL_SLUGS (see the models.py docstring)"
    )

    resolved = {alias: _resolve(alias, aliases, routing) for alias in _UTILITY_ALIASES}
    drifted = {
        alias: slug
        for alias, slug in resolved.items()
        if slug not in GEMINI_UTILITY_MODEL_SLUGS
    }
    assert not drifted, (
        "gemini-cli utility aliases now resolve to slugs outside "
        f"GEMINI_UTILITY_MODEL_SLUGS={sorted(GEMINI_UTILITY_MODEL_SLUGS)}: "
        f"{drifted} (None = chain dead-ended). Add the new slug(s) to "
        "GEMINI_UTILITY_MODEL_SLUGS and refresh the provenance note in "
        "src/inspect_swe/_gemini_cli/models.py."
    )


@skip_if_github_action
def test_live_gemini_primary_alias_targets_cover_routing_table() -> None:
    _, routing = _parse(_fetch_upstream())

    upstream_aliases = {
        name for name in routing if not name.startswith(_CONCRETE_PREFIXES)
    }
    unmapped = sorted(upstream_aliases - GEMINI_PRIMARY_ALIAS_TARGETS.keys())
    assert not unmapped, (
        f"upstream modelIdResolutions declares primary aliases {unmapped} with "
        "no entry in GEMINI_PRIMARY_ALIAS_TARGETS; add them (default plus every "
        "contexts[].target) in src/inspect_swe/_gemini_cli/models.py."
    )

    gone = sorted(GEMINI_PRIMARY_ALIAS_TARGETS.keys() - routing.keys())
    assert not gone, (
        f"GEMINI_PRIMARY_ALIAS_TARGETS names aliases {gone} that upstream "
        "modelIdResolutions no longer routes; drop or rename them."
    )

    missing_targets = {
        alias: sorted(routing[alias].targets - known)
        for alias, known in GEMINI_PRIMARY_ALIAS_TARGETS.items()
        if not routing[alias].targets <= known
    }
    assert not missing_targets, (
        "gemini-cli primary aliases now resolve to targets missing from "
        f"GEMINI_PRIMARY_ALIAS_TARGETS: {missing_targets}. Add them in "
        "src/inspect_swe/_gemini_cli/models.py and refresh the provenance note."
    )
