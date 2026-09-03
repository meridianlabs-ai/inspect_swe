"""Internal model-name knowledge for Google Gemini CLI.

Shared by both the native (`gemini_cli.py`) and ACP
(`acp/_agents/gemini_cli/gemini_cli.py`) agent variants, since both run the
same gemini-cli binary and are therefore subject to the same internal
routing.

Gemini CLI resolves its own internal utility calls -- loop detection, web
search/fetch, edit correction, next-speaker checks, context snapshotting,
chat compaction -- through an internal `modelConfigService` alias table
*before* the request reaches the wire, so the bridge never sees the alias
key (e.g. `"loop-detection"`); it sees whatever concrete model slug that
alias currently resolves to (e.g. `"gemini-3-flash-preview"`). This module
collects the concrete slugs those aliases are known to resolve to (gathered
from `@google/gemini-cli-core`'s `defaultModelConfigs.js` alias table).

Provenance: `@google/gemini-cli-core@0.58.0` (resolved via `npm view
@google/gemini-cli-core version`, then fetched with `npm pack` and read
directly from `dist/src/config/defaultModelConfigs.js` /
`dist/src/core/baseLlmClient.js` -- first gathered from 0.54.4 on 2026-08-08,
re-verified unchanged against 0.58.0 and upstream main on 2026-09-03).
`gemini_cli(version="auto")` installs whatever is latest, so
`tests/test_gemini_models_drift.py` fetches the alias table from upstream main
and fails naming the alias/slug pair if any utility alias resolves outside
`GEMINI_UTILITY_MODEL_SLUGS`; that test is the signal to refresh this set and
this note.

Two of gemini-cli's utility aliases (`chat-compression-*` family members,
selected by whichever model family is actually presented) always resolve to
*the same concrete slug as whatever model is presented as primary* -- so
they're structurally indistinguishable from root traffic by slug alone and
are intentionally left out of `GEMINI_UTILITY_MODEL_SLUGS` (a
`slug_map_classifier` checks `root_slugs` first anyway, so a same-slug
compaction call is classified "root", not misclassified -- just
under-attributed, which is the best any slug-only classifier can do here).
`"gemini-3-pro-preview"` *is* included below because one utility alias
(`loop-detection-double-check`) resolves to it unconditionally, regardless
of the presented model's family -- making it a genuine (if imperfect, see
above) utility signal whenever the presented model isn't itself
`gemini-3-pro-preview`.

Gemini CLI's subagent feature has no reserved model slug of its own: a
subagent definition's `modelConfig.model` (see `agents/local-executor.js`)
defaults to the *same* `DEFAULT_GEMINI_MODEL` used as gemini-cli's own
primary model when unset, and to whatever else a subagent definition names
it when set. There is no known-name to key a `"subagent"` classification on,
so this module doesn't attempt one.
"""

from typing import Literal

GEMINI_UTILITY_MODEL_SLUGS: frozenset[str] = frozenset(
    {
        # loop-detection, web-search, web-fetch(-fallback), llm-edit-fixer,
        # next-speaker-checker, context-snapshotter, chat-compression-3-flash,
        # agent-history-provider-summarizer: all resolve via the
        # `gemini-3-flash-base` alias.
        "gemini-3-flash-preview",
        # classifier, prompt-completion, fast-ack-helper, edit-corrector,
        # summarizer-default, summarizer-shell, chat-compression-3.1-flash-lite:
        # all resolve via the `flash-lite` alias.
        "gemini-3.1-flash-lite",
        # loop-detection-double-check (unconditional; see module docstring),
        # chat-compression-3-pro, chat-compression-default.
        "gemini-3-pro-preview",
    }
)
"""Concrete model slugs gemini-cli's internal utility calls resolve to."""

GEMINI_UTILITY_MODEL_KINDS: dict[str, Literal["subagent", "utility"]] = {
    slug: "utility" for slug in GEMINI_UTILITY_MODEL_SLUGS
}
"""`GEMINI_UTILITY_MODEL_SLUGS`, shaped for `slug_map_classifier`'s `kind_by_slug`."""
