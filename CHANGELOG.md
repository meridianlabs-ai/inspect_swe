## Unreleased

- Support multiple attempts for Codex CLI via `codex exec <...> resume --last` (requires Codex v0.36.0 or later).

## 0.2.12 (08 September 2026)

- Close stdin when running agent binaries (needed for k8s provider to work properly)

## 0.2.11 (06 September 2026)

- Codex CLI: New `codex_cli()` agent for OpenAI Codex.
- Added `filter` parameter to agents for intercepting model generations.

## 0.2.10 (03 September 2026)

- Add trace logging for claude code debug/verbose output.

## v0.2.8 (02 September 2026)

- Claude Code: `allowed_tools` and `disallowed_tools` options.

## v0.2.6 (01 September 2025)

- Claude Code: Add support for multiple agent `attempts`.

## v0.2.5 (01 September 2025)

Initial release.