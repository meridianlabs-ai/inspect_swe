## 0.2.20 (07 October 20)

- Codex CLI now uses 0.44.0 as its default version (since later versions include the `apply_patch` tool which relies on "custom" tool types not currently supported by Inspect).

## 0.2.19 (05 October 2025)

- Automatically use a new port for each unique agent bridge invocation within a sample.
- Added `cached_agent_binaries()` function to list previously downloaded and cached agent binaries.

## 0.2.18 (23 September 2025)

- Update for Claude Code 2.0 (don't call `config list` after installation as it has been removed).
- Update `inspect_ai` requirement to >= 0.3.135.

## 0.2.17 (23 September 2025)

- Update `inspect_ai` requirement to >=0.3.134.

## 0.2.16 (23 September 2025)

- Add support for the `update_plan()` tool for Codex CLI.

## 0.2.15 (23 September 2025)

- Use `gpt-5-codex` as the default model config for Codex CLI (e.g. results in use of the `gpt-5-codex` specific system instructions).

## 0.2.14 (22 September 2025)

- Support multiple attempts for Codex CLI via `codex exec <...> resume --last` (requires Codex v0.36.0 or later).
- Add `retry_refusals` option to set a configurable number of retries for requests refused due to content filtering.
- Update `inspect_ai` requirement to >=0.3.133.

## 0.2.13 (12 September 2025)

- Update `inspect_ai` requirement to >=0.3.132.

## 0.2.12 (08 September 2025)

- Close stdin when running agent binaries (needed for k8s provider to work properly)

## 0.2.11 (06 September 2025)

- Codex CLI: New `codex_cli()` agent for OpenAI Codex.
- Added `filter` parameter to agents for intercepting model generations.

## 0.2.10 (03 September 2025)

- Add trace logging for claude code debug/verbose output.

## v0.2.8 (02 September 2025)

- Claude Code: `allowed_tools` and `disallowed_tools` options.

## v0.2.6 (01 September 2025)

- Claude Code: Add support for multiple agent `attempts`.

## v0.2.5 (01 September 2025)

Initial release.