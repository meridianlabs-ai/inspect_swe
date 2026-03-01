## Unreleased

- Improvements to the structure and content of claude code tool events.

## 0.2.38 (28 February 2026)

- Mini SWE Agent: New `mini_swe_agent()` agent implementation.
- Agent Binaries: Make only a single request for the latest agent binary versions (vs. a request per sample).
- Model aliases: Enable passing `model_aliases` to all agents (passed through to `sandbox_agent_bridge()`).

## 0.2.37 (24 February 2026)

- Claude Code: Capture `stream-json` output to create agent spans within transcript.

## 0.2.36 (23 February 2026)

- Gemini CLI: New `gemini_cli()` agent for Google Gemini.
- Sandbox: Use `exec_remote()` interface for impoved robustness of long running processes.
- Claude Code: Remove `retry_timeouts` options (not longer necessary due to use of `exec_remote()`).
- Claude Code: Add `debug` option to enable `--debug` and `--verbose` CLI flags.
- Claude Code: Pre-seed auth token config to fix silent auth failure in sandbox.

## 0.2.34 (10 February 2026)

- Agent execution: Redirect stdin using `exec 0</dev/null;`
- Agent execution: Print returncode for failed processes.

## 0.2.33 (10 February 2026)

- Claude Code: Look for request timeout errors in stdout as well as stderr.
- Claude Code: Restore default "auto" version behavior (download latest stable version).

## 0.2.32 (31 January 2026)

- Claude Code: Set "auto" version to 2.1.3 (which avoids a not yet fixed compaction bug with gateways).
- Claude Code: Add `retry_timeouts` option for retrying "Request timed out" errors a configurable number of times.
- Claude Code: Add CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS environment variable.

## 0.2.31 (03 January 2026)

- [Centaur Mode](https://meridianlabs-ai.github.io/inspect_swe/claude_code.html#centaur-mode) for running Claude Code and Codex CLI alongside the Inspect [Human Agent](https://inspect.aisi.org.uk/human-agent.html).
- Update Inspect dependency to 0.3.159 (required for human agent integration).

## 0.2.30 (24 December 2025)

- Agent Skills: Support for adding [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to Claude Code and Codex CLI agents.
- Update Inspect dependency to 0.3.158 (required for skills implementation).

## 0.2.29 (22 December 2025)

- Claude Code: Enable explicit specification of `opus_model`, `sonnet_model`, `haiku_model`, and `subagent_model`.
- Update Inspect dependency to 0.3.157 (required for changes to codex cli web search).

## 0.2.28 (05 December 2025)

- Codex CLI and Claude Code: Added `bridged_tools` parameter for exposing host-side Inspect tools to sandboxed agents via MCP protocol.
- Codex CLI: Use GPT 5.1 system prompt by default (includes guidance on using the `update_plan()` tool which is excluded from gpt-5-codex system prompt).
- Codex CLI: Add `config_overrides` parameter for overriding arbitrary config values.
- Update Inspect dependency to 0.3.152 (required for bridged tools).

## 0.2.27 (27 November 2025)

- Codex CLI: Use `RUST_LOG=warning` (rather than `debug`) to reduce amount of output produced by `exec()`.
- Codex CLI: Enable setting a custom `home_dir` (override default of ~).
- Handle multiple `run()` calls to the same agent (resume session and send only new messages in prompt).

## 0.2.26 (15 November 2025)

- Copy agent binary executables to /var/tmp/ (more liberal default permissions)
- Agent binary chmod+x using "root" user for scenarios where they agent isn't root.

## 0.2.25 (07 November 2025)

- Codex CLI: Execute `codex` using the `user` passed to `codex_cli()`.

## 0.2.24 (01 November 2025)

- Claude Code: Pass `model` through to agent bridge to accomodate scenarios where a non-Inspect model is specified.
- Claude Code: Remove `small_model` setting as it conflicts with aforementioned fix to non-Inspect model.

## 0.2.23 (27 October 2025)

- Codex: Eliminate use of `--include-plan-tool` option (no longer available as plan mode is now [always on](https://github.com/openai/codex/pull/5384)).

## 0.2.22 (16 October 2025)

- Codex CLI now uses the latest version of codex, which includes the `apply_patch` tool.
- Update `inspect_ai` requirement to >= 0.3.138.

## 0.2.20 (07 October 2025)

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
