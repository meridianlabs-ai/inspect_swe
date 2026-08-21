# Changelog

## [0.2.71](https://github.com/meridianlabs-ai/inspect_swe/compare/0.2.70...0.2.71) (2026-08-21)


### Features

* **codex_cli:** configure sandbox and approval policies ([#88](https://github.com/meridianlabs-ai/inspect_swe/issues/88)) ([a038ece](https://github.com/meridianlabs-ai/inspect_swe/commit/a038ece49a6f296fb52625aa3b875efa61a3572b))


### Bug Fixes

* default Claude Code env to disable SDK retry watchdogs behind the bridge ([#99](https://github.com/meridianlabs-ai/inspect_swe/issues/99)) ([0ccc009](https://github.com/meridianlabs-ai/inspect_swe/commit/0ccc009694cc6f75023bf1dffffcf119885a5eef))
* gate agent launch on bridged MCP readiness and block Claude Code's first model call on MCP connect ([#106](https://github.com/meridianlabs-ai/inspect_swe/issues/106)) ([cd03b95](https://github.com/meridianlabs-ai/inspect_swe/commit/cd03b957dc7ecdd0fddbef0c2794e078f043b1e5))
* make bridged MCP readiness timeouts configurable ([#123](https://github.com/meridianlabs-ai/inspect_swe/issues/123)) ([d63ca56](https://github.com/meridianlabs-ai/inspect_swe/commit/d63ca5699c72033272e35a655731424d5f4c2e5d))
* resolve npm agent versions once per process to avoid GitHub rate limits ([#127](https://github.com/meridianlabs-ai/inspect_swe/issues/127)) ([b4a988b](https://github.com/meridianlabs-ai/inspect_swe/commit/b4a988b5823bf515879018fef3ee7db24b5509d9))
* retry agent binary downloads and raise the http timeout ([#130](https://github.com/meridianlabs-ai/inspect_swe/issues/130)) ([cdf1010](https://github.com/meridianlabs-ai/inspect_swe/commit/cdf10108e271bcf5f00fd9668c1ab519458c7cdd))

## [0.2.70](https://github.com/meridianlabs-ai/inspect_swe/compare/0.2.69...0.2.70) (2026-08-09)


### Features

* pass web_search to sandbox_agent_bridge based on web search config ([#116](https://github.com/meridianlabs-ai/inspect_swe/issues/116)) ([5ba7549](https://github.com/meridianlabs-ai/inspect_swe/commit/5ba754967343786f79e8c59808bce2df531daaf5))

## [0.2.69](https://github.com/meridianlabs-ai/inspect_swe/compare/0.2.68...0.2.69) (2026-08-08)


### Bug Fixes

* **codex_cli:** require inspect_ai&gt;=0.3.253 for Multi-Agent V2 delegation ([#109](https://github.com/meridianlabs-ai/inspect_swe/issues/109)) ([249479d](https://github.com/meridianlabs-ai/inspect_swe/commit/249479d31b79f2f481cebcab0f47dbe040ca30f6))
* **codex_cli:** support Codex Multi-Agent V2 — package-archive install and sub-agent span reconstruction ([#107](https://github.com/meridianlabs-ai/inspect_swe/issues/107)) ([51adaf0](https://github.com/meridianlabs-ai/inspect_swe/commit/51adaf05990ef33825ea06fc9e38103935235cf2))

## [0.2.68](https://github.com/meridianlabs-ai/inspect_swe/compare/0.2.67...0.2.68) (2026-08-04)


### Features

* **antigravity:** add native Antigravity SDK agent ([#94](https://github.com/meridianlabs-ai/inspect_swe/issues/94)) ([7d85ee9](https://github.com/meridianlabs-ai/inspect_swe/commit/7d85ee9bacc9955068c4333018067d726b7ce2c9))
* support replacing Claude Code system prompt ([#98](https://github.com/meridianlabs-ai/inspect_swe/issues/98)) ([58638bc](https://github.com/meridianlabs-ai/inspect_swe/commit/58638bcb4cef191fd5bff7b1a961792f0d16e190))

## [0.2.67](https://github.com/meridianlabs-ai/inspect_swe/compare/0.2.66...0.2.67) (2026-07-29)


### Features

* **claude_code:** add `permission_mode` option (deprecates `auto_mode`) ([5cc9e6d](https://github.com/meridianlabs-ai/inspect_swe/commit/5cc9e6d9a92ed71efc9d72d075528c598b4498d4))
* **claude_code:** control static MCP allowlisting ([#91](https://github.com/meridianlabs-ai/inspect_swe/issues/91)) ([5cc9e6d](https://github.com/meridianlabs-ai/inspect_swe/commit/5cc9e6d9a92ed71efc9d72d075528c598b4498d4))
* **codex_cli:** add auto_review option for automated approval review ([#102](https://github.com/meridianlabs-ai/inspect_swe/issues/102)) ([cc854fd](https://github.com/meridianlabs-ai/inspect_swe/commit/cc854fd00f6939f40d2004eb6e7d077283d8af18))
* **kimi:** add `kimi_code()` agent with Inspect bridging, MCP, skills, and Centaur support ([#73](https://github.com/meridianlabs-ai/inspect_swe/issues/73)) ([7fca27f](https://github.com/meridianlabs-ai/inspect_swe/commit/7fca27f4481a02e39122efd0fd98b306274251bb))


### Bug Fixes

* **claude_code:** correct "all" tools MCP allowlist wildcard to ([5cc9e6d](https://github.com/meridianlabs-ai/inspect_swe/commit/5cc9e6d9a92ed71efc9d72d075528c598b4498d4))
* **codex_cli:** refresh bundled model catalog snapshot to track live latest ([#95](https://github.com/meridianlabs-ai/inspect_swe/issues/95)) ([596be22](https://github.com/meridianlabs-ai/inspect_swe/commit/596be229bb3efb47adaab0edbb2997b7ae3c7d68))

## [0.2.66](https://github.com/meridianlabs-ai/inspect_swe/compare/0.2.65...0.2.66) (2026-07-14)


### Bug Fixes

* update ACP client for agent-client-protocol 0.11.0 compatibility ([#76](https://github.com/meridianlabs-ai/inspect_swe/issues/76)) ([#82](https://github.com/meridianlabs-ai/inspect_swe/issues/82)) ([22328dc](https://github.com/meridianlabs-ai/inspect_swe/commit/22328dc0082e0ed49f751d0f905ee249f640db70))

## 0.2.65 (05 July 2026)

- Codex CLI and Claude Code: Support checkpointing and resuming runs via `checkpointer()`, restoring session/attempt state across resumes.
- Claude Code: Handle Anthropic refusals gracefully — a content-filter refusal now scores the sample as incorrect and continues the eval (matching native Inspect solvers) instead of raising an error that aborts the run.
- Gemini CLI (ACP): Register bridged tools and MCP servers with the CLI (write `settings.json` and pass `--allowed-mcp-server-names`) so host-side tools are available to the agent.
- All agents: When no `cwd` is specified and the sandbox's default working directory is `/` (i.e. the image has no `WORKDIR`), run the agent in the user's home directory instead of the container root.

## 0.2.63 (10 June 2026)

- Remove ACP patch for connection initialization order issue (resolved in ACP 0.10.1).
- Codex CLI: Resolve "stable"/"latest" via the GitHub `releases/latest` endpoint — the full releases listing for openai/codex frequently 504s.

## 0.2.62 (05 June 2026)

- Claude Code: Present the real served model as the agent's own identity (e.g. its "You are powered by the model …" system prompt) instead of the bridge sentinel. Added a `model_config` parameter to override the presented identity.
- Claude Code: `opus_model` / `sonnet_model` / `haiku_model` / `subagent_model` are now actually honored — previously the bridge fallback collapsed them onto a single model.
- Tracing: Only trace full session output when `debug` option is passed.

## 0.2.61 (03 June 2026)

- Claude Code: Fix system prompt being re-sent on resumed turns.
- Codex CLI: Align `OpenAIAPI`-derived custom providers by their declared `service_model_name()`.
- Codex CLI: Fall back to a bundled model catalog when the version-matched `models.json` can't be fetched.

## 0.2.60 (02 June 2026)

- Update Inspect AI dependency to 0.3.234

## 0.2.59 (31 May 2026)

- Codex CLI: Detect and apply agent spans automatically from Codex event stream.

## 0.2.57 (30 May 2026)

- Codex CLI: Align the system prompt and tool set with the real bridged model. `model_config` now defaults to `None`, deriving Codex's `--model` slug from the actual model. Pass an explicit `model_config` to override.
- Codex CLI: Add options for enabling/disabling `web_search` and `goals` tools (both default to enabled).
- Codex CLI: Improve MCP tests to check for correct tool names and assert results.
- Gemini CLI: Explicity set auth type to "gemini-api-key" (required for recent versions of the CLI).

## 0.2.56 (25 May 2026)

- OpenCode: Install ripgrep for access to native skills.
- Claude Code: Improved live event capture (no longer requires injection of spans into transcript).
- Claude Code: Support for importing transcripts with new JSONL sub-agent file layout.
- Claude Code (ACP): Use officially supported ACP client (@agentclientprotocol/claude-agent-acp)

## 0.2.55 (16 May 2026)

- Workaround for `agent_client_protocol` v0.10.0 connection bug.

## 0.2.54 (13 May 2026)

- Claude Code (ACP): Allocate per-invocation bridge port.

## 0.2.53 (13 May 2026)

- Pin `agent-client-protocol>=0.9.0,<0.10` until [init ordering bug](https://github.com/agentclientprotocol/python-sdk/issues/97) is fixed.

## 0.2.52 (09 May 2026)

- OpenCode: New agent backend wrapping [OpenCode](https://github.com/anomalyco/opencode).
- Codex CLI: Disable telemetry by default.
- Gemini CLI: Disable telemetry by default.

## 0.2.51 (07 May 2026)

- Gemini CLI: Fix MCP registration via GEMINI_CLI_TRUST_WORKSPACE.
- Mini SWE Agent: Ensure that pip is available before attempting installation.

## 0.2.50 (29 April 2026)

- Codex CLI: Run ACP mode with approval_policy: never and sandbox_mode: danger_full_access
- Codex CLI: Fix for MCP tool calling (bump to Inspect v0.3.214 which has the fix).

## 0.2.48 (26 April 2026)

- Update download location for Claude Code binaries.

## 0.2.47 (13 April 2026)

- Codex CLI: Set 60 minute timeout for OpenAI streaming requests (default was 5 minutes).
- Gemini CLI: Use `gemini-3.1-pro-preview` as model for tests.
- Claude Code: Always use `--resume` for re-attaching to sessions (`--continue` sometimes drops history).

## 0.2.46 (26 March 2026)

- Update to latest ACP types.

## 0.2.45 (20 March 2026)

- ACP protocol improvements.

## 0.2.44 (18 March 2026)

- Claude Code: Retry uncaught errors (unexpected crashes of scaffold) up to 3 times by default.
- Claude Code: Retry refusals up to 3 times by default.

## 0.2.43 (16 March 2026)

- Improve ACP error reporting for adapter failures.

## 0.2.42 (09 March 2026)

- Claude Code: Improved detection of final agent span messages.
- Claude Code: Capture stdout and stderr to store in debug mode.

## 0.2.41 (04 March 2026)

- Claude Code: Annotate event stream with agent spans.
- Support for sub-agents defined using agent teams.
- Attach agent_span_id to tool events for agent spawning tools.
- Pass `sandbox` argument to `sandbox_agent_bridge()`.

## 0.2.40 (02 March 2026)

- Claude Code: Revert streaming events while we refine our approach.
- Use normal heading/body for task tool views.

## 0.2.39 (01 March 2026)

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
