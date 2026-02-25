## Problem

When running Codex CLI evals with multiple concurrent samples (e.g. `max_tasks=50`), samples fail with `HTTPStatusError: Client error '403 rate limit exceeded'` from the GitHub API. This happens because each sample independently calls `_fetch_latest_stable_version()` and `_fetch_release_assets()`, which hit `https://api.github.com/repos/openai/codex/releases` — an unauthenticated endpoint with a limit of 60 requests/hour.

## What was observed

1. The error occurs during binary installation (`ensure_agent_binary_installed` → `resolve_version` → `_fetch_latest_stable_version`), not during LLM API calls.
2. Inspect AI's built-in rate limiting only covers LLM calls via `model.generate()`. The GitHub API calls in `download_text_file()` use a bare `httpx.AsyncClient()` with no retry logic or backoff.
3. With `max_tasks=50`, up to 100 GitHub API calls are made per eval run (2 per sample: one to list releases, one to fetch the specific release). This quickly exhausts the 60 req/hour unauthenticated limit.
4. The existing concurrency lock (`concurrency("codex-install", 1)`) serializes binary downloads but doesn't prevent repeated version resolution calls — each sample still resolves the version independently.
5. The equivalent Claude Code agent is unaffected because it resolves versions from Google Cloud Storage, which has no such rate limit.

## Change

Cache the results of `_fetch_latest_stable_version()` and `_fetch_release_assets()` in module-level variables, protected by an `asyncio.Lock`. The first sample to call these functions hits the GitHub API and caches the result; all subsequent samples return the cached value immediately.

This reduces GitHub API calls from 2×N (where N = number of samples) to exactly 2 per eval run. The cache lives for the process lifetime, so each new eval run starts fresh and picks up any new Codex releases.
