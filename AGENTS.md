# AGENTS.md

This file provides guidance to AI coding agents working in this repository.

## Build/Lint/Test Commands

- Install dev environment: `pip install -e ".[dev]"`, or with uv: `make sync`
- Format, lint, and type check: `make check` (runs `ruff format`, `ruff check --fix`, `mypy`)
- Run fast tests: `make test` (or `pytest`)
- Run a single test: `pytest tests/test_file.py::test_name -v`

## Type Checking

- mypy runs with `strict = true` over `src` and `tests`. All functions need type annotations, and an unused `type: ignore` comment is a hard error (`warn_unused_ignores`).
- CI installs the latest unpinned mypy, so a new mypy release can turn `main` red with no code change (e.g. a `type: ignore` that a newer mypy considers unused). If the mypy check fails on a line your diff didn't touch, check whether `main` has the same failure before debugging your change — and if so, fix it in your PR and note that it was pre-existing.

## Test Suite Topology

- Three opt-in gates skip tests by default: `--runslow` (integration tests that drive real agents end-to-end), `--runapi` (tests needing model API access), and `--runflaky`. PR CI runs only the fast suite; the slow suite runs nightly via `meridianlabs-ai/actions` ("Inspect SWE Nightly Tests").
- Slow tests call real model APIs (skipped without the relevant `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GOOGLE_API_KEY`) and need a sandbox: docker and/or k8s are auto-detected from your environment and sandbox-dependent tests are parametrized over whatever is available. `pytest --co` shows what would actually run.
- The end-to-end example tests all route through the `run_example` helper in `tests/conftest.py`, which bounds each eval with `time_limit` and `token_limit` (see the comments there for the rationale before changing them). The nightly runs pytest with `--timeout=900`, so any per-eval time limit must stay comfortably under that or a timeout surfaces as an opaque pytest kill instead of a clean eval failure.

## Pull Requests

- Title PRs as Conventional Commits (`<type>: <description>`)—we squash-merge, so the PR title becomes the commit message that drives releases; `pr-title-lint` enforces it
- `feat:`/`fix:` are for user-facing changes only: they headline the release notes and bump the version. `perf:`/`revert:` also appear in the notes (no bump); `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` are hidden
- In the description part of the title, state the user-facing outcome — the problem a user hit or the capability they gain — not the mechanism of the fix: `fix: agent hang when sandbox startup races container pull`, not `fix: add lock around container init`. For changes with no user-facing outcome (refactoring, CI, docs), describe the change itself.
- Body lines starting with `<type>:` are parsed as extra changelog entries—don't begin description lines with a conventional-commit prefix unless that's intended
- Never edit `CHANGELOG.md`, version numbers, or `.release-please-manifest.json`—Release Please owns them
- After opening a PR, don't stop at creation: watch its checks (`gh pr checks <number> --watch`) until they complete, report the outcome, and investigate and fix any failures. If the branch falls behind `main`, update it so CI runs against current code.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines
