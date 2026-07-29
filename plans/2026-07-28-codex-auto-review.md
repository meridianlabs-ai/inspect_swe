# Codex CLI `auto_review` Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `auto_review` option to `codex_cli()` (and `interactive_codex_cli()`) that enables Codex CLI's automated approval review (guardian) feature.

**Architecture:** `auto_review` is *not* a code-review pass — it is Codex's automated approval adjudicator: a "guardian" model session that approves/denies sandbox-escalation requests (network access, out-of-workspace writes, etc.) that would otherwise prompt a human. Enabling it requires re-engaging Codex's own sandbox machinery, so when `auto_review` is on we (a) stop passing `--dangerously-bypass-approvals-and-sandbox`, and (b) configure `approval_policy = "on-request"`, `sandbox_mode = "workspace-write"`, `approvals_reviewer = "auto_review"` via both config.toml and `-c` overrides (the existing `goals`/`web_search` pattern). The guardian's model calls arrive at the Inspect sandbox bridge under the slug `codex-auto-review`; by default they fall through to the task's main model, and an optional `model` field binds them to an Inspect model role or explicit model via `model_aliases`.

**Tech Stack:** Python 3.13, pydantic v2, inspect_ai (`sandbox_agent_bridge`, `model_roles`, `get_model`), pytest, mypy `--strict` (covers `tests/` too), ruff.

---

## Background facts (verified 2026-07-28)

An implementing engineer needs these; they are not guessable from the repo:

- **What auto_review does (upstream):** With `approval_policy = "on-request"`, Codex's own OS-level sandbox (Landlock/seccomp on Linux) restricts commands; when the model requests escalation (e.g. network access, writes outside the workspace, MCP tool approval), a separately-prompted guardian session assesses the action and returns allow/deny. Fails closed on timeout (90s)/malformed output. A circuit breaker interrupts the turn after 3 consecutive denials (or 10 in the last 50 reviews). Docs: https://developers.openai.com/codex/concepts/sandboxing/auto-review
- **Upstream config surface:** top-level `approvals_reviewer = "user" | "auto_review"` (default `"user"`); optional `[auto_review]` table with a single `policy` string (extra instructions injected into the guardian prompt); feature flag `[features] guardian_approval` (Stable, default-on). There is **no** dedicated CLI flag; `-c approvals_reviewer="auto_review"` works. Codex's own TUI "Approve for me" preset sets exactly: `approval_policy = on-request` + `approvals_reviewer = auto_review` + workspace-write sandbox — we mirror that trio.
- **Why the bypass flag must be dropped:** codex-rs applies typed CLI-flag overrides (`ConfigOverrides`) *after* `-c` key-value overrides (`approval_policy_override.or(cfg.approval_policy)` in `codex-rs/core/src/config/mod.rs`), so `--dangerously-bypass-approvals-and-sandbox` (which forces `approval_policy=never` + `sandbox_mode=danger-full-access`) cannot be beaten by `-c`. It must simply not be passed.
- **Headless support:** `codex exec` normally forces `approval_policy=never`, but since upstream PR #23763 it re-resolves without that override when the configured reviewer is `auto_review`. **First release containing this: `rust-v0.137.0`** (verified by probing `codex-rs/exec/src/lib.rs` for `preserve_headless_approval_policy` across release tags; 0.136.0 lacks it, 0.137.0 has it, `rust-v0.137.1` does not exist). Older versions silently disable the feature headlessly — hence the version gate below.
- **Guardian model routing:** the guardian session's requests hit our bridge under model slug `codex-auto-review` (upstream `DEFAULT_APPROVAL_REVIEW_PREFERRED_MODEL`; present in our bundled catalog at `src/inspect_swe/_codex_cli/_bundled_catalog.py:63`). Bridge resolution order (`inspect_ai/agent/_bridge/util.py:329-347`): `model_aliases` exact match → fallback to the bridge model → names matching a defined Inspect model role resolve via `get_model(role=...)`. So with no alias, the guardian is served by the task's main model — a safe default.
- **Observability:** upstream `GuardianAssessment` events are transient (never written to rollout files, dropped by exec JSON output). The guardian's *model calls* still flow through our bridge, so they appear in the Inspect transcript as ordinary model events.
- **Runtime requirement:** `workspace-write` engages Codex's Landlock sandbox inside the Inspect container — requires Linux kernel ≥ 5.13 with Landlock enabled. Documented in the docs task; not enforced in code.

## Decisions locked in

1. **API:** `auto_review: bool | CodexAutoReview = False` on `codex_cli()`; `CodexAutoReview(policy: str | None, model: str | Model | None)` pydantic options type exported from the package root (mirrors `CentaurOptions`).
2. Enabling `auto_review` implicitly re-engages Codex's sandbox (`workspace-write`) + on-request approvals — the only semantics under which the feature does anything. Prominently documented.
3. `policy` is emitted **only** into config.toml (never as `-c`) to avoid shell/TOML escaping of multiline strings. The other keys are emitted both ways per the existing belt-and-suspenders pattern.
4. Guardian model binding: a `model` string naming a defined Inspect model role binds via `get_model(role=...)`; any other string or `Model` instance is passed through as a `model_aliases` value. Default (`None`): main task model serves guardian calls.
5. Version gate: `RuntimeError` if the installed codex binary is `< 0.137.0` and `auto_review` is enabled (skip check if version undetectable). No gate in the ACP variant (adapter manages its own codex).
6. Ships as PR titled `feat(codex_cli): add auto_review option for automated approval review`. **Never edit CHANGELOG.md** (Release Please owns it).

## File structure

- Modify: `src/inspect_swe/_util/toml.py` — escape `\n`/`\r`/`\t` in strings (needed for multiline `policy`)
- Modify: `src/inspect_swe/_codex_cli/config.py` — `CodexAutoReview`, resolvers, extended emitters
- Modify: `src/inspect_swe/_codex_cli/codex_cli.py` — parameter, version gate, conditional bypass flag, guardian aliases
- Modify: `src/inspect_swe/acp/_agents/codex_cli/codex_cli.py` — same parameter for the ACP/interactive agent
- Modify: `src/inspect_swe/__init__.py` — export `CodexAutoReview`
- Modify: `docs/codex_cli.qmd` — options row + "Auto Review" section
- Modify: `tests/test_codex_config.py` — unit tests
- Create: `tests/test_codex_auto_review.py` — live integration test (slow; Docker + model API)

Run everything with the project venv: `uv run pytest ...`, `make check` (ruff format + ruff check --fix + mypy strict).

---

### Task 1: TOML emitter — escape control characters in strings

`_format_value` in `src/inspect_swe/_util/toml.py:33-38` escapes only `\` and `"`. A multiline guardian `policy` would emit a literal newline inside a TOML basic string, which is invalid TOML.

**Files:**
- Modify: `src/inspect_swe/_util/toml.py:33-38`
- Test: `tests/test_codex_config.py` (already imports `to_toml`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codex_config.py`:

```python
def test_to_toml_escapes_control_characters() -> None:
    toml = to_toml({"policy": 'line one\nline "two"\ttabbed'})
    assert toml == 'policy = "line one\\nline \\"two\\"\\ttabbed"'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codex_config.py::test_to_toml_escapes_control_characters -v`
Expected: FAIL — assertion error (output contains a raw newline).

- [ ] **Step 3: Implement the escaping**

In `src/inspect_swe/_util/toml.py`, replace the `str` branch of `_format_value`:

```python
    if isinstance(value, str):
        # Escape special characters and quote
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS (including pre-existing tests — the new escapes don't affect strings without control chars).

- [ ] **Step 5: Commit**

```bash
git add src/inspect_swe/_util/toml.py tests/test_codex_config.py
git commit -m "fix: escape control characters in TOML string values"
```

---

### Task 2: `CodexAutoReview` options type and resolver

**Files:**
- Modify: `src/inspect_swe/_codex_cli/config.py`
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codex_config.py` (extend the existing `inspect_swe._codex_cli.config` import block with `CodexAutoReview`, `resolve_codex_auto_review`):

```python
def test_resolve_codex_auto_review_false_is_none() -> None:
    assert resolve_codex_auto_review(False) is None


def test_resolve_codex_auto_review_true_is_defaults() -> None:
    resolved = resolve_codex_auto_review(True)
    assert resolved == CodexAutoReview()
    assert resolved is not None
    assert resolved.policy is None
    assert resolved.model is None


def test_resolve_codex_auto_review_passes_through_options() -> None:
    options = CodexAutoReview(policy="Deny all network access.")
    assert resolve_codex_auto_review(options) is options
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_codex_config.py -v -k auto_review`
Expected: FAIL — `ImportError: cannot import name 'CodexAutoReview'`.

- [ ] **Step 3: Implement the type and resolver**

In `src/inspect_swe/_codex_cli/config.py`, add imports at the top:

```python
from inspect_ai.model import Model, get_model, model_roles
from pydantic import BaseModel, ConfigDict, Field
```

Then add after the `CodexWebSearch` alias:

```python
class CodexAutoReview(BaseModel):
    """Options for Codex automated approval review (`auto_review`).

    When enabled, Codex runs with its own sandbox active (`workspace-write`)
    and `approval_policy` set to `on-request`; escalation requests are
    adjudicated by a guardian model session rather than a human.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: str | None = Field(default=None)
    """Additional policy instructions inserted into the guardian review prompt."""

    model: str | Model | None = Field(default=None)
    """Model that serves guardian review requests.

    A `str` naming an Inspect model role binds that role; any other string is
    treated as a model name. Defaults to `None`, which serves guardian
    requests with the task's main model.
    """


def resolve_codex_auto_review(
    auto_review: bool | CodexAutoReview,
) -> CodexAutoReview | None:
    if auto_review is False:
        return None
    if auto_review is True:
        return CodexAutoReview()
    return auto_review
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codex_config.py -v -k auto_review`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_swe/_codex_cli/config.py tests/test_codex_config.py
git commit -m "feat: add CodexAutoReview options type"
```

---

### Task 3: Config emitters — `approvals_reviewer` and friends

The two emitters at `src/inspect_swe/_codex_cli/config.py:40-53` feed config.toml (`codex_config_options`) and `-c` CLI flags (`codex_cli_config_overrides`) respectively. Extend both with a keyword `auto_review` param (existing call sites pass positionally, so a keyword default keeps them working until updated).

**Files:**
- Modify: `src/inspect_swe/_codex_cli/config.py:40-53`
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codex_config.py`:

```python
def test_codex_config_options_auto_review_off_by_default() -> None:
    config = codex_config_options("live", True)
    assert "approvals_reviewer" not in config
    assert "approval_policy" not in config
    assert "sandbox_mode" not in config


def test_codex_config_options_auto_review_enabled() -> None:
    config = codex_config_options("live", True, auto_review=CodexAutoReview())
    assert config["approval_policy"] == "on-request"
    assert config["sandbox_mode"] == "workspace-write"
    assert config["approvals_reviewer"] == "auto_review"
    assert config["features.guardian_approval"] is True
    assert "auto_review" not in config  # no [auto_review] table without a policy
    toml = to_toml(config)
    assert 'approvals_reviewer = "auto_review"' in toml
    assert 'approval_policy = "on-request"' in toml


def test_codex_config_options_auto_review_policy_table() -> None:
    config = codex_config_options(
        "live", True, auto_review=CodexAutoReview(policy="Never allow curl.\nAllow pip.")
    )
    assert config["auto_review"] == {"policy": "Never allow curl.\nAllow pip."}
    toml = to_toml(config)
    assert "[auto_review]" in toml
    assert 'policy = "Never allow curl.\\nAllow pip."' in toml


def test_codex_cli_config_overrides_auto_review() -> None:
    overrides = codex_cli_config_overrides(
        "live", True, auto_review=CodexAutoReview(policy="Never allow curl.")
    )
    assert overrides["approval_policy"] == '"on-request"'
    assert overrides["sandbox_mode"] == '"workspace-write"'
    assert overrides["approvals_reviewer"] == '"auto_review"'
    assert overrides["features.guardian_approval"] == "true"
    # policy goes only into config.toml (multiline-safe), never -c
    assert not any(key.startswith("auto_review") for key in overrides)


def test_codex_cli_config_overrides_auto_review_off_by_default() -> None:
    overrides = codex_cli_config_overrides("live", True)
    assert "approvals_reviewer" not in overrides
    assert "approval_policy" not in overrides
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_codex_config.py -v -k "config_options_auto_review or overrides_auto_review"`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'auto_review'`.

- [ ] **Step 3: Extend the emitters**

Replace `codex_config_options` and `codex_cli_config_overrides` in `src/inspect_swe/_codex_cli/config.py`:

```python
def codex_config_options(
    web_search: CodexWebSearch,
    goals: bool,
    auto_review: "CodexAutoReview | None" = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "web_search": web_search,
        "features.goals": goals,
    }
    if auto_review is not None:
        # auto_review only functions with on-request approvals and Codex's own
        # sandbox engaged (mirrors Codex's "Approve for me" preset)
        options["approval_policy"] = "on-request"
        options["sandbox_mode"] = "workspace-write"
        options["approvals_reviewer"] = "auto_review"
        options["features.guardian_approval"] = True
        if auto_review.policy is not None:
            options["auto_review"] = {"policy": auto_review.policy}
    return options


def codex_cli_config_overrides(
    web_search: CodexWebSearch,
    goals: bool,
    auto_review: "CodexAutoReview | None" = None,
) -> dict[str, str]:
    overrides = {
        "web_search": f'"{web_search}"',
        "features.goals": "true" if goals else "false",
    }
    if auto_review is not None:
        overrides["approval_policy"] = '"on-request"'
        overrides["sandbox_mode"] = '"workspace-write"'
        overrides["approvals_reviewer"] = '"auto_review"'
        overrides["features.guardian_approval"] = "true"
        # auto_review.policy is emitted only via config.toml: -c values are
        # parsed as TOML and multiline policies don't survive shell quoting
    return overrides
```

- [ ] **Step 4: Run the full config test file**

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS (pre-existing positional call tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/inspect_swe/_codex_cli/config.py tests/test_codex_config.py
git commit -m "feat: emit auto_review approval configuration for codex"
```

---

### Task 4: Guardian model-alias and version-check helpers

**Files:**
- Modify: `src/inspect_swe/_codex_cli/config.py`
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codex_config.py` (add `from typing import Any` and `from inspect_ai.model import Model` if not present; extend the config import block with `GUARDIAN_MODEL_SLUG`, `check_codex_auto_review_version`, `resolve_codex_auto_review_model_aliases`):

```python
def test_auto_review_model_aliases_none_passthrough() -> None:
    existing = {"alias": "openai/gpt-4o"}
    assert (
        resolve_codex_auto_review_model_aliases(CodexAutoReview(), existing) is existing
    )
    assert resolve_codex_auto_review_model_aliases(None, existing) is existing


def test_auto_review_model_aliases_adds_guardian_string() -> None:
    # outside a task, model_roles() is {}, so plain strings pass through
    aliases = resolve_codex_auto_review_model_aliases(
        CodexAutoReview(model="openai/gpt-4o"), {"alias": "x"}
    )
    assert aliases == {"alias": "x", "codex-auto-review": "openai/gpt-4o"}
    assert GUARDIAN_MODEL_SLUG == "codex-auto-review"


def test_auto_review_model_aliases_binds_role(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect_swe._codex_cli.config as config_mod

    guardian_model = object()

    def fake_model_roles() -> dict[str, Any]:
        return {"guardian": object()}

    def fake_get_model(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("role") == "guardian"
        return guardian_model

    monkeypatch.setattr(config_mod, "model_roles", fake_model_roles)
    monkeypatch.setattr(config_mod, "get_model", fake_get_model)

    aliases = resolve_codex_auto_review_model_aliases(
        CodexAutoReview(model="guardian"), None
    )
    assert aliases == {"codex-auto-review": guardian_model}


def test_check_codex_auto_review_version() -> None:
    check_codex_auto_review_version("0.137.0")
    check_codex_auto_review_version("0.145.0")
    check_codex_auto_review_version(None)  # undetectable: proceed
    with pytest.raises(RuntimeError, match="0.137.0"):
        check_codex_auto_review_version("0.136.0")
    with pytest.raises(RuntimeError, match="0.137.0"):
        check_codex_auto_review_version("0.99.0")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_codex_config.py -v -k "aliases or version"`
Expected: FAIL — ImportError on the new names.

- [ ] **Step 3: Implement the helpers**

Append to `src/inspect_swe/_codex_cli/config.py`:

```python
GUARDIAN_MODEL_SLUG = "codex-auto-review"
"""Model slug Codex uses for guardian (auto_review) requests."""

CODEX_AUTO_REVIEW_MIN_VERSION = "0.137.0"
"""First Codex CLI release where `codex exec` preserves auto_review approvals."""


def resolve_codex_auto_review_model_aliases(
    auto_review: CodexAutoReview | None,
    model_aliases: dict[str, str | Model] | None,
) -> dict[str, str | Model] | None:
    """Bind the guardian model slug to the configured auto_review model.

    Call within a running task (role resolution reads task context). A `str`
    naming a defined Inspect model role binds via `get_model(role=...)`;
    other values pass through for the bridge to resolve.
    """
    if auto_review is None or auto_review.model is None:
        return model_aliases
    guardian: str | Model = auto_review.model
    if isinstance(guardian, str) and guardian in model_roles():
        guardian = get_model(role=guardian)
    return {**(model_aliases or {}), GUARDIAN_MODEL_SLUG: guardian}


def check_codex_auto_review_version(version: str | None) -> None:
    """Raise if the installed Codex CLI can't run auto_review headlessly."""
    if version is None:
        return
    installed = tuple(int(part) for part in version.split(".")[:3])
    required = tuple(int(part) for part in CODEX_AUTO_REVIEW_MIN_VERSION.split("."))
    if installed < required:
        raise RuntimeError(
            f"auto_review requires Codex CLI >= {CODEX_AUTO_REVIEW_MIN_VERSION} "
            f"(found {version}). Pass version='latest' (or an explicit newer "
            "version) to codex_cli()."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_swe/_codex_cli/config.py tests/test_codex_config.py
git commit -m "feat: guardian model aliasing and version gate for auto_review"
```

---

### Task 5: Thread `auto_review` through `codex_cli()`

**Files:**
- Modify: `src/inspect_swe/_codex_cli/codex_cli.py` (imports at :45-52, signature at :76, docstring at :114, execute() at :152-156, :171-185, :190-192, :260-268, :276-279, :287)
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_codex_config.py` (add `from inspect_swe import codex_cli` at top):

```python
def test_codex_cli_accepts_auto_review() -> None:
    codex_cli(auto_review=True)
    codex_cli(auto_review=False)
    codex_cli(
        auto_review=CodexAutoReview(policy="Deny package installs.", model="guardian")
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codex_config.py::test_codex_cli_accepts_auto_review -v`
Expected: FAIL — `TypeError: codex_cli() got an unexpected keyword argument 'auto_review'`.

- [ ] **Step 3: Implement in `codex_cli.py`**

3a. Extend the `.config` import block (lines 45-52):

```python
from .config import (
    CodexAutoReview,
    CodexDeprecatedArgs,
    CodexWebSearch,
    check_codex_auto_review_version,
    codex_cli_config_overrides,
    codex_config_options,
    resolve_codex_auto_review,
    resolve_codex_auto_review_model_aliases,
    resolve_codex_deprecated_args,
    resolve_codex_web_search,
)
```

3b. Add the parameter after `goals: bool = True,` (line 76):

```python
    goals: bool = True,
    auto_review: bool | CodexAutoReview = False,
```

3c. Add to the docstring after the `goals:` line (line 114):

```python
        auto_review: Enable Codex automated approval review (guardian). When enabled,
            Codex runs with its own sandbox active (`workspace-write`) and `on-request`
            approvals; escalation requests (e.g. network access, writes outside the
            workspace) are adjudicated by a guardian model rather than auto-approved.
            Pass `CodexAutoReview` to customize the guardian policy and model.
            Requires Codex CLI >= 0.137.0. Defaults to `False`.
```

3d. Resolve it alongside the other options (after line 155, `effective_web_search = ...`):

```python
    resolved_auto_review = resolve_codex_auto_review(auto_review)
```

3e. In `execute()`, bind the guardian model alias — replace `model_aliases=model_aliases,` in the `sandbox_agent_bridge(...)` call (line 176) with:

```python
                model_aliases=resolve_codex_auto_review_model_aliases(
                    resolved_auto_review, model_aliases
                ),
```

3f. Version gate — after `codex_binary = await ensure_agent_binary_installed(...)` (lines 190-192):

```python
            # auto_review requires codex exec support for on-request approvals
            if resolved_auto_review is not None:
                check_codex_auto_review_version(
                    await codex_binary_version(sandbox_env(sandbox), codex_binary, user)
                )
```

3g. Make the bypass flag conditional — replace lines 260-268:

```python
            # default cli args
            cmd.extend(
                [
                    # the real model is served via the bridge; this slug only
                    # selects Codex's system prompt + tool set (see codex_model above)
                    "--model",
                    codex_model,
                ]
            )
            # with auto_review, approvals/sandbox come from config (on-request +
            # workspace-write); the bypass flag would force approval_policy=never
            # at a precedence -c can't beat
            if resolved_auto_review is None:
                cmd.append("--dangerously-bypass-approvals-and-sandbox")
```

3h. Pass `auto_review` to both emitter call sites — line 276-279 becomes:

```python
            for key, value in codex_cli_config_overrides(
                effective_web_search, goals, resolved_auto_review
            ).items():
                cmd.extend(["-c", f"{key}={value}"])
```

and line 287 becomes:

```python
            toml_config.update(
                codex_config_options(effective_web_search, goals, resolved_auto_review)
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_swe/_codex_cli/codex_cli.py tests/test_codex_config.py
git commit -m "feat: add auto_review option to codex_cli"
```

---

### Task 6: Thread `auto_review` through the ACP (interactive) agent

The ACP variant writes `approval_policy = "never"` / `sandbox_mode = "danger-full-access"` directly into its toml dict (`src/inspect_swe/acp/_agents/codex_cli/codex_cli.py:132-133`) and then calls `toml_config.update(codex_config_options(...))` (line 144) — so passing `auto_review` to the emitter automatically overwrites both keys. No bypass CLI flag exists on this path. No version gate (the codex-acp adapter pins its own codex).

**Files:**
- Modify: `src/inspect_swe/acp/_agents/codex_cli/codex_cli.py` (imports :21-27, `CodexCli.__init__` :50-69, `_start_agent` bridge call :84-92 and config :144, `interactive_codex_cli` :197-229)
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_codex_config.py` (add `from inspect_swe import interactive_codex_cli`):

```python
def test_interactive_codex_cli_accepts_auto_review() -> None:
    interactive_codex_cli(auto_review=True)
    interactive_codex_cli(auto_review=CodexAutoReview(policy="Deny network."))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codex_config.py::test_interactive_codex_cli_accepts_auto_review -v`
Expected: FAIL — `TypeError` unexpected keyword argument.

- [ ] **Step 3: Implement**

3a. Extend the `inspect_swe._codex_cli.config` import block (lines 21-27) with `CodexAutoReview`, `resolve_codex_auto_review`, `resolve_codex_auto_review_model_aliases`.

3b. `CodexCli.__init__` — add parameter after `goals: bool = True,` and store the resolved value after `self._goals = goals`:

```python
        auto_review: bool | CodexAutoReview = False,
```
```python
        self._auto_review = resolve_codex_auto_review(auto_review)
```

3c. In `_start_agent`, bind the guardian alias — replace `model_aliases=self.model_map,` (line 87) with:

```python
            model_aliases=resolve_codex_auto_review_model_aliases(
                self._auto_review, self.model_map
            ),
```

3d. Pass to the emitter — line 144 becomes:

```python
            toml_config.update(
                codex_config_options(self._web_search, self._goals, self._auto_review)
            )
```

3e. `interactive_codex_cli` — add `auto_review: bool | CodexAutoReview = False,` after `goals: bool = True,`, forward `auto_review=auto_review,` in the `CodexCli(...)` call, and add to the docstring Args:

```python
        auto_review: Enable Codex automated approval review (guardian): Codex runs
            with its own ``workspace-write`` sandbox and ``on-request`` approvals,
            with escalations adjudicated by a guardian model. Pass
            :class:`CodexAutoReview` to customize the guardian policy and model.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_swe/acp/_agents/codex_cli/codex_cli.py tests/test_codex_config.py
git commit -m "feat: add auto_review option to interactive_codex_cli"
```

---

### Task 7: Export `CodexAutoReview` from the package root

**Files:**
- Modify: `src/inspect_swe/__init__.py`
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codex_config.py`:

```python
def test_codex_auto_review_exported_from_package_root() -> None:
    import inspect_swe

    assert inspect_swe.CodexAutoReview is CodexAutoReview
    assert "CodexAutoReview" in inspect_swe.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codex_config.py::test_codex_auto_review_exported_from_package_root -v`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Add the export**

In `src/inspect_swe/__init__.py`, add after line 2 (`from ._codex_cli.codex_cli import codex_cli`):

```python
from ._codex_cli.config import CodexAutoReview
```

and add `"CodexAutoReview",` to `__all__` (next to `"CentaurOptions",`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_swe/__init__.py tests/test_codex_config.py
git commit -m "feat: export CodexAutoReview from package root"
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/codex_cli.qmd`

- [ ] **Step 1: Add the options-table row**

After the `goals` row (`docs/codex_cli.qmd:23`), add:

```markdown
| `auto_review` | Enable Codex [automated approval review](https://developers.openai.com/codex/concepts/sandboxing/auto-review) (see [Auto Review](#auto-review) below for details). Defaults to `False`. |
```

- [ ] **Step 2: Add an "Auto Review" section**

Insert after the options example block (after line 45, before the MCP include):

```markdown
## Auto Review

By default, the agent runs Codex with `--dangerously-bypass-approvals-and-sandbox`: Codex's internal sandbox and approval prompts are disabled, since the agent already runs inside the Inspect sandbox. The `auto_review` option instead runs Codex the way its own "Approve for me" mode ships: Codex's `workspace-write` sandbox is active, `approval_policy` is `on-request`, and escalation requests (network access, writes outside the workspace, MCP tool approvals) are adjudicated by an automated "guardian" reviewer model rather than a human:

```python
{{< meta agent >}}(auto_review=True)
```

Note that enabling `auto_review` materially constrains the agent relative to the default: commands run under Codex's OS-level sandbox (requires Linux kernel >= 5.13 with Landlock in the sandbox container), the guardian may deny escalations, and repeated denials interrupt the turn. Requires Codex CLI >= 0.137.0.

Under `workspace-write`, the workspace root is the agent's working directory (the `cwd` option): commands can read the whole filesystem but can only write within the working directory plus `/tmp` and `$TMPDIR`, top-level `.git` and `.agents` directories are read-only even inside the workspace (so e.g. `git commit` requires an approved escalation), and commands have no network access by default. Actions outside these bounds fail in the sandbox and proceed only if the model requests escalation and the guardian approves. If your task layout requires more, pass `config_overrides` (applied before the auto_review settings, so they compose) — for example `{"sandbox_workspace_write.network_access": "true"}` or `{"sandbox_workspace_write.writable_roots": '["/data"]'}` — and/or set `cwd` so the workspace root matches where the task's files live.

Use `CodexAutoReview` to customize the guardian policy (extra instructions inserted into the guardian review prompt) and the model that serves guardian requests (an Inspect model role name or a model; by default guardian requests are served by the task's main model):

```python
from inspect_swe import CodexAutoReview

{{< meta agent >}}(
    auto_review=CodexAutoReview(
        policy="Deny anything that installs packages.",
        model="guardian",  # binds the 'guardian' model role
    )
)
```
```

- [ ] **Step 3: Verify docs render (if quarto available)**

Run: `command -v quarto && (cd docs && quarto render codex_cli.qmd --to html) || echo "quarto not installed - visually inspect the diff"`
Expected: renders without errors, or fall back to visual inspection.

- [ ] **Step 4: Commit**

```bash
git add docs/codex_cli.qmd
git commit -m "docs: document codex_cli auto_review option"
```

---

### Task 9: Live integration test (slow: Docker + model API)

Modeled on `tests/test_codex_align.py`: run a real eval, capture bridged requests via a `GenerateFilter`, and assert a guardian review request occurred. The task prompt forces a network escalation (workspace-write blocks network by default). The guardian-request detection heuristics may need adjustment after a first live run — the assertion inspects captured system prompts for guardian markers; if upstream prompt wording differs, update `_GUARDIAN_MARKERS` from the captured output.

**Files:**
- Create: `tests/test_codex_auto_review.py`

- [ ] **Step 1: Write the integration test**

```python
"""End-to-end test that auto_review engages Codex's guardian reviewer.

Runs Codex CLI with ``auto_review=True`` in a real sandbox and captures all
bridged requests via a ``GenerateFilter``. The task requires network access,
which ``workspace-write`` blocks, forcing an escalation that the guardian
must review — so we expect at least one bridged request whose system prompt
is the guardian review prompt (identified by marker phrases).

Slow: requires Docker + a live model API (mirrors ``tests/test_codex_align.py``).
"""

from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe import codex_cli

from tests.conftest import skip_if_no_docker, skip_if_no_openai

_DOCKERFILE = str(Path(__file__).parent.parent / "examples" / "mcp" / "Dockerfile")

# phrases expected in the guardian review prompt (from codex-rs guardian
# policy templates); loosen/update from captured output if upstream rewords
_GUARDIAN_MARKERS = ["risk", "approval"]


class _CaptureSystemPrompts:
    """Bridge ``GenerateFilter`` that records every request's system prompt."""

    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        self.system_prompts.append(
            "\n".join(m.text for m in messages if isinstance(m, ChatMessageSystem))
        )
        return None  # passthrough


def _is_guardian_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return all(marker in lowered for marker in _GUARDIAN_MARKERS)


@skip_if_no_docker
@skip_if_no_openai
def test_auto_review_triggers_guardian_review() -> None:
    capture = _CaptureSystemPrompts()
    task = Task(
        dataset=[
            Sample(
                input="Fetch https://example.com with curl and print the HTTP "
                "status code. If you need approval for network access, request it."
            )
        ],
        solver=codex_cli(auto_review=True, version="latest", filter=capture),
        sandbox=("docker", _DOCKERFILE),
    )
    eval(task, model="openai/gpt-5.1-codex", limit=1)
    assert capture.system_prompts, "Codex made no bridged requests"
    assert any(_is_guardian_prompt(p) for p in capture.system_prompts), (
        "no guardian review request observed; captured system prompts:\n"
        + "\n---\n".join(p[:500] for p in capture.system_prompts)
    )
```

- [ ] **Step 2: Run it once live (requires Docker + OPENAI_API_KEY)**

Run: `uv run pytest tests/test_codex_auto_review.py -v -s`
Expected: PASS. If the guardian-marker assertion fails, read the printed captured prompts, update `_GUARDIAN_MARKERS` to phrases actually present in the guardian prompt, and re-run. If no escalation occurs at all (model solves the task without network), strengthen the sample input to require fetching the URL. Also sanity-check the Landlock requirement: if the eval errors with a codex sandbox failure, note it in the docs troubleshooting and mark the test with the appropriate skip.

Additionally, open the produced log in `inspect view` and confirm the guardian calls render as utility agents: guardian requests use a foreign system prompt and produce no tool calls, so inspect_ai's timeline builder (`_wrap_utility_events` in `inspect_ai/event/_timeline.py:1152`) should wrap each one in a synthetic `utility=True` span within the main agent span. If they instead render inline as main-agent turns, record what the events look like (system prompt, tool calls, `role` field) and file a follow-up — do not block the PR on viewer rendering.

- [ ] **Step 3: Commit**

```bash
git add tests/test_codex_auto_review.py
git commit -m "test: live integration test for codex auto_review"
```

---

### Task 10: Full verification and PR

- [ ] **Step 1: Run the full check suite**

Run: `make check` (ruff format, ruff check --fix, mypy strict over src + tests)
Expected: no errors. Fix any strict-mypy complaints (e.g. annotate test helpers).

- [ ] **Step 2: Run the full unit test suite**

Run: `uv run pytest -q --ignore tests/test_codex_auto_review.py -k "not (mcp or align or bridged or attempts or skills or multi_call or system_explorer or web_search)"`
Expected: PASS (integration-style suites need Docker/live APIs; run them too if the environment allows: `uv run pytest -q`).

- [ ] **Step 3: Push branch and open PR**

Branch `feat/codex-auto-review` already exists and is checked out. Do **not** edit `CHANGELOG.md`, version numbers, or `.release-please-manifest.json` (Release Please owns them).

```bash
git push -u origin feat/codex-auto-review
gh pr create \
  --title "feat(codex_cli): add auto_review option for automated approval review" \
  --body "$(cat <<'EOF'
Adds an `auto_review` option to `codex_cli()` and `interactive_codex_cli()` enabling Codex CLI's automated approval review (guardian): Codex runs with its own `workspace-write` sandbox and `on-request` approvals, with escalation requests adjudicated by a guardian model instead of auto-approved. `CodexAutoReview` customizes the guardian policy and model (Inspect model roles supported); requires Codex CLI >= 0.137.0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

- Spec coverage: API (Tasks 2, 5, 6, 7), config plumbing (Task 3), command construction (Task 5 step 3g), guardian model routing incl. roles (Task 4, wired in 5/6), version gating (Tasks 4, 5), observability verification + docs (Tasks 8, 9), tests/docs/PR conventions (Tasks 1-10). ✓
- Emitter keyword default keeps existing positional call sites compiling between Tasks 3 and 5/6. ✓
- Type consistency: `CodexAutoReview`, `resolve_codex_auto_review` → `CodexAutoReview | None`, `resolve_codex_auto_review_model_aliases(auto_review, model_aliases)`, `check_codex_auto_review_version(version)`, `GUARDIAN_MODEL_SLUG` used identically across tasks. ✓
- Known risks called out where they live: Landlock availability in containers (Tasks 8, 9), guardian prompt markers (Task 9), catalog `auto_review_model_override` could route guardian requests under a different slug for some parent models (alias would then not match; the fallback still serves the main model, so behavior degrades gracefully — revisit if Task 9 shows a different slug).
