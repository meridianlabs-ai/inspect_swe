# Codex auto_review Guardian Gate Mode (new default) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `auto_review` default to a "guardian gate" mode that reviews commands via the guardian *before* execution with no OS sandbox involved — eliminating the bubblewrap panic-per-command overhead in containers — while keeping the current OS-sandbox pairing available as an opt-in.

**Architecture:** Codex routes approvals to the guardian when `approval_policy ∈ {on-request, granular}` and `approvals_reviewer = "auto_review"`, and approval resolution happens *before* the sandbox stage in its tool orchestrator. Execpolicy `prompt` rules (Starlark `prefix_rule(...)` files in `$CODEX_HOME/rules/*.rules`) force a pre-execution approval for matching commands; with `sandbox_mode = "danger-full-access"` the sandbox stage is skipped entirely, so no bwrap/landlock launcher ever runs. Gate mode = that combination plus a default rules file we install next to the config.toml we already write. The previous pairing (`workspace-write`, OS sandbox) remains as `CodexAutoReview(sandbox=True)`.

**Tech Stack:** Python 3.13, pydantic v2, inspect_ai sandbox API, pytest, mypy `--strict` (covers `tests/`), ruff.

---

## Background facts (all verified live or in codex rust-v0.145.0 source, 2026-07-29)

- **Guardian routing predicate** (`codex-rs/core/src/guardian/review.rs:151`): approvals route to the guardian iff `approval_policy` is `on-request` or `granular` AND `approvals_reviewer == auto_review`. (`untrusted` approvals are NOT guardian-adjudicated — verified live: they fail headless with `Rejected("approval request failed")`, zero guardian calls.)
- **Approval-before-sandbox** (`codex-rs/core/src/tools/orchestrator.rs:150-240`): the orchestrator resolves approval (step 1) before sandbox selection/attempt (step 2). Under `danger-full-access`, `should_sandbox` is false → no sandbox stage → the bwrap launcher never runs.
- **Execpolicy prompt rules**: user rules load from `$CODEX_HOME/rules/*.rules` (`codex-rs/core/src/exec_policy.rs`: `RULES_DIR_NAME = "rules"`, `DEFAULT_POLICY_FILE = "default.rules"`), Starlark syntax `prefix_rule(pattern = ["cmd"], decision = "prompt")` (`codex-rs/execpolicy/README.md`). Verified present at rust-v0.137.0, 0.140.0, 0.145.0 — **no version-floor change needed** (floor stays `0.137.0`, required for headless auto_review).
- **Live verification of gate mode** (2026-07-29 experiment, default `python:3.12-bookworm` container, no bwrap, codex 0.145.0, `openai/gpt-5.5`): prompt rules + `danger-full-access` + `auto_review` → zero panics, guardian adjudications occur pre-execution, all commands (including an out-of-workspace root write) succeed after approval. Transcript markers: `bwrap_panic: 0`, `panic_any: 0`, guardian verdicts present.
- **Default decision for unmatched commands** (`exec_policy.rs`, `default_decision`-style logic at ~:741-830): under `on-request` + unrestricted filesystem, unmatched non-dangerous commands → `Allow`; commands matching codex's built-in *dangerous command* detection → `Prompt` (guardian) even with no rules. So the rules file is the coverage dial *on top of* a dangerous-command baseline.
- **Known-safe commands**: the built-in safe-list shortcut (`is_known_safe_command`) auto-allows only under `untrusted` policy or a Windows edge case — it does NOT bypass rules under `on-request`. Observed guardian review counts lower than command counts in the experiment are attributable to codex's continuation-review batching ("history added since your last approval assessment" reviews); Task 6 verifies coverage per command explicitly.
- **Cost of the status quo** (degraded mode, no bwrap): every command = panic (exit 101) + model turn digesting failure + escalation + guardian review (~3× model interactions per tool call, guardian reviews 100% of commands). Gate mode: 1 pre-execution guardian review per rule-matched command, non-matched commands run free.

## Does the execpolicy discovery change the deprecated legacy-landlock story?

**No — it makes it moot for the default path, and it stays dead for the enforcing path.** The legacy `use_legacy_landlock` backend fails for auto_review because modern workspace-write permission profiles "require direct runtime enforcement" (verified live: helper panic at `linux_run_main.rs:318`), and the flag is deprecated for removal. Gate mode sidesteps the backend question entirely: there is no restricted profile and no sandbox stage, so neither bwrap nor landlock is invoked. The only remaining consumer of a working OS sandbox is the opt-in `sandbox=True` mode, whose fidelity path is the parked bwrap auto-provisioning branch (`feat/codex-bwrap`) — landlock plays no role there either. Recommendation: never emit `features.use_legacy_landlock`; do not build anything on it.

## Decisions locked in

1. **`auto_review=True` defaults to gate mode** (per repo owner decision 2026-07-29). `CodexAutoReview` gains `sandbox: bool = False`; `sandbox=True` restores the current `workspace-write` pairing (OS sandbox; degraded escalate-everything without bwrap; full enforcement with the future bwrap PR).
2. Gate mode emits `approval_policy = "on-request"`, `sandbox_mode = "danger-full-access"`, `approvals_reviewer = "auto_review"`, `features.guardian_approval = true` (both config channels, existing pattern), and installs a rules file at `<codex_home>/rules/default.rules`.
3. `CodexAutoReview` gains `rules: str | None = None` — raw execpolicy rules-file content. `None` → the bundled `DEFAULT_GATE_RULES` in gate mode, and *no rules file* in sandbox mode (custom `rules` are honored in both modes).
4. `DEFAULT_GATE_RULES` covers shells/interpreters, file-mutation, network, package-manager, and privilege prefixes (below). Codex's built-in dangerous-command prompting supplements it. Read-only inspection commands (`ls`, `cat`, `grep`, …) deliberately unmatched → run free.
5. Version floor unchanged (`CODEX_AUTO_REVIEW_MIN_VERSION = "0.137.0"`).
6. Docs present three modes honestly: gate (default; guardian-as-the-gate, no OS enforcement), sandbox (opt-in; OS enforcement, requires bwrap + compose seccomp — currently degraded without them), and the non-auto_review default (bypass everything).
7. Ships on the existing unmerged `feat/codex-auto-review` branch (PR #102); PR title/body updated to describe gate mode as default. Never edit CHANGELOG.md.

## File structure

- Modify: `src/inspect_swe/_codex_cli/config.py` — `sandbox`/`rules` fields, `DEFAULT_GATE_RULES`, emitter changes, `codex_auto_review_rules()` helper
- Modify: `src/inspect_swe/_codex_cli/codex_cli.py` — rules-file install in `execute()`
- Modify: `src/inspect_swe/acp/_agents/codex_cli/codex_cli.py` — same for the ACP agent
- Modify: `tests/test_codex_config.py` — emitter/helper unit tests
- Modify: `tests/test_codex_auto_review.py` — gate-mode live test becomes primary; sandbox-mode expectations preserved
- Modify: `docs/codex_cli.qmd` — Auto Review section rewrite

Run with the project venv: `uv run pytest ...`, `make check` before every commit.

---

### Task 1: `CodexAutoReview.sandbox` / `.rules` fields and `DEFAULT_GATE_RULES`

**Files:**
- Modify: `src/inspect_swe/_codex_cli/config.py`
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_codex_config.py` (extend the config import block with `DEFAULT_GATE_RULES`, `codex_auto_review_rules`):

```python
def test_codex_auto_review_defaults_to_gate_mode() -> None:
    options = CodexAutoReview()
    assert options.sandbox is False
    assert options.rules is None


def test_codex_auto_review_rules_selection() -> None:
    # gate mode: default rules
    assert codex_auto_review_rules(CodexAutoReview()) == DEFAULT_GATE_RULES
    # gate mode with custom rules: custom wins
    custom = 'prefix_rule(pattern = ["mycmd"], decision = "prompt")\n'
    assert codex_auto_review_rules(CodexAutoReview(rules=custom)) == custom
    # sandbox mode: no rules file unless custom provided
    assert codex_auto_review_rules(CodexAutoReview(sandbox=True)) is None
    assert codex_auto_review_rules(CodexAutoReview(sandbox=True, rules=custom)) == custom
    # auto_review disabled: never a rules file
    assert codex_auto_review_rules(None) is None


def test_default_gate_rules_are_prompt_prefix_rules() -> None:
    assert 'decision = "prompt"' in DEFAULT_GATE_RULES
    # spot-check shell and network coverage
    assert '["bash"]' in DEFAULT_GATE_RULES
    assert '["curl"]' in DEFAULT_GATE_RULES
    assert '["rm"]' in DEFAULT_GATE_RULES
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `uv run pytest tests/test_codex_config.py -v -k "gate or rules_selection"`
Expected: FAIL — ImportError on the new names.

- [ ] **Step 3: Implement.** In `src/inspect_swe/_codex_cli/config.py`, add to `CodexAutoReview` after the `model` field (keep existing field docstring style):

```python
    sandbox: bool = Field(default=False)
    """Run Codex's own OS sandbox (`workspace-write`).

    Defaults to `False`: Codex runs with `danger-full-access` and the guardian
    reviews commands matched by execpolicy prompt rules *before* execution
    ("gate mode") — no OS-level enforcement, works in any container. Set to
    `True` to pair the guardian with Codex's `workspace-write` sandbox instead
    (requires bubblewrap in the sandbox for OS enforcement; without it every
    command fails into the escalation path).
    """

    rules: str | None = Field(default=None)
    """Custom execpolicy rules installed to `$CODEX_HOME/rules/default.rules`.

    Starlark `prefix_rule(...)` syntax (see the Codex execpolicy docs).
    Defaults to `None`: gate mode installs `DEFAULT_GATE_RULES`; sandbox mode
    installs no rules file.
    """
```

Add the module-level constant and helper after `CODEX_AUTO_REVIEW_MIN_VERSION`:

```python
DEFAULT_GATE_RULES = '''\
# Guardian gate-mode rules installed by inspect_swe: commands matching these
# prefixes require a pre-execution approval, which Codex routes to the
# auto_review guardian. Codex additionally prompts for its built-in
# dangerous-command detections. Unmatched commands run without review.

# shells / interpreters (codex evaluates inner commands of `bash -lc` too)
prefix_rule(pattern = [["bash", "sh", "zsh", "dash", "ksh"]], decision = "prompt")
prefix_rule(pattern = [["python", "python3", "node", "deno", "bun", "perl", "ruby", "Rscript"]], decision = "prompt")

# network
prefix_rule(pattern = [["curl", "wget", "nc", "ncat", "socat", "ssh", "scp", "rsync", "nmap"]], decision = "prompt")
prefix_rule(pattern = [["git"]], decision = "prompt")

# package managers / installers
prefix_rule(pattern = [["pip", "pip3", "uv", "npm", "npx", "yarn", "pnpm", "cargo", "gem", "apt", "apt-get", "dpkg", "brew"]], decision = "prompt")

# file mutation / system state
prefix_rule(pattern = [["rm", "mv", "cp", "dd", "chmod", "chown", "ln", "tee", "truncate", "mkfs", "mount", "umount"]], decision = "prompt")
prefix_rule(pattern = [["kill", "pkill", "killall", "reboot", "shutdown", "systemctl", "service", "sudo", "su"]], decision = "prompt")
'''
"""Default execpolicy rules for auto_review gate mode."""


def codex_auto_review_rules(auto_review: CodexAutoReview | None) -> str | None:
    """Resolve the execpolicy rules file content for an auto_review config."""
    if auto_review is None:
        return None
    if auto_review.rules is not None:
        return auto_review.rules
    return None if auto_review.sandbox else DEFAULT_GATE_RULES
```

NOTE on rule syntax: a `pattern` element that is itself a list denotes alternatives (per the execpolicy README), so `[["bash", "sh", ...]]` is one rule matching any of those first tokens. Task 6's live run validates the file parses; if codex rejects the alternatives form at load, fall back to one `prefix_rule` per command.

- [ ] **Step 4: Run the full test file.**

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS (existing tests unaffected — new fields have defaults).

- [ ] **Step 5: Lint/format, then commit.**

Run: `source .venv/bin/activate && make check`, then:

```bash
git add src/inspect_swe/_codex_cli/config.py tests/test_codex_config.py
git commit -m "feat: add gate/sandbox modes and execpolicy rules to CodexAutoReview

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Emitters — mode-dependent `sandbox_mode`

**Files:**
- Modify: `src/inspect_swe/_codex_cli/config.py` (the two emitters)
- Test: `tests/test_codex_config.py`

- [ ] **Step 1: Write the failing tests.** Append:

```python
def test_codex_config_options_gate_mode_full_access() -> None:
    config = codex_config_options("live", True, auto_review=CodexAutoReview())
    assert config["approval_policy"] == "on-request"
    assert config["sandbox_mode"] == "danger-full-access"
    assert config["approvals_reviewer"] == "auto_review"
    assert config["features.guardian_approval"] is True


def test_codex_config_options_sandbox_mode_workspace_write() -> None:
    config = codex_config_options(
        "live", True, auto_review=CodexAutoReview(sandbox=True)
    )
    assert config["sandbox_mode"] == "workspace-write"
    assert config["approval_policy"] == "on-request"


def test_codex_cli_config_overrides_mode_dependent_sandbox() -> None:
    gate = codex_cli_config_overrides("live", True, auto_review=CodexAutoReview())
    assert gate["sandbox_mode"] == '"danger-full-access"'
    enforced = codex_cli_config_overrides(
        "live", True, auto_review=CodexAutoReview(sandbox=True)
    )
    assert enforced["sandbox_mode"] == '"workspace-write"'
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `uv run pytest tests/test_codex_config.py -v -k "gate_mode or sandbox_mode or mode_dependent"`
Expected: FAIL — gate-mode assertions see `"workspace-write"`.

- [ ] **Step 3: Implement.** In both emitters, replace the hardcoded sandbox_mode line. In `codex_config_options`:

```python
        options["approval_policy"] = "on-request"
        options["sandbox_mode"] = (
            "workspace-write" if auto_review.sandbox else "danger-full-access"
        )
```

In `codex_cli_config_overrides`:

```python
        overrides["approval_policy"] = '"on-request"'
        overrides["sandbox_mode"] = (
            '"workspace-write"' if auto_review.sandbox else '"danger-full-access"'
        )
```

Adjust the emitter comment ("mirrors Codex's 'Approve for me' preset") to note the two pairings: gate mode (full access + prompt rules; guardian reviews pre-execution) vs sandbox mode (workspace-write; matches Codex's preset).

- [ ] **Step 4: Fix now-stale existing tests.** `test_codex_config_options_auto_review_enabled` and `test_codex_cli_config_overrides_auto_review` (from the earlier plan) assert `workspace-write` for default `CodexAutoReview()` — update them to expect `danger-full-access`, and keep a `sandbox=True` variant asserting `workspace-write` (covered by the new tests above; delete duplication rather than keeping two copies).

Run: `uv run pytest tests/test_codex_config.py -v`
Expected: all PASS.

- [ ] **Step 5: `make check`, then commit.**

```bash
git add src/inspect_swe/_codex_cli/config.py tests/test_codex_config.py
git commit -m "feat: default auto_review to guardian gate mode (danger-full-access)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Install the rules file in `codex_cli()`

**Files:**
- Modify: `src/inspect_swe/_codex_cli/codex_cli.py`
- Test: `tests/test_codex_config.py` (construction smoke only — sandbox interaction is covered live in Task 6)

- [ ] **Step 1: Failing smoke test.** Append:

```python
def test_codex_cli_accepts_gate_and_sandbox_modes() -> None:
    codex_cli(auto_review=CodexAutoReview(sandbox=False))
    codex_cli(auto_review=CodexAutoReview(sandbox=True))
    codex_cli(auto_review=CodexAutoReview(rules='prefix_rule(pattern = ["x"], decision = "prompt")'))
```

Run: `uv run pytest tests/test_codex_config.py::test_codex_cli_accepts_gate_and_sandbox_modes -v`
Expected: PASS already if fields exist (Task 1) — if it passes, keep it as a regression guard and continue (the real change in this task is behavioral, verified in Task 6).

- [ ] **Step 2: Implement the install.** In `src/inspect_swe/_codex_cli/codex_cli.py`, extend the `.config` import with `codex_auto_review_rules`. Then, immediately after the config.toml write (`await sbox.write_file(await codex_config_toml(), to_toml(toml_config))`), add:

```python
            # install execpolicy rules (guardian gate mode / custom rules)
            auto_review_rules = codex_auto_review_rules(resolved_auto_review)
            if auto_review_rules is not None:
                await sbox.write_file(
                    join_path(codex_home, "rules", "default.rules"),
                    auto_review_rules,
                )
```

(`join_path` is already imported; `sbox.write_file` creates parent directories. `codex_home` is in scope — it's computed above for the config.toml path; if the local variable name differs, use the same expression the config write uses.)

- [ ] **Step 3: Run the full unit file + `make check`, then commit.**

```bash
git add src/inspect_swe/_codex_cli/codex_cli.py tests/test_codex_config.py
git commit -m "feat: install guardian gate-mode execpolicy rules in codex_cli

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Same for the ACP (interactive) agent

**Files:**
- Modify: `src/inspect_swe/acp/_agents/codex_cli/codex_cli.py`
- Test: `tests/test_codex_config.py`

- [ ] **Step 1:** Extend the `inspect_swe._codex_cli.config` import with `codex_auto_review_rules`. In `_start_agent`, after the config.toml `write_file`, add the same conditional rules write, using that file's `codex_home` variable and `join_path(codex_home, "rules", "default.rules")`.

- [ ] **Step 2:** Extend the existing ACP smoke test (`test_interactive_codex_cli_accepts_auto_review`) with a `CodexAutoReview(sandbox=True)` and a `rules=...` construction.

- [ ] **Step 3:** `uv run pytest tests/test_codex_config.py -v` → all PASS; `make check`; commit:

```bash
git add src/inspect_swe/acp/_agents/codex_cli/codex_cli.py tests/test_codex_config.py
git commit -m "feat: gate-mode rules install for interactive_codex_cli

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Docs — Auto Review section rewrite

**Files:**
- Modify: `docs/codex_cli.qmd`

- [ ] **Step 1:** Rewrite the "Auto Review" section to present the two auto_review modes. Replace the existing paragraphs between the `auto_review=True` example block and the "Under `workspace-write` ..." paragraph with:

```markdown
By default, `auto_review` runs in **gate mode**: Codex executes with full filesystem/network access inside the Inspect sandbox (no OS-level sandbox of its own), and commands matching a set of [execpolicy](https://github.com/openai/codex/tree/main/codex-rs/execpolicy) prompt rules — shells, interpreters, network tools, package managers, file-mutating and privileged commands, plus Codex's built-in dangerous-command detection — require a guardian approval *before* they run. Unmatched commands (e.g. read-only inspection like `ls` or `cat`) run without review. Pass `rules` to replace the default rule set:

```python
from inspect_swe import CodexAutoReview

{{< meta agent >}}(
    auto_review=CodexAutoReview(
        rules='prefix_rule(pattern = [["curl", "wget"]], decision = "prompt")',
    )
)
```

Gate mode works in any sandbox image with no special configuration. Note that the guardian is the *only* gate in this mode — there is no OS-level enforcement behind its decisions, and rule coverage (not a sandbox boundary) determines which commands are reviewed.

Set `sandbox=True` to instead pair the guardian with Codex's own `workspace-write` OS sandbox (the way Codex's "Approve for me" mode ships): commands run sandboxed, and only genuine escalations (network access, out-of-workspace writes) reach the guardian. Codex's OS sandbox requires a `bwrap` binary and a container that permits namespace creation; without them, every command fails into the escalation path and is guardian-reviewed post-failure (functional, but roughly triple the per-command overhead of gate mode).
```

Keep the existing "Under `workspace-write` ..." paragraph but introduce it with "With `sandbox=True`, ..." so its scope is clear. Update the options-table row for `auto_review` if its wording implies workspace-write by default.

- [ ] **Step 2:** Visual-inspect the diff (fences balanced, table intact — note the nested fence inside the markdown block above needs care), then commit:

```bash
git add docs/codex_cli.qmd
git commit -m "docs: document auto_review gate mode as the default

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Live integration tests — gate mode primary

**Files:**
- Modify: `tests/test_codex_auto_review.py`

- [ ] **Step 1: Add the gate-mode test** (this is the new default, so it's the primary test). Reuse the capture-filter pattern already in the file:

```python
@skip_if_no_docker
@skip_if_no_openai
def test_auto_review_gate_mode_reviews_before_execution() -> None:
    """Default (gate) mode: no sandbox panics; guardian reviews rule-matched
    commands before they run; unmatched read-only commands run unreviewed."""
    capture = _CaptureSystemPrompts()
    task = Task(
        dataset=[
            Sample(
                input="Run exactly these three commands one at a time and "
                "report each result: (1) `ls -la`, (2) `curl -sI https://example.com "
                "| head -1`, (3) `echo done`."
            )
        ],
        solver=codex_cli(auto_review=True, version="latest", filter=capture),
        sandbox=("docker", _DOCKERFILE),
    )
    log = eval(task, model="openai/gpt-5.5", limit=1)[0]
    assert log.status == "success", f"eval failed: {log.error}"
    # guardian engaged (curl matches the default rules)
    assert any(_is_guardian_prompt(p) for p in capture.system_prompts)
    # no sandbox launcher panics anywhere in the transcript
    sample = read_eval_log_sample(log.location, id=log.samples[0].id) if False else None
    # (simpler: scan the captured full text via the log)
    text = _all_text(log.location)
    assert "bubblewrap is unavailable" not in text
    assert "panicked" not in text
```

Implement `_all_text(location)` as a small helper using `read_eval_log_samples(location, all_samples_required=False, resolve_attachments=True)` concatenating message texts and event reprs (mirror the pattern used during the experiments). Clean up the dead `read_eval_log_sample` line — write the final version properly; the executor should produce a tidy test, not the sketch above verbatim.

- [ ] **Step 2: Re-scope the existing test.** `test_auto_review_triggers_guardian_review` currently exercises the escalate-after-panic flow, which is now the `sandbox=True` degraded path. Update its solver to `codex_cli(auto_review=CodexAutoReview(sandbox=True), version="latest", filter=capture)` (import `CodexAutoReview`) and update its docstring to say it verifies the sandbox-mode degraded path (bwrap absent → escalation → guardian).

- [ ] **Step 3: Run both live** (Docker + OPENAI_API_KEY; generous timeouts):

Run: `uv run pytest tests/test_codex_auto_review.py -v -s`
Expected: 2 passed. While iterating, additionally verify from the gate-mode log that (a) the rules file parsed (no execpolicy load errors in transcript), and (b) reviews precede execution (no failed-then-retried commands). If the alternatives `[[...]]` pattern syntax fails to load, split `DEFAULT_GATE_RULES` into one rule per command (Task 1 note) and re-run.

- [ ] **Step 4: `make check`, commit.**

```bash
git add tests/test_codex_auto_review.py
git commit -m "test: live coverage for auto_review gate mode default

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Full verification and PR update

- [ ] **Step 1:** `make check` and `uv run pytest -q --ignore tests/test_codex_auto_review.py -k "not (mcp or align or bridged or attempts or skills or multi_call or system_explorer or web_search)"` → all green (run the live suites too if the environment allows).
- [ ] **Step 2:** Push and update PR #102's body: description of gate mode as default, `sandbox=True` opt-in, and the honest security framing (guardian-only gate vs OS enforcement). Do not edit CHANGELOG.md. Keep the PR title `feat(codex_cli): add auto_review option for automated approval review`.
- [ ] **Step 3:** File the upstream issue from `plans/2026-07-29-upstream-issue-bwrap-panic.md` after repo-owner review (owner action).

---

## Self-review notes

- Coverage: API fields (T1), emitters (T2), rules install exec path (T3), ACP path (T4), docs (T5), live verification incl. rule-syntax risk (T6), verification/PR (T7). The "does execpolicy change the landlock story" question is answered in Background (no; moot for gate mode, dead otherwise).
- Type consistency: `codex_auto_review_rules(auto_review: CodexAutoReview | None) -> str | None` used identically in T1/T3/T4; `sandbox`/`rules` field names consistent throughout.
- Known risks, called out where they live: execpolicy alternatives-pattern syntax (T1 note + T6 fallback); guardian review batching means review-count assertions must stay loose (T6 asserts presence, not exact counts); `parse_shell_lc` inner-command matching means shell-wrapper rules may double-fire with inner rules (acceptable; guardian batches continuation reviews).
- Follow-on (not this plan): bwrap auto-provisioning branch (`feat/codex-bwrap`) upgrades `sandbox=True` from degraded to enforcing; reviewer-noted improvements there (tarball integrity check, negative-result caching, rebase conflict) are tracked in that branch's review.
