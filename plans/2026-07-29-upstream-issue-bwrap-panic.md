# DRAFT — upstream issue for github.com/openai/codex

> Review before filing. References real observations from inspect_swe live runs
> on codex-cli 0.145.0 (Docker containers, `codex exec`, `approvals_reviewer =
> "auto_review"`). Related issues to link: #24873, #35547.

---

**Title:** When no Linux sandbox backend is available, degrade to escalated execution (guardian/approval-gated) instead of panicking on every command

### What version of Codex CLI is running?

codex-cli 0.145.0 (installed from the GitHub release tarball `codex-<arch>.tar.gz`)

### What platform is your computer?

Linux containers (Docker, default seccomp profile; kernel 6.12 LinuxKit and
various CI hosts). Codex runs headlessly via `codex exec` inside the container.
This is an automated-evaluation harness setup ([inspect_swe]), where Codex's
model traffic is proxied and Codex itself runs inside a sandboxed container —
a configuration in which bubblewrap generally cannot work: the release tarball
does not bundle `codex-resources/bwrap`, images rarely ship `bwrap`, and even
when it is present, namespace creation is blocked by the container runtime's
default seccomp policy.

[inspect_swe]: https://github.com/meridianlabs-ai/inspect_swe

### What issue are you seeing?

With `approval_policy = "on-request"`, `sandbox_mode = "workspace-write"`, and
`approvals_reviewer = "auto_review"`, **every sandboxed command** fails with:

```text
thread 'main' (494) panicked at linux-sandbox/src/launcher.rs:43:13:
bubblewrap is unavailable: no system bwrap was found on PATH and no bundled
codex-resources/bwrap binary was found next to the Codex executable
```

The model then sees the failure, retries with
`sandbox_permissions: "require_escalated"`, the guardian (auto_review)
adjudicates, and the approved retry runs unsandboxed. The session *works*, but
every single command costs: one failed exec (exit 101) + one model turn to
digest the panic + one escalation + one guardian review. In our measurements
this roughly **triples the model interactions per tool call** across an entire
session — and the guardian ends up reviewing 100% of commands rather than the
minority that genuinely cross a sandbox boundary.

Two adjacent behaviors compound the problem:

1. The startup warning says: *"Codex could not find bubblewrap on PATH. …
   **Codex will use the bundled bubblewrap in the meantime.**"* — but the
   release-tarball install has no bundled bubblewrap, so the promise is
   immediately followed by per-command panics.
2. The legacy fallback is not viable: `-c features.use_legacy_landlock=true`
   is accepted, but the helper then panics with
   `permission profiles requiring direct runtime enforcement are incompatible
   with --use-legacy-landlock` — current workspace-write profiles (protected
   `.git`/`.agents` metadata carveouts) cannot be expressed in the legacy
   Landlock policy, so the deprecated escape hatch cannot help precisely in
   the environments that need it (also reported in #24873).

### What did you expect to happen?

When the sandbox launcher is unavailable, degrade **gracefully and loudly**
rather than panicking per command. Concretely, in descending order of
preference:

1. **Auto-escalate**: if `bwrap` cannot launch (detected once, up front),
   treat sandboxed commands as `require_escalated` so the existing approval
   machinery (`approvals_reviewer = "auto_review"`, or the user prompt in
   interactive sessions) adjudicates each command *before* execution — the
   same security posture the panic path converges to today, at one review per
   command instead of a panic + retry + review.
2. Alternatively, expose the behavior as configuration (e.g. a
   `sandbox_unavailable = "escalate" | "error"` policy), so headless/container
   deployments can opt into escalated execution explicitly.
3. At minimum, detect launcher unavailability once at session start and fail
   (or warn) with an actionable message — as also requested in #35547 — and
   make the startup warning stop claiming a bundled bubblewrap exists when it
   does not.

### What steps can reproduce the bug?

In any Linux container without `bwrap` (e.g. `python:3.12-bookworm`, default
Docker seccomp profile):

```bash
codex exec --skip-git-repo-check \
  -c approval_policy='"on-request"' \
  -c sandbox_mode='"workspace-write"' \
  -c approvals_reviewer='"auto_review"' \
  "run: echo hello"
```

Every command the model runs panics with the `bubblewrap is unavailable`
message, then succeeds on the escalated retry after guardian review.

### Additional context

- #24873 reports the same panic (plus bwrap discovery problems and the same
  legacy-Landlock profile incompatibility) on WSL2.
- #35547 asks for clearer diagnostics when neither backend is available; this
  issue asks for a *behavioral* fallback in addition to better diagnostics,
  because in container/CI use the environment often cannot be changed.
- We currently work around this in our harness by (a) provisioning the
  npm-package `codex-resources/bwrap` next to the binary and relaxing the
  container seccomp profile where permitted, or (b) accepting the
  panic-per-command overhead. A supported degrade path would remove the need
  for both.
