# DRAFT — upstream feature request for github.com/openai/codex

> Review before filing. Companion to the bwrap-panic issue draft
> (`2026-07-29-upstream-issue-bwrap-panic.md`). Based on codex-rs source at
> rust-v0.145.0 and live runs from an automated-evaluation harness
> ([inspect_swe]) driving `codex exec` with `approvals_reviewer = "auto_review"`.

[inspect_swe]: https://github.com/meridianlabs-ai/inspect_swe

---

**Title:** Expose `strict_auto_review` (review every tool call) as configuration, not only via a per-turn permission grant

### What version of Codex CLI is running?

codex-cli 0.145.0

### Feature request

Codex already implements a "review every tool call" mode internally:
`strict_auto_review`. When enabled, the tool orchestrator routes **every**
exec-family tool call through the auto_review guardian *before* it runs —
including calls whose approval requirement is `Skip` (i.e. would otherwise be
auto-allowed):

```rust
// codex-rs/core/src/tools/orchestrator.rs (~line 170)
ExecApprovalRequirement::Skip { .. } => {
    if strict_auto_review {
        // route through ApprovalReviewer::Guardian anyway
        ...
    }
}
```

The problem is that **this mode cannot be turned on through configuration.**
`strict_auto_review` lives in per-turn state
(`codex-rs/core/src/state/turn.rs`), and the only thing that sets it is a
`request_permissions` **response** carrying `strict_auto_review: true`
(`codex-rs/core/src/session/mod.rs`, around the `PermissionGrantScope::Turn`
match). Every codex-internal construction site hardcodes the field to `false`.
There is no `config.toml` key, no `-c` override, and no feature flag that
enables it — confirmed by searching the tree: `strict_auto_review` appears in
protocol, session, orchestrator, and test files, but nowhere in the `config`
crate.

We would like a configuration surface for it, for example:

```toml
[auto_review]
strict = true
```

or a `features.strict_auto_review` flag, so that headless / automated
deployments can opt into "guardian reviews every tool call" deterministically.

### Why

We run Codex non-interactively (`codex exec`) inside containers as an
evaluation harness, with `approval_policy = "on-request"` and
`approvals_reviewer = "auto_review"`. For studying guardian behavior we want a
**deterministic guarantee** that the guardian sees every command. Today the
only approximations are both unsatisfactory:

1. **execpolicy `prompt` rules.** We can install
   `$CODEX_HOME/rules/*.rules` with `prefix_rule(..., decision = "prompt")`
   entries, which route matching commands through the guardian. But rules are
   matched by literal first token — `PatternToken` is only `Single`/`Alts`
   (`codex-rs/execpolicy/src/rule.rs`), an empty pattern is rejected
   ("prefix cannot be empty", `policy.rs`), and there is no policy-wide default
   decision. So there is **no way to express "prompt on every command"**; we
   must enumerate program names, which is inherently incomplete, and it does
   not cover non-shell tool calls such as `apply_patch` or MCP tools.

2. **Relying on model behavior.** With `workspace-write` and no usable sandbox
   backend (a common container situation — see the companion bwrap issue), a
   capable model tends to re-issue commands with `require_escalated`, which the
   guardian then reviews, and to carry that flag forward. But this is emergent
   and model-dependent: it is not guaranteed, it wastes a failed attempt per
   command until the model adapts, and any command the model does *not*
   escalate runs with **no** review at all.

`strict_auto_review` is exactly the primitive that removes both problems — it
guarantees pre-execution guardian review of every tool call regardless of rule
coverage or model behavior. Only the config surface is missing.

### Additional context

- This is complementary to the companion request that Codex degrade gracefully
  (auto-escalate) when the sandbox launcher is unavailable. That request is
  about *not panicking* when bubblewrap is missing; this one is about
  *deterministically reviewing everything* regardless of sandbox state.
- Scope question for maintainers: an ideal config-level `strict` would also be
  clear about coverage of non-exec tools (`apply_patch`, MCP tool calls) —
  today the orchestrator's guardian routing covers exec-family tools; whether
  strict mode should extend to MCP/patch review is worth specifying.
