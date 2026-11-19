# Adding a coding agent

This guide lists what a new agent in `inspect_swe` (a `foo_code()` solver that
runs a third-party coding CLI in the sandbox) has to provide. Almost every item
comes from something an existing agent needed, either when it was added or in a
later fix. The commit or PR that taught us each one is cited in brackets. A few
items come from fixes still in open PRs; those are marked *(pending)*, and the
reference agents don't do them yet. Contributors
can use the guide as a checklist, and reviewers can check a PR against it item
by item.

Getting the CLI to run and answer is the easy part. Most of the work is making
the **trajectory** right:

- classifying every model call (top-level agent, sub-agent, internal utility
  call, or unknown);
- putting each call in the right transcript span;
- making the run read correctly in the Inspect log viewer.

The existing agents needed most of their follow-up fixes here, and new agent
PRs most often skip it. A PR that runs end to end but files every model call
flat under one span is not finished.

Items are marked:

- **MUST**: required to merge.
- **SHOULD**: expected. If you leave one out, say why in the PR.
- **IF**: required only when the CLI has the feature. When it doesn't, say so
  in the PR (for example "N/A: the CLI has no delegation").

## 0. Inventory the CLI first

Before you write code, answer these questions and put the answers in the PR
description. Section B (attribution) and section C (trajectory) depend on them.
When an answer is "I don't know", that is a finding too, and the PR should
report it.

1. **Wire API.** Which provider API does the CLI speak (Anthropic Messages,
   OpenAI Responses or Chat Completions, Google)? How do you point it at a base
   URL and a model name?
2. **Distribution.** Is it a standalone per-platform binary with checksums, an
   npm package, a PyPI wheel, or an SDK? Are glibc and musl builds both
   available? Can the sandbox install it with no network access?
3. **Headless mode.** Is there a non-interactive mode, and does it have a
   machine-readable event stream (JSONL or similar)? What does it print on a
   refusal, an error or an exit?
4. **Delegation.** Can the CLI delegate to sub-agents (a task or agent tool,
   spawned threads, agent teams)? Can those run in the background, in
   parallel, or be resumed? Which model name does a sub-agent request?
5. **Internal model calls.** Does the CLI make calls of its own: title
   generation, compaction or summarization, permission or safety classifiers,
   loop detection, next-speaker checks, web-fetch summarizers? Which model name
   does each request, and how does each look on the wire?
6. **Sessions.** How do you resume a session (`--resume <id>`, `--continue`,
   `resume --last`)? What does the CLI regenerate when it resumes? Does the
   system prompt change between calls within one session (see C10)?
7. **Extension surfaces.** How does the CLI configure MCP servers, which
   directory does it read skills from, and does it have a native web search
   tool? How do you append to or replace its system prompt?
8. **Side effects.** Does it send telemetry, update itself, keep memory across
   sessions, or ask for login or OAuth?

## 1. Architecture

Each agent runs inside one `sandbox_agent_bridge()` block (inspect_ai). The
bridge serves the CLI's model API in the sandbox, routes each request to an
Inspect `Model`, and records each call as a `ModelEvent`.

```
CLI process in the sandbox ──HTTP──▶ sandbox_agent_bridge (inspect_ai)
                                      │  filter = classify_filter(user_filter, classify, ...)   (B)
                                      │     └─ set_agent_bridge_context(root|subagent|utility|unknown)
                                      │  model / model_aliases / model_resolver                 (A3)
                                      │  model_event_sink = <Agent>Consumer                     (C)
                                      │     on_start  → the span this generate runs under
                                      │     on_pending/on_complete → record the event, open
                                      │       sub-agent spans, attach tool views
                                      ▼
                                   Inspect Model ──▶ ModelEvent in the transcript ──▶ timeline_build() ──▶ viewer
```

These are the parts and where their reference implementations live:

| Part | Reference implementation |
|---|---|
| Binary acquisition | `_util/agentbinary.py` (`AgentBinarySource`, `ensure_agent_binary_installed`, `record_agent_binary_install`), `_<agent>/agentbinary.py`. For npm packages: `_util/node.py` (`create_npm_bundle`, `resolve_npm_package_version`). For wheels: `_util/agentwheel.py`. |
| Entry point | `_<agent>/<agent>.py`: `@agent` factory → `execute(state)` → `agent_with(execute, name=..., description=...)` |
| Classifier | `_util/agentcontext.py`: `classify_filter`, `static_root_classifier`, `slug_map_classifier`, `AgentContextClassifier` |
| Sink and span builder | `_claude_code/_events/live_consumer.py` (`LiveConsumer`), `_codex_cli/_events/consumer.py` (`CodexConsumer`), shared state in `_util/model_event_sink.py` |
| Tool views | `_claude_code/_events/toolview.py`, `_codex_cli/_events/toolview.py` |
| Shared plumbing | `_util/bridged_tools.py`, `_util/mcp_ready.py`, `_util/websearch.py`, `_util/centaur.py`, `_util/messages.py`, `_util/sandbox.py` |

Use `kimi_code` as the template for a CLI with no delegation. Use `claude_code`
and `codex_cli` as the templates for a CLI that delegates.

## A. Packaging, binary, and model routing

**A1. MUST: get the CLI onto the host, then copy it into the sandbox.**
Download the CLI on the host, verify its checksum, cache it, and write it into
the sandbox. Never install by running `npm`, `pip` or `curl` inside the
sandbox: network-isolated sandboxes hang there (#133). When the CLI publishes
standalone binaries, implement `<agent>_binary_source() -> AgentBinarySource`
and call `ensure_agent_binary_installed(source, version, user, sandbox)`.
Kimi's `_kimi_code/agentbinary.py` is the smallest example.

- If the release ships companion executables, install the whole package
  archive (`package_entrypoint`, `cached_package_path`) (#107).
- If the CLI ships only on npm, use `create_npm_bundle` and
  `install_npm_bundle`. They give a 7-day release cooldown and `--ignore-scripts`
  (#234), and must not be bypassed.
- Raise an error at version resolution for a platform the vendor doesn't build,
  for example musl (see `_platform_to_kimi_platform`). Don't let a wrong binary
  fail later at exec.
- Downloads go through `_util/download.py`, which handles retries, timeouts and
  `GITHUB_TOKEN` (#130, #199).
- Cache paths go through `appdirs.package_cache_dir`.

*Verify:* the install makes no network call from inside the sandbox. A unit test
covers version resolution and checksum mismatch (see
`tests/test_agentbinary_install.py`, `tests/test_kimi_setup.py`).
`tests/test_home_isolation.py` still passes.

**A2. MUST: accept the standard `version` values.**
`version` accepts `"auto" | "sandbox" | "stable" | "latest" | "x.y.z"` with the
same meaning as in the other agents. `"stable"` and `"latest"` resolve once per
process through the shared cache (#127, `_util/versioncache.py`). When the
agent has an `AgentBinarySource`, add it to the `Literal`s and the dispatcher in
`_tools/download.py` (`download_agent_binary`, `cached_agent_binaries`,
`resolve_agent_version`, `_agent_binary_source`). Kimi was missing from these
until #147. *Verify:* `resolve_agent_version("<agent>", "stable")` returns a
concrete version.

**A3. MUST: record which binary ran.**
Binary installs go through `ensure_agent_binary_installed`, which emits the
`InfoEvent(source="inspect_swe")` carrying `AgentBinaryInstall` (#183). Include
`_snippets/agent_binary_provenance.md` in the agent's docs page. *Verify:* a
sample's events contain one `AgentBinaryInstall` record per agent run. If the
agent installs another way (npm, wheel, SDK), the PR states that no record is
written, as `gemini_cli`, `mini_swe_agent` and `antigravity` (a wheel) do today.

**A4. MUST: serve every model call with the Inspect model.**
Pass `model=f"inspect/{model}" if model else "inspect"` to
`sandbox_agent_bridge`, and pass `model_aliases` through. Every name the CLI can
request must reach the agent's model, including hard-coded utility model names
and sub-agent model names. Gemini's utility names crashed the bridge until they
were aliased (#48). The interactive agents pass
`model_resolver=lambda _name: model` so that unaliased names keep the agent's
`Model` config (#182). *Verify:* list every model name the CLI sends and check
that each one resolves. `ModelEvent.model` is never a provider default the user
didn't choose.

**A5. MUST: tell the CLI the real model.**
The CLI should see the real served model name in its own identity and system
prompt, not the `inspect` sentinel (#63, `_claude_code/model.py`
`resolve_claude_code_models`). Use `model_config` when the CLI needs a catalog
entry or context size (`_codex_cli/model_catalog.py`, kimi
`max_context_size`). *Verify:* the first `ModelEvent`'s system prompt doesn't
contain "inspect" as the model name.

**A6. MUST: keep credentials out of the sandbox.**
Follow the AGENTS.md credential boundary. Only the bridge's dummy key enters the
sandbox (claude_code writes an `apiKeyHelper` for it). Real provider keys
never do. Real MCP credentials may cross the boundary for transport but must
stay out of every path the model or its tools can read: no config file in the
workspace, and no env var a `bash` tool can print. *Verify:* `grep` the
generated config and env for `*_API_KEY` and auth headers. Caller-supplied
`env=` is documented as sandbox-exposed.

**A7. MUST: turn off CLI side effects.**
Disable telemetry and update checks (#50, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`,
kimi `telemetry = false`, gemini `settings.json`). Disable memory that persists
across sessions and can override task-provided tools (#145, auto-memory). Make
sure no login or OAuth flow can start silently (`_seed_claude_config`). If the
sandbox's default working directory is `/`, fall back to the home directory
(`resolve_agent_cwd`). *Verify:* review the generated config and env in the PR.

## B. Agent context classification

Every bridged request is classified before the user's `filter` runs, so that
a filter gated on `is_root_agent()` can skip sub-agent and utility calls. The goal is
that steering written for the top-level agent never leaks into sub-agents or
internal calls (b69a5f9).

**B1. MUST: install `classify_filter` at the bridge call.**
Wrap the user's filter: `filter=classify_filter(user_filter, classify, ...)`.
Wrap it unconditionally, even when the caller passed no filter. Choose
`classify` by what the CLI exposes:

| The CLI... | Use | Example |
|---|---|---|
| has no delegation and no internal calls | `static_root_classifier` | `mini_swe_agent`, `antigravity` |
| uses distinct model names for internal or sub-agent calls | `slug_map_classifier(root_slugs, kind_by_slug)` (reads `current_bridge_request().model`) | `gemini_cli`, `opencode`, ACP variants |
| marks internal calls in the request content | a custom classifier (see `kimi_classifier`) | `kimi_code` compaction |
| has delegation with state to track | the consumer's `classify` method (section C) | `claude_code`, `codex_cli` |

If the agent has a consumer, also pass `on_filter_output=consumer.on_filter_output`
and `sink=consumer`. A filter that returns its own `ModelOutput` produces no
`ModelEvent`, so the delegations in that output would otherwise go
untracked. *Verify:* a unit test drives the wrapped filter and checks
`current_agent_bridge_context()` inside the user's filter (see
`tests/test_agent_context_static.py`, `tests/test_agent_context_opencode.py`).

**B2. MUST: treat "unknown" as an honest answer and fail in the safe direction.**
Don't guess. The harmful direction is a sub-agent call that isn't labelled
`"subagent"`, because a root-only filter then steers the sub-agent. Note that
`is_root_agent()` returns `True` for `"unknown"` (and for no context), so under
the usual `is_root_agent()` gate an `"unknown"` sub-agent call is steered just
like a mislabelled `"root"` one; `"unknown"` is only safer for filters that
check `kind == "root"` explicitly. So when you know a call comes from *some*
sub-agent but not which one, return `"subagent"` with `subagent_id=None`.
Reserve `"unknown"` for calls you genuinely can't place as root or sub-agent. Labelling a
main-thread call `"utility"` is just as harmful. Under-attributing a utility
call as `"root"` is acceptable and should be documented. Match markers by a
structural position (for example "the latest user message *starts with* X"),
not anywhere in the content, so that a tool output quoting the marker can't
flip the classification (kimi's `_COMPACTION_INSTRUCTION_PREAMBLE`, codex's
`COMPACTION_MARKER`). *Verify:* there are negative tests in which a tool result
or the task prompt quotes the marker or a sub-agent prompt.

**B3. MUST: classify the CLI's internal calls.**
Every internal call from inventory question 5 is classified `"utility"` or
documented as under-attributed. Where you can switch a pointless call off, do:
opencode's title call was wasted tokens and a spurious event (#159), and it
could also become the sample's answer (#151). *Verify:* the PR lists the
internal calls and how each one is classified.

**B4. MUST: add drift checks for any snapshot of CLI internals.**
If classification depends on a model-name table, a marker string, or a model
name the CLI must honour, add one of these:

- a live drift test gated `--runslow` or `--runapi` (examples:
  `tests/test_gemini_models_drift.py`,
  `tests/test_kimi_setup.py::test_live_compaction_instruction_marker`);
- a runtime canary that logs a warning once (example:
  `LiveConsumer._check_subagent_slug_drift`).

"auto" and "latest" pick up new CLI releases without warning, so these
assumptions will drift.

**B5. MUST: add a row to `docs/sub-agent-context.qmd`.**
Add a row to the Per-Agent Guarantees table covering sub-agent detection,
utility detection, notes and known gaps, and what fills `subagent_id` and
`subagent_type`. "None, the CLI has no delegation" is a valid row (see
`kimi_code`). The row must describe the code as it is: if a test
contradicts the row, the row is wrong.

## C. Transcript and trajectory (the main work)

The viewer builds its timeline from spans with `timeline_build()` in
inspect_ai (`event/_timeline.py`). These are the rules that matter for a
bridged CLI:

- A span with `type="agent"` becomes an agent swimlane (`TimelineSpan(span_type="agent")`).
- A span id of the form `agent-{tool_call_id}` lets the viewer find the
  sub-agent's answer: the `ChatMessageTool` for that call id in a later
  `ModelEvent` input (`_extract_agent_results`).
- Within a span, a model call whose system prompt differs from the prompt of
  the first tool-calling call, and that makes no tool calls, is wrapped as a
  `utility` span. So are cache-warmup calls: `max_tokens=1` and a last user
  message that is a single word (`_wrap_utility_events`, `_is_warmup_call`).
- A child agent span with a single turn (or a single tool-calling turn) and its
  own system prompt, under a parent that loops, is marked utility
  (`_classify_utility_agents`). This applies to the sub-agent spans your
  consumer emits (they aren't `tool_invoked`), so a sub-agent that answers in
  one turn will show as utility in the viewer. That's a known viewer effect,
  not a bug in your spans.
- If a sub-agent's calls stay in the parent's span, the viewer shows one
  conversation with several threads interleaved, and its heuristics label the
  pieces "utility". That is the bug #117 and #114 describe.

**C1. IF the CLI delegates, MUST: rebuild sub-agent spans with a `ModelEventSink` consumer.**
Pass `model_event_sink=consumer` to the bridge. In the consumer:

- **Open the span when the parent's call completes.** In `on_complete`, when
  the parent's output contains a delegation tool call, emit
  `SpanBeginEvent(id=f"agent-{tool_call.id}", parent_id=<the parent's span>,
  type="agent", name=<subagent type or name>)` before the response goes back
  to the CLI. Driving span opening from stdout loses the race with the
  sub-agent's first request, which arrives about 1–2 s earlier (be5910d;
  `LiveConsumer._register_tasks`, `CodexConsumer._register_delegation`).
- **Attribute each request to its span.** Use structural signals first: a
  sub-agent model name, recipient paths (#107), headers. Use the delegated
  prompt as a substring match only as a fallback. Search every opening user
  message, not only the first, because CLIs prepend reminders *(pending: #244;
  `LiveConsumer` on main still matches the first)*. Zero matches or several
  matches go to the outer span.
- **Close the span on real completion.** A background or async sub-agent's
  first tool result is only a launch acknowledgement; keep the span open until
  the CLI reports that the sub-agent finished *(pending: #114, #244, and the
  #117 review)*.
  Parallel sub-agents each get their own span.
- **Handle resumed sub-agents.** A sub-agent resumed with a new prompt
  (opencode `task_id`, Codex `followup_task`, `send_input`) continues in a
  span with the same name or, when you can't tell which sub-agent resumed, is
  classified `"subagent"` with no id (B2; `codex_cli` under Multi-Agent V1
  still returns `"unknown"` here). It is never
  silently filed under the root (`CodexConsumer._resume`, 51adaf0; opencode
  `task_id` is *pending* in #117, since opencode has no consumer on main).
- **Classify and place from one decision.** The filter-time classification
  (`classify`) and the span placement must come from the same decision, so
  that they can't disagree. Store the decision with `remember_decision` and
  read it back with `remembered_decision` in `on_start`, `on_complete` and
  `on_filter_output`, so that a filter which rewrites the messages can't move
  the event (b69a5f9, `_util/model_event_sink.py`).

*Verify:* open a real delegating run in the viewer. Each sub-agent shows as its
own swimlane under the right parent, its result is filled in, and the main
conversation's input lengths increase monotonically, with no interleaving. Unit
tests replay recorded request shapes (see the next item and section F).

**C2. IF there is a sink, MUST: implement `on_start` with the fallback.**
`on_start(input) -> span_id` is called once per generate and decides the span
the whole generate runs under. A live sentinel follows that span. Implement it,
call `mark_started(self)`, and in `on_pending` place the event yourself only
when `not started(self)` (older inspect_ai). Do not move `event.span_id` after
`on_start`. Otherwise the live sentinel's conversation and the conversation that
replay rebuilds from the log disagree (#206, #117 review). *Verify:* a test
checks both paths, and the span from `on_start` equals the span of the recorded
event.

**C3. IF there is a sink, MUST: resolve the outer span lazily and keep spans balanced.**
Resolve the outer span with `current_span_id()` each time you emit, never once
at construction. Checkpointing rotates the enclosing span (b0879c2). Provide
`reset()`, which closes every open agent span (innermost first). Call it
between subprocess attempts and in a `finally` after the run loop, so spans
can't leak past the `@agent` boundary. `claude_code.execute` does both for its
non-centaur path; its centaur path and `codex_cli` (which resets only after each
exec) don't yet, so a new agent should do better than the reference here. *Verify:* every `SpanBeginEvent` has a matching
`SpanEndEvent`, including after a crash or retry.

**C4. SHOULD: record compaction as a `CompactionEvent`.**
When the CLI compacts its context, emit `CompactionEvent(source="<agent>",
span_id=<the compacted agent's span>, tokens_before=..., ...)` in the span of
the agent that compacted (claude_code: `compact_boundary` JSONL in
`_handle_compact_boundary` (6b1c3dc); codex: `is_compaction_request` in
`on_pending`). Without it, the viewer shows an unexplained drop in input length,
and the summarizer call looks like a turn. *Verify:* a forced or long run shows
a compaction marker in the right lane.

**C5. SHOULD: add tool views for the CLI's built-in tools.**
The CLI runs its own tools (Bash, Write, apply_patch, task or spawn), so
inspect_ai has no `ToolDef` to render them with. In `on_complete`, set
`tc.view = ToolCallContent(...)` where `tc.view is None`
(`_claude_code/_events/toolview.py` `tool_view`, `_codex_cli/_events/toolview.py`;
2cea321, 0f8e1af). The delegation tool's view should show the delegated
prompt. Agents without a sink currently show raw JSON arguments (gemini, kimi,
opencode). That is acceptable but worse. *Verify:* the viewer shows commands
and diffs as text, not JSON.

**C6. MUST: make sure the main thread wins `state.messages`.**
The bridge decides which conversation becomes `state.messages` (the sample's
answer) by matching each call against the task input (`_track_state` in
inspect_ai `agent/_bridge/types.py`). Pass the prompt to the CLI byte for byte
(stdin if the CLI escapes argv) so the main thread matches. opencode's
title call became the answer for prompts containing `"` (#151). *Verify:* the
fast tests cover a prompt with quotes and newlines (see
`tests/test_opencode_prompt.py`). In a live run, the final assistant message is
the main agent's answer.

**C7. IF the CLI resumes sessions, MUST: keep resumed turns and attempts consistent.**
Multi-turn (`multi_call`), `attempts`, and retries of uncaught errors relaunch
the CLI. Use the CLI's resume mechanism so that the model sees the full history
through the bridge (d4ae16b), and:

- don't send the system prompt again on resume; it doubled on every turn
  (#64);
- if the CLI rebuilds environment-dependent parts of its system prompt on
  resume, pin the root system prompt to its first value (#142,
  `_claude_code/systemprompt.py` `pin_system_prompt_filter`). Otherwise every
  turn misses the prompt cache, and the agent can notice that it was restarted;
- reset the consumer state per attempt (C3).

*Verify:* in a multi-turn run, each turn's system prompt is byte-identical, and
the history is not duplicated.

**C8. SHOULD: support checkpoint and resume.**
`claude_code` and `codex_cli` run the bridge under `checkpointer()`: they pass
`checkpointer=cp`, return early on `cp.attempt == "resume_for_scoring"`, and
`cp.track(...)` the session id and attempt count. A new agent should do the
same or state that it doesn't support checkpointing.

**C9. MUST: make exits and refusals map to scoreable outcomes.**
A refusal the bridge has already recorded must not turn into a sample error
just because the CLI exits non-zero (c415688, 7c52458). Take the stop reason
from the consumer (`last_stop_reason`), not by parsing stdout again. Pass
`retry_refusals` through.

**C10. MUST: check whether the CLI's system prompt varies between calls, and handle it.**
The timeline takes the system prompt of a span's first tool-calling call as the
main thread's prompt. A later call that makes no tool calls and has a
*different* prompt is shown as a utility call (`_wrap_utility_events`). If the
CLI rebuilds its system prompt per call, the main agent's text-only turns
(including its final answer) disappear into utility spans, even though every
request was classified `"root"`. Common sources of variation are timestamps,
per-request hashes or ids, cwd/git status, todo or plan state, token or
context counters, and tool lists that change mid-run.

- *Find out (inventory question 6):* capture the requests of a multi-turn,
  tool-using run and diff the system prompts of consecutive calls.
- *Prefer a CLI setting* that keeps the prompt stable (turn off the dynamic
  section, or move it into a user message), when the CLI has one.
- *Environment noise only:* if the varying part is noise the model doesn't need
  (a billing header, a request hash, the time), you may pin it to its first
  value with a filter, as `claude_code` does for resumed turns (#142,
  `pin_system_prompt_filter`). Say so in the PR. Never rewrite a part that
  carries task state the model relies on (a todo list, a plan, remaining
  context). That changes what the agent sees, and so the evaluation.
- *Fixed-format header:* if the variation is a fixed-format header or line,
  normalize it in inspect_ai's `_normalize_system_prompt` instead
  (Claude Code's `x-anthropic-billing-header:` line is the precedent), with a
  timeline test. That needs an inspect_ai PR first (see the last section).
- *Otherwise:* document the gap in the PR and in the agent's
  `sub-agent-context.qmd` notes.

*Verify:* run `timeline_build()` on a captured multi-turn log and check that
every main-thread call, including the final answer, is a turn and not a
`utility` span. A unit test over recorded requests is enough; the viewer just
renders what `timeline_build()` returns.

## D. Feature parity

Each item below is a standard parameter. Support it, or reject it with a clear
error and say so in the docs and the PR.

| Item | Requirement | Reference |
|---|---|---|
| D1 `mcp_servers`, `bridged_tools` | MUST. Accept plain `Tool`s (`resolve_bridged_tools`, #201). Before launch, and before every retry, gate on `wait_for_mcp_endpoints` (#106, #123; `mcp_ready_timeout`). Honour each server's tools allowlist and transport (#240). | `claude_code.execute` |
| D2 `skills` | SHOULD. `read_skills` at construction, `install_skills` into the directory the CLI reads (opencode also needs ripgrep, #54). | `claude_code`, `kimi_code` |
| D3 web search | SHOULD. Pass `web_search=` to the bridge to match the CLI's native tool config (#116, `_util/websearch.py`). | `claude_code.resolve_web_search_enabled` |
| D4 `attempts` | MUST. `AgentAttempts`: score after each attempt and resume with `incorrect_message`. | `claude_code.execute` loop |
| D5 multi-turn | IF the CLI can resume a session, MUST (otherwise N/A, stated in the PR). `build_user_prompt(state.messages)` returns `has_assistant_response`, which switches to resume. | all agents |
| D6 `system_prompt` | MUST. Append (and replace, if the CLI allows it) without duplication (C7). | `_system_prompt_args` |
| D7 `filter` | MUST. Wrapped by `classify_filter` (B1). Legacy `str`-first filters still work. | `_util/agentcontext.py` |
| D8 `centaur` | SHOULD. `CentaurOptions` and `run_centaur`; the sink still runs, and spans are closed afterwards (C3; the reference centaur path doesn't call `reset()` yet). | `run_claude_code_centaur` |
| D9 `cwd`, `env`, `user`, `sandbox`, `model_aliases`, `debug`, `retry_refusals` | MUST. Same names and meanings as the other agents. | all agents |
| D10 interactive (ACP) variant | IF the CLI speaks ACP. Use `model_resolver` (#182) and a slug-only classifier, and add a docs table row. | `acp/_agents/*` |
| D11 sandbox identity scope | MUST. Either call `refuse_sandbox_identity` before any setup, or install through the shared installer (`install_shared_artifact`) and run as the member (`resolve_agent_identity`). | `gemini_cli`; `claude_code`, `codex_cli` |

## E. Public API and docs

**E1. MUST: follow the API conventions.**
The factory is `@agent`, returns `agent_with(execute, name=..., description=...)`,
and is exported from `_registry.py` and `__init__.py` (`__all__`). Follow the
AGENTS.md public API rules: `Literal` types with runtime validation, and new
parameters added at the end. Reuse the parameter vocabulary in D9. Don't add a
new name for an existing concept.

**E2. MUST: write the docs.**

- Add `docs/<agent>.qmd` with the standard snippets (`agent_intro`,
  `agent_mcp_servers`, `agent_bridged_tools`, `agent_installation`,
  `agent_binary_provenance`, `agent_centaur`, `agent_troubleshooting`).
- Add an entry to `docs/_quarto.yml`.
- Add a row to the agent table in `docs/index.qmd` and a mention in its
  opening sentence.
- Add a `### <agent>` entry to `docs/reference/index.qmd`. `kimi_code` is
  missing from it today; don't copy that omission.
- Add the `docs/sub-agent-context.qmd` row (B5).

Don't edit `CHANGELOG.md`.

**E3. MUST: keep dependencies host-safe.**
The agent's runtime must not become a host dependency. A package used only in
the sandbox gets a mypy override rather than an entry in `dependencies`
(#94). If you need an unreleased inspect_ai, follow CONTRIBUTING.md (the
floor has to be published first).

## F. Tests

**F1. MUST: fast unit tests (run in PR CI, no sandbox or API key needed).**

- Setup and command building: config written, env, argv or stdin, resume flags
  (`tests/test_kimi_setup.py`, `tests/test_opencode_setup.py`,
  `tests/test_opencode_prompt.py`).
- Binary source: version resolution, platform mapping, checksum, cache
  (`tests/test_<agent>_agentbinary.py`, `tests/test_agentbinary_install.py`).
- Classifier truth table, including negative cases (B2), the user filter
  seeing the context, and a filter that returns `GenerateInput` or
  `ModelOutput` (`tests/test_agent_context_*.py`).
- **IF there is a consumer:** span tests that replay *real* request and output
  shapes captured from the CLI. Cover:
  - a foreground sub-agent;
  - a background sub-agent;
  - parallel sub-agents;
  - a resumed sub-agent;
  - compaction;
  - `reset()` balance;
  - `on_start` alongside the fallback without it.

  Use a stub transcript (`tests/test_claude_code_subagent_spans.py`,
  `tests/test_codex_subagent_spans.py`). Say which CLI version the fixtures
  were captured from.

**F2. MUST: add the agent to the shared example tests.**
Add a `case` for the agent to `examples/*/task.py` and add it to the `agent`
`Literal` in `tests/conftest.py::run_example`. Add a test to each of:
`test_system_explorer`, `test_bridged_tools`, `test_mcp`, `test_skills`,
`test_multi_call`, `test_attempts`, `test_web_search`. Use the gates
`@pytest.mark.slow`, `skip_if_no_<provider>` and `skip_if_no_docker`, and
parametrize over `get_available_sandboxes()`. Leave out an example only with a
reason in the PR (today `kimi_code` is in two of them). `run_example` already
enforces completion and limits; don't loosen its `time_limit` or
`token_limit` without reading the comment there.

**F3. MUST: run the slow tests locally and report them.**
The PR reports which `--runslow --runapi` tests ran locally, with which
provider and sandbox, and the result. PR CI runs only the fast suite. A run in
the slow suite that delegates is the only end-to-end check of section C.

## G. PR description

Include:

1. The inventory from section 0.
2. The classification table for the agent (the `sub-agent-context.qmd` row)
   and the list of internal calls.
3. A screenshot or description of a delegating run (or a compaction run) in
   the viewer: the swimlanes, sub-agent results, and no interleaving. Say
   whether the system prompt varies between calls and how C10 is handled.
4. The CLI versions tested and fixtures captured, and the slow tests run (F3).
5. Every MUST or SHOULD item marked N/A or left out, with the reason.
6. Any inspect_ai change the agent needs (see below).
7. The `### Agent review` disclosure required by AGENTS.md.

## When inspect_ai needs a change

Most of what an agent needs is generic: the bridge, `AgentBridgeContext`, the
`ModelEventSink`, and the span rules in the timeline. Some past agents needed
inspect_ai changes, though, and a new one may too. Find out early, because an
inspect_swe PR can't merge until the inspect_ai change is released.

- **The bridge can't parse the CLI's wire format.** Codex V2 `agent_message`
  items needed inspect_ai#4787, and opencode tool calls with `id: null` needed
  inspect_ai#5426.
- **The CLI hides the prompt from main-thread tracking.** opencode's quote
  escaping needed inspect_ai#4768. The bridge still doesn't match the escaped
  form; inspect_swe works around it by sending the prompt on stdin (#151).
- **Classification needs signals beyond the model name.** The bridge exposes
  only `BridgeRequest.model`. Per-session headers aren't available yet (#189).
- **The CLI varies its system prompt per call** in a fixed-format way that
  should be normalized for every user (see C10). Claude Code's per-call
  `x-anthropic-billing-header:` line is the precedent, in
  `_normalize_system_prompt`.
- **Utility classification doesn't reach the log.** `AgentBridgeContext("utility")`
  is not recorded on the `ModelEvent`. Bridged calls are recorded with
  `origin="agent"`, so the viewer's prompt heuristics alone decide what is
  utility: an internal call that shares the main system prompt shows as a
  main-thread turn, whether or not it makes tool calls. inspect_ai has
  `model_origin("helper")`, which the timeline and sentinels respect. It is a
  `with` block around the generate, though, so short of the filter running the
  generate itself (and skipping the bridge's own generate path), there is no
  supported way to connect the classifier's `"utility"` to `origin` without a
  bridge change. Until then, a `CompactionEvent` (C4) is the only other marker
  an agent can add.
