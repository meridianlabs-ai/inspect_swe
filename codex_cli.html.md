# Codex CLI – Inspect SWE

## Overview

The `codex_cli()` agent uses the unattended mode of OpenAI [Codex CLI](https://github.com/openai/codex) to execute agentic tasks within the Inspect sandbox. Model API calls that occur in the sandbox are proxied back to Inspect for handling by the model provider for the current task.

> **NOTE: NoteCodex CLI Installation**
>
> By default, the agent will download the current stable version of Codex CLI and copy it to the sandbox. You can also exercise more explicit control over which version of Codex CLI is used—see the [Installation](#installation) section below for details.

## Basic Usage

Use the `codex_cli()` agent as you would any Inspect agent. For example, here we use it as the solver in an Inspect task:

``` python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

from inspect_swe import codex_cli

@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=codex_cli(),
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

You can also pass the agent as a `--solver` on the command line:

``` bash
inspect eval ctf.py --solver inspect_swe/codex_cli
```

If you want to try this out locally, see the [system_explorer](https://github.com/meridianlabs-ai/inspect_swe/tree/main/examples/system_explorer/task.py) example.

## Options

The following options are supported for customizing the behavior of the agent:

| Option | Description |
|----|----|
| `system_prompt` | Additional system prompt to append to default system prompt. |
| `model_config` | Codex model slug used to select the system prompt and tool set. Defaults to `None`, which derives the slug from the model used by the agent so Codex’s prompt/tooling aligns with what’s actually running. |
| `skills` | Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent. |
| `mcp_servers` | MCP servers (see [MCP Servers](#mcp-servers) below for details). |
| `bridged_tools` | Host-side Inspect tools to expose via MCP (see [Bridged Tools](#bridged-tools) below for details). |
| `web_search` | Web search mode. Use `"live"` for live web search, `"cached"` for cached web search, or `"disabled"` to disable web search. Defaults to `"live"`. |
| `goals` | Enable Codex goal tools. Defaults to `True`. |
| `auto_review` | Enable Codex [automated approval review](https://developers.openai.com/codex/concepts/sandboxing/auto-review) (see [Auto Review](#auto-review) below for details). Defaults to `False`. |
| `centaur` | Run in [Centaur Mode](#centaur-mode), which makes Codex CLI available to an Inspect `human_cli()` agent rather than running it unattended. |
| `attempts` | Allow the agent to have multiple scored attempts at solving the task. |
| `model` | Model name to use for agent (defaults to main model for task). |
| `filter` | Filter for intercepting bridged model requests. |
| `retry_refusals` | Should refusals be retried? (pass number of times to retry) |
| `home_dir` | Home directory to use for codex cli. When set, AGENTS.md and the MCP configuration will be written here rather than to .codex |
| `cwd` | Working directory for Codex CLI session. |
| `env` | Environment variables to set for Codex CLI. |
| `version` | Version of Codex CLI to use (see [Installation](#installation) below for details) |
| `config_overrides` | Additional Codex CLI configuration overrides. |

For example, here we specify a custom system prompt and disable the web search and goals tools:

``` python
codex_cli(
    system_prompt="You are an ace system researcher.",
    web_search="disabled",
    goals=False,
)
```

## MCP Servers

You can specify one or more [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro) (MCP) servers to provide additional tools to Codex CLI. Servers are specified via the [`MCPServerConfig`](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#mcpserverconfig) class and its Stdio and HTTP variants.

For example, here is a Dockerfile that makes the `server-memory` MCP server available in the sandbox container:

``` dockerfile
FROM python:3.12-bookworm

# nodejs (required by mcp server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# memory mcp server
RUN npx --yes @modelcontextprotocol/server-memory --version

# run forever
CMD ["tail", "-f", "/dev/null"]
```

Note that we run the `npx` server during the build of the Dockerfile so that it is cached for use offline (below we’ll run it with the `--offline` option).

We can then use this MCP server in a task as follows:

``` python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.tool import MCPServerConfigStdio
from inspect_swe import codex_cli

@task
def investigator() -> Task:
    return Task(
        dataset=[
            Sample(
                input="What transport protocols are supported in "
                + " the 2025-03-26 version of the MCP spec?"
            )
        ],
        solver=codex_cli(
            system_prompt="Please use the web search tool to "
            + "research this question and the memory tools "
            + "to keep track of your research.",
            mcp_servers=[
                MCPServerConfigStdio(
                    name="memory",
                    command="npx",
                    args=[
                        "--offline",
                        "@modelcontextprotocol/server-memory"
                    ],
                )
            ]
        ),
        sandbox=("docker", "Dockerfile"),
    )
```

Note that we run the MCP server using the `--offline` option so that it doesn’t require an internet connection (which it would normally use to check for updates to the package).

## Bridged Tools

You can expose host-side Inspect tools to the sandboxed agent via the MCP protocol using the `bridged_tools` parameter. This allows you to run tools on the host (e.g. tools that access host resources, databases, or APIs) but make them available to the agent running inside the sandbox.

Tools are specified via [`BridgedToolsSpec`](https://inspect.aisi.org.uk/reference/inspect_ai.agent.html#bridgedtoolsspec) which wraps a list of Inspect tools:

``` python
from inspect_ai import Task, task
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.dataset import Sample
from inspect_ai.tool import tool
from inspect_swe import codex_cli

@tool
def search_database():
    async def execute(query: str) -> str:
        """Search the internal database.

        Args:
            query: The search query.
        """
        # This runs on the host, not in the sandbox
        return f"Results for: {query}"
    return execute

@task
def investigator() -> Task:
    return Task(
        dataset=[
            Sample(input="Search for information about MCP protocols.")
        ],
        solver=codex_cli(
            system_prompt="Use the search tool to research.",
            bridged_tools=[
                BridgedToolsSpec(
                    name="host_tools",
                    tools=[search_database()]
                )
            ]
        ),
        sandbox=("docker", "Dockerfile"),
    )
```

The `name` field identifies the MCP server and will be visible to the agent as a tool prefix. You can specify multiple `BridgedToolsSpec` instances to create separate MCP servers for different tool groups.

See the [Bridged Tools](https://inspect.aisi.org.uk/agent-bridge.html#bridged-tools) documentation for more details on the architecture and how tool execution flows between host and sandbox.

## Auto Review

By default, the agent runs Codex with `--dangerously-bypass-approvals-and-sandbox`: Codex’s internal sandbox and approval prompts are disabled, since the agent already runs inside the Inspect sandbox. The `auto_review` option instead runs Codex the way its own “Approve for me” mode ships: Codex’s `workspace-write` sandbox is active, `approval_policy` is `on-request`, and escalation requests (network access, writes outside the workspace, MCP tool approvals) are adjudicated by an automated “guardian” reviewer model rather than a human:

``` python
codex_cli(auto_review=True)
```

Enabling `auto_review` requires Codex CLI \>= 0.137.0 and adds constraints on what the agent can do without review. It works whether or not Codex’s own OS-level sandbox is available, but *how* those constraints are enforced differs. On Linux, Codex enforces the `workspace-write` sandbox by launching each command through [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`), which needs both a `bwrap` binary in the sandbox and a container configured to permit the namespace operations it requires (see [Running the sandbox in a container](#running-the-sandbox-in-a-container) below). When both are present, commands run under the OS sandbox and only genuine escalations (network access, out-of-workspace writes) reach the guardian.

When `bwrap` is unavailable, which is the case for most sandbox images out of the box, `auto_review` still works, but review is model-driven rather than OS-enforced. Codex cannot start its sandbox, so each sandboxed command fails with a `bubblewrap is unavailable` error returned to the model; Codex does *not* fall back to a review-only mode on its own. A capable model recovers by re-issuing the command with escalated permissions (`require_escalated`), which the guardian then reviews and, if approved, runs outside the sandbox. Models typically carry that escalation forward. The trade-off is that review now depends on the model choosing to escalate (an un-escalated command simply hits the sandbox error again), rather than being enforced by Codex. Either way, the guardian may deny an escalation, and repeated denials interrupt the turn.

When `bwrap` is available, the `workspace-write` policy defines what a command may do before it needs an approved escalation. The workspace root is the agent’s working directory (the `cwd` option): commands can read the whole filesystem but can only write within the working directory plus `/tmp` and `$TMPDIR`, top-level `.git` and `.agents` directories are read-only even inside the workspace (so e.g. `git commit` requires an approved escalation), and commands have no network access by default. Actions outside these bounds fail in the sandbox and proceed only if the model requests escalation and the guardian approves. If your task layout requires more, pass `config_overrides` (applied before the auto_review settings, so they compose), for example `{"sandbox_workspace_write.network_access": "true"}` or `{"sandbox_workspace_write.writable_roots": '["/data"]'}`, and/or set `cwd` so the workspace root matches where the task’s files live.

### Running the sandbox in a container

To make Codex’s `workspace-write` sandbox actually enforce in a Docker sandbox, rather than falling back to the model-driven path above, do both of the following.

**1. Provide `bwrap` in the sandbox image.** Codex looks for `bwrap` on the `PATH` (or bundled at `codex-resources/bwrap` next to its binary). Install it in your Dockerfile:

``` dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*
```

**2. Permit namespace creation.** Docker’s default `seccomp` profile blocks the namespace and mount syscalls `bwrap` uses, so it fails with `Creating new namespace failed: Operation not permitted` even when installed. Launch the sandbox container with `seccomp` unconfined via a compose file:

``` yaml
# compose.yaml
services:
  default:
    build: .
    command: "tail -f /dev/null"
    init: true
    security_opt:
      - seccomp=unconfined
```

Reference that compose file from the task’s sandbox, e.g. `sandbox=("docker", "compose.yaml")`. With both pieces in place, sandboxed commands run under `bwrap` and only genuine escalations (network access, out-of-workspace writes) reach the guardian, with no `bubblewrap is unavailable` failures.

> **WARNING: Warning**
>
> `seccomp=unconfined` removes a layer of the container’s isolation. By default, Docker applies a seccomp profile that blocks the container from making a set of Linux system calls which containerized code rarely needs but which have historically been the vector for kernel-level privilege escalation and container escapes (for example namespace creation, `mount`/`pivot_root`, and the keyring and `bpf` calls). Setting `seccomp=unconfined` lifts that filter, making the kernel’s entire system-call surface reachable from inside the sandbox.
>
> Concretely, this removes defense-in-depth against a compromised or adversarial command escaping the container by exploiting a bug in the host kernel: it does not grant an escape on its own, but it removes a mitigation that would otherwise have to be defeated first. (This matters here precisely because `bwrap`’s own sandboxing depends on some of those same namespace/mount syscalls, which is why enabling it requires unblocking them.)
>
> Enable it only for sandboxes running trusted or already-isolated workloads, and weigh it against the fact that the agent may run untrusted, model-generated commands. For security-sensitive deployments, prefer a tailored seccomp profile that unblocks *only* the syscalls bubblewrap needs (`unshare`, `clone`, `mount`, `pivot_root`, `umount2`, `setns`) and keeps the rest of the default block-list intact, rather than disabling seccomp wholesale.

A few caveats:

- The host kernel must support unprivileged user namespaces (standard on modern Linux distributions). `--privileged` also works but grants far more than necessary.
- On macOS/Windows Docker, containers run in a Linux VM, so these settings apply to the VM’s kernel; behavior matches native Linux.

Use `CodexAutoReview` to customize the guardian policy (extra instructions inserted into the guardian review prompt) and the model that serves guardian requests (an Inspect model role name or a model; by default guardian requests are served by the model the agent is running with):

``` python
from inspect_swe import CodexAutoReview

codex_cli(
    auto_review=CodexAutoReview(
        policy="Deny anything that installs packages.",
        model="guardian",  # binds the 'guardian' model role
    )
)
```

## Installation

By default, the agent will download the current stable version of Codex CLI and copy it to the sandbox. You can override this behaviour using the `version` option:

| Option | Description |
|----|----|
| `"auto"` | Use any available version of Codex CLI in the sandbox, otherwise download the latest version. |
| `"sandbox"` | Use the version of Codex CLI in the sandbox (raises `RuntimeError` if not available in the sandbox) |
| `"latest"` | Download and use the very latest version. |
| `"x.x.x"` | Download and use a specific version number. |

If you don’t ever want to rely on automatic downloads of Codex CLI (e.g. if you run your evaluations offline), you can use one of two approaches:

1.  Pre-install the version of Codex CLI you want to use in the sandbox, then use `version="sandbox"`:

    ``` python
    codex_cli(version="sandbox")
    ```

2.  Download the version of Codex CLI you want to use into the cache, then specify that version explicitly:

    ``` python
    # download the agent binary during installation/configuration
    download_agent_binary("codex_cli", "0.29.0", "linux-x64")

    # reference that version in your task (no download will occur)
    codex_cli(version="0.29.0")
    ```

    Note that the 5 most recently downloaded versions are retained in the cache. Use the [cached_agent_binaries()](./reference/index.html.md#cached_agent_binaries) function to list the contents of the cache.

## Centaur Mode

The `codex_cli()` agent can also be run in “centaur” mode which uses the Inspect AI [Human Agent](https://inspect.aisi.org.uk/human-agent.html) as the solver and makes [Codex CLI](https://github.com/openai/codex) available to the human user for help with the task. So rather than strictly measuring human vs. model performance, you are able to measure performance of humans working collaboratively with a model.

Enable centaur mode by passing `centaur=True` to the `codex_cli()` agent:

``` python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

from inspect_swe import codex_cli

@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=codex_cli(centaur=True),
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

You can also enable centaur mode from the CLI using a solver arg (`-S`):

``` bash
inspect eval ctf.py --solver inspect_swe/codex_cli -S centaur=true
```

You can also pass `CentaurOptions` to further customize the behavior of the human agent. For example:

``` python
from inspect_swe import CentaurOptions

Task(
    dataset=json_dataset("dataset.json"),
    solver=codex_cli(centaur=CentaurOptions(answer=False)),
    scorer=model_graded_qa(),
    sandbox="docker",
)
```

See the [human_cli()](https://inspect.aisi.org.uk/reference/inspect_ai.agent.html#human_cli) documentation for details on available options.

## Troubleshooting

If Codex CLI doesn’t appear to be working or working as expected, you can troubleshoot by dumping the Codex CLI debug log after an evaluation task is complete. You can do this with:

``` bash
inspect trace dump --filter "Codex CLI"
```
