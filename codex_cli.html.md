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
| `model_config` | Codex model slug used to select the system prompt and tool set. Defaults to `None`, which derives the slug from the model used by the agent. |
| `skills` | Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent. |
| `mcp_servers` | MCP servers (see [MCP Servers](#mcp-servers) below for details). |
| `bridged_tools` | Host-side Inspect tools to expose via MCP (see [Bridged Tools](#bridged-tools) below for details). |
| `web_search` | Web search mode. Use `"live"` for live web search, `"cached"` for cached web search, or `"disabled"` to disable web search. Defaults to `"live"`. |
| `goals` | Enable Codex goal tools. Defaults to `True`. |
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
