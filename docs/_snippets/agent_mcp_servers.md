## MCP Servers {#mcp-servers}

You can specify one or more [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro) (MCP) servers to provide additional tools to Codex CLI. Servers are specified via the [`MCPServerConfig`](https://inspect.aisi.org.uk/reference/inspect_ai.tool.html#mcpserverconfig) class and its Stdio and HTTP variants.

For example, here is a Dockerfile that makes the `server-memory` MPC server available in the sandbox container:

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

Note that we run the `npx` server during the build of the Dockerfile so that it is cached for use offline (below we'll run it with the `--offline` option).

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
        solver={{< meta agent >}}(
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

Note that we run the MCP server using the `--offline` option so that it doesn't require an internet connection (which it would normally use to check for updates to the package).
