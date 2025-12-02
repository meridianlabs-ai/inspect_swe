## Bridged Tools {#bridged-tools}

You can expose host-side Inspect tools to the sandboxed agent via the MCP protocol using the `bridged_tools` parameter. This is useful when you have tools that need to run on the host (e.g. tools that access host resources, databases, or APIs) but want them available to the agent running inside the sandbox.

Tools are specified via [`BridgedToolsSpec`](https://inspect.aisi.org.uk/reference/inspect_ai.agent.html#bridgedtoolsspec) which wraps a list of Inspect tools:

``` python
from inspect_ai import Task, task
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.dataset import Sample
from inspect_ai.tool import tool
from inspect_swe import {{< meta agent >}}

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
        solver={{< meta agent >}}(
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
