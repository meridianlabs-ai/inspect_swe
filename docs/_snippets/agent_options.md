## Options

The following options are supported for customizing the behavior of the agent:

| Option | Description |
|------------------------------------|------------------------------------|
| `system_prompt` | Additional system prompt to append to default system prompt. |
| `mcp_servers` | MCP servers (see [MCP Servers](#mcp-servers) below for details). |
| `disallowed_tools` | Optionally disallow tools (e.g. `"{{< meta agent_disallowed_tool >}}"`) |
| `attempts` | Allow the agent to have multiple scored attempts at solving the task. |
| `model` | Model name to use for agent (defaults to main model for task). |
| `filter` | Filter for intercepting bridged model requests. |
| `retry_refusals` | Should refusals be retried? (pass number of times to retry) |
| `cwd` | Workding directory for {{< meta agent_name >}} session. |
| `env` | Environment variables to set for {{< meta agent_name >}}. |
| `version` | Version of {{< meta agent_name >}} to use (see [Installation](#installation) below for details) |

: {tbl-colwidths=\[25,75\]}

For example, here we specify a custom system prompt and disallow the `{{< meta agent_disallowed_tool >}}` tool:

```python
{{< meta agent >}}(
    system_prompt="You are an ace system researcher.",
    disallowed_tools=["{{< meta agent_disallowed_tool >}}"]
)
```
