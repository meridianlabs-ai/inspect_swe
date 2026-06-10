# Reference – Inspect SWE

## Agents

### claude_code

Claude Code agent.

Agent that uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) running in a sandbox.

The agent can either use a version of Claude Code installed in the sandbox, or can download a version and install it in the sandbox (see docs on `version` option below for details).

Use `disallowed_tools` to control access to tools. See [Tools available to Claude](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for the list of built-in tools which can be disallowed.

Use the `attempts` option to enable additional submissions if the initial submission(s) are incorrect (by default, no additional attempts are permitted).

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_claude_code/claude_code.py#L51)

``` python
@agent
def claude_code(
    name: str = "Claude Code",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    disallowed_tools: list[str] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_config: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    opus_model: str | None = None,
    sonnet_model: str | None = None,
    haiku_model: str | None = None,
    subagent_model: str | None = None,
    filter: GenerateFilter | None = None,
    auto_mode: bool = False,
    retry_refusals: int | None = 3,
    retry_uncaught_errors: int | None = 3,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
) -> Agent
```

`name` str  
Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)

`description` str  
Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)

`system_prompt` str \| None  
Additional system prompt to append to default system prompt.

`skills` Sequence\[str \| Path \| Skill\] \| None  
Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.

`mcp_servers` Sequence\[MCPServerConfig\] \| None  
MCP servers to make available to the agent.

`bridged_tools` Sequence\[BridgedToolsSpec\] \| None  
Host-side Inspect tools to expose to the agent via MCP. Each BridgedToolsSpec creates an MCP server that makes the specified tools available to the agent running in the sandbox.

`disallowed_tools` list\[str\] \| None  
List of tool names to disallow entirely.

`centaur` bool \| CentaurOptions  
Run in ‘centaur’ mode, which makes Claude Code available to an Inspect `human_cli()` agent rather than running it unattended.

`attempts` int \| AgentAttempts  
Configure agent to make multiple attempts. When this is specified, the task will be scored when the agent stops calling tools. If the scoring is successful, execution will stop. Otherwise, the agent will be prompted to pick up where it left off for another attempt.

`model` str \| None  
Model name to use for Opus and Sonnet calls (defaults to main model for task).

`model_config` str \| None  
Model id used to select the identity Claude Code presents to itself (its “You are powered by the model …” system prompt) and any model-gated client behavior. Defaults to `None`, which derives it from the real served model so the presented identity matches what’s actually running. Purely the displayed identity — calls are still bridged to the served Inspect model regardless. (Claude Code renders the genuine name/cutoff for recognized Anthropic ids and shows other ids verbatim.)

`model_aliases` dict\[str, str \| Model\] \| None  
Optional mapping of model names to Model instances or model name strings. Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models. When a model name in the mapping is referenced, the corresponding Model/string is used.

`opus_model` str \| None  
The model to use for `opus`, or for `opusplan` when Plan Mode is active. Defaults to `model`.

`sonnet_model` str \| None  
The model to use for `sonnet`, or for `opusplan` when Plan Mode is not active. Defaults to `model`.

`haiku_model` str \| None  
The model to use for haiku, or [background functionality](https://code.claude.com/docs/en/costs#background-token-usage). Defaults to `model`.

`subagent_model` str \| None  
The model to use for [subagents](https://code.claude.com/docs/en/sub-agents). Defaults to `model`.

`filter` GenerateFilter \| None  
Filter for intercepting bridged model requests.

`auto_mode` bool  
Use `auto` permission mode rather than `--dangerously-skip-permissions`. Note that this can result in rejected tool calls so only enable if your evaluation can tolerate this.

`retry_refusals` int \| None  
Should refusals be retried? Defaults to retrying up to 3 times.

`retry_uncaught_errors` int \| None  
Should uncaught errors (unexpected crashes of Claude Code) be retried. Defaults to retrying up to 3 times.

`cwd` str \| None  
Working directory to run claude code within.

`env` dict\[str, str\] \| None  
Environment variables to set for claude code.

`user` str \| None  
User to execute claude code with.

`sandbox` str \| None  
Optional sandbox environment name.

`version` Literal\['auto', 'sandbox', 'stable', 'latest'\] \| str  
Version of claude code to use. One of: - “auto”: Use any available version of claude code in the sandbox, otherwise download the current stable version. - “sandbox”: Use the version of claude code in the sandbox (raises `RuntimeError` if claude is not available in the sandbox) - “stable”: Download and use the current stable version of claude code. - “latest”: Download and use the very latest version of claude code. - “x.x.x”: Download and use a specific version of claude code.

`debug` bool \| None  
Add `--debug` cli flag and trace all debug output.

### codex_cli

Codex CLI.

Agent that uses OpenAI [Codex CLI](https://github.com/openai/codex) running in a sandbox.

Use the `attempts` option to enable additional submissions if the initial submission(s) are incorrect (by default, no additional attempts are permitted).

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_codex_cli/codex_cli.py#L64)

``` python
def codex_cli(
    name: str = ...,
    description: str = ...,
    system_prompt: str | None = ...,
    model_config: str | None = ...,
    skills: Sequence[str | Path | Skill] | None = ...,
    mcp_servers: Sequence[MCPServerConfig] | None = ...,
    bridged_tools: Sequence[BridgedToolsSpec] | None = ...,
    web_search: CodexWebSearch = ...,
    goals: bool = ...,
    centaur: bool | CentaurOptions = ...,
    attempts: int | AgentAttempts = ...,
    model: str | None = ...,
    model_aliases: dict[str, str | Model] | None = ...,
    filter: GenerateFilter | None = ...,
    retry_refusals: int | None = ...,
    home_dir: str | None = ...,
    cwd: str | None = ...,
    env: dict[str, str] | None = ...,
    user: str | None = ...,
    sandbox: str | None = ...,
    version: Literal['auto', 'sandbox', 'latest'] | str = ...,
    config_overrides: dict[str, str] | None = ...,
    debug: bool | None = ...,
    *,
    disallowed_tools: list[Literal['web_search']] | None = ...,
) -> Agent
```

`name` str  
Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)

`description` str  
Agent description (used in multi-agent systems with `as_tool()` and `handoff()`)

`system_prompt` str \| None  
Additional system prompt to append to default system prompt.

`model_config` str \| None  
Codex model slug used to select the system prompt and tool set. Defaults to `None`, which derives the slug from the real model so Codex’s prompt/tooling aligns with what’s actually running. Pass an explicit slug to override.

`skills` Sequence\[str \| Path \| Skill\] \| None  
Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.

`mcp_servers` Sequence\[MCPServerConfig\] \| None  
MCP servers to make available to the agent.

`bridged_tools` Sequence\[BridgedToolsSpec\] \| None  
Host-side Inspect tools to expose to the agent via MCP. Each BridgedToolsSpec creates an MCP server that makes the specified tools available to the agent running in the sandbox.

`web_search` CodexWebSearch  
Web search mode. Use “live” for live web search, “cached” for cached web search, or “disabled” to disable web search. Defaults to “live”.

`goals` bool  
Enable Codex goal tools (defaults to `True`).

`centaur` bool \| CentaurOptions  
Run in ‘centaur’ mode, which makes Codex CLI available to an Inspect `human_cli()` agent rather than running it unattended.

`attempts` int \| AgentAttempts  
Configure agent to make multiple attempts. When this is specified, the task will be scored when the agent stops calling tools. If the scoring is successful, execution will stop. Otherwise, the agent will be prompted to pick up where it left off for another attempt.

`model` str \| None  
Model name to use (defaults to main model for task).

`model_aliases` dict\[str, str \| Model\] \| None  
Optional mapping of model names to Model instances or model name strings. Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models. When a model name in the mapping is referenced, the corresponding Model/string is used.

`filter` GenerateFilter \| None  
Filter for intercepting bridged model requests.

`retry_refusals` int \| None  
Should refusals be retried? (pass number of times to retry)

`home_dir` str \| None  
Home directory to use for codex cli. If set, AGENTS.md, skills, and the MCP configuration will be written here.

`cwd` str \| None  
Working directory to run codex cli within.

`env` dict\[str, str\] \| None  
Environment variables to set for codex cli

`user` str \| None  
User to execute codex cli with.

`sandbox` str \| None  
Optional sandbox environment name.

`version` Literal\['auto', 'sandbox', 'latest'\] \| str  
Version of codex cli to use. One of: - “auto”: Use any available version of codex cli in the sandbox, otherwise download the latest version. - “sandbox”: Use the version of codex cli in the sandbox (raises `RuntimeError` if codex is not available in the sandbox) - “latest”: Download and use the very latest version of codex cli. - “x.x.x”: Download and use a specific version of codex cli.

`config_overrides` dict\[str, str\] \| None  
Additional Codex CLI configuration overrides. Each key-value pair is passed as `-c key=value` to the CLI.

`debug` bool \| None  
Trace all debug output.

`disallowed_tools` list\[Literal\['web_search'\]\] \| None  

### gemini_cli

Gemini CLI agent.

Agent that uses Google [Gemini CLI](https://github.com/google-gemini/gemini-cli) running in a sandbox with Inspect model bridging.

Use the `attempts` option to enable additional submissions if the initial submission(s) are incorrect (by default, no additional attempts are permitted).

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_gemini_cli/gemini_cli.py#L33)

``` python
@agent
def gemini_cli(
    name: str = "Gemini CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    gemini_model: str = "gemini-2.5-pro",
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
) -> Agent
```

`name` str  
Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)

`description` str  
Agent description

`system_prompt` str \| None  
Additional system prompt to append

`skills` Sequence\[str \| Path \| Skill\] \| None  
Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.

`mcp_servers` Sequence\[MCPServerConfig\] \| None  
MCP servers to make available to the agent

`bridged_tools` Sequence\[BridgedToolsSpec\] \| None  
Host-side Inspect tools to expose to the agent via MCP

`centaur` bool \| CentaurOptions  
Run in ‘centaur’ mode, which makes Gemini CLI available to an Inspect `human_cli()` agent rather than running it unattended.

`attempts` int \| AgentAttempts  
Configure agent to make multiple attempts

`model` str \| None  
Model name to use for inspect bridge (defaults to main model for task)

`model_aliases` dict\[str, str \| Model\] \| None  
Optional mapping of model names to Model instances or model name strings. Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models. When a model name in the mapping is referenced, the corresponding Model/string is used.

`gemini_model` str  
Gemini model name to pass to CLI. This bypasses the auto-router. Use “gemini-2.5-pro” (default) or “gemini-2.5-flash”. The actual model calls still go through the inspect bridge, but this disables the router.

`filter` GenerateFilter \| None  
Filter for intercepting bridged model requests

`retry_refusals` int \| None  
Should refusals be retried? (pass number of times to retry)

`cwd` str \| None  
Working directory to run gemini cli within

`env` dict\[str, str\] \| None  
Environment variables to set for gemini cli

`user` str \| None  
User to execute gemini cli with

`sandbox` str \| None  
Optional sandbox environment name

`version` Literal\['auto', 'sandbox', 'stable', 'latest'\] \| str  
Version of gemini cli to use. One of: - “auto”: Use any available version in sandbox, otherwise download latest - “sandbox”: Use sandbox version (raises RuntimeError if not available) - “stable”/“latest”: Download and use the latest version - “x.x.x”: Download and use a specific version

`debug` bool \| None  
Trace all debug output.

### opencode

OpenCode agent.

Agent that uses [OpenCode](https://github.com/anomalyco/opencode) running in a sandbox with Inspect model bridging.

Use the `attempts` option to enable additional submissions if the initial submission(s) are incorrect (by default, no additional attempts are permitted).

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_opencode/opencode.py#L32)

``` python
@agent
def opencode(
    name: str = "OpenCode",
    description: str = dedent("""
       Open-source autonomous coding agent for the terminal, capable
       of writing, testing, debugging, and iterating on code across
       multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    opencode_model: str = "anthropic/claude-sonnet-4-5",
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
) -> Agent
```

`name` str  
Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)

`description` str  
Agent description

`system_prompt` str \| None  
Additional system prompt to append

`skills` Sequence\[str \| Path \| Skill\] \| None  
Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.

`mcp_servers` Sequence\[MCPServerConfig\] \| None  
MCP servers to make available to the agent

`bridged_tools` Sequence\[BridgedToolsSpec\] \| None  
Host-side Inspect tools to expose to the agent via MCP

`centaur` bool \| CentaurOptions  
Run in ‘centaur’ mode, which makes OpenCode available to an Inspect `human_cli()` agent rather than running it unattended.

`attempts` int \| AgentAttempts  
Configure agent to make multiple attempts

`model` str \| None  
Model name to use for inspect bridge (defaults to main model for task)

`model_aliases` dict\[str, str \| Model\] \| None  
Optional mapping of model names to Model instances or model name strings. Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models. When a model name in the mapping is referenced, the corresponding Model/string is used.

`opencode_model` str  
OpenCode model identifier to pass to the CLI in the form `provider/model` (default: `"anthropic/claude-sonnet-4-5"`). The actual model calls still go through the Inspect bridge; this just selects which provider client OpenCode uses to format the request.

`filter` GenerateFilter \| None  
Filter for intercepting bridged model requests

`retry_refusals` int \| None  
Should refusals be retried? (pass number of times to retry)

`cwd` str \| None  
Working directory to run opencode within

`env` dict\[str, str\] \| None  
Environment variables to set for opencode

`user` str \| None  
User to execute opencode with

`sandbox` str \| None  
Optional sandbox environment name

`version` Literal\['auto', 'sandbox', 'stable', 'latest'\] \| str  
Version of opencode to use. One of: - “auto”: Use any available version in sandbox, otherwise download latest - “sandbox”: Use sandbox version (raises RuntimeError if not available) - “stable”/“latest”: Download and use the latest version - “x.x.x”: Download and use a specific version

`debug` bool \| None  
Trace all debug output.

### mini_swe_agent

mini-swe-agent agent.

Agent that uses [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) running in a sandbox. Mini-swe-agent is a minimal 100-line agent that solves GitHub issues using only bash commands.

The agent can either use a version of mini-swe-agent installed in the sandbox, or can download and install it via pip (see docs on `version` option below).

Use `attempts` to enable additional submissions if initial submission(s) are incorrect (by default, no additional attempts are permitted).

This agent does not handle compaction natively. Use `compaction` to specify a compaction strategy.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_mini_swe_agent/mini_swe_agent.py#L48)

``` python
@agent
def mini_swe_agent(
    name: str = "mini-swe-agent",
    description: str = dedent("""
       Minimal AI agent that solves software engineering tasks using bash commands.
       100 lines of Python, radically simple, scores >74% on SWE-bench verified.
    """),
    system_prompt: str | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    compaction: CompactionStrategy | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["stable", "sandbox", "latest"] | str = "stable",
    debug: bool | None = None,
) -> Agent
```

`name` str  
Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)

`description` str  
Agent description (used in multi-agent systems)

`system_prompt` str \| None  
Additional system prompt to include (appended to any system messages from the task).

`centaur` bool \| CentaurOptions  
Run in ‘centaur’ mode, which makes mini-swe-agent available to an Inspect `human_cli()` agent rather than running it unattended.

`attempts` int \| AgentAttempts  
Configure agent to make multiple attempts.

`model` str \| None  
Model name to use (defaults to main model for task).

`model_aliases` dict\[str, str \| Model\] \| None  
Optional mapping of model names to Model instances or model name strings. Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models. When a model name in the mapping is referenced, the corresponding Model/string is used.

`filter` GenerateFilter \| None  
Filter for intercepting bridged model requests.

`retry_refusals` int \| None  
Should refusals be retried? (pass number of times to retry)

`compaction` CompactionStrategy \| None  
Compaction strategy for managing context window overflow.

`cwd` str \| None  
Working directory to run mini-swe-agent within.

`env` dict\[str, str\] \| None  
Environment variables to set for mini-swe-agent.

`user` str \| None  
User to execute mini-swe-agent with.

`sandbox` str \| None  
Optional sandbox environment name.

`version` Literal\['stable', 'sandbox', 'latest'\] \| str  
Version of mini-swe-agent to use. One of: - “stable”: Download and install the default pinned version. - “sandbox”: Use version in sandbox (raises RuntimeError if not available) - “latest”: Download and install latest version from PyPI. - “x.x.x”: Install and use a specific version.

`debug` bool \| None  
Trace all debug output.

## Binaries

### download_agent_binary

Download agent binary.

Download an agent binary. This version will be added to the cache of downloaded versions (which retains the 5 most recently downloaded versions).

Use this if you need to ensure that a specific version of an agent binary is downloaded in advance (e.g. if you are going to run your evaluations offline). After downloading, explicit requests for the downloaded version (e.g. `claude_code(version="1.0.98")`) will not require network access.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_tools/download.py#L53)

``` python
def download_agent_binary(
    binary: Literal["claude_code", "codex_cli"],
    version: Literal["stable", "latest"] | str,
    platform: SandboxPlatform,
) -> None
```

`binary` Literal\['claude_code', 'codex_cli'\]  
Type of binary to download

`version` Literal\['stable', 'latest'\] \| str  
Version to download (“stable”, “latest”, or an explicit version number).

`platform` [SandboxPlatform](../reference/index.html.md#sandboxplatform)  
Target platform (“linux-x64”, “linux-arm64”, “linux-x64-musl”, or “linux-arm64-musl”)

### cached_agent_binaries

List the agent binaries which have been cached on this system.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_tools/download.py#L80)

``` python
def cached_agent_binaries(
    binary: Literal["claude_code", "codex_cli"] | None = None, quiet: bool = False
) -> AgentBinaries
```

`binary` Literal\['claude_code', 'codex_cli'\] \| None  
Type of binary to list (lists all of if not specified).

`quiet` bool  
Do not print the binaries as a side effect

### download_wheels_tarball

Download all wheels for a package and its dependencies.

Downloads wheels from PyPI for the specified platform and Python version, then bundles them into a tarball for offline installation in sandbox. Downloaded wheels are cached locally (retaining 5 most recent versions).

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_util/agentwheel.py#L304)

``` python
def download_wheels_tarball(
    package_name: str,
    version: str | None,
    platform: SandboxPlatform,
    python_version: str,
) -> tuple[bytes, str]
```

`package_name` str  
PyPI package name (e.g., “mini-swe-agent”)

`version` str \| None  
Package version or None for latest

`platform` [SandboxPlatform](../reference/index.html.md#sandboxplatform)  
Target sandbox platform

`python_version` str  
Python version without dots (e.g., “312”)

### AgentBinary

Agent binary.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_tools/download.py#L15)

``` python
class AgentBinary(NamedTuple)
```

#### Attributes

`agent` Literal\['claude_code', 'codex_cli'\]  
Agent type.

`version` str  
Agent version.

`path` Path  
“Agent path.

### SandboxPlatform

Target platform identifier for sandbox binary and wheel downloads.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/_util/sandbox.py#L5)

``` python
SandboxPlatform: TypeAlias = Literal[
    "linux-x64", "linux-arm64", "linux-x64-musl", "linux-arm64-musl"
]
```

## ACP

### interactive_claude_code

Claude Code agent via ACP.

Uses the `claude-agent-acp` adapter in a sandbox. Supports multi-turn sessions and mid-turn interrupts.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/acp/_agents/claude_code/claude_code.py#L167)

``` python
def interactive_claude_code(
    *,
    disallowed_tools: list[str] | None = ...,
    skills: list[str | Path | Skill] | None = ...,
    opus_model: str | Model | None = ...,
    sonnet_model: str | Model | None = ...,
    haiku_model: str | Model | None = ...,
    subagent_model: str | Model | None = ...,
    model: str | Model | None = ...,
    filter: GenerateFilter | None = ...,
    bridged_tools: list[BridgedToolsSpec] | None = ...,
    mcp_servers: list[MCPServerConfig] | None = ...,
    system_prompt: str | None = ...,
    retry_refusals: int | None = ...,
    model_map: dict[str, str | Model] | None = ...,
    cwd: str | None = ...,
    env: dict[str, str] | None = ...,
    user: str | None = ...,
    sandbox: str | None = ...,
) -> ACPAgent
```

`disallowed_tools` list\[str\] \| None  
Tool names to disallow.

`skills` list\[str \| Path \| Skill\] \| None  
Additional skills to make available.

`opus_model` str \| Model \| None  
Model for opus calls.

`sonnet_model` str \| Model \| None  
Model for sonnet calls.

`haiku_model` str \| Model \| None  
Model for haiku / background calls.

`subagent_model` str \| Model \| None  
Model for subagents.

`model` str \| Model \| None  

`filter` GenerateFilter \| None  

`bridged_tools` list\[BridgedToolsSpec\] \| None  

`mcp_servers` list\[MCPServerConfig\] \| None  

`system_prompt` str \| None  

`retry_refusals` int \| None  

`model_map` dict\[str, str \| Model\] \| None  

`cwd` str \| None  

`env` dict\[str, str\] \| None  

`user` str \| None  

`sandbox` str \| None  

### interactive_codex_cli

Codex CLI agent via ACP.

Uses the `codex-acp` adapter in a sandbox. Supports multi-turn sessions and mid-turn interrupts.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/acp/_agents/codex_cli/codex_cli.py#L197)

``` python
def interactive_codex_cli(
    *,
    web_search: CodexWebSearch = ...,
    goals: bool = ...,
    skills: list[str | Path | Skill] | None = ...,
    home_dir: str | None = ...,
    config_overrides: dict[str, str] | None = ...,
    disallowed_tools: list[Literal['web_search']] | None = ...,
) -> ACPAgent
```

`web_search` CodexWebSearch  
Web search mode. Use `"live"` for live web search, `"cached"` for cached web search, or `"disabled"` to disable web search.

`goals` bool  
Enable Codex goal tools.

`skills` list\[str \| Path \| Skill\] \| None  
Additional skills to make available.

`home_dir` str \| None  
Override for `CODEX_HOME` directory in the sandbox.

`config_overrides` dict\[str, str\] \| None  
Extra Codex config.toml key-value pairs.

`disallowed_tools` list\[Literal\['web_search'\]\] \| None  

### interactive_gemini_cli

Gemini CLI agent via ACP.

Uses gemini’s native `--experimental-acp` flag in a sandbox. Supports multi-turn sessions and mid-turn interrupts.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/acp/_agents/gemini_cli/gemini_cli.py#L152)

``` python
def interactive_gemini_cli(
    *,
    skills: list[str | Path | Skill] | None = ...,
    version: Literal['auto', 'sandbox', 'stable', 'latest'] | str = ...,
    debug: bool = ...,
    model: str | Model | None = ...,
    filter: GenerateFilter | None = ...,
    bridged_tools: list[BridgedToolsSpec] | None = ...,
    mcp_servers: list[MCPServerConfig] | None = ...,
    system_prompt: str | None = ...,
    retry_refusals: int | None = ...,
    model_map: dict[str, str | Model] | None = ...,
    cwd: str | None = ...,
    env: dict[str, str] | None = ...,
    user: str | None = ...,
    sandbox: str | None = ...,
) -> ACPAgent
```

`skills` list\[str \| Path \| Skill\] \| None  
Additional skills to make available.

`version` Literal\['auto', 'sandbox', 'stable', 'latest'\] \| str  
Version of gemini CLI to use. One of: `"auto"`, `"sandbox"`, `"stable"`, `"latest"`, or a specific semver version string.

`debug` bool  
Run gemini-cli with `--debug` and `GEMINI_DEBUG_LOG_FILE` set to `$HOME/gemini-debug.log` in the sandbox (in ACP mode console output is patched away from stderr, so the log file is the only way to surface internals).

`model` str \| Model \| None  

`filter` GenerateFilter \| None  

`bridged_tools` list\[BridgedToolsSpec\] \| None  

`mcp_servers` list\[MCPServerConfig\] \| None  

`system_prompt` str \| None  

`retry_refusals` int \| None  

`model_map` dict\[str, str \| Model\] \| None  

`cwd` str \| None  

`env` dict\[str, str\] \| None  

`user` str \| None  

`sandbox` str \| None  

### bridge_mcp_to_acp

Convert bridge `MCPServerConfigHTTP` objects to ACP `HttpMcpServer`.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/acp/agent.py#L29)

``` python
def bridge_mcp_to_acp(configs: list[MCPServerConfigHTTP]) -> list[HttpMcpServer]
```

`configs` list\[MCPServerConfigHTTP\]  

### ACPAgent

Base class for ACP-based agents running in sandboxes.

Manages the ACP lifecycle (connection, session, MCP announcement, cleanup). Subclasses implement :meth:`_start_agent` for agent-specific setup.

Sets up the ACP lifecycle, exposes `.conn` and `.session_id`, signals `.ready`, then blocks until the task is cancelled. The caller drives all prompts via `conn.prompt()` / `conn.cancel()`.

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/acp/agent.py#L76)

``` python
class ACPAgent(Agent)
```

### ACPAgentParams

Keyword arguments accepted by :class:[ACPAgent](../reference/index.html.md#acpagent).

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/acp/agent.py#L45)

``` python
class ACPAgentParams(TypedDict, total=False)
```

### acp_connection

Bridge an `ExecRemoteProcess` to ACP. Yield `(conn, feeder, error_info)`.

Bridges `ExecRemoteProcess` to the SDK’s `connect_to_agent()` via a transport wrapper, then cleans up on exit.

*feeder* is a background task that reads process stdout and feeds it into the ACP reader. It completes when the process exits, so callers can `await feeder` to detect unexpected process termination.

*proc_info* collects stderr output and the exit code as the process runs. Inspect after `await feeder` for full diagnostics.

Usage::

    async with acp_connection(proc) as (conn, feeder, proc_info):
        await conn.initialize(...)
        session = await conn.new_session(...)
        await conn.prompt(...)

[Source](https://github.com/meridianlabs-ai/inspect_swe/blob/49a7c3004a872b87f9c22fc53036b397b660716f/src/inspect_swe/acp/client.py#L255)

``` python
@contextlib.asynccontextmanager
async def acp_connection(
    proc: ExecRemoteProcess,
) -> AsyncIterator[tuple[ClientSideConnection, asyncio.Task[None], ErrorInfo]]
```

`proc` ExecRemoteProcess  
