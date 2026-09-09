
If you don't ever want to rely on automatic downloads of {{< meta agent_name >}} (e.g. if you run your evaluations offline), you can use one of two approaches:

1.  Pre-install the version of {{< meta agent_name >}} you want to use in the sandbox, then use `version="sandbox"`:

    ``` python
    {{< meta agent >}}(version="sandbox")
    ```

2.  For the agents that `download_agent_binary()` supports — `antigravity_cli`, `claude_code`, `codex_cli`, `kimi_code`, and `opencode` — download the version you want into the cache, then specify that version explicitly:

    ``` python
    # download the agent binary during installation/configuration
    download_agent_binary("claude_code", "2.1.258", "linux-x64")

    # reference that version in your task (no download will occur)
    claude_code(version="2.1.258")
    ```

    Note that for those agents the 3 most recently downloaded versions are retained in the cache. Use the `cached_agent_binaries()` function to list the contents of the cache.