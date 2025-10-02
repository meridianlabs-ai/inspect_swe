
If you don't ever want to rely on automatic downloads of {{< meta agent_name >}} (e.g. if you run your evaluations offline), you can use one of two approaches:

1.  Pre-install the version of {{< meta agent_name >}} you want to use in the sandbox, then use `version="sandbox"`:

    ``` python
    {{< meta agent >}}(version="sandbox")
    ```

2.  Download the version of {{< meta agent_name >}} you want to use into the cache, then specify that version explicitly:

    ``` python
    # download the agent binary during installation/configuration
    download_agent_binary("{{< meta agent >}}", "0.29.0", "linux-x64")

    # reference that version in your task (no download will occur)
    {{< meta agent >}}(version="0.29.0")
    ```

    Note that the 5 most recently downloaded versions are retained in the cache. Use the `cached_agent_binaries()` function to list the contents of the cache.