### Recording What Ran

The version installed is not always the one a task appears to name. `version="auto"` uses whatever the sandbox image already contains, and `"stable"`/`"latest"` resolve once per process with no expiry, so a resumed run, an `eval_set` retry, or a multi-process run can install a newer release for the remaining samples.

Each sample therefore records the binary it actually ran as an `InfoEvent` with `source="inspect_swe"`, carrying an `AgentBinaryInstall`:

``` python
from inspect_ai.event import InfoEvent
from inspect_ai.log import read_eval_log
from inspect_swe import AgentBinaryInstall

log = read_eval_log("logs/2026-09-15_task.eval")
for event in log.samples[0].events:
    if isinstance(event, InfoEvent) and event.source == "inspect_swe":
        install = AgentBinaryInstall.model_validate(event.data)
        print(install.agent, install.version, install.origin)
```

One record is written per agent invocation, so a sample that runs two agents carries two. `origin` says where the bytes came from:

| Origin | Description |
|------------------------------------|------------------------------------|
| `"download"` | Fetched over the network and verified against the resolved digest. |
| `"cache"` | Served from the local cache, either for a pinned version or because the resolved version was already downloaded. Not a claim of verification: a pinned version has no resolved digest to check against, and neither does a cached artifact for an agent that transforms its download. |
| `"sandbox"` | Already present in the image; nothing was installed, so no version or checksum is reported (neither is knowable without running it). |
| `"cache_unverified"` | The offline fallback: resolution failed, so no digest could be obtained to check the cached bytes against. |

: {tbl-colwidths=\[25,75\]}

The event renders in the viewer under the agent's span. To compare what ran across a set of logs, request the fields as explicit columns:

``` python
from inspect_ai.analysis import EventColumn, EventInfo, events_df

events = events_df(
    "logs",
    columns=EventInfo
    + [
        EventColumn("source", path="source"),
        EventColumn("agent", path="data.agent"),
        EventColumn("requested", path="data.requested"),
        EventColumn("version", path="data.version"),
        EventColumn("origin", path="data.origin"),
    ],
)
installs = events[events["source"] == "inspect_swe"]
```
