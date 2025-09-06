## Overview

The `{{< meta agent >}}()` agent uses the the unattended mode of {{< meta agent_provider >}} [{{< meta agent_name >}}]({{< meta agent_url >}}) to execute agentic tasks within the Inspect sandbox. Model API calls that occur in the sandbox are proxied back to Inspect for handling by the model provider for the current task.

::: callout-note
#### {{< meta agent_name >}} Installation

By default, the agent will download the current stable version of {{< meta agent_name >}} and copy it to the sandbox. You can also exercise more explicit control over which version of {{< meta agent_name >}} is used---see the [Installation](#installation) section below for details.
:::

## Basic Usage

Use the `{{< meta agent >}}()` agent as you would any Inspect agent. For example, here we use it as the solver in an Inspect task:

``` python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

from inspect_swe import codex_cli

@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver={{< meta agent >}}(),
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

You can also pass the agent as a `--solver` on the command line:

```bash
inspect eval ctf.py --solver inspect_swe/{{< meta agent >}}
```

If you want to try this out locally, see the [system_explorer](https://github.com/meridianlabs-ai/inspect_swe/tree/main/examples/system_explorer/task.py) example.
