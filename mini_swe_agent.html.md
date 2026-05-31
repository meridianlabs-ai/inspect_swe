# Mini SWE Agent – Inspect SWE

## Overview

The `mini_swe_agent()` agent uses the unattended mode of SWE-agent [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) to execute agentic tasks within the Inspect sandbox. Model API calls that occur in the sandbox are proxied back to Inspect for handling by the model provider for the current task.

> **NOTE: Notemini-swe-agent Installation**
>
> By default, the agent will download the current stable version of mini-swe-agent and copy it to the sandbox. You can also exercise more explicit control over which version of mini-swe-agent is used—see the [Installation](#installation) section below for details.

## Basic Usage

Use the `mini_swe_agent()` agent as you would any Inspect agent. For example, here we use it as the solver in an Inspect task:

``` python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

from inspect_swe import mini_swe_agent

@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=mini_swe_agent(),
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

You can also pass the agent as a `--solver` on the command line:

``` bash
inspect eval ctf.py --solver inspect_swe/mini_swe_agent
```

If you want to try this out locally, see the [system_explorer](https://github.com/meridianlabs-ai/inspect_swe/tree/main/examples/system_explorer/task.py) example.

## Options

The following options are supported for customizing the behavior of the agent:

| Option | Description |
|----|----|
| `system_prompt` | Additional system prompt to append to default system prompt. |
| `centaur` | Run in [Centaur Mode](#centaur-mode), which makes mini-swe-agent available to an Inspect `human_cli()` agent rather than running it unattended. |
| `attempts` | Allow the agent to have multiple scored attempts at solving the task. |
| `model` | Model name to use for agent (defaults to main model for task). |
| `filter` | Filter for intercepting bridged model requests. |
| `retry_refusals` | Should refusals be retried? (pass number of times to retry) |
| `compaction` | Compaction strategy for managing context window overflow. |
| `cwd` | Working directory for mini-swe-agent session. |
| `env` | Environment variables to set for mini-swe-agent. |
| `user` | User to execute mini-swe-agent as in the sandbox. |
| `sandbox` | Sandbox environment name. |
| `version` | Version of mini-swe-agent to use (see [Installation](#installation) below for details) |

For example, here we specify a custom system prompt:

``` python
mini_swe_agent(
    system_prompt="You are an ace system researcher.",
)
```

## Installation

By default, the agent will install the current stable version of mini-swe-agent in the sandbox via Python wheels. You can override this behaviour using the `version` option:

| Option | Description |
|----|----|
| `"stable"` | Install and use the default pinned stable version. |
| `"sandbox"` | Use the version of mini-swe-agent in the sandbox (raises `RuntimeError` if not available in the sandbox) |
| `"latest"` | Install and use the latest version from PyPI. |
| `"x.x.x"` | Install and use a specific version number. |

Unlike the other agents which use standalone binaries, mini-swe-agent is installed via Python wheels using `uv`. If you don’t ever want to rely on automatic installation of mini-swe-agent (e.g. if you run your evaluations offline), you can use one of two approaches:

1.  Pre-install the version of mini-swe-agent you want to use in the sandbox, then use `version="sandbox"`:

    ``` python
    mini_swe_agent(version="sandbox")
    ```

2.  Pre-install mini-swe-agent in your sandbox Dockerfile:

    ``` dockerfile
    RUN pip install mini-swe-agent==2.2.3
    ```

    Then reference it with `version="sandbox"` in your task.

## Centaur Mode

The `mini_swe_agent()` agent can also be run in “centaur” mode which uses the Inspect AI [Human Agent](https://inspect.aisi.org.uk/human-agent.html) as the solver and makes [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) available to the human user for help with the task. So rather than strictly measuring human vs. model performance, you are able to measure performance of humans working collaboratively with a model.

Enable centaur mode by passing `centaur=True` to the `mini_swe_agent()` agent:

``` python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

from inspect_swe import mini_swe_agent

@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=mini_swe_agent(centaur=True),
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

You can also enable centaur mode from the CLI using a solver arg (`-S`):

``` bash
inspect eval ctf.py --solver inspect_swe/mini_swe_agent -S centaur=true
```

You can also pass `CentaurOptions` to further customize the behavior of the human agent. For example:

``` python
from inspect_swe import CentaurOptions

Task(
    dataset=json_dataset("dataset.json"),
    solver=mini_swe_agent(centaur=CentaurOptions(answer=False)),
    scorer=model_graded_qa(),
    sandbox="docker",
)
```

See the [human_cli()](https://inspect.aisi.org.uk/reference/inspect_ai.agent.html#human_cli) documentation for details on available options.

## Troubleshooting

If mini-swe-agent doesn’t appear to be working or working as expected, you can troubleshoot by dumping the mini-swe-agent debug log after an evaluation task is complete. You can do this with:

``` bash
inspect trace dump --filter "mini-swe-agent"
```
