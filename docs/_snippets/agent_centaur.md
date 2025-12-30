## Centaur Mode

::: {.callout-note appearance="minimal"}
The centaur mode feature described below requires the development version of the `inspect_swe` package. You can install the development version with:

```bash
pip install git+https://github.com/meridianlabs-ai/inspect_swe
```
:::

The `{{< meta agent >}}()` agent can also be run in "centaur" mode which uses the Inspect AI [Human Agent](https://inspect.aisi.org.uk/human-agent.html) as the solver and makes [{{< meta agent_name >}}]({{< meta agent_url >}}) available to the human user for help with the task. So rather than strictly measuring human vs. model performance, you are able to measure performance of humans working collaboratively with a model.

Enable centaur mode by passing `centaur=True` to the `{{< meta agent >}}()` agent:

``` python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa

from inspect_swe import codex_cli

@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver={{< meta agent >}}(centaur=True),
        scorer=model_graded_qa(),
        sandbox="docker",
    )
```

You can also enable centaur mode from the CLI using a solver arg (`-S`):

```bash
inspect eval ctf.py --solver inspect_swe/{{< meta agent >}} -S centaur=True
```

You can also pass `CentaurOptions` to further customize the behavior of the human agent. For example:

```python
from inspect_swe import CentaurOptions

Task(
    dataset=json_dataset("dataset.json"),
    solver={{< meta agent >}}(centaur=CentaurOptions(answer=False)),
    scorer=model_graded_qa(),
    sandbox="docker",
)
```

See the [human_cli()](https://inspect.aisi.org.uk/reference/inspect_ai.agent.html#human_cli) documentation for details on available options.

