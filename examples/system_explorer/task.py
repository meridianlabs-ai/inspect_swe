from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_swe import claude_code


@task
def system_explorer() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=claude_code(),
        scorer=model_graded_qa(),
        sandbox=("docker", "Dockerfile"),
    )
