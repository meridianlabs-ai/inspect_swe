from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_swe import claude_code

from tests.conftest import skip_if_no_k8s


@task
def t() -> Task:
    return Task(
        dataset=[Sample(input="what is 1+1?")],
        solver=claude_code(),
        sandbox="k8s",
    )


@skip_if_no_k8s
def test_k8s() -> None:
    log = eval(
        t(),
        model=["anthropic/claude-sonnet-4-20250514"],
        token_limit=5000,
    )[0]
    assert log.status == "success"
