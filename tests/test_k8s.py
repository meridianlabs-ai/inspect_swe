from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_swe import claude_code

from tests.conftest import skip_if_no_anthropic, skip_if_no_k8s


@task
def t() -> Task:
    return Task(
        dataset=[Sample(input="what is 1+1?")],
        solver=claude_code(),
        sandbox="k8s",
    )


# Reaches a live Anthropic model, so it needs the key as well as the cluster.
# Without the provider gate this is the one test in the suite that bills a real
# generation from a bare `pytest`, and on a keyless machine it fails rather than
# skipping -- which is how it turned up.
@skip_if_no_anthropic
@skip_if_no_k8s
def test_k8s() -> None:
    log = eval(
        t(),
        model=["anthropic/claude-sonnet-4-20250514"],
        token_limit=5000,
    )[0]
    assert log.status == "success"
