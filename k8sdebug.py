from datetime import datetime
from inspect_ai import Task, eval_set, task
from inspect_ai.dataset import Sample
from inspect_swe import claude_code


@task
def system_explorer() -> Task:
    return Task(
        dataset=[
            Sample(
                input="Explore the system and tell me: 1) What version of Linux is running, and 2) How many CPU cores the system has"
            )
        ],
        solver=claude_code(),
        sandbox="k8s",
    )


date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
_ = eval_set(
    system_explorer,
    log_dir="/tmp/" + date_time,
    model=["anthropic/claude-sonnet-4-20250514"],
    token_limit=5000,
)