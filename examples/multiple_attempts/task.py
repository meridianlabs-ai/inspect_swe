from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_swe import claude_code


@task
def multiple_attempts() -> Task:
    return Task(
        dataset=[
            Sample(
                input="Try to guess the magic number (it's hidden somewhere on this system)",
                target="56198347654",
            )
        ],
        solver=claude_code(
            system_prompt="You will be given two attempts and you should take only a short time (maximum 3 tool calls) to come with each attempt. If you haven't yet found the magic number after 3 tool calls you should just make a guess.",
            attempts=2,
        ),
        scorer=includes(),
        sandbox="docker",
    )
