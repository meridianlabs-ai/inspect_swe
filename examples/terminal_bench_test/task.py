from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.tool import bash, python
from inspect_swe import mini_swe_agent


def _get_challenges_dir() -> Path:
    """Get the path to Terminal Bench 2.0 challenges directory.

    Attempts to find the challenges directory from inspect_evals installation.
    """
    try:
        import inspect_evals.terminal_bench_2 as tb2_module

        module_dir = Path(tb2_module.__file__).parent
        challenges_dir = module_dir / "challenges"
        if challenges_dir.exists():
            return challenges_dir
    except ImportError:
        pass

    raise ImportError(
        "Could not find Terminal Bench 2.0 challenges directory. "
        "Please install inspect-evals[terminal_bench_2]:\n"
        "  pip install inspect-evals[terminal_bench_2]"
    )


@task
def terminal_bench_task(
    eval_names: list[str] | None = None,
    variant_names: list[str] | None = None,
    model: str | None = None,
) -> Task:
    """Terminal Bench 2.0 with mini-swe-agent solver.

    Runs Terminal Bench 2.0 challenges using mini-swe-agent instead of
    the default ReAct solver.

    Args:
        eval_names: Filter to specific challenge names (e.g., ["constraints-scheduling"]).
            If None, runs all challenges.
        variant_names: Filter to specific variants. Defaults to ["default"].
        model: Model name to use for mini-swe-agent. If None, uses default inspect model.

    Returns:
        Task configured with mini-swe-agent solver and Terminal Bench scorer.

    Note:
        This task uses pre-built Docker images from Docker Hub. For local builds,
        use the original inspect_evals/terminal_bench_2 task directly.
    """
    # Import dependencies from inspect_evals
    try:
        from inspect_cyber import create_agentic_eval_dataset
        from inspect_evals.terminal_bench_2.terminal_bench_2 import (
            terminal_bench_2_scorer,
        )
    except ImportError:
        raise ImportError(
            "inspect_cyber and inspect_evals are required for Terminal Bench 2.0. "
            "Please install inspect-evals[terminal_bench_2]:\n"
            "  pip install inspect-evals[terminal_bench_2]"
        ) from None

    # Get challenges directory
    challenges_dir = _get_challenges_dir()

    # Load dataset
    dataset = create_agentic_eval_dataset(root_dir=challenges_dir.absolute())

    # Filter by eval_names if specified
    if eval_names:
        dataset = dataset.filter_by_metadata_field("eval_name", eval_names)

    # Filter by variant_names (default to "default" variant)
    if variant_names is None:
        variant_names = ["default"]
    dataset = dataset.filter_by_metadata_field("variant_name", variant_names)

    # Create the solver
    solver = mini_swe_agent(model=model)

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=terminal_bench_2_scorer(),
        tools=[bash(timeout=60), python(timeout=60)],
    )
