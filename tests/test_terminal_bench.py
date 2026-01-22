import importlib.util

import pytest
from inspect_ai import eval
from inspect_ai.log import EvalLog

from tests.conftest import skip_if_no_docker, skip_if_no_openai


def is_terminal_bench_available() -> bool:
    """Check if inspect_evals[terminal_bench_2] is installed."""
    return (
        importlib.util.find_spec("inspect_evals") is not None
        and importlib.util.find_spec("inspect_cyber") is not None
    )


skip_if_no_terminal_bench = pytest.mark.skipif(
    not is_terminal_bench_available(),
    reason="Test requires inspect-evals[terminal_bench_2] to be installed",
)


def run_terminal_bench(
    model: str,
    eval_names: list[str] | None = None,
) -> list[EvalLog]:
    """Run terminal_bench_mini_swe task."""
    task_args: dict[str, str | list[str]] = {}
    if eval_names is not None:
        task_args["eval_names"] = eval_names
    return eval(
        "examples/terminal_bench_test",
        model=model,
        limit=1,
        task_args=task_args,
    )


@skip_if_no_openai
@skip_if_no_docker
@skip_if_no_terminal_bench
@pytest.mark.slow
@pytest.mark.parametrize(
    "model, challenge, score",
    [
        ("openai/gpt-5-mini", "constraints-scheduling", 1.0),
        ("openai/gpt-5-mini", "break-filter-js-from-html", 0),
    ],
)
def test_mini_swe_agent_terminal_bench_challenges(model, challenge, score) -> None:
    """Test mini-swe-agent on constraints-scheduling challenges.

    This test verifies the integration works. Agent success rate may vary.
    Challenges were picked based on their consistency on the official leaderboard and cost running.
    constraints-scheduling: this challenge has 5/5 pass rate with GPT-5-Mini on the official leaderboard. See https://www.tbench.ai/leaderboard/terminal-bench/2.0/mini-swe-agent/unknown/gpt-5-mini%40openai/c69027b9099bb7ac63a753b310e31cfef9cc20dce2d1c6c88956c8a854d7ac16
    break-filter-js-from-html: this challenge has 0/5 pass rate. See https://www.tbench.ai/leaderboard/terminal-bench/2.0/mini-swe-agent/unknown/gpt-5-mini%40openai/b33421492e86b07de2a01bc2205f7e7d2fdc36446a84787265d99e5e138fdaaa
    """
    logs = run_terminal_bench(
        model=model,
        eval_names=[
            challenge,
        ],
    )
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success", f"Eval failed to complete: {log.error}"

    # Check the actual score
    assert log.results is not None, "No results in log"
    assert log.results.scores is not None, "No scores in results"
    assert len(log.results.scores) > 0, "Empty scores list"

    expected_score = score  # Preserve parameter before shadowing
    score_result = log.results.scores[0]
    assert score_result.metrics is not None, "No metrics in score"
    accuracy = score_result.metrics.get("accuracy")
    assert accuracy is not None, "No accuracy metric"
    assert accuracy.value == expected_score, f"Task failed: accuracy={accuracy.value}"
