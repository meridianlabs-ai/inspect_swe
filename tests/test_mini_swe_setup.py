"""Tests for mini-swe-agent sandbox setup utilities."""

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox, store
from inspect_swe._mini_swe_agent.setup import (
    _TRAJECTORY_STORE_KEY,
    RESUMABLE_AGENT_PATH,
    _read_resumable_agent_source,
    get_trajectory_path,
    install_resumable_agent,
    validate_version,
)

from tests.conftest import skip_if_no_docker


@pytest.mark.parametrize(
    "version", ["stable", "sandbox", "latest", "2.0.0", "2.2.3", "3.0.0"]
)
def test_validate_version_valid(version: str) -> None:
    validate_version(version)  # should not raise


@pytest.mark.parametrize(
    "version,match",
    [
        ("1.17.4", "not supported"),
        ("1.0.0", "not supported"),
        ("0.9.0", "not supported"),
        ("not-a-version", "Invalid"),
        ("abc", "Invalid"),
        ("", "Invalid"),
    ],
)
def test_invalid_version(version: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_version(version)


# Sandbox tests


@solver
def verify_setup_in_sandbox() -> Solver:
    """Install resumable_agent.py and verify it in the sandbox."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sbox = sandbox()

        await install_resumable_agent(sbox)

        # Verify the file exists and contains ResumableAgent
        result = await sbox.exec(
            ["python3", "-c", f"print(open('{RESUMABLE_AGENT_PATH}').read()[:50])"]
        )
        state.metadata["file_exists"] = result.success
        state.metadata["file_content_start"] = result.stdout.strip()

        # Verify the file matches what we'd read from the package
        content = await sbox.read_file(RESUMABLE_AGENT_PATH)
        expected = _read_resumable_agent_source()
        state.metadata["content_matches"] = content == expected

        # Test get_trajectory_path idempotency (same path returned twice)
        path1 = get_trajectory_path()
        path2 = get_trajectory_path()
        state.metadata["path_idempotent"] = path1 == path2
        state.metadata["path_prefix"] = path1.startswith(
            "/var/tmp/.mini-swe-trajectory-"
        )
        state.metadata["path_suffix"] = path1.endswith(".json")

        # Test that the store holds the path
        stored = store().get(_TRAJECTORY_STORE_KEY, None)
        state.metadata["path_stored"] = stored == path1

        return state

    return solve


@scorer(metrics=[])
def check_setup_results() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        checks = {
            "file_exists": state.metadata.get("file_exists"),
            "content_matches": state.metadata.get("content_matches"),
            "path_idempotent": state.metadata.get("path_idempotent"),
            "path_prefix": state.metadata.get("path_prefix"),
            "path_suffix": state.metadata.get("path_suffix"),
            "path_stored": state.metadata.get("path_stored"),
        }

        failures = {k: v for k, v in checks.items() if v is not True}
        if failures:
            return Score(value=0, explanation=f"Failed checks: {failures}")

        return Score(value=1, explanation=f"All {len(checks)} checks passed")

    return score


@skip_if_no_docker
@pytest.mark.slow
def test_install_resumable_agent_and_trajectory_path() -> None:
    """Verify resumable_agent.py is deployed correctly and trajectory paths are stable."""
    task = Task(
        dataset=[Sample(input="test", target="pass")],
        solver=verify_setup_in_sandbox(),
        scorer=check_setup_results(),
        sandbox="docker",
    )
    logs = eval(task, model="mockllm/model", limit=1)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success", f"Task failed: {log.error}"
    assert log.samples and log.samples[0].scores

    score_value = list(log.samples[0].scores.values())[0]
    assert score_value.value == 1, f"Checks failed: {score_value.explanation}"
