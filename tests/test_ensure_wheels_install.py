import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from inspect_swe._mini_swe_agent.mini_swe_agent import MINI_SWE_AGENT_SOURCE
from inspect_swe._util.agentwheel import ensure_agent_wheel_installed

from tests.conftest import skip_if_no_docker


@solver
def install_mini_swe_agent(version: str = "1.17.4") -> Solver:
    """Solver that installs mini-swe-agent in the sandbox."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        binary_path = await ensure_agent_wheel_installed(
            source=MINI_SWE_AGENT_SOURCE,
            version=version,
        )
        state.metadata["binary_path"] = binary_path
        state.metadata["expected_version"] = version
        return state

    return solve


@scorer(metrics=[])
def verify_mini_installation() -> Scorer:
    """Scorer that verifies mini-swe-agent installation."""

    async def score(state: TaskState, target: Target) -> Score:
        sbox = sandbox()
        expected_version = state.metadata.get("expected_version", "unknown")
        expected_path = state.metadata.get("binary_path", "unknown")

        # Check which mini
        result = await sbox.exec(["bash", "-c", "which mini"])
        if not result.success:
            return Score(
                value=0,
                explanation=f"'which mini' failed: {result.stderr}",
            )

        binary_path = result.stdout.strip()
        if not binary_path:
            return Score(
                value=0,
                explanation="mini binary not found in PATH",
            )

        # Verify binary path matches what ensure_agent_wheel_installed returned
        if expected_path != "unknown" and binary_path != expected_path:
            return Score(
                value=0,
                explanation=f"Binary path mismatch: expected {expected_path}, got {binary_path}",
            )

        # Verify binary is executable
        result = await sbox.exec(["test", "-x", binary_path])
        if not result.success:
            return Score(
                value=0,
                explanation=f"{binary_path} is not executable",
            )

        # Verify installed version via pip show
        result = await sbox.exec(["pip", "show", "mini-swe-agent"])
        if result.success:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    actual_version = line.split(":", 1)[1].strip()
                    if actual_version != expected_version:
                        return Score(
                            value=0,
                            explanation=f"Version mismatch: expected {expected_version}, got {actual_version}",
                        )
                    break

        return Score(
            value=1,
            explanation=f"mini-swe-agent {expected_version} installed at {binary_path}",
        )

    return score


@skip_if_no_docker
@pytest.mark.slow
def test_mini_swe_agent_sandbox_installation() -> None:
    """Test that mini-swe-agent installs correctly in a Docker sandbox.

    This test verifies the agentwheel utility by:
    1. Installing mini-swe-agent v1.17.4 wheels in the sandbox
    2. Verifying 'which mini' returns a valid path
    3. Verifying the installed version matches the expected version

    No API key needed (uses mockllm).
    """
    task = Task(
        dataset=[Sample(input="Install test", target="installed")],
        solver=install_mini_swe_agent(version="1.17.4"),
        scorer=verify_mini_installation(),
        sandbox="docker",
    )

    logs = eval(task, model="mockllm/model", limit=1)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success", f"Task failed: {log.error}"
    assert log.samples and log.samples[0].scores

    scores = log.samples[0].scores
    assert len(scores) > 0, "No scores returned"
    score_value = list(scores.values())[0]
    assert score_value.value == 1, f"Installation failed: {score_value.explanation}"
