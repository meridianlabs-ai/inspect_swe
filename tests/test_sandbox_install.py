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
        print(f"Installing mini-swe-agent version {version}...")
        binary_path = await ensure_agent_wheel_installed(
            source=MINI_SWE_AGENT_SOURCE,
            version=version,
        )
        print(f"Installed mini-swe-agent to: {binary_path}")
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

        print(
            f"Verifying mini-swe-agent installation (expected version: {expected_version})..."
        )

        # Check which mini
        print("Running: which mini")
        result = await sbox.exec(["bash", "-c", "which mini"])
        if not result.success:
            print(f"ERROR: 'which mini' failed: {result.stderr}")
            return Score(
                value=0,
                explanation=f"'which mini' failed: {result.stderr}",
            )

        binary_path = result.stdout.strip()
        if not binary_path:
            print("ERROR: mini binary not found in PATH")
            return Score(
                value=0,
                explanation="mini binary not found in PATH",
            )
        print(f"Found mini at: {binary_path}")

        # Check version
        print(f"Running: {binary_path} --version")
        result = await sbox.exec([binary_path, "--version"])
        print(f"  Exit code: {result.returncode}, Success: {result.success}")
        print(f"  Stdout: '{result.stdout.strip()}'")
        print(f"  Stderr: '{result.stderr.strip()}'")

        actual_version = "unknown"
        if result.success:
            version_output = result.stdout.strip()
            # Parse version - mini-swe-agent outputs "mini-swe-agent X.Y.Z"
            version_line = version_output.split("\n")[0]
            version_parts = version_line.split()
            actual_version = version_parts[-1] if version_parts else "unknown"
        else:
            # Try -v as fallback
            print(f"\n--version failed, trying: {binary_path} -v")
            result = await sbox.exec([binary_path, "-v"])
            print(f"  Exit code: {result.returncode}, Success: {result.success}")
            print(f"  Stdout: '{result.stdout.strip()}'")
            print(f"  Stderr: '{result.stderr.strip()}'")

            if result.success:
                version_output = result.stdout.strip()
                version_line = version_output.split("\n")[0]
                version_parts = version_line.split()
                actual_version = version_parts[-1] if version_parts else "unknown"
            else:
                # Maybe version is in stderr? Or try running just the binary
                print(f"\n-v also failed, trying to run: {binary_path} (no args)")
                result = await sbox.exec([binary_path])
                print(f"  Exit code: {result.returncode}")
                print(f"  Stdout (first 300 chars): '{result.stdout[:300]}'")
                print(f"  Stderr (first 300 chars): '{result.stderr[:300]}'")

                # Try to find version in any output
                combined_output = result.stdout + result.stderr
                for line in combined_output.split("\n"):
                    # Look for version patterns like "1.17.4" or "mini-swe-agent 1.17.4"
                    words = line.split()
                    for word in words:
                        # Check if word looks like a version (starts with digit, has dots)
                        if word and word[0].isdigit() and "." in word:
                            # Clean up any surrounding characters (including trailing periods)
                            version_candidate = word.strip("(),[].")
                            if version_candidate.count(".") >= 1:
                                actual_version = version_candidate
                                print(f"\nFound version in output: {actual_version}")
                                break
                    if actual_version != "unknown":
                        break

        # Verify version matches expected
        if expected_version != "unknown" and actual_version != expected_version:
            print(
                f"ERROR: Version mismatch! Expected {expected_version}, got {actual_version}"
            )
            return Score(
                value=0,
                explanation=f"Version mismatch: expected {expected_version}, got {actual_version}",
            )

        print(f"SUCCESS: mini installed at {binary_path}, version: {actual_version}")
        return Score(
            value=1,
            explanation=f"mini installed at {binary_path}, version: {actual_version} (expected: {expected_version})",
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
    Run with -s flag to see verbose output:
        pytest tests/test_sandbox_install.py --runslow -v -s
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
