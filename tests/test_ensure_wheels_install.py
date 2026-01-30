import asyncio
import json
import urllib.request

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from inspect_swe._mini_swe_agent.mini_swe_agent import MINI_SWE_AGENT_SOURCE
from inspect_swe._util.agentwheel import (
    detect_python_version,
    ensure_agent_wheel_installed,
)

from tests.conftest import create_mock_sandbox_with_result, skip_if_no_docker


# https://docs.pypi.org/api/json/#get-a-project
def get_pypi_latest_version(package_name: str) -> str:
    """Get the latest version of a package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode())
        return str(data["info"]["version"])


# Default version for testing
TEST_MINI_SWE_VERSION = "1.17.4"


@pytest.mark.parametrize(
    "version_output,expected",
    [
        ("Python 3.12.0\n", "312"),
        ("Python 3.11.5\n", "311"),
        ("Python 3.10.14\n", "310"),
        ("Python 3.9.7\n", "39"),
        ("Python 3.8.0\n", "38"),
    ],
    ids=["py312", "py311", "py310", "py39", "py38"],
)
def test_detect_python_version_parsing(version_output: str, expected: str) -> None:
    """Test parsing various Python version strings."""
    mock_sandbox = create_mock_sandbox_with_result(success=True, stdout=version_output)
    result = asyncio.run(detect_python_version(mock_sandbox))
    assert result == expected


def test_detect_python_version_not_found() -> None:
    """Test error when python3 command fails."""
    mock_sandbox = create_mock_sandbox_with_result(
        success=False, stderr="python3: command not found"
    )
    with pytest.raises(RuntimeError, match="Python 3 not found in sandbox"):
        asyncio.run(detect_python_version(mock_sandbox))


def test_detect_python_version_unparseable() -> None:
    """Test error when version output cannot be parsed."""
    mock_sandbox = create_mock_sandbox_with_result(
        success=True, stdout="some garbage output\n"
    )
    with pytest.raises(RuntimeError, match="Could not parse Python version"):
        asyncio.run(detect_python_version(mock_sandbox))


# Integration tests for ensure_agent_wheel_installed in sandbox


@solver
def install_mini_swe_agent(version: str = TEST_MINI_SWE_VERSION) -> Solver:
    """Solver that installs mini-swe-agent in the sandbox."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        binary_path = await ensure_agent_wheel_installed(
            source=MINI_SWE_AGENT_SOURCE,
            version=version,
        )
        state.metadata["binary_path"] = binary_path
        state.metadata["requested_version"] = version
        return state

    return solve


# scorer for single run/score no need for metrics or data.
@scorer(metrics=[])
def verify_mini_installation() -> Scorer:
    """Scorer that verifies mini-swe-agent installation."""

    async def score(state: TaskState, target: Target) -> Score:
        sbox = sandbox()
        requested_version = state.metadata.get("requested_version", "unknown")
        expected_path = state.metadata.get("binary_path", "unknown")

        # Resolve expected version
        expected_version: str
        if requested_version == "stable":
            expected_version = MINI_SWE_AGENT_SOURCE.default_version
        elif requested_version == "latest":
            try:
                expected_version = get_pypi_latest_version(
                    MINI_SWE_AGENT_SOURCE.package
                )
            except Exception as e:
                pytest.skip(f"Could not fetch latest version from PyPI: {e}")
        else:
            expected_version = requested_version

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

        # Verify installed version via uv tool list (uv tool install, not pip)
        result = await sbox.exec(["bash", "-c", "/var/tmp/.uv tool list"])
        if result.success:
            # Output format: "mini-swe-agent v1.17.4"
            for line in result.stdout.split("\n"):
                if "mini-swe-agent" in line:
                    # Extract version from "mini-swe-agent v1.17.4" format
                    parts = line.split()
                    if len(parts) >= 2:
                        actual_version = parts[1].lstrip("v")
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
@pytest.mark.parametrize(
    "version",
    [TEST_MINI_SWE_VERSION, "stable", "latest"],
    ids=["pinned", "stable", "latest"],
)
def test_mini_swe_agent_installation(version: str) -> None:
    """Test that mini-swe-agent installs correctly in a Docker sandbox."""
    task = Task(
        dataset=[Sample(input="Install test", target="installed")],
        solver=install_mini_swe_agent(version=version),
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


@skip_if_no_docker
@pytest.mark.slow
def test_mini_swe_agent_invalid_version() -> None:
    """Test that installing mini-swe-agent with an invalid version raises an error."""
    task = Task(
        dataset=[Sample(input="Install test", target="installed")],
        solver=install_mini_swe_agent(version="99.99.99"),
        sandbox="docker",
    )
    logs = eval(task, model="mockllm/model", limit=1)
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "error", f"Expected error status, got: {log.status}"
    assert "pip download failed" in str(log.error), (
        f"Expected pip download error, got: {log.error}"
    )
