"""Tests for the OpenCode agent install/setup utilities."""

import asyncio

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from inspect_swe._opencode.agentbinary import (
    ensure_opencode_setup,
    resolve_opencode_version,
)

from tests.conftest import skip_if_no_docker


def test_resolve_version_literal() -> None:
    """Explicit semver strings are returned as-is without hitting the network."""
    assert asyncio.run(resolve_opencode_version("1.14.30")) == "1.14.30"
    assert asyncio.run(resolve_opencode_version("0.42.0")) == "0.42.0"


@solver
def install_opencode_in_sandbox(version: str = "stable") -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sbox = sandbox()
        opencode_binary, node_binary = await ensure_opencode_setup(
            sbox, version=version, user=None
        )
        state.metadata["opencode_binary"] = opencode_binary
        state.metadata["node_binary"] = node_binary

        version_result = await sbox.exec(
            [node_binary, opencode_binary, "--version"], user=None
        )
        state.metadata["version_ok"] = version_result.success
        state.metadata["reported_version"] = version_result.stdout.strip()
        return state

    return solve


@scorer(metrics=[])
def check_install() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        version_ok = state.metadata.get("version_ok")
        binary = state.metadata.get("opencode_binary")
        if not version_ok or not binary:
            return Score(
                value=0,
                explanation=f"Install failed: binary={binary} version_ok={version_ok}",
            )
        return Score(
            value=1,
            explanation=f"Installed at {binary}, --version => {state.metadata.get('reported_version')!r}",
        )

    return score


@skip_if_no_docker
@pytest.mark.slow
def test_install_opencode_in_docker_sandbox() -> None:
    """Verify opencode-ai installs into a docker sandbox and reports a version."""
    task = Task(
        dataset=[Sample(input="install", target="ok")],
        solver=install_opencode_in_sandbox(),
        scorer=check_install(),
        sandbox="docker",
    )
    logs = eval(task, model="mockllm/model", limit=1)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success", f"Task failed: {log.error}"
    assert log.samples and log.samples[0].scores

    score_value = list(log.samples[0].scores.values())[0]
    assert score_value.value == 1, score_value.explanation
