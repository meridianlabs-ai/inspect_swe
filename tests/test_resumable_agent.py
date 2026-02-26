"""Tests for ResumableAgent trajectory handling in sandbox.

Uses mini_swe_agent() as a proper Inspect agent. Injects trajectory files
into the sandbox before calling run() with messages that trigger the resume
path (has_assistant_response=True → MSWEA_RESUME=true).
"""

import json

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import run
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox, store
from inspect_swe import mini_swe_agent
from inspect_swe._mini_swe_agent.setup import _TRAJECTORY_STORE_KEY

from tests.conftest import skip_if_no_docker

TRAJ_PATH = "/var/tmp/test_trajectory.json"

# Valid v2 trajectory with prior cost data for resume testing.
VALID_TRAJECTORY = json.dumps(
    {
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {
            "model_stats": {"instance_cost": 10.0, "api_calls": 5},
            "config": {"model": {"model_name": "test"}, "agent_type": "test"},
        },
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "The answer is 2."},
            {
                "role": "exit",
                "content": "2",
                "extra": {"exit_status": "Submitted", "submission": "2"},
            },
        ],
    }
)


@solver
def resume_with_trajectory(content: str | None) -> Solver:
    """Inject a trajectory file and run the agent in resume mode.

    Sets the trajectory store key before calling run(), then passes
    messages with a prior assistant response so the agent takes the
    resume path (MSWEA_RESUME=true).

    If content is None, no file is written (tests missing file error).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sbox = sandbox()

        # Pre-seed trajectory path in store so get_trajectory_path() returns it
        store().set(_TRAJECTORY_STORE_KEY, TRAJ_PATH)

        # Write trajectory content (or skip for missing file test)
        if content is not None:
            await sbox.write_file(TRAJ_PATH, content)

        agent = mini_swe_agent(system_prompt="test")

        # Messages with a prior assistant response trigger the resume path
        messages: list[ChatMessage] = [
            ChatMessageUser(content="What is 1+1?"),
            ChatMessageAssistant(content="2"),
            ChatMessageUser(content="What is 2+2?"),
        ]

        await run(agent, messages)
        return state

    return solve


# -- Bad trajectory tests: agent should fail with clear error messages --

BAD_TRAJECTORY_CASES = [
    ("missing_file", None, "not found"),
    ("invalid_json", "{ not valid json }", "invalid JSON"),
    (
        "wrong_format",
        json.dumps({"trajectory_format": "mini-swe-agent-1", "messages": []}),
        "not supported",
    ),
    ("non_dict", json.dumps([1, 2, 3]), "Cannot resume"),
]


@skip_if_no_docker
@pytest.mark.slow
@pytest.mark.parametrize(
    "content,expected_error",
    [(c, e) for _, c, e in BAD_TRAJECTORY_CASES],
    ids=[tid for tid, _, _ in BAD_TRAJECTORY_CASES],
)
def test_resumable_agent_bad_trajectory(
    content: str | None, expected_error: str
) -> None:
    """Agent should fail with a clear error when trajectory data is corrupt."""
    task = Task(
        dataset=[Sample(input="test", target="pass")],
        solver=resume_with_trajectory(content),
        sandbox="docker",
    )
    logs = eval(task, model="mockllm/model", limit=1)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "error", f"Expected error but got: {log.status}"
    assert expected_error in str(log.error), (
        f"Expected '{expected_error}' in error, got:\n{str(log.error)[:500]}"
    )


# Valid trajectory test: agent should load and resume successfully


@skip_if_no_docker
@pytest.mark.slow
def test_resumable_agent_valid_trajectory() -> None:
    """Agent should load a valid trajectory and resume without trajectory errors.

    The agent will fail when trying to call the mock model after loading,
    but the key assertion is that trajectory loading itself succeeded
    (no "Cannot resume", "not supported", or "invalid JSON" errors).
    """
    task = Task(
        dataset=[Sample(input="test", target="pass")],
        solver=resume_with_trajectory(VALID_TRAJECTORY),
        sandbox="docker",
    )
    logs = eval(task, model="mockllm/model", limit=1)

    assert len(logs) == 1
    log = logs[0]

    # The agent may error due to mockllm/model, but trajectory-related
    # errors mean the loading path is broken.
    if log.status == "error":
        error_str = str(log.error)
        assert "Cannot resume" not in error_str, (
            f"Trajectory load failed: {error_str[:500]}"
        )
        assert "not supported" not in error_str, f"Format rejected: {error_str[:500]}"
        assert "invalid JSON" not in error_str, f"JSON parse failed: {error_str[:500]}"
