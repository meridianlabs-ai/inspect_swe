"""Real MiniMax CLI and Inspect bridge integration (no provider key required)."""

from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import run
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from inspect_swe import minimax_code
from inspect_swe._minimax_code.agentbinary import ensure_minimax_code_setup

from tests.conftest import skip_if_no_docker


@skip_if_no_docker
@pytest.mark.slow
@pytest.mark.parametrize("alias", [None, "fast"])
def test_minimax_cli_through_real_bridge(tmp_path: Path, alias: str | None) -> None:
    # The model alone is deterministic; installation, SQLite, stdin delivery,
    # CLI parsing, Responses HTTP, bridge recording, and retry are all real.
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm/model", "incorrect"),
            ModelOutput.from_content("mockllm/model", "ready"),
        ],
    )
    task = Task(
        dataset=[Sample(input='--help\nReply with the word "ready".', target="ready")],
        solver=minimax_code(
            version="0.4.12",
            max_context_size=32768,
            attempts=2,
            model=alias,
            model_aliases={"fast": model},
        ),
        scorer=match(),
        sandbox="docker",
        time_limit=240,
        token_limit=100000,
    )
    (log,) = eval(task, model=model, log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    assert log.samples
    sample = log.samples[0]
    assert sample.error is None, sample.error
    assert sample.scores and sample.scores["match"].value == "C"
    assert sample.output.completion == "ready"


@solver
def minimax_conversation() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent = minimax_code(
            version="0.4.12", max_context_size=32768, system_prompt="Answer concisely"
        )
        conversation = await run(agent, state.messages)
        assert conversation.output.completion == "one"
        conversation.messages.append(ChatMessageUser(content="Now say two"))
        conversation = await run(agent, conversation)
        state.messages = conversation.messages
        state.output = conversation.output
        return state

    return solve


@skip_if_no_docker
@pytest.mark.slow
def test_minimax_resumes_across_agent_calls(tmp_path: Path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm/model", text) for text in ["one", "two"]
        ],
    )
    (log,) = eval(
        Task(
            dataset=[Sample(input="Say one", target="two")],
            solver=minimax_conversation(),
            scorer=match(),
            sandbox="docker",
            time_limit=240,
            token_limit=100000,
        ),
        model=model,
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success", log.error
    assert log.samples and log.samples[0].error is None
    assert log.samples[0].output.completion == "two"


@solver
def install_minimax() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        binary, node = await ensure_minimax_code_setup(sandbox(), "0.4.12", None)
        result = await sandbox().exec([node, binary, "--version"])
        assert result.success, result.stderr
        assert result.stdout.strip() == "0.4.12"
        return state

    return solve


@skip_if_no_docker
@pytest.mark.slow
def test_minimax_native_runtime_on_ubuntu_20_04(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  default:\n    image: ubuntu:20.04\n    command: sleep infinity\n"
    )
    (log,) = eval(
        Task(
            dataset=[Sample(input="install")],
            solver=install_minimax(),
            sandbox=("docker", str(compose)),
            time_limit=240,
        ),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success", log.error
    assert log.samples and log.samples[0].error is None
