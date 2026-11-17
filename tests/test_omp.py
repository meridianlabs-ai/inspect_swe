from pathlib import Path

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import sandbox
from inspect_swe import omp
from inspect_swe._omp.omp import _omp_error

from tests.conftest import skip_if_no_docker


def test_structured_omp_error() -> None:
    stdout = "\n".join(
        [
            "diagnostic before JSON",
            '{"type":"message_end","message":{"errorMessage":"upstream unavailable"}}',
            '{"type":"agent_end","messages":[]}',
        ]
    )
    assert _omp_error(stdout, "generic process error") == "upstream unavailable"
    assert _omp_error("", "native addon failed") == "native addon failed"


@pytest.mark.parametrize(
    "context_window,max_tokens", [(0, 1), (32, 0), (32, 32), (32, 33)]
)
def test_invalid_omp_capacity(context_window: int, max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens < context_window"):
        omp(context_window=context_window, max_tokens=max_tokens)


def _compose(tmp_path: Path) -> str:
    path = tmp_path / "compose.yaml"
    path.write_text(
        "services:\n"
        "  default:\n"
        "    image: python:3.11-slim\n"
        "    command: sleep infinity\n"
        "    working_dir: /workspace\n"
        "    network_mode: none\n"
    )
    return str(path)


@pytest.mark.slow
@skip_if_no_docker
def test_omp_tools_and_session_resume(tmp_path: Path) -> None:
    requests: list[list[ChatMessage]] = []
    outputs = iter(
        [
            ModelOutput.for_tool_call(
                "mockllm/model",
                "write",
                {
                    "path": "/workspace/answer.py",
                    "content": "def answer():\n    return 42\n",
                },
            ),
            ModelOutput.for_tool_call(
                "mockllm/model",
                "bash",
                {
                    "command": "python -c 'import answer; assert answer.answer() == 42; print(\"OMP_TEST_PASS\")'"
                },
            ),
            ModelOutput.from_content("mockllm/model", "OMP_DONE"),
            ModelOutput.from_content("mockllm/model", "OMP_RESUMED"),
        ]
    )

    def respond(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        requests.append(messages)
        return next(outputs)

    @agent
    def two_turns() -> Agent:
        run = omp(system_prompt="PRESERVE_FACTORY_INSTRUCTION")

        async def execute(state: AgentState) -> AgentState:
            state.messages.insert(
                0, ChatMessageSystem(content="PRESERVE_STATE_INSTRUCTION")
            )
            state = await run(state)
            assert (
                await sandbox().read_file("/workspace/answer.py")
                == "def answer():\n    return 42\n"
            )
            assert state.output.completion == "OMP_DONE"
            state.messages.append(
                ChatMessageUser(content="Confirm the previous test result.")
            )
            resumed = await run(state)
            with pytest.raises(ValueError, match="matching OMP session"):
                await run(
                    AgentState(
                        messages=[
                            ChatMessageUser(content="An unrelated question"),
                            ChatMessageAssistant(content="An unrelated answer"),
                            ChatMessageUser(content="Continue that unrelated answer"),
                        ]
                    )
                )
            return resumed

        return execute

    logs = eval(
        Task(
            dataset=[Sample(input='Write and test a function returning "42".')],
            solver=two_turns(),
            sandbox=("docker", _compose(tmp_path)),
        ),
        model=get_model("mockllm/model", custom_outputs=respond),
        time_limit=180,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples is not None
    assert logs[0].samples[0].output.completion == "OMP_RESUMED"
    assert len(requests) == 4
    assert any(
        message.role == "tool" and "OMP_TEST_PASS" in message.text
        for message in requests[2]
    )
    assert any(
        message.role == "assistant" and message.text == "OMP_DONE"
        for message in requests[3]
    )
    for request in (requests[0], requests[3]):
        system = "\n".join(
            message.text for message in request if message.role == "system"
        )
        assert system.count("PRESERVE_FACTORY_INSTRUCTION") == 1
        assert system.count("PRESERVE_STATE_INSTRUCTION") == 1


@pytest.mark.slow
@skip_if_no_docker
def test_omp_explicit_model_overrides_task_model(tmp_path: Path) -> None:
    logs = eval(
        Task(
            dataset=[Sample(input="Return a short answer.")],
            solver=omp(model="mockllm/explicit"),
            sandbox=("docker", _compose(tmp_path)),
        ),
        model=get_model(
            "mockllm/task",
            custom_outputs=[
                ModelOutput.from_content("mockllm/task", "WRONG_TASK_MODEL")
            ],
        ),
        time_limit=180,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples is not None
    assert logs[0].samples[0].output.completion == "Default output from mockllm/model"


@pytest.mark.slow
@skip_if_no_docker
def test_omp_cancellation_kills_process(tmp_path: Path) -> None:
    started = anyio.Event()
    provider_cancelled = False
    checked_processes = False

    async def respond(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        nonlocal provider_cancelled
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            provider_cancelled = True
        raise AssertionError("unreachable")

    @agent
    def cancel_agent() -> Agent:
        run = omp()

        async def execute(state: AgentState) -> AgentState:
            nonlocal checked_processes
            async with anyio.create_task_group() as group:
                group.start_soon(run, state)
                await started.wait()
                group.cancel_scope.cancel()
            result = await sandbox().exec(
                [
                    "python",
                    "-c",
                    "import pathlib\n"
                    "matches = []\n"
                    "for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):\n"
                    " try:\n"
                    "  command = path.read_bytes()\n"
                    "  if b'/.5c95f967ca830048/omp-' in command.split(b'\\0')[0]:\n"
                    "   matches.append((str(path), command))\n"
                    " except FileNotFoundError:\n"
                    "  pass\n"
                    "assert not matches, matches\n",
                ]
            )
            assert result.success, result.stderr
            checked_processes = True
            return state

        return execute

    logs = eval(
        Task(
            dataset=[Sample(input="Wait for cancellation.")],
            solver=cancel_agent(),
            sandbox=("docker", _compose(tmp_path)),
        ),
        model=get_model("mockllm/model", custom_outputs=respond),
        time_limit=180,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert provider_cancelled
    assert checked_processes
