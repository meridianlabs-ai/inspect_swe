from pathlib import Path
from typing import Literal

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    GenerateInput,
    Model,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe import pi

from tests.conftest import skip_if_no_docker


@pytest.mark.slow
@skip_if_no_docker
@pytest.mark.parametrize(
    ("thinking", "expected_effort"), [("off", None), ("high", "high")]
)
def test_pi_bridge_failure_with_real_cli(
    tmp_path: Path, thinking: Literal["off", "high"], expected_effort: str | None
) -> None:
    """Exercise the actual bridge and CLI with a deliberately unavailable provider."""
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        data="""services:
  default:
    image: python:3.12-slim
    command: sleep infinity
    network_mode: none
    working_dir: /tmp
"""
    )
    requests: list[GenerateInput] = []

    async def record_request(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> None:
        requests.append(
            GenerateInput(
                input=messages, tools=tools, tool_choice=tool_choice, config=config
            )
        )

    # This is the real OpenAI provider, not a replacement ModelAPI or canned
    # response. It fails at localhost, so no credentials or paid calls are needed.
    model = get_model(
        model="openai/gpt-5-mini",
        api_key="host-only-test-key",
        base_url="http://127.0.0.1:1/v1",
        config=GenerateConfig(max_retries=0, timeout=5),
        memoize=False,
    )
    logs = eval(
        tasks=Task(
            dataset=[Sample(input="@not-a-file\nSay hello")],
            solver=pi(
                user="nobody",
                system_prompt="pi-bridge-system-marker",
                skills=[
                    Path(__file__).resolve().parents[1]
                    / "examples/skills/welcome-banner"
                ],
                thinking=thinking,
                filter=record_request,
                model="alias/with-slash",
                model_aliases={"alias/with-slash": model},
                max_context_size=32000,
            ),
            sandbox=("docker", str(compose)),
        ),
        model=model,
        log_dir=str(tmp_path / "logs"),
        time_limit=60,
        token_limit=1000,
        max_samples=1,
        display="none",
    )
    assert requests, logs[0].error
    first_request = requests[0]
    assert first_request.input[0].text.count("pi-bridge-system-marker") == 1
    assert "welcome-banner" in first_request.input[0].text
    assert all(
        "host-only-test-key" not in message.model_dump_json()
        for message in first_request.input
    )
    assert first_request.config.reasoning_effort == expected_effort
    assert first_request.input[-1].text == "@not-a-file\nSay hello"
    assert {tool.name for tool in first_request.tools} >= {
        "read",
        "write",
        "edit",
        "bash",
    }
    assert logs[0].samples
    assert logs[0].samples[0].error is not None
