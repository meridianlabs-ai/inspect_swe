import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

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
from inspect_swe._pi.config import pi_models_json

from tests.conftest import skip_if_no_docker


def test_pi_152_session_replays_tool_and_answer(tmp_path: Path) -> None:
    """Exercise Pi's real v3 session replay against a local HTTP endpoint."""
    binary = shutil.which("pi")
    if binary is None:
        pytest.skip("Pi 1.5.2 is not installed")
    version = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=False, timeout=10
    )
    if version.returncode != 0 or version.stdout.strip() != "1.5.2":
        pytest.skip("This test requires Pi 1.5.2")

    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            requests.append(request)
            if len(requests) == 1:
                delta: dict[str, Any] = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(
                                    {"command": "printf pi-tool-marker"}
                                ),
                            },
                        }
                    ]
                }
                finish = "tool_calls"
            else:
                delta = {
                    "content": "first-answer" if len(requests) == 2 else "second-answer"
                }
                finish = "stop"
            chunks = [
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "openai/gpt-5-mini",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "openai/gpt-5-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                },
            ]
            body = (
                "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body.encode())))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        (tmp_path / "models.json").write_text(
            pi_models_json(
                port=server.server_port,
                model="openai/gpt-5-mini",
                context_size=32000,
                max_tokens=1000,
                reasoning=False,
            )
        )
        (tmp_path / "settings.json").write_text(
            json.dumps({"retry": {"enabled": False}})
        )
        session = tmp_path / "session.jsonl"
        session.write_text(
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": "test-session",
                    "timestamp": "2024-12-03T14:00:00.000Z",
                    "cwd": str(tmp_path),
                }
            )
            + "\n"
        )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PI_CODING_AGENT_DIR": str(tmp_path),
            "PI_OFFLINE": "1",
        }
        command = [
            binary,
            "--provider",
            "inspect",
            "--model",
            "openai/gpt-5-mini",
            "--session",
            str(session),
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-approve",
            "--print",
            "--mode",
            "json",
        ]
        prompt = 'A "quoted" prompt\n\nwith a second line'
        for text in (prompt, "Second request"):
            result = subprocess.run(
                command,
                input=text,
                text=True,
                capture_output=True,
                cwd=tmp_path,
                env=env,
                timeout=30,
                check=False,
            )
            assert result.returncode == 0, result.stderr + result.stdout
        assert len(requests) == 3
        first, after_tool, resumed = [request["messages"] for request in requests]
        assert any(
            message.get("role") == "user"
            and message.get("content") == [{"type": "text", "text": prompt}]
            for message in first
        )
        assert any(
            message.get("role") == "tool"
            and "pi-tool-marker" in str(message.get("content"))
            for message in after_tool
        )
        assert any(
            message.get("role") == "tool"
            and "pi-tool-marker" in str(message.get("content"))
            for message in resumed
        )
        assert any(
            message.get("role") == "assistant"
            and message.get("content") == "first-answer"
            for message in resumed
        )
        assert [
            message["content"] for message in first if message["role"] == "system"
        ] == [message["content"] for message in resumed if message["role"] == "system"]
        entries = [json.loads(line) for line in session.read_text().splitlines()]
        assert entries[0]["type"] == "session" and entries[0]["version"] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
