import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, run
from inspect_ai.dataset import Sample
from inspect_ai.event import InfoEvent, ModelEvent
from inspect_ai.event._timeline import TimelineEvent, TimelineSpan, timeline_build
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import match
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe import pi

from tests.conftest import skip_if_no_docker


@contextmanager
def completion_server(
    *, refusal: bool = False, compact: bool = False
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Serve Chat Completions locally while the real bridge and CLI execute."""
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "first-answer"
                if len(requests) == (4 if compact else 2)
                else "second-answer",
            }
            finish_reason = "stop"
            if compact and not request.get("tools"):
                message["content"] = (
                    "The user asked a quoted prompt. A bash tool returned pi-tool-marker and the assistant answered first-answer. Continue with the next user request."
                )
            if refusal:
                message["content"] = "I cannot help with that request."
                finish_reason = "content_filter"
            elif len(requests) == 1 or (compact and len(requests) <= 3):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_marker_{len(requests)}",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(
                                    {
                                        "command": "printf 'pi-tool-marker\\nmodel=%s\\n' \"$PI_MODEL\""
                                        + ("; printf '%040000d' 0" if compact else "")
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish_reason = "tool_calls"
            body = json.dumps(
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request["model"],
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish_reason}
                    ],
                    "usage": {
                        "prompt_tokens": 31000
                        if compact and len(requests) == 4
                        else 100,
                        "completion_tokens": 10,
                        "total_tokens": 31010
                        if compact and len(requests) == 4
                        else 110,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(
        server_address=("127.0.0.1", 0), RequestHandlerClass=Handler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@solver
def call_twice(*, agent: Agent, fresh_agent: bool = False) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent_state = await run(agent=agent, input=state.messages)
        agent_state.messages.append(ChatMessageUser(content="Second request"))
        agent_state = await run(agent=pi() if fresh_agent else agent, input=agent_state)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


def timeline_spans(*, span: TimelineSpan) -> Iterator[TimelineSpan]:
    yield span
    for child in [*span.content, *span.branches]:
        if isinstance(child, TimelineSpan):
            yield from timeline_spans(span=child)


@pytest.mark.slow
@skip_if_no_docker
@pytest.mark.parametrize(
    "mode",
    ["multi_call", "attempts", "refusal", "compaction", "handoff", "filter_error"],
)
def test_pi_offline_trajectory(
    tmp_path: Path,
    mode: Literal[
        "multi_call", "attempts", "refusal", "compaction", "handoff", "filter_error"
    ],
) -> None:
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
    prompt = '@not-a-file\nA "quoted" prompt\n\nwith a second line'
    with completion_server(
        refusal=mode in ("refusal", "filter_error"), compact=mode == "compaction"
    ) as (
        base_url,
        requests,
    ):
        model = get_model(
            model="openai/gpt-5-mini",
            api_key="host-only-test-key",
            base_url=base_url,
            responses_api=False,
            config=GenerateConfig(max_retries=0, timeout=10),
            memoize=False,
        )

        async def fail_after_helper(
            model: Model,
            messages: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice | None,
            config: GenerateConfig,
        ) -> None:
            await model.generate(
                input=messages, tools=tools, tool_choice=tool_choice, config=config
            )
            raise RuntimeError("deliberate filter failure")

        agent = pi(
            version="1.5.2",
            system_prompt="pi-system-marker",
            agents_md="pi-global-agents-marker",
            attempts=2 if mode == "attempts" else 1,
            max_context_size=32000,
            model="pi-alias" if mode == "attempts" else None,
            model_aliases={"pi-alias": model} if mode == "attempts" else None,
            filter=fail_after_helper if mode == "filter_error" else None,
        )
        logs = eval(
            tasks=Task(
                dataset=[Sample(input=prompt, target="second-answer")],
                solver=call_twice(agent=agent, fresh_agent=mode == "handoff")
                if mode in ("multi_call", "compaction", "handoff")
                else agent,
                scorer=match(location="exact"),
                sandbox=("docker", str(compose)),
            ),
            model=model,
            log_dir=str(tmp_path / "logs"),
            time_limit=120,
            token_limit=40000 if mode == "compaction" else 2000,
            max_samples=1,
            display="none",
        )
    log = read_eval_log(log_file=logs[0].location, resolve_attachments=True)
    assert log.samples, log.error
    sample = log.samples[0]
    if mode == "filter_error":
        assert sample.error is not None
        assert len(requests) == 1
        return
    if mode == "handoff":
        assert sample.error is not None
        assert "Pi cannot import an existing conversation" in sample.error.message
        assert len(requests) == 2
        return
    assert sample.error is None, sample.error
    assert logs[0].status == "success", logs[0].error
    events = [event for event in sample.events if isinstance(event, ModelEvent)]
    assert events
    assert all(event.model == "openai/gpt-5-mini" for event in events)
    system = events[0].input[0].text
    assert "pi-system-marker" in system
    assert "pi-global-agents-marker" in system
    assert "host-only-test-key" not in system
    assert events[0].input[-1].text == prompt
    installs = [
        event
        for event in sample.events
        if isinstance(event, InfoEvent)
        and event.source == "inspect_swe"
        and isinstance(event.data, dict)
        and event.data.get("agent") == "pi"
    ]
    assert len(installs) == (2 if mode in ("multi_call", "compaction") else 1)
    assert all(
        isinstance(event.data, dict) and event.data["version"] == "1.5.2"
        for event in installs
    )
    if mode == "refusal":
        assert len(requests) == 1
        assert events[-1].output.stop_reason == "content_filter"
        assert isinstance(sample.messages[-1], ChatMessageAssistant)
        assert "cannot help" in sample.messages[-1].text
        return
    if mode == "compaction":
        summaries = [
            event
            for event in events
            if event.input[0].text.startswith(
                "You are a context summarization assistant."
            )
        ]
        assert len(summaries) == 1
        assert not summaries[0].tools
        timeline = timeline_build(events=sample.events)
        assert any(
            span.utility or span.span_type == "utility"
            for span in timeline_spans(span=timeline.root)
        )
        assert sample.messages[-1].text == "second-answer"
        assert all(
            event.input[0].text == system for event in events if event not in summaries
        )
        assert any(
            span.span_type == "agent"
            and not span.utility
            and any(
                isinstance(item, TimelineEvent) and item.event.uuid == events[-1].uuid
                for item in span.content
            )
            for span in timeline_spans(span=timeline.root)
        )
        (tmp_path / "timeline.txt").write_text(data=timeline.render())
        return
    assert len(requests) == len(events) == 3
    resumed = events[-1].input
    assert any(
        isinstance(message, ChatMessageTool) and "pi-tool-marker" in message.text
        for message in resumed
    )
    assert any(
        isinstance(message, ChatMessageTool) and "model=gpt-5-mini" in message.text
        for message in resumed
    )
    assert any(
        isinstance(message, ChatMessageAssistant) and message.text == "first-answer"
        for message in resumed
    )
    assert sum(message.text == prompt for message in resumed) == 1
    assert all(event.input[0].text == system for event in events)
    assert (
        system.count("pi-system-marker") == system.count("pi-global-agents-marker") == 1
    )
    assert sample.messages[-1].text == "second-answer"
    timeline = timeline_build(events=sample.events)
    assert not any(
        span.utility or span.span_type == "utility"
        for span in timeline_spans(span=timeline.root)
    )
    (tmp_path / "timeline.txt").write_text(data=timeline.render())
