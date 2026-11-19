"""A recorded dsh run replayed through the real agent bridge.

`tests/data/deepseek_harness/delegation-0.2.2rc2.json` holds every model
request of one `deepseek_harness()` sample (dsh 0.2.2rc2, headless profile,
openrouter/deepseek/deepseek-v4.1-flash-1017, inspect_ai 0.3.291) in the order
the bridge received them. In it the root agent starts three children in one
message (two named, one left for dsh to name), waits for their settlement
notices, resumes one with `send_message`, forks a fourth from its own history
and answers. The replay feeds those requests to an `AgentBridge` wired as
`deepseek_harness()` wires it, with the recorded outputs as the model's
replies, and checks what ends up in the transcript, the timeline and the
sample's answer.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from inspect_ai._sentinel._scope import sentinel_solving
from inspect_ai.agent import (
    AgentBridgeContext,
    AgentState,
    current_agent_bridge_context,
    is_root_agent,
)
from inspect_ai.agent._bridge.types import AgentBridge
from inspect_ai.agent._bridge.util import bridge_generate
from inspect_ai.event import (
    Event,
    SpanBeginEvent,
    SpanEndEvent,
    TimelineEvent,
    TimelineSpan,
    timeline_build,
)
from inspect_ai.event._model import ModelEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentText,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_ai.util._span import current_span_id, span
from inspect_swe._deepseek_harness._events.consumer import DshConsumer
from inspect_swe._deepseek_harness.deepseek_harness import (
    _apply_root_generation,
    build_dsh_filter,
)

from tests.dsh_shapes import (
    assistant,
    bash_call,
    child_task,
    conversation,
    delegation_call,
    notice,
    root_task,
    started,
    tool_result,
)

RECORDING = (
    Path(__file__).parent / "data" / "deepseek_harness" / "delegation-0.2.2rc2.json"
)

# the model name dsh requests (the bridge keys a request's decision on it)
SLUG = "deepseek/deepseek-v4.1-flash-1017"


class Request(NamedTuple):
    agent: str
    """The dsh agent that sent the request."""
    input: list[ChatMessage]
    output: ModelOutput


def _content(content: str | list[str]) -> str | list[Content]:
    if isinstance(content, str):
        return content
    return [ContentText(text=text) for text in content]


def _message(entry: dict[str, Any]) -> ChatMessage:
    content = _content(entry["content"])
    match entry["role"]:
        case "user":
            return ChatMessageUser(content=content)
        case "tool":
            return ChatMessageTool(
                content=content,
                tool_call_id=entry["tool_call_id"],
                function=entry["function"],
            )
        case "assistant":
            return ChatMessageAssistant(
                content=content,
                tool_calls=[ToolCall(**call) for call in entry.get("tool_calls", [])]
                or None,
            )
    raise ValueError(entry["role"])


def _recording() -> list[Request]:
    data = json.loads(RECORDING.read_text(encoding="utf-8"))
    requests: list[Request] = []
    for request in data["requests"]:
        reply = _message(request["output"])
        assert isinstance(reply, ChatMessageAssistant)
        output = ModelOutput.from_message(reply)
        output.choices[0].stop_reason = request["output"]["stop_reason"]
        requests.append(
            Request(
                agent=request["agent"],
                input=[
                    ChatMessageSystem(content=data["system"]),
                    *(_message(data["pool"][index]) for index in request["messages"]),
                ],
                output=output,
            )
        )
    return requests


class Replay(NamedTuple):
    events: list[Event]
    outer: str
    """The `deepseek_harness()` agent span the requests ran under."""
    contexts: list[AgentBridgeContext | None]
    """The agent context the user's filter saw, per request."""
    steered: list[bool]
    """`is_root_agent()` in the user's filter, per request."""
    state: AgentState
    """The sample's state as `deepseek_harness()` would return it."""
    tracked: ModelOutput
    """The answer the bridge's own tracking reported."""


async def _replay(requests: list[Request], task: str) -> Replay:
    init_transcript(Transcript())
    consumer = DshConsumer()
    contexts: list[AgentBridgeContext | None] = []
    steered: list[bool] = []

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        contexts.append(current_agent_bridge_context())
        steered.append(is_root_agent())
        return None

    bridge = AgentBridge(
        AgentState(messages=[ChatMessageUser(content=task)]),
        filter=build_dsh_filter(user_filter, consumer),
        model_event_sink=consumer,
    )
    model = get_model(
        "mockllm/model", custom_outputs=[request.output for request in requests]
    )
    # as during a sample's solve, where a bridged call is recorded as an agent
    # turn (`ModelEvent.origin`), which the timeline reads
    with sentinel_solving():
        return await _replay_solve(requests, bridge, model, consumer, contexts, steered)


async def _replay_solve(
    requests: list[Request],
    bridge: AgentBridge,
    model: Model,
    consumer: DshConsumer,
    contexts: list[AgentBridgeContext | None],
    steered: list[bool],
) -> Replay:
    async with span("DeepSeek Harness", type="agent"):
        outer = current_span_id()
        assert outer is not None
        for request in requests:
            await bridge_generate(
                bridge, model, request.input, [], None, GenerateConfig(), SLUG
            )
        tracked = bridge.state.output
        _apply_root_generation(bridge.state, consumer)
        consumer.reset()
    return Replay(
        list(transcript().events), outer, contexts, steered, bridge.state, tracked
    )


def _model_events(events: list[Event]) -> list[ModelEvent]:
    return [event for event in events if isinstance(event, ModelEvent)]


def _main_agent(events: list[Event]) -> TimelineSpan:
    """The `deepseek_harness()` agent in the timeline the viewer builds."""
    [main] = [
        item
        for item in timeline_build(events).root.content
        if isinstance(item, TimelineSpan)
    ]
    assert (main.name, main.span_type, main.utility) == (
        "deepseek harness",
        "agent",
        False,
    )
    return main


def _turns(agent: TimelineSpan) -> list[ModelEvent]:
    """An agent's own model calls: its turns, not wrapped as utility calls."""
    return [
        item.event
        for item in agent.content
        if isinstance(item, TimelineEvent) and isinstance(item.event, ModelEvent)
    ]


def _task() -> str:
    task = json.loads(RECORDING.read_text(encoding="utf-8"))["task"]
    assert isinstance(task, str)
    return task


# ---------------------------------------------------------------------------
# the recording itself
# ---------------------------------------------------------------------------


def test_recording_covers_parallel_resumed_and_forked_children() -> None:
    agents = [request.agent for request in _recording()]
    assert agents == [
        "lead",
        "lead",
        # three children started by one message, interleaved
        "alpha",
        "beta",
        "agent-1",
        "agent-1",
        "alpha",
        "beta",
        "lead",
        "lead",
        "lead",
        # alpha again, after its first settlement
        "alpha",
        "alpha",
        "lead",
        "lead",
        # a fork, whose requests open with the root's own turns
        "gamma",
        "gamma",
        "lead",
    ]


def test_system_prompt_is_the_same_on_every_request() -> None:
    """No per-request fact (cwd, policy, identity) is in dsh's system prompt.

    The timeline shows a text-only turn whose system prompt differs from the
    span's first tool-calling call as a utility call, so a varying prompt
    would hide the final answer. The root, its children and a resumed child
    all send the same one.
    """
    prompts = {request.input[0].text for request in _recording()}
    assert len(prompts) == 1


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_each_request_is_classified_as_its_sender() -> None:
    requests = _recording()
    replay = await _replay(requests, _task())

    tools = {
        "alpha": "subagent",
        "beta": "subagent",
        "agent-1": "subagent",
        "gamma": "subagent_fork",
    }
    assert replay.contexts == [
        AgentBridgeContext("root")
        if request.agent == "lead"
        else AgentBridgeContext(
            "subagent", subagent_id=request.agent, subagent_type=tools[request.agent]
        )
        for request in requests
    ]
    # a filter gated on is_root_agent() steers the root and nothing else
    assert replay.steered == [request.agent == "lead" for request in requests]


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_each_agent_gets_its_own_balanced_span() -> None:
    requests = _recording()
    replay = await _replay(requests, _task())
    events = replay.events

    begins = [
        e
        for e in events
        if isinstance(e, SpanBeginEvent) and e.type == "agent" and e.id != replay.outer
    ]
    # the unnamed child's span is named by its description until dsh names it
    assert [(e.name, e.parent_id == replay.outer) for e in begins] == [
        ("alpha", True),
        ("beta", True),
        ("Run whoami", True),
        ("alpha", True),
        ("gamma", True),
    ]
    alpha, beta, unnamed, alpha_again, gamma = (e.id for e in begins)
    assert alpha_again == f"{alpha}-resumed-1"
    assert begins[3].metadata is not None and begins[3].metadata["continues"] == alpha

    spans = [event.span_id for event in _model_events(events)]
    by_agent = {"beta": beta, "agent-1": unnamed, "gamma": gamma}
    expected: list[str | None] = []
    alpha_requests = 0
    for request in requests:
        if request.agent == "lead":
            expected.append(replay.outer)
        elif request.agent == "alpha":
            alpha_requests += 1
            expected.append(alpha if alpha_requests <= 2 else alpha_again)
        else:
            expected.append(by_agent[request.agent])
    assert spans == expected

    # every span closed, each before the next one of the same agent opens
    ends = [e.id for e in events if isinstance(e, SpanEndEvent)]
    assert sorted(e.id for e in begins) == sorted(i for i in ends if i != replay.outer)
    order = [
        (type(e).__name__, e.id)
        for e in events
        if isinstance(e, (SpanBeginEvent, SpanEndEvent))
        and e.id in (alpha, alpha_again)
    ]
    assert order == [
        ("SpanBeginEvent", alpha),
        ("SpanEndEvent", alpha),
        ("SpanBeginEvent", alpha_again),
        ("SpanEndEvent", alpha_again),
    ]


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_root_conversation_is_not_interleaved() -> None:
    """The root's span holds one conversation whose input only ever grows."""
    replay = await _replay(_recording(), _task())
    root_inputs = [
        len(event.input)
        for event in _model_events(replay.events)
        if event.span_id == replay.outer
    ]
    assert len(root_inputs) == 8
    assert root_inputs == sorted(set(root_inputs))


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_sample_answer_is_the_root_agents_final_answer() -> None:
    requests = _recording()
    replay = await _replay(requests, _task())

    final = requests[-1]
    assert final.agent == "lead"
    assert replay.state.output.completion == final.output.completion
    assert replay.state.output.completion.rstrip().endswith("DONE")
    # the conversation is the root's: its task first, its answer last
    assert replay.state.messages[1].text.startswith(_task())
    assert replay.state.messages[-1].text == final.output.completion
    assert [m.role for m in replay.state.messages[:-1]] == [m.role for m in final.input]


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_timeline_shows_the_final_answer_as_a_main_thread_turn() -> None:
    requests = _recording()
    replay = await _replay(requests, _task())
    main = _main_agent(replay.events)

    # every root call, the text-only ones and the final answer included, is a
    # turn of the main agent rather than a utility call
    turns = _turns(main)
    assert len(turns) == sum(request.agent == "lead" for request in requests)
    assert turns[-1].output.completion == requests[-1].output.completion

    children = [item for item in main.content if isinstance(item, TimelineSpan)]
    assert [(child.name, child.span_type, child.utility) for child in children] == [
        ("alpha", "agent", False),
        ("beta", "agent", False),
        ("run whoami", "agent", False),
        ("alpha", "agent", False),
        ("gamma", "agent", False),
    ]
    # each child's calls are turns in its own lane
    assert [len(_turns(child)) for child in children] == [2, 2, 2, 2, 2]


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_short_root_answer_after_a_long_child_run() -> None:
    """The case the bridge's own thread tracking gets wrong.

    A child's multi-call loop takes over the bridge's tracked thread; the
    root wins it back only with two more requests, or one longer than the
    child's thread. Here the root hands off, the child works for many steps,
    and the root answers in a single short request.
    """
    task = "Find out which kernel this machine runs and tell me."
    prompt = "Run `uname -a`, look around, and report the kernel."
    start = assistant("", [delegation_call("c1", prompt, "scout")])
    root = conversation(root_task(task))

    requests = [Request("lead", root, ModelOutput.from_message(start))]
    child = conversation(child_task(prompt, "scout"))
    for step in range(8):
        call = assistant("", [bash_call(f"b{step}", f"echo step {step}")])
        requests.append(Request("scout", list(child), ModelOutput.from_message(call)))
        child = [*child, call, tool_result(f"b{step}", f"step {step}")]
    report = assistant("The child's report: kernel 7.2.")
    requests.append(Request("scout", list(child), ModelOutput.from_message(report)))

    answer = assistant("It runs kernel 7.2.")
    requests.append(
        Request(
            "lead",
            [*root, start, started("c1", "scout"), notice("scout", report.text)],
            ModelOutput.from_message(answer),
        )
    )

    replay = await _replay(requests, task)
    assert replay.state.output.completion == "It runs kernel 7.2."
    assert replay.state.messages[1].text.startswith(task)
    assert replay.state.messages[-1].text == "It runs kernel 7.2."
    # (the bridge alone reports `replay.tracked`, the child's report here; not
    # asserted, so that a bridge that learns to track the root doesn't fail us)

    turns = _turns(_main_agent(replay.events))
    assert [turn.output.completion for turn in turns] == ["", "It runs kernel 7.2."]
