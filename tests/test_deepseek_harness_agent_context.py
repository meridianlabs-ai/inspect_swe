"""Agent-context classification for `deepseek_harness()` (root, subagent, utility, unknown).

What the user's `filter` sees in `current_agent_bridge_context()` for each
kind of request dsh makes, through the filter `deepseek_harness()` hands the
bridge. Request shapes are in `tests/dsh_shapes.py` (dsh 0.2.2rc2).
"""

from typing import Any

import pytest
from inspect_ai.agent import (
    AgentBridgeContext,
    current_agent_bridge_context,
    is_root_agent,
)
from inspect_ai.agent._bridge.context import bridged_request_scope
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe._deepseek_harness._events.consumer import DshConsumer
from inspect_swe._deepseek_harness._events.detection import WEB_SEARCH_REQUEST_PREFIX
from inspect_swe._deepseek_harness.deepseek_harness import build_dsh_filter

from tests.dsh_shapes import (
    COMPACTION_INSTRUCTION,
    agent_message,
    assistant,
    bash_call,
    blocks,
    checkpoint,
    child_reminder,
    child_task,
    conversation,
    delegation_call,
    fork_task,
    notice,
    root_reminder,
    root_task,
    started,
    tool_result,
)

TASK = "Check the three services and summarize what is broken."
ALPHA = "Run `uname -a` and report its exact output."

ROOT = AgentBridgeContext("root")
UTILITY = AgentBridgeContext("utility")
ALPHA_CONTEXT = AgentBridgeContext(
    "subagent", subagent_id="alpha", subagent_type="subagent"
)

_START = assistant("", [delegation_call("c1", ALPHA, "alpha")])
_ROOT_HISTORY: list[ChatMessage] = [root_task(TASK), _START, started("c1", "alpha")]

# request → the context the user's filter must see, with a child named alpha
# running (started by the root's first reply)
CASES: dict[str, tuple[list[ChatMessage], AgentBridgeContext]] = {
    "root-first-request": (conversation(root_task(TASK)), ROOT),
    "root-after-starting-a-child": (conversation(*_ROOT_HISTORY), ROOT),
    "root-with-a-settlement-notice": (
        conversation(*_ROOT_HISTORY, assistant("waiting"), notice("alpha", "Linux")),
        ROOT,
    ),
    "root-after-compaction": (
        conversation(
            checkpoint("## Current Work\n- waiting for alpha"),
            ChatMessageUser(content=blocks(root_reminder(replaces=True))),
        ),
        ROOT,
    ),
    "child-first-request": (conversation(child_task(ALPHA, "alpha")), ALPHA_CONTEXT),
    "child-mid-task": (
        conversation(
            child_task(ALPHA, "alpha"),
            assistant("", [bash_call("b1", "uname -a")]),
            tool_result("b1", "Linux"),
        ),
        ALPHA_CONTEXT,
    ),
    "child-resumed-by-a-message": (
        conversation(
            child_task(ALPHA, "alpha"),
            assistant("Linux"),
            agent_message("lead", "Also run `id -u`."),
        ),
        ALPHA_CONTEXT,
    ),
    "forked-child": (
        conversation(
            root_task(TASK),
            assistant("Both are running."),
            fork_task("Report the year.", "gamma"),
        ),
        # a child whose start the consumer never saw: no type
        AgentBridgeContext("subagent", subagent_id="gamma"),
    ),
    "root-compaction": (
        conversation(*_ROOT_HISTORY, ChatMessageUser(content=COMPACTION_INSTRUCTION)),
        UTILITY,
    ),
    "child-compaction": (
        conversation(
            child_task(ALPHA, "alpha"),
            assistant("", [bash_call("b1", "uname -a")]),
            tool_result("b1", "Linux"),
            ChatMessageUser(content=COMPACTION_INSTRUCTION),
        ),
        UTILITY,
    ),
    "compaction-of-a-span-without-a-reminder": (
        conversation(
            ChatMessageUser(
                content=[
                    *checkpoint("## Current Work\n- step 3").content,
                    *blocks(COMPACTION_INSTRUCTION),
                ]
            )
        ),
        UTILITY,
    ),
    # the model request behind dsh's web_search tool: the query, nothing else
    "web-search-request": (
        [ChatMessageUser(content=blocks(f"{WEB_SEARCH_REQUEST_PREFIX}MCP transports"))],
        UTILITY,
    ),
    # negatives: text that only quotes dsh's markers must not flip a request
    "task-worded-like-a-web-search-request": (
        conversation(root_task(f"{WEB_SEARCH_REQUEST_PREFIX}MCP transports")),
        ROOT,
    ),
    "task-quoting-a-child-reminder": (
        conversation(root_task(f"Explain this text:\n{child_reminder('alpha')}")),
        ROOT,
    ),
    "tool-output-quoting-a-child-reminder": (
        conversation(
            *_ROOT_HISTORY,
            assistant("", [bash_call("b1", "cat session.jsonl")]),
            tool_result("b1", child_reminder("alpha")),
        ),
        ROOT,
    ),
    "task-quoting-the-delegated-prompt": (
        conversation(root_task(f"{TASK}\nEarlier you delegated: {ALPHA}")),
        ROOT,
    ),
    "child-told-about-the-root-reminder": (
        conversation(child_task(f"Explain:\n{root_reminder()}", "alpha")),
        ALPHA_CONTEXT,
    ),
    "task-quoting-the-compaction-prompt": (
        conversation(root_task(f"Explain this prompt:\n{COMPACTION_INSTRUCTION}")),
        ROOT,
    ),
    "tool-output-quoting-the-compaction-prompt": (
        conversation(
            *_ROOT_HISTORY,
            assistant("", [bash_call("b1", "cat summarizer.ts")]),
            tool_result("b1", COMPACTION_INSTRUCTION),
        ),
        ROOT,
    ),
    "compaction-prompt-earlier-in-the-conversation": (
        conversation(
            root_task(COMPACTION_INSTRUCTION),
            assistant("That is dsh's compaction prompt."),
            ChatMessageUser(content="Thanks, carry on."),
        ),
        ROOT,
    ),
    # dsh gave the sender no identity while a child is running: it could be
    # either, so neither is claimed
    "no-reminder-while-a-child-runs": (
        conversation(ChatMessageUser(content=TASK)),
        AgentBridgeContext("unknown"),
    ),
}


class _Seen:
    def __init__(self) -> None:
        self.context: AgentBridgeContext | None = None
        self.root: bool | None = None
        self.result: Any = None

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        self.context = current_agent_bridge_context()
        self.root = is_root_agent()
        return self.result  # type: ignore[no-any-return]


async def _classify(
    request: list[ChatMessage], seen: _Seen | None
) -> tuple[AgentBridgeContext | None, Any]:
    """Run one request through the filter, after the root started `alpha`."""
    init_transcript(Transcript())
    consumer = DshConsumer()
    consumer.on_filter_output(
        conversation(root_task(TASK)), ModelOutput.from_message(_START)
    )
    filter = build_dsh_filter(seen, consumer)
    with bridged_request_scope("deepseek/deepseek-v4.1-flash-1017"):
        result = await filter(
            get_model("mockllm/model"),
            request,
            [],
            None,
            GenerateConfig(reasoning_effort="low"),
        )
        return current_agent_bridge_context(), result


@pytest.mark.parametrize("case", sorted(CASES))
async def test_request_is_classified(case: str) -> None:
    request, expected = CASES[case]
    stamped, result = await _classify(request, None)
    assert stamped == expected
    # no filter of the caller's: the request goes through unchanged
    assert result is None


@pytest.mark.parametrize("case", sorted(CASES))
async def test_user_filter_sees_the_context(case: str) -> None:
    request, expected = CASES[case]
    seen = _Seen()
    await _classify(request, seen)
    assert seen.context == expected
    # the usual steering gate: on for the root (and for "unknown"), off for
    # sub-agents and dsh's own compaction
    assert seen.root == (expected.kind in ("root", "unknown"))


async def test_filter_results_pass_through() -> None:
    """A filter may rewrite the request or answer it; the context is set first."""
    request = conversation(child_task(ALPHA, "alpha"))

    seen = _Seen()
    seen.result = GenerateInput(
        input=[ChatMessageUser(content="rewritten")],
        tools=[],
        tool_choice=None,
        config=GenerateConfig(),
    )
    stamped, result = await _classify(request, seen)
    assert (stamped, result) == (ALPHA_CONTEXT, seen.result)

    seen = _Seen()
    seen.result = ModelOutput.from_content("mockllm/model", "answered by the filter")
    stamped, result = await _classify(request, seen)
    assert (stamped, result) == (ALPHA_CONTEXT, seen.result)


async def test_default_reasoning_effort_reaches_the_user_filter() -> None:
    """Dsh asks for high effort itself; the bridge drops that, so the agent restores it."""
    captured: list[GenerateConfig] = []

    async def user_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        captured.append(config)
        return None

    init_transcript(Transcript())
    filter = build_dsh_filter(user_filter, DshConsumer())
    with bridged_request_scope("deepseek/deepseek-v4.1-flash-1017"):
        result = await filter(
            get_model("mockllm/model"),
            conversation(root_task(TASK)),
            [],
            None,
            GenerateConfig(),
        )
    assert captured[0].reasoning_effort == "high"
    assert isinstance(result, GenerateInput)
    assert result.config.reasoning_effort == "high"
