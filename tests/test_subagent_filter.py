"""`is_sub_agent()` — telling a caller's GenerateFilter which thread a bridge call belongs to.

Offline: synthetic message lists, no sandbox, no model, no bridge. Two things are pinned here.

1. The consumers' `is_sub_agent_call`, which exposes attribution they already compute.
2. `with_sub_agent_attribution`, which binds a filter to ITS OWN consumer at the single call
   site that constructs both — the property that makes cross-sample contamination impossible
   under concurrency.

The over-detection cases (a parent quoting the text it spawned with) are the reason
attribution excludes a conversation's own issued calls; they are called out individually below,
alongside the nested-delegation cases that exclusion exists to preserve.
"""

import asyncio
from typing import Any

from inspect_ai._util.content import ContentText
from inspect_ai.model import GenerateConfig, Model, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_swe import is_sub_agent
from inspect_swe._claude_code._events.live_consumer import LiveConsumer, _OpenAgent
from inspect_swe._codex_cli._events.consumer import CodexConsumer
from inspect_swe._codex_cli._events.consumer import _OpenAgent as _CodexOpenAgent
from inspect_swe._util.subagent import with_sub_agent_attribution

# A prompt comfortably past _MIN_PROMPT_LENGTH (16).
SUBTASK = "audit the parser for out-of-bounds reads and report findings"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _claude_consumer(*prompts: str) -> LiveConsumer:
    """A consumer with sub-agent spans already open, as `on_complete` would leave them."""
    consumer = LiveConsumer()
    for i, prompt in enumerate(prompts):
        tool_use_id = f"toolu_{i}"
        consumer._pending_subagents[tool_use_id] = prompt
        consumer._open_agents[tool_use_id] = _OpenAgent(span_id=f"agent-{tool_use_id}")
    return consumer


def _codex_consumer(*prompts: str) -> CodexConsumer:
    consumer = CodexConsumer()
    for i, prompt in enumerate(prompts):
        call_id = f"call_{i}"
        consumer._agents[call_id] = _CodexOpenAgent(
            call_id=call_id, span_id=f"agent-{call_id}", prompt=prompt, name=f"a{i}"
        )
    return consumer


def _sub_agent_thread(prompt: str) -> list[ChatMessage]:
    """A sub-agent's forked conversation: opens at the Task prompt, no parent history."""
    return [ChatMessageUser(content=prompt), ChatMessageAssistant(content="on it")]


def _parent_thread(
    first_user: str, *, spawned: list[tuple[str, str]] | None = None
) -> list[ChatMessage]:
    """The spawning agent's own conversation, carrying its Task/spawn_agent tool calls."""
    out: list[ChatMessage] = [
        ChatMessageSystem(content="you are claude code"),
        ChatMessageUser(content=first_user),
    ]
    for call_id, prompt in spawned or []:
        out.append(
            ChatMessageAssistant(
                content="delegating",
                tool_calls=[
                    ToolCall(id=call_id, function="Task", arguments={"prompt": prompt})
                ],
            )
        )
    return out


# ---------------------------------------------------------------------------
# 1. claude_code attribution
# ---------------------------------------------------------------------------


def test_claude_sub_agent_thread_is_detected() -> None:
    consumer = _claude_consumer(SUBTASK)
    assert consumer.is_sub_agent_call(_sub_agent_thread(SUBTASK)) is True


def test_claude_main_thread_is_not_a_sub_agent() -> None:
    consumer = _claude_consumer(SUBTASK)
    assert consumer.is_sub_agent_call(_parent_thread("do the task")) is False


def test_claude_no_pending_sub_agents_is_never_a_sub_agent() -> None:
    assert LiveConsumer().is_sub_agent_call(_sub_agent_thread(SUBTASK)) is False


def test_claude_parent_quoting_its_own_task_prompt_is_not_a_sub_agent() -> None:
    """The over-detection case, and why `_attribute` excludes a conversation's own calls.

    The parent's first user message is the task instruction, which can quote the very text the
    agent then hands to Task. Substring-matched naively, the parent matches its own child: its
    events get parented under the sub-agent, and a caller gating on `is_sub_agent_call` loses
    its steering on the real agent for the rest of the episode.
    """
    consumer = _claude_consumer(SUBTASK)
    parent = _parent_thread(
        f"Here is the job. One subtask is to {SUBTASK}. Delegate as you see fit.",
        spawned=[("toolu_0", SUBTASK)],
    )
    assert consumer.is_sub_agent_call(parent) is False


def test_claude_short_prompts_are_never_matched() -> None:
    """Upstream's `_MIN_PROMPT_LENGTH` guard: under-detection, the safe direction."""
    short = "look"
    consumer = _claude_consumer(short)
    assert consumer.is_sub_agent_call(_sub_agent_thread(short)) is False


def test_claude_ambiguous_overlapping_prompts_fall_back_to_main() -> None:
    consumer = _claude_consumer(SUBTASK, SUBTASK + " twice over")
    assert (
        consumer.is_sub_agent_call(_sub_agent_thread(SUBTASK + " twice over")) is False
    )


def test_claude_pending_entry_without_an_open_span_is_not_a_sub_agent() -> None:
    consumer = _claude_consumer(SUBTASK)
    consumer._open_agents.clear()
    assert consumer.is_sub_agent_call(_sub_agent_thread(SUBTASK)) is False


# ---------------------------------------------------------------------------
# 2. codex attribution
# ---------------------------------------------------------------------------


def _agent_message(author: str, recipient: str) -> ChatMessageUser:
    raw: dict[str, Any] = {
        "type": "agent_message",
        "author": author,
        "recipient": recipient,
        "content": [{"type": "input_text", "text": "go"}],
    }
    return ChatMessageUser(
        content=[ContentText(text="Agent message", internal={"agent_message": raw})]
    )


def test_codex_recipient_identifies_a_sub_agent() -> None:
    consumer = _codex_consumer(SUBTASK)
    consumer._agents["call_0"].name = "worker"
    assert consumer.is_sub_agent_call([_agent_message("/root", "/root/worker")]) is True


def test_codex_main_thread_is_not_a_sub_agent() -> None:
    consumer = _codex_consumer(SUBTASK)
    assert consumer.is_sub_agent_call(_parent_thread("do the task")) is False


def test_codex_parent_quoting_its_own_spawn_prompt_is_not_a_sub_agent() -> None:
    """The V1 substring fallback's over-detection case; V2 recipients are exact and unaffected."""
    consumer = _codex_consumer(SUBTASK)
    parent = _parent_thread(
        f"Here is the job. One subtask is to {SUBTASK}.",
        spawned=[("call_0", SUBTASK)],
    )
    assert consumer.is_sub_agent_call(parent) is False


def test_codex_v1_substring_still_matches_a_real_sub_agent() -> None:
    """The exclusion must not cost V1 its attribution when the thread really is forked."""
    consumer = _codex_consumer(SUBTASK)
    assert consumer.is_sub_agent_call(_sub_agent_thread(SUBTASK)) is True


# ---------------------------------------------------------------------------
# 3. the ambient API and its binding
# ---------------------------------------------------------------------------


class _Recorder:
    """A caller's filter: records what `is_sub_agent()` said on each invocation."""

    def __init__(self) -> None:
        self.seen: list[bool] = []

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        self.seen.append(is_sub_agent())
        return None


class _FakeModel:
    """Stands in for the resolved Model (only `.name` is ever read)."""

    name = "test-model"


async def _invoke(filter_fn: Any, messages: list[ChatMessage]) -> Any:
    return await filter_fn(_FakeModel(), messages, [], None, GenerateConfig())


def test_is_sub_agent_is_false_outside_a_bridged_call() -> None:
    assert is_sub_agent() is False


def test_wrapped_filter_sees_the_verdict_and_it_is_reset_after() -> None:
    recorder = _Recorder()
    wrapped = with_sub_agent_attribution(recorder, _claude_consumer(SUBTASK))
    assert wrapped is not None

    asyncio.run(_invoke(wrapped, _sub_agent_thread(SUBTASK)))
    asyncio.run(_invoke(wrapped, _parent_thread("do the task")))

    assert recorder.seen == [True, False]
    assert is_sub_agent() is False, "the scope must not leak past the call"


def test_two_samples_never_see_each_others_attribution() -> None:
    """The structural property: each filter is bound to its own sample's consumer.

    Sample B has an open sub-agent whose prompt would match A's conversation; A must still
    report False, because A's filter can only ever consult A's consumer.
    """
    a_recorder, b_recorder = _Recorder(), _Recorder()
    a = with_sub_agent_attribution(a_recorder, _claude_consumer())
    b = with_sub_agent_attribution(b_recorder, _claude_consumer(SUBTASK))
    assert a is not None and b is not None

    asyncio.run(_invoke(a, _sub_agent_thread(SUBTASK)))
    asyncio.run(_invoke(b, _sub_agent_thread(SUBTASK)))

    assert a_recorder.seen == [False]
    assert b_recorder.seen == [True]


def test_scope_is_reset_when_the_inner_filter_raises() -> None:
    async def _boom(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        raise RuntimeError("filter exploded")

    wrapped = with_sub_agent_attribution(_boom, _claude_consumer(SUBTASK))
    assert wrapped is not None
    try:
        asyncio.run(_invoke(wrapped, _sub_agent_thread(SUBTASK)))
    except RuntimeError:
        pass
    assert is_sub_agent() is False


def test_attribution_failure_reports_main_agent() -> None:
    """Fail SAFE in one direction: over-reporting would strip the real agent of its steering."""

    class _Broken:
        def is_sub_agent_call(self, input_messages: list[ChatMessage]) -> bool:
            raise RuntimeError("attribution exploded")

    recorder = _Recorder()
    wrapped = with_sub_agent_attribution(recorder, _Broken())
    assert wrapped is not None
    asyncio.run(_invoke(wrapped, _sub_agent_thread(SUBTASK)))
    assert recorder.seen == [False]


def test_agents_without_an_attributing_consumer_are_untouched() -> None:
    """Agents that install no consumer (antigravity, ACP) must get their filter back as-is."""
    recorder = _Recorder()
    assert with_sub_agent_attribution(recorder, None) is recorder
    assert with_sub_agent_attribution(recorder, object()) is recorder
    assert with_sub_agent_attribution(None, _claude_consumer(SUBTASK)) is None


def test_a_legacy_str_filter_still_receives_a_model_name() -> None:
    """`GenerateFilter` still admits the deprecated str-first shape; the wrapper must honour it."""
    seen: list[Any] = []

    async def _legacy(
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        seen.append((model, is_sub_agent()))
        return None

    wrapped = with_sub_agent_attribution(_legacy, _claude_consumer(SUBTASK))
    assert wrapped is not None

    asyncio.run(_invoke(wrapped, _sub_agent_thread(SUBTASK)))
    assert seen == [("test-model", True)]


# ---------------------------------------------------------------------------
# 4. nested delegation (the issued-call exclusion, not a short-circuit)
# ---------------------------------------------------------------------------

NESTED = "trace every call site of the allocator and summarise the lifetimes"


def test_claude_nested_delegator_still_matches_its_own_spawn_prompt() -> None:
    """A sub-agent that spawns its own sub-agent must keep its identity.

    Excluding only the calls a conversation ISSUED — rather than sending the whole conversation
    to the outer span the moment it carries any pending Task — is what preserves this. The
    spawning agent is not necessarily the top-level one; `on_complete` parents to
    `event.span_id or outer_span_id` precisely so nesting works.
    """
    consumer = _claude_consumer(
        SUBTASK, NESTED
    )  # toolu_0 = A (ours), toolu_1 = A's child
    thread_a: list[ChatMessage] = [
        ChatMessageUser(content=SUBTASK),
        ChatMessageAssistant(
            content="delegating further",
            tool_calls=[
                ToolCall(id="toolu_1", function="Task", arguments={"prompt": NESTED})
            ],
        ),
    ]
    assert consumer._attribute(thread_a) == "agent-toolu_0"
    assert consumer.is_sub_agent_call(thread_a) is True


def test_claude_parent_quoting_survives_the_nested_fix() -> None:
    """The over-detection guard must still hold with the exclusion in place."""
    consumer = _claude_consumer(SUBTASK)
    parent = _parent_thread(
        f"Here is the job. One subtask is to {SUBTASK}.", spawned=[("toolu_0", SUBTASK)]
    )
    assert consumer.is_sub_agent_call(parent) is False


def test_codex_v1_nested_delegator_still_matches_its_own_spawn_prompt() -> None:
    consumer = _codex_consumer(SUBTASK, NESTED)  # call_0 = A (ours), call_1 = A's child
    thread_a: list[ChatMessage] = [
        ChatMessageUser(content=SUBTASK),
        ChatMessageAssistant(
            content="delegating further",
            tool_calls=[
                ToolCall(
                    id="call_1", function="spawn_agent", arguments={"message": NESTED}
                )
            ],
        ),
    ]
    assert consumer._attribute(thread_a) == "agent-call_0"
    assert consumer.is_sub_agent_call(thread_a) is True


def test_codex_v1_parent_quoting_survives_the_nested_fix() -> None:
    consumer = _codex_consumer(SUBTASK)
    parent = _parent_thread(
        f"Here is the job. One subtask is to {SUBTASK}.", spawned=[("call_0", SUBTASK)]
    )
    assert consumer.is_sub_agent_call(parent) is False
