"""Which Claude Code sub-agent made a bridged model call, from its history.

The fallback when requests don't carry their agent id. Pure: reads the
`SubAgents` state, changes nothing. A call's history identifies its agent by:

- a fork's "Fork started" placeholders and directive;
- a compacted agent's summary;
- a tool call one of the agent's earlier calls made;
- otherwise the agent's prompt. A history that launched the agent or one of
  its ancestors is not the agent's, whatever it quotes. Identical first
  calls of agents with the same prompt go to them in launch order; calls
  still ambiguous go to the main agent.

The run boundaries in the history tell which run of the agent it is.
"""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from logging import getLogger
from typing import cast

from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)

from .shapes import (
    FORK_STARTED,
    continued_summary,
    first_user_message,
    leading_text,
    run_boundaries,
    strip_reminders,
)
from .subagents import SubAgent, SubAgents

logger = getLogger(__name__)

# shortest prompt to match: short ones can appear in unrelated content
MIN_PROMPT_LENGTH = 16

# start of a compaction summary that identifies a compacted sub-agent when
# the whole summary is not found
SUMMARY_EXCERPT_LENGTH = 200


@dataclass(frozen=True)
class Caller:
    """The sub-agent that made a call."""

    agent: SubAgent
    # finished runs its history shows, including those before a compaction
    finished_runs: int


def resolve(
    agents: SubAgents,
    messages: list[ChatMessage],
    among: Collection[SubAgent] | None = None,
) -> Caller | None:
    """The sub-agent that made a call with this history (None: the main agent).

    Args:
        agents: The sub-agents launched so far.
        messages: The call's history.
        among: Only consider these sub-agents.
    """

    def allowed(agent: SubAgent) -> bool:
        return among is None or agent in among

    fork = _match_fork(agents, messages, allowed)
    if fork is not None:
        agent, start = fork
        return Caller(agent, run_boundaries(messages[start:]))

    text = leading_text(first_user_message(messages))
    if not text:
        return None
    boundaries = run_boundaries(messages)
    # only a parent's history holds the tool call that launched an agent
    launched_here = {
        tc.id
        for msg in messages
        if isinstance(msg, ChatMessageAssistant)
        for tc in msg.tool_calls or []
    }

    summary = continued_summary(text)

    def since_compaction(agent: SubAgent) -> int:
        # a compacted history shows only the runs since its summary
        if summary is not None and _summary_leads(agent, summary):
            return agent.summary_run - 1
        return 0

    # a history holding a sub-agent's own tool calls, if it starts with its
    # prompt or summary and launched no one else's agents (so one wrongly
    # attributed call doesn't take its real caller's later calls)
    owner = _match_tool_calls(agents, messages)
    if owner is not None and not allowed(owner):
        owner = None
    launched_by_others = owner is not None and any(
        launched.launcher is not owner
        for launched in map(agents.get, launched_here)
        if launched is not None
    )
    if (
        owner is not None
        and not launched_by_others
        and not _launched_in(owner, launched_here)
        and (
            text.startswith(strip_reminders(owner.prompt))
            or (summary is not None and _summary_leads(owner, summary))
        )
    ):
        return Caller(owner, boundaries + since_compaction(owner))

    if summary is not None:
        compacted = _match_summary(agents, summary, launched_here, allowed)
        if compacted is not None:
            return Caller(compacted, boundaries + since_compaction(compacted))
        # the main agent's compaction, or a summary we did not capture
        logger.debug("compacted history matches no sub-agent summary")

    first_call = not any(isinstance(m, ChatMessageAssistant) for m in messages)
    prompted = _match_prompt(
        [a for a in agents if allowed(a)],
        text,
        boundaries,
        launched_here,
        compacted=summary is not None,
        first_call=first_call,
    )
    return Caller(prompted, boundaries) if prompted is not None else None


def _launched_in(agent: SubAgent, launched_here: set[str]) -> bool:
    """Whether a history launched `agent` or one of its ancestors."""
    launched: SubAgent | None = agent
    while launched is not None:
        if launched.origin in launched_here:
            return True
        launched = launched.launcher
    return False


def _match_tool_calls(
    agents: SubAgents, messages: list[ChatMessage]
) -> SubAgent | None:
    """The sub-agent whose earlier call made the latest tool call in a history."""
    for msg in reversed(messages):
        if isinstance(msg, ChatMessageAssistant):
            for tc in msg.tool_calls or []:
                owner = agents.tool_call_owners.get(tc.id)
                if owner is not None:
                    return owner
    return None


def _match_summary(
    agents: SubAgents,
    summary: str,
    launched_here: set[str],
    allowed: Callable[[SubAgent], bool],
) -> SubAgent | None:
    """The sub-agent whose compaction summary a continuation starts with."""
    candidates = [
        a for a in agents if allowed(a) and not _launched_in(a, launched_here)
    ]
    found = [a for a in candidates if _summary_leads(a, summary, whole=True)] or [
        a for a in candidates if _summary_leads(a, summary)
    ]
    return found[0] if len(found) == 1 else None


def _summary_leads(agent: SubAgent, summary: str, whole: bool = False) -> bool:
    """Whether a continuation starts with an agent's summary (or its start)."""
    if agent.summary is None:
        return False
    part = agent.summary if whole else agent.summary[:SUMMARY_EXCERPT_LENGTH]
    return any(
        body.startswith(part) for body in (summary, summary.removeprefix("Summary: "))
    )


def _match_prompt(
    agents: list[SubAgent],
    text: str,
    boundaries: int,
    launched_here: set[str],
    compacted: bool,
    first_call: bool,
) -> SubAgent | None:
    """The sub-agent whose prompt a history starts with (or, running, contains)."""

    def prompt(agent: SubAgent) -> str:
        return strip_reminders(agent.prompt)

    def matches(agent: SubAgent) -> bool:
        p = prompt(agent)
        if (
            agent.fork
            or len(p) < MIN_PROMPT_LENGTH
            or _launched_in(agent, launched_here)
        ):
            return False
        running = agent.open_run is not None
        if text.startswith(p):
            # a finished agent only matches a call that shows a resume, or
            # one its last run may still make
            return running or boundaries >= 1 or agent.may_continue
        return not compacted and p in text and running

    candidates = [a for a in agents if matches(a)]
    # a history that starts with a prompt is that agent's, whatever other
    # prompts it quotes
    starts = [a for a in candidates if text.startswith(prompt(a))] or candidates
    # the longest prompt wins over prompts it contains
    longest = [
        a
        for a in starts
        if not any(prompt(a) != prompt(b) and prompt(a) in prompt(b) for b in starts)
    ]
    if len(longest) <= 1:
        return longest[0] if longest else None

    # the same prompt: an agent whose current run fits the call, then one the
    # call resumes; still tied is ambiguous
    def fit(agent: SubAgent) -> int:
        current = agent.open_run is not None or agent.may_continue
        if current and boundaries == agent.started - 1:
            return 2
        return 1 if boundaries >= agent.started else 0

    top = max(fit(a) for a in longest)
    best = [a for a in longest if fit(a) == top]
    if len(best) > 1 and first_call:
        # identical first calls: the first agent yet to make one
        best = [a for a in best if a.calls == 0][:1]
    return best[0] if len(best) == 1 else None


def _match_fork(
    agents: SubAgents,
    messages: list[ChatMessage],
    allowed: Callable[[SubAgent], bool],
) -> tuple[SubAgent, int] | None:
    """(fork, index of its directive) if a call is from a fork.

    The last group of "Fork started" placeholders is the innermost fork's
    launch; its directive picks which of the forks launched there it is.
    """
    last = max(
        (
            i
            for i, m in enumerate(messages)
            if isinstance(m, ChatMessageTool)
            and m.text.lstrip().startswith(FORK_STARTED)
        ),
        default=None,
    )
    if last is None or last + 1 >= len(messages):
        return None
    directive = messages[last + 1]
    if not isinstance(directive, ChatMessageUser):
        return None
    group: list[SubAgent] = []
    i = last
    while i >= 0 and isinstance(messages[i], ChatMessageTool):
        agent = agents.get(str(cast(ChatMessageTool, messages[i]).tool_call_id))
        if agent is not None and agent.fork and allowed(agent):
            group.append(agent)
        i -= 1
    if len(group) > 1:
        hits = [a for a in group if a.prompt.strip() in directive.text]
        group = [
            a
            for a in hits
            if not any(
                a.prompt.strip() != h.prompt.strip()
                and a.prompt.strip() in h.prompt.strip()
                for h in hits
            )
        ]
    return (group[0], last + 1) if len(group) == 1 else None
