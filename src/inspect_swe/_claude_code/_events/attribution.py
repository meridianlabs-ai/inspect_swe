"""Which Claude Code sub-agent made a bridged model call.

Pure: reads the `SubAgents` state, changes nothing. Claude Code re-sends an
agent's whole history with each call, so the history identifies the agent:

- A **fork** inherits its parent's history; the "Fork started" placeholders
  for the launching tool calls, then its directive (its prompt), identify it.
- A sub-agent that **compacted** restarts its history from its summary,
  which it wrote in an earlier call.
- A history that holds a **tool call** one of a sub-agent's earlier calls
  made (tool_use_ids are unique), and starts with its prompt or summary, is
  that sub-agent's.
- Otherwise a sub-agent's history starts with its **prompt** (after Claude
  Code's `<system-reminder>`s). A still-running sub-agent also matches a
  message that merely contains its prompt, unless the caller launched it
  itself (a parent quoting the prompt) or the message is a compaction
  summary. The longest prompt wins over prompts it contains. Agents with
  the same prompt are told apart by the run the call fits, and a first call
  goes to the first of them yet to make one; agents still tied are
  ambiguous, and the call goes to the main agent. That first-call guess is
  a heuristic: identical requests carry nothing to tell the agents apart,
  so if one agent's first call fails and another's starts before the
  retry, the two can swap spans. Each keeps its calls in spans of its own,
  but those spans open and close with the other agent's runs.

The run boundaries in the history (finished runs) tell which run of the
agent the call is from.
"""

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

# Shortest prompt to match. Short prompts could appear in unrelated content;
# real Task prompts are full sentences.
MIN_PROMPT_LENGTH = 16

# Length of the start of a compaction summary that identifies a compacted
# sub-agent's later calls when the whole summary is not found.
SUMMARY_EXCERPT_LENGTH = 200


@dataclass(frozen=True)
class Caller:
    """The sub-agent that made a call."""

    agent: SubAgent
    # finished runs the call's history shows, counting runs before a
    # compaction: the call is from run `finished_runs + 1`, or continues
    # an earlier one
    finished_runs: int


def resolve(agents: SubAgents, messages: list[ChatMessage]) -> Caller | None:
    """The sub-agent that made a call with this history (None: the main agent)."""
    fork = _match_fork(agents, messages)
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

    # a history holding a sub-agent's own tool calls, if it is also one the
    # sub-agent could have: it starts with its prompt or summary, and holds no
    # launches but its own (so one wrongly attributed call doesn't take every
    # later call of its real caller with it)
    owner = _match_tool_calls(agents, messages)
    launched_by_others = owner is not None and any(
        launched.launcher is not owner
        for launched in map(agents.get, launched_here)
        if launched is not None
    )
    if (
        owner is not None
        and not launched_by_others
        and (
            text.startswith(strip_reminders(owner.prompt))
            or (summary is not None and _summary_leads(owner, summary))
        )
    ):
        return Caller(owner, boundaries + since_compaction(owner))

    if summary is not None:
        compacted = _match_summary(agents, summary, launched_here)
        if compacted is not None:
            return Caller(compacted, boundaries + since_compaction(compacted))
        # the main agent's compaction, or a summary we did not capture
        logger.debug("compacted history matches no sub-agent summary")

    first_call = not any(isinstance(m, ChatMessageAssistant) for m in messages)
    prompted = _match_prompt(
        agents,
        text,
        boundaries,
        launched_here,
        compacted=summary is not None,
        first_call=first_call,
    )
    return Caller(prompted, boundaries) if prompted is not None else None


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
    agents: SubAgents, summary: str, launched_here: set[str]
) -> SubAgent | None:
    """The sub-agent whose compaction summary a continuation starts with.

    A compacted parent's own summary can quote a child's further on, so it
    must start with it: the whole summary, or failing that (Claude Code
    reformatted its end) its start. An untagged summary may still be
    labelled "Summary:".
    """
    candidates = [a for a in agents if a.origin not in launched_here]
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
    agents: SubAgents,
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
        if agent.fork or len(p) < MIN_PROMPT_LENGTH or agent.origin in launched_here:
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
        # identical first calls: the first agent (in launch order) yet to make
        # one; later calls hold its tool calls
        best = [a for a in best if a.calls == 0][:1]
    return best[0] if len(best) == 1 else None


def _match_fork(
    agents: SubAgents, messages: list[ChatMessage]
) -> tuple[SubAgent, int] | None:
    """(fork, index of its directive) if a call is from a fork.

    Claude Code answers every tool_use in the launching message with the same
    "Fork started" placeholder, then gives the fork its directive (its prompt)
    in one user message. The last such group is the innermost fork; among its
    fork launches, the directive picks which one this is (the longest prompt
    wins over prompts it contains).
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
        if agent is not None and agent.fork:
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
