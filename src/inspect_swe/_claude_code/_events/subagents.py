"""Claude Code sub-agents and their runs (one span each).

A sub-agent is named by the tool_use_id that launched it (its `origin`); each
resume starts a new run. At most one run of a sub-agent is open, its latest.
A run opened for a SendMessage that then failed is `undone`: it keeps its span
number but is not a run.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class Run:
    """One run of a sub-agent, and its span."""

    agent: "SubAgent"
    key: str
    span_id: str
    # which of the agent's runs that happened this is (1 = the first). Not
    # always the span id's `-runN`, which also counts undone runs
    number: int
    open: bool = True
    # keyed by a SendMessage, whose (successful) result means the run
    # started, not that it finished
    via_send: bool = False
    # opened for a SendMessage before any call showed the run; undone if the
    # SendMessage fails
    speculative: bool = False
    undone: bool = False
    # Claude Code's run id (from task_started), and those of earlier runs
    # whose late notifications must not close this one
    run_id: str | None = None
    stale_run_ids: set[str] = field(default_factory=set)
    # closed at its final reply, before anything confirmed the run was over;
    # calls can still follow (e.g. a blocking Stop hook)
    ended_by_reply: bool = False


@dataclass(eq=False)
class SubAgent:
    """A sub-agent launched this session; kept after it finishes (it can resume)."""

    origin: str
    prompt: str
    span_name: str
    description: str
    # the span current when it was launched
    parent_span_id: str | None
    # the sub-agent that launched this one (None: the main agent)
    launcher: "SubAgent | None"
    name: str | None
    fork: bool
    # Claude Code's id for it, and whether its launch result confirmed it
    agent_id: str | None = None
    id_confirmed: bool = False
    # launched in the background: each run's end is notified
    background: bool = False
    # every run, oldest first, including undone ones
    runs: list[Run] = field(default_factory=list)
    # calls attributed to it
    calls: int = 0
    # Claude Code run ids from task_started, in order: the n-th is run n
    run_ids: list[str] = field(default_factory=list)
    # SendMessage seen while the latest run still looked open; keys the run
    # it starts if that run opens lazily, under the sender's span
    pending_send: str | None = None
    pending_send_parent: str | None = None
    # keys of runs rolled over before their notification arrived, which must
    # not close the run after them (`rolled_unnotified` until we know the
    # agent is in the background)
    owed: list[str] = field(default_factory=list)
    rolled_unnotified: list[str] = field(default_factory=list)
    # its latest compaction summary (as `shapes.summary_body`), which starts
    # its history from then on, and the run it was in
    summary: str | None = None
    summary_run: int = 0
    # compactions marked from the bridge that stream-json may report again
    compactions: int = 0

    @property
    def started(self) -> int:
        """Runs that happened."""
        return sum(1 for run in self.runs if not run.undone)

    @property
    def latest(self) -> Run | None:
        """Its latest run that happened."""
        return next((run for run in reversed(self.runs) if not run.undone), None)

    @property
    def open_run(self) -> Run | None:
        latest = self.latest
        return latest if latest is not None and latest.open else None

    @property
    def may_continue(self) -> bool:
        """Its latest run ended with its own reply, and may still make calls."""
        latest = self.latest
        return latest is not None and not latest.open and latest.ended_by_reply

    def run_number(self, number: int) -> Run | None:
        """Its `number`-th run that happened."""
        return next((r for r in self.runs if not r.undone and r.number == number), None)


class SubAgents:
    """Every sub-agent this session, and lookups by each id."""

    def __init__(self) -> None:
        self._agents: dict[str, SubAgent] = {}
        self._runs: list[Run] = []
        # key → the latest run it names
        self._by_key: dict[str, Run] = {}
        self._by_span: dict[str, Run] = {}
        # SendMessage tool_use_ids, whose results say whether a run started
        self.send_ids: set[str] = set()
        # tool_use_id → the sub-agent whose call made that tool call (its
        # later calls' histories hold it). Never pruned: a retry or a fork can
        # show an old history again
        self.tool_call_owners: dict[str, SubAgent] = {}

    def __iter__(self) -> Iterator[SubAgent]:
        return iter(self._agents.values())

    def __contains__(self, origin: object) -> bool:
        return origin in self._agents

    def get(self, origin: str | None) -> SubAgent | None:
        return self._agents.get(origin) if origin is not None else None

    def add(self, agent: SubAgent) -> None:
        self._agents[agent.origin] = agent

    def new_run(self, agent: SubAgent, key: str) -> Run:
        """Record a new open run of `agent`, keyed by `key`."""
        assert agent.open_run is None, "a sub-agent has at most one open run"
        span_id = (
            f"agent-{agent.origin}-run{len(agent.runs) + 1}"
            if agent.runs
            else f"agent-{agent.origin}"
        )
        run = Run(
            agent=agent,
            key=key,
            span_id=span_id,
            number=agent.started + 1,
            via_send=key != agent.origin,
        )
        agent.runs.append(run)
        self._runs.append(run)
        self._by_key[key] = run
        self._by_span[span_id] = run
        return run

    def run(self, key: str | None) -> Run | None:
        """The latest run a key names, open or not."""
        return self._by_key.get(key) if key is not None else None

    def open_run(self, key: str | None) -> Run | None:
        run = self.run(key)
        return run if run is not None and run.open else None

    def open_runs(self) -> list[Run]:
        """Open runs, in the order they opened."""
        return [run for run in self._runs if run.open]

    def of_span(self, span_id: str | None) -> SubAgent | None:
        """The sub-agent a span belongs to (None: the main agent)."""
        run = self.run_of_span(span_id)
        return run.agent if run is not None else None

    def run_of_span(self, span_id: str | None) -> Run | None:
        return self._by_span.get(span_id) if span_id is not None else None

    def by_agent_id(self, agent_id: str | None) -> SubAgent | None:
        if not agent_id:
            return None
        found = [a for a in self._agents.values() if a.agent_id == agent_id]
        return found[-1] if found else None

    def find(self, ref: Any, caller_span_id: str | None = None) -> SubAgent | None:
        """The sub-agent a SendMessage `to` names.

        An agent id is unique; a name is only unique among one agent's
        children, so name matches prefer the caller's own launches.
        """
        if not isinstance(ref, str) or not ref:
            return None
        by_id = self.by_agent_id(ref)
        if by_id is not None:
            return by_id
        by_name = [a for a in self._agents.values() if a.name == ref]
        caller = self.of_span(caller_span_id)
        own = [a for a in by_name if a.launcher is caller]
        matches = own or by_name
        return matches[-1] if matches else None
