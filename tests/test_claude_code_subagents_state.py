"""The sub-agent and run records `LiveConsumer` keeps (`subagents`)."""

import pytest
from inspect_swe._claude_code._events.subagents import SubAgent, SubAgents


def _agent(origin: str = "toolu_1", launcher: SubAgent | None = None) -> SubAgent:
    return SubAgent(
        origin=origin,
        prompt="Count the TODO comments in src/",
        span_name="general-purpose",
        description="",
        parent_span_id="outer",
        launcher=launcher,
        name="worker",
        fork=False,
    )


def test_runs_get_unique_span_ids_and_numbers() -> None:
    agents = SubAgents()
    agent = _agent()
    agents.add(agent)
    first = agents.new_run(agent, "toolu_1")
    assert (first.span_id, first.number, first.via_send) == ("agent-toolu_1", 1, False)
    first.open = False
    second = agents.new_run(agent, "send_1")
    assert (second.span_id, second.number, second.via_send) == (
        "agent-toolu_1-run2",
        2,
        True,
    )
    assert agents.of_span("agent-toolu_1-run2") is agent
    assert agents.of_span("outer") is None


def test_an_undone_run_keeps_its_span_number_but_is_not_a_run() -> None:
    agents = SubAgents()
    agent = _agent()
    agents.add(agent)
    first = agents.new_run(agent, "toolu_1")
    first.open = False
    first.ended_by_reply = True
    failed = agents.new_run(agent, "send_1")
    assert not agent.may_continue
    failed.open, failed.undone = False, True
    assert (agent.started, agent.latest) == (1, first)
    assert agent.may_continue  # the run before is the latest again
    third = agents.new_run(agent, "toolu_1")
    assert (third.span_id, third.number) == ("agent-toolu_1-run3", 2)
    assert agents.run("toolu_1") is third  # a reused key names its latest run
    assert (agent.run_number(1), agent.run_number(2)) == (first, third)


def test_a_sub_agent_has_at_most_one_open_run() -> None:
    agents = SubAgents()
    agent = _agent()
    agents.add(agent)
    agents.new_run(agent, "toolu_1")
    with pytest.raises(AssertionError):
        agents.new_run(agent, "send_1")


def test_names_resolve_among_the_callers_own_launches_first() -> None:
    agents = SubAgents()
    parent = _agent("toolu_p")
    agents.add(parent)
    agents.new_run(parent, "toolu_p")
    mine = _agent("toolu_mine", launcher=parent)
    other = _agent("toolu_other")
    agents.add(mine)
    agents.add(other)
    assert agents.find("worker", "agent-toolu_p") is mine
    assert agents.find("worker", "outer") is other
    mine.agent_id = "a1"
    assert agents.find("a1") is mine
    assert agents.find("") is None
