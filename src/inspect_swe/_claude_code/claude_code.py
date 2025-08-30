from inspect_ai.agent import Agent, AgentState, agent


@agent
def claude_code() -> Agent:
    async def execute(state: AgentState) -> AgentState:
        raise RuntimeError("claude_code() not yet implemented.")

    return execute
