from inspect_ai.agent import Agent, AgentState, agent


@agent
def claude_code() -> Agent:
    async def execute(state: AgentState) -> AgentState:
        return state

    return execute
