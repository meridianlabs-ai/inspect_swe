from inspect_ai import Task, eval, task
from inspect_ai.agent import Agent
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant
from inspect_swe import claude_code

from tests.conftest import run_example, skip_if_no_anthropic, skip_if_no_docker


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_example() -> None:
    log = run_example("system_explorer", "anthropic/claude-sonnet-4-0")[0]
    assert log.status == "success"


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_options() -> None:
    SYSTEM_PROMPT_CANARY = "32C507F0-9347-4DB2-8061-907682DD34EB"
    PASSED_MODEL = "anthropic/claude-sonnet-4-0"
    MAIN_MODEL = "anthropic/claude-3-7-sonnet-20250219"
    SMALL_MODEL = "anthropic/claude-3-5-haiku-20241022"

    log = eval(
        system_explorer(
            claude_code(
                system_prompt=f"This is a part of the system prompt {SYSTEM_PROMPT_CANARY}. When solving this task you should use a mix of the main model and the smaller model that you typically use for backgrounds tasks.",
                model=MAIN_MODEL,
                small_model=SMALL_MODEL,
                env={"MAX_THINKING_TOKENS": "16666"},
            )
        ),
        model=PASSED_MODEL,
    )[0]
    assert log.status == "success"
    log_json = log.model_dump_json(exclude_none=True)
    assert SYSTEM_PROMPT_CANARY in log_json
    assert MAIN_MODEL in log_json
    assert SMALL_MODEL in log_json
    assert "16666" in log_json


@skip_if_no_anthropic
@skip_if_no_docker
def test_claude_code_tools() -> None:
    log = run_example("web_search", "anthropic/claude-sonnet-4-0")[0]
    assert log.status == "success"
    assert log.samples
    assistant_messages = [
        m for m in log.samples[0].messages if isinstance(m, ChatMessageAssistant)
    ]
    tool_calls = [tc for m in assistant_messages for tc in (m.tool_calls or [])]
    assert next((tc for tc in tool_calls if tc.function == "WebSearch"), None)
    assert next(
        (tc for tc in tool_calls if tc.function == "mcp__memory__create_entities"), None
    )

@task
def system_explorer(agent: Agent) -> Task:
    return Task(
        dataset=[
            Sample(
                input="Without using the internet, investigate the network configuration and report: 1) What network interfaces are present on the system, 2) What is the IP address of the loopback interface, and 3) What port does SSH typically listen on according to its configuration file?",
                target="The output should list the network interfaces, correctly identify 127.0.0.1 as the loopback IP address, and report port 22 as the SSH port",
            )
        ],
        solver=agent,
        sandbox="docker",
    )
