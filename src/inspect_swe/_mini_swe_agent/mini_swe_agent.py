from textwrap import dedent
from typing import Literal

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter
from inspect_ai.scorer import score
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store

from .._util._async import is_callable_coroutine
from .._util.agentwheel import AgentWheelSource, ensure_agent_wheel_installed
from .._util.messages import build_user_prompt
from .._util.trace import trace

# mini-swe-agent wheel source configuration
# Pin to v1.x by default - v2.0 has breaking changes (migration guide pending)
# See: https://mini-swe-agent.com/latest/advanced/v2_migration/
MINI_SWE_AGENT_SOURCE = AgentWheelSource(
    agent="mini-swe-agent",
    package="mini-swe-agent",
    binary="mini",  # CLI entrypoint
    default_version="1.17.4",
)


@agent
def mini_swe_agent(
    name: str = "mini-swe-agent",
    description: str = dedent("""
       Minimal AI agent that solves software engineering tasks using bash commands.
       100 lines of Python, radically simple, scores >74% on SWE-bench verified.
    """),
    system_prompt: str | None = None,
    attempts: int | AgentAttempts = 1,  # TODO: currently supports single attempt
    model: str | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cost_limit: float | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["stable", "sandbox", "latest"] | str = "stable",
) -> Agent:
    """mini-swe-agent agent.

    Agent that uses [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
    running in a sandbox. Mini-swe-agent is a minimal 100-line agent that solves
    GitHub issues using only bash commands.

    The agent can either use a version of mini-swe-agent installed in the sandbox,
    or can download and install it via pip (see docs on `version` option below).

    Use `attempts` to enable additional submissions if initial submission(s)
    are incorrect (by default, no additional attempts are permitted).

    Use `cost_limit` to set a maximum cost for the agent run (in USD).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems)
        system_prompt: Additional system prompt to append to default.
        attempts: Configure agent to make multiple attempts.
        model: Model name to use (defaults to main model for task).
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cost_limit: Maximum cost limit for the agent run.
        cwd: Working directory to run mini-swe-agent within.
        env: Environment variables to set for mini-swe-agent.
        user: User to execute mini-swe-agent with.
        sandbox: Optional sandbox environment name.
        version: Version of mini-swe-agent to use. One of:
            - "stable": Download and install the default pinned version.
            - "sandbox": Use version in sandbox (raises RuntimeError if not available)
            - "latest": Download and install latest version from PyPI.
            - "x.x.x": Install and use a specific version.
    """
    # resolve models
    inspect_model = f"inspect/{model}" if model is not None else "inspect"

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "mini_swe_agent_model_port"
        port = store().get(MODEL_PORT, 4000) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=inspect_model,
            filter=filter,
            retry_refusals=retry_refusals,
            port=port,
        ) as bridge:
            # ensure mini-swe-agent is installed
            mini_binary = await ensure_agent_wheel_installed(
                source=MINI_SWE_AGENT_SOURCE,
                version=version,
                user=user,
                sandbox=sandbox_env(sandbox),
            )

            # base command options
            cmd = [
                mini_binary,
                "--yolo",  # run without confirmations (like --print for claude)
                "--exit-immediately",  # exit when agent finishes instead of prompting
            ]

            # add cost limit if specified
            if cost_limit is not None:
                cmd.extend(["--cost-limit", str(cost_limit)])

            # build user prompt
            prompt, has_assistant_response = build_user_prompt(state.messages)

            # add system prompt context if provided
            full_prompt = prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)
            if system_messages:
                # Prepend system context to the task
                system_context = "\n\n".join(system_messages)
                full_prompt = (
                    f"System instructions:\n{system_context}\n\nTask:\n{prompt}"
                )

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # execute the agent
            debug_output: list[str] = []
            agent_prompt = full_prompt
            attempt_count = 0

            while True:  # Kept for consistency with other agents but currently only single attempt supported
                # TODO: build command with task. This only works for single-turn tasks currently. Need to update to support multi-turn (perhaps via file output option)
                agent_cmd = cmd + ["--task", agent_prompt]

                # run agent
                # Use /dev/null for stdin instead of closing it, so prompt_toolkit
                # doesn't fail at module import time when checking stdin.fileno()
                result = await sbox.exec(
                    cmd=["bash", "-c", 'exec 0</dev/null "$@"', "bash"] + agent_cmd,
                    cwd=cwd,
                    env={
                        "MSWEA_CONFIGURED": "true",  # Skip interactive setup wizard
                        "MSWEA_MODEL_NAME": model,
                        "OPENAI_API_BASE": f"http://localhost:{bridge.port}/v1",
                        "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                    }
                    | (env or {}),
                    user=user,
                    concurrency=False,
                )

                # track debug output
                debug_output.append(result.stdout)
                debug_output.append(result.stderr)

                # raise for error
                if not result.success:
                    raise RuntimeError(
                        f"Error executing mini-swe-agent (cwd={cwd or 'default'}):\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}"
                    )

                # exit if we are at max_attempts
                attempt_count += 1
                if attempt_count >= attempts.attempts:
                    break

                # score this attempt
                answer_scores = await score(state)

                # break if we score 'correct'
                if attempts.score_value(answer_scores[0].value) == 1.0:
                    break

                # otherwise update prompt with incorrect message and continue
                else:
                    if callable(attempts.incorrect_message):
                        if not is_callable_coroutine(attempts.incorrect_message):
                            raise ValueError(
                                "The incorrect_message function must be async."
                            )
                        agent_prompt = await attempts.incorrect_message(
                            state, answer_scores
                        )
                    else:
                        agent_prompt = attempts.incorrect_message

            # trace debug info
            debug_output.insert(0, "mini-swe-agent Debug Output:")
            trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)
