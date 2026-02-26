import posixpath
import shlex
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
from inspect_ai.model import (
    ChatMessageSystem,
    CompactionStrategy,
    GenerateFilter,
    get_model,
)
from inspect_ai.scorer import score
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from .._util._async import is_callable_coroutine
from .._util.agentwheel import AgentWheelSource, ensure_agent_wheel_installed
from .._util.centaur import CentaurOptions, run_centaur
from .._util.messages import build_user_prompt
from .._util.trace import trace
from .setup import (
    RESUMABLE_AGENT_PATH,
    get_trajectory_path,
    install_resumable_agent,
    validate_version,
)

# mini-swe-agent wheel source configuration (v2.x)
MINI_SWE_AGENT_SOURCE = AgentWheelSource(
    agent="mini-swe-agent",
    package="mini-swe-agent",
    binary="mini",  # CLI entrypoint
    default_version="2.2.3",
)


@agent
def mini_swe_agent(
    name: str = "mini-swe-agent",
    description: str = dedent("""
       Minimal AI agent that solves software engineering tasks using bash commands.
       100 lines of Python, radically simple, scores >74% on SWE-bench verified.
    """),
    system_prompt: str | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    compaction: CompactionStrategy | None = None,
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

    This agent does not handle compaction natively. Use `compaction` to specify a compaction strategy.

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description (used in multi-agent systems)
        system_prompt: Additional system prompt to include (appended to any system messages from the task).
        centaur: Run in 'centaur' mode, which makes mini-swe-agent available
            to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts.
        model: Model name to use (defaults to main model for task).
        filter: Filter for intercepting bridged model requests.
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        compaction: Compaction strategy for managing context window overflow.
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
    # validate version before anything else
    validate_version(version)

    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve models
    inspect_model = f"inspect/{model}" if model is not None else "inspect"

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "mini_swe_agent_model_port"
        port = store().get(MODEL_PORT, 4000) + 1
        store().set(MODEL_PORT, port)

        # resolve full model name (e.g. "openai/gpt-5-mini") for provider
        # detection and bridge model creation
        model_name = model if model is not None else str(get_model())
        provider = model_name.split("/")[0] if "/" in model_name else "openai"
        if provider == "openai":
            # Note: force responses_api=False to avoid conversion issue with tool use.
            bridge_model = get_model(model_name, responses_api=False, memoize=False)
        else:
            bridge_model = get_model(model_name, memoize=False)

        async with sandbox_agent_bridge(
            state,
            model=inspect_model,
            model_aliases={inspect_model: bridge_model},
            filter=filter,
            retry_refusals=retry_refusals,
            compaction=compaction,
            port=port,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # ensure mini-swe-agent is installed
            mini_binary = await ensure_agent_wheel_installed(
                source=MINI_SWE_AGENT_SOURCE,
                version=version,
                user=user,
                sandbox=sbox,
            )

            # MSWEA_MODEL_NAME uses "{provider}/inspect" so litellm routes
            # to the correct provider while
            # "inspect" avoids litellm's model-specific routing logic.
            agent_env = {
                "MSWEA_CONFIGURED": "true",
                "MSWEA_MODEL_NAME": f"{provider}/inspect",
                "OPENAI_BASE_URL": f"http://localhost:{bridge.port}/v1",
                "OPENAI_API_BASE": f"http://localhost:{bridge.port}/v1",
                "OPENAI_API_KEY": "sk-none",
                "ANTHROPIC_BASE_URL": f"http://localhost:{bridge.port}",
                "ANTHROPIC_API_KEY": "sk-none",
            } | (env or {})

            # centaur mode uses human_cli with custom instructions and bashrc
            if centaur:
                await _run_mini_swe_centaur(
                    options=centaur,
                    mini_cmd=[mini_binary],
                    agent_env=agent_env,
                    state=state,
                )
            else:
                # install resumable agent to sandbox
                await install_resumable_agent(sbox)
                trajectory_path = get_trajectory_path()

                # build prompt (incorporating system messages)
                prompt, has_assistant_response = build_user_prompt(state.messages)
                system_messages = [
                    m.text for m in state.messages if isinstance(m, ChatMessageSystem)
                ]
                if system_prompt is not None:
                    system_messages.append(system_prompt)
                if system_messages:
                    system_context = "\n\n".join(system_messages)
                    prompt = (
                        f"System instructions:\n{system_context}\n\nTask:\n{prompt}"
                    )

                # base command with v2 agent-class and output flags
                cmd = [
                    mini_binary,
                    "--yolo",
                    "--exit-immediately",
                    "--agent-class",
                    "resumable_agent.ResumableAgent",
                    "--output",
                    trajectory_path,
                ]

                # add cost limit if specified
                if cost_limit is not None:
                    cmd.extend(["--cost-limit", str(cost_limit)])

                # env for mini CLI (adds PYTHONPATH for resumable agent import)
                mini_env = agent_env | {
                    "PYTHONPATH": posixpath.dirname(RESUMABLE_AGENT_PATH)
                }

                # execute the agent
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    is_resume = has_assistant_response or attempt_count > 0
                    run_env = mini_env | {
                        "MSWEA_RESUME": "true" if is_resume else "false"
                    }
                    agent_cmd = cmd + ["--task", agent_prompt]

                    result = await sbox.exec_remote(
                        cmd=["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
                        + agent_cmd,
                        options=ExecRemoteAwaitableOptions(
                            cwd=cwd,
                            env=run_env,
                            user=user,
                            concurrency=False,
                        ),
                        stream=False,
                    )

                    # track debug output
                    debug_output.append(f"[stdout]\n{result.stdout}")
                    debug_output.append(f"[stderr]\n{result.stderr}")

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


async def _run_mini_swe_centaur(
    options: CentaurOptions,
    mini_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    instructions = (
        "mini-swe-agent:\n\n - You may use mini-swe-agent via the 'mini' command."
    )

    # build .bashrc content
    agent_env_vars = [f"export {k}={shlex.quote(v)}" for k, v in agent_env.items()]
    alias_cmd = shlex.join(mini_cmd)
    alias_cmd = "alias mini='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(agent_env_vars + ["", alias_cmd])

    # run the human cli
    await run_centaur(options, instructions, bashrc, state)
