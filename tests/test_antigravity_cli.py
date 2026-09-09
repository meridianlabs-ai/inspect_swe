"""End-to-end coverage for the native Antigravity CLI (`agy`) agent.

`tests/test_antigravity_cli_config.py` and
`tests/test_antigravity_cli_error_reporting.py` cover the on-disk configuration
and the failure-reporting helper as pure units. Neither runs the CLI, so neither
can catch the launch itself not working: a flag the binary rejects, a settings
key that no longer selects the direct-Gemini route, or a tool call that reaches
the CLI and executes nothing.

The two round-trip tests here run the pinned binary for real. Only the *model* is
scripted -- via the keyless `mockllm` provider, so no Google credentials and no
network generation are involved -- and it is reached through the real
`sandbox_agent_bridge` over loopback, exactly as a live run reaches a real
model. The CLI, its tools, the bridge and the sandbox are all genuine.

The first has the CLI execute its own `run_command` tool to write and read back
a harmless fixture file, asserted against that file's real content in the
sandbox. The second exposes a host-side Inspect tool over the bridge's MCP
endpoint and has the CLI call it, asserted against that host tool's real return
value arriving back in the sandbox.

Both scripted argument sets are the ones the pinned CLI's own declarations mark
required: `CommandLine`, `Cwd`, `WaitMsBeforeAsync`, `toolAction` and
`toolSummary` for `run_command`; the tool's own `message` plus `toolSummary`
and `toolAction` for the bridged tool, which the CLI declares under a
server-qualified name of its own making rather than the host's. Omitting a
required key, or using the host-side name, is rejected before the tool runs.

Three further Docker tests inspect the launch rather than run one. They install
the pinned binary for real and capture what the agent builds -- the command,
the environment it carries and the settings file it writes -- to pin which
controls belong to the unattended path, which of them are withheld from the
command a human is handed in centaur mode, and that a disabled reasoning effort
is really left off. They are launch-contract inspections, not native proof:
the unattended two stop the run at the launch rather than let a CLI that never
ran appear to have produced a result. `effort=None` cannot be a real launch at
all -- the CLI's default model rejects the flag's absence -- which is the same
reason these are inspections.

A last test covers the shared agent-binary tooling and needs neither Docker nor
the network.
"""

import json
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState, BridgedToolsSpec, agent, run
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalSample, resolve_sample_attachments
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolChoice, ToolInfo, tool
from inspect_ai.util import (
    ExecRemoteAwaitableOptions,
    ExecResult,
    SandboxEnvironment,
    sandbox,
)
from inspect_swe import antigravity_cli
from inspect_swe._antigravity_cli import agentbinary
from inspect_swe._antigravity_cli.antigravity_cli import (
    AntigravityEffort,
    _native_conversation_id,
)
from inspect_swe._tools import download as download_tool
from inspect_swe._util import centaur as centaur_module
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR

from tests.conftest import skip_if_no_docker

# The package rebinds `antigravity_cli` to the agent factory, so both
# `from inspect_swe._antigravity_cli import antigravity_cli` and
# `import inspect_swe._antigravity_cli.antigravity_cli as m` hand back that
# function rather than the module -- and `patch.object` on a function fails.
# Naming a member, as the import above does, reaches the module properly;
# resolve the module itself here for the launch tests to patch.
agy_module = import_module("inspect_swe._antigravity_cli.antigravity_cli")

# Pinned rather than "auto"/"latest" so the run is reproducible, and so an
# upstream flag or tool-schema change surfaces here as a deliberate version
# bump rather than as a test that quietly started exercising something else.
_CLI_VERSION = "1.1.27"

_MODEL = "mockllm/model"

_FIXTURE_DIR = "/tmp"
_FIXTURE_PATH = f"{_FIXTURE_DIR}/antigravity-cli-fixture.txt"
_FIXTURE_CONTENT = "antigravity-cli-fixture"

# Absolute paths inside the command, so what the assertions read back does not
# depend on how the CLI resolves the tool's `Cwd`.
_COMMAND_LINE = (
    f"printf %s '{_FIXTURE_CONTENT}' > {_FIXTURE_PATH} && cat {_FIXTURE_PATH}"
)

_RUN_COMMAND_ARGS: dict[str, Any] = {
    "CommandLine": _COMMAND_LINE,
    "Cwd": _FIXTURE_DIR,
    # Synchronous: wait for the command rather than backgrounding it, so its
    # output lands in the tool result. 10000ms is the declared maximum.
    "WaitMsBeforeAsync": 10000,
    "toolAction": "Running command",
    "toolSummary": "Command execution",
}

_FINAL_ANSWER = f"The fixture file contains {_FIXTURE_CONTENT}."


class _ScriptedModel:
    """Drive the CLI to one `run_command` call, then let it finish.

    Scripted on conversation *state* rather than on call count: the tool call is
    returned until a tool result comes back, then the final text. A CLI that
    issues an extra request (a retry, a summarization turn) therefore cannot
    desynchronize the script or exhaust it -- which a fixed list of outputs
    would, since `mockllm` raises once its outputs run out.
    """

    def __init__(self) -> None:
        self.declared_tools: list[str] = []
        self.requests = 0

    def __call__(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.requests += 1
        if not self.declared_tools:
            self.declared_tools = [info.name for info in tools]

        answered = any(isinstance(message, ChatMessageTool) for message in input)
        if answered:
            return ModelOutput.from_content(model=_MODEL, content=_FINAL_ANSWER)
        return ModelOutput.for_tool_call(
            model=_MODEL,
            tool_name="run_command",
            tool_arguments=_RUN_COMMAND_ARGS,
        )


@scorer(metrics=[])
def fixture_file_written() -> Scorer:
    """Read the fixture file back out of the sandbox the CLI actually ran in."""

    async def score(state: TaskState, target: Target) -> Score:
        result = await sandbox().exec(["cat", _FIXTURE_PATH])
        if not result.success:
            return Score(
                value=0,
                explanation=(
                    f"{_FIXTURE_PATH} was not created by the CLI: {result.stderr}"
                ),
            )
        if result.stdout != _FIXTURE_CONTENT:
            return Score(
                value=0,
                explanation=(
                    f"{_FIXTURE_PATH} holds {result.stdout!r}, "
                    f"expected {_FIXTURE_CONTENT!r}"
                ),
            )
        return Score(value=1, explanation="the CLI wrote the fixture file")

    return score


def _bridged_requests(sample: EvalSample) -> list[list[Any]]:
    """Every conversation the bridge served, in order, as the model saw it.

    Read from the model events rather than from `sample.messages`.
    `bridge.state` holds ONE conversation, and `agy` opens a second of its own:
    a tool-free request with its own system prompt ("You are a conversation
    title generator") carrying the task prompt as its user message. The
    sample's own message list is whichever of the two the bridge ended up
    bound to, which is not necessarily the task's -- in the run this was
    written against it was the title generator's, so the task's tool result
    and final answer were absent from `sample.messages` entirely while both
    were plainly present in the requests.

    Attachments are resolved first, exactly as `_env_block_counts` in
    tests/test_multi_call.py does: a sample straight out of `eval()` carries
    `attachment://` placeholders in place of long message text.
    """
    sample = resolve_sample_attachments(sample, "full")
    return [
        list(getattr(event, "input", []))
        for event in sample.events
        if getattr(event, "event", None) == "model"
    ]


@skip_if_no_docker
@pytest.mark.slow
def test_native_cli_executes_a_tool_call_over_the_bridge() -> None:
    scripted = _ScriptedModel()
    task = Task(
        dataset=[
            Sample(
                input=(
                    f"Write {_FIXTURE_CONTENT} to {_FIXTURE_PATH}, read it back, "
                    "and tell me what it contains."
                ),
                target=_FIXTURE_CONTENT,
            )
        ],
        solver=antigravity_cli(cwd=_FIXTURE_DIR, version=_CLI_VERSION),
        scorer=fixture_file_written(),
        sandbox="docker",
    )

    # Bounded on time for the reason conftest.run_example bounds its evals:
    # nothing else caps a wedged CLI, and an unbounded stall reports as an
    # opaque pytest timeout rather than as a failure. No token limit -- the
    # scripted model reports no usage, so a token ceiling would never trip.
    logs = eval(
        task,
        model=get_model(_MODEL, custom_outputs=scripted),
        limit=1,
        time_limit=300,
    )

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success", f"CLI run failed: {log.error}"
    assert log.samples

    # The CLI reached the bridge at all, and offered the model the tool this
    # test drives: a CLI that stopped declaring `run_command` would otherwise
    # be indistinguishable from a model that simply chose not to call it.
    assert scripted.requests > 0, "the CLI made no bridged model request"
    assert "run_command" in scripted.declared_tools, scripted.declared_tools

    # The command really ran. This reads the file in the sandbox, not a
    # transcript of an intent to write it.
    sample = log.samples[0]
    assert sample.scores
    score_value = list(sample.scores.values())[0]
    assert score_value.value == 1, score_value.explanation

    # --- what the bridge carried (supplement) -----------------------------
    #
    # The requests are every conversation the bridge served. They localize a
    # failure: if these pass and the canonical assertions below do not, the
    # round trip worked and the eval lost it.
    requests = _bridged_requests(sample)
    assert requests, "the eval recorded no bridged model requests"

    # The call the CLI executed is the call that was scripted. Only
    # `CommandLine` is asserted: these messages are the CLI's own replay of its
    # history, and which advisory fields it echoes there is not this contract.
    #
    # The ids are the CLI's, taken from that same replay. The bridge hands back
    # an id of its own making and the CLI substitutes its own, so a result can
    # only be paired to its call within one replay -- not against the id the
    # bridge returned.
    executed = {
        call.id: call
        for messages in requests
        for message in messages
        if isinstance(message, ChatMessageAssistant)
        for call in (message.tool_calls or [])
        if call.function == "run_command"
    }
    assert executed, "no run_command call was recorded"
    assert any(
        call.arguments.get("CommandLine") == _COMMAND_LINE for call in executed.values()
    ), [call.arguments for call in executed.values()]

    # ...and its output came back to the model through the bridge, linked by id
    # to the call it answers rather than merely sharing its name.
    tool_results = [
        message
        for messages in requests
        for message in messages
        if isinstance(message, ChatMessageTool)
        and message.function == "run_command"
        and message.tool_call_id in executed
    ]
    assert tool_results, "no run_command result was bridged back"
    assert any(_FIXTURE_CONTENT in message.text for message in tool_results), [
        message.text for message in tool_results
    ]

    # --- what the eval recorded (acceptance) ------------------------------
    #
    # `sample.output` and `sample.messages` are the canonical record: they are
    # all an ordinary scorer ever sees, so `includes()` or `model_graded_qa()`
    # would grade these and nothing else. A round trip the eval failed to
    # record is not a round trip a user can score, which is why acceptance
    # lives here and not on the requests above.
    #
    # `agy` also opens a tool-free "conversation title generator" conversation
    # whose user message is the task prompt, and `bridge.state` can end up
    # bound to that one instead of the task's. A sample recording it carries
    # the title generator's system prompt, no tool result and no answer, and
    # has to fail here.
    assert _FINAL_ANSWER in sample.output.completion, (
        "the sample did not record the answer the bridge served -- canonical "
        f"state lost: {sample.output.completion!r}"
    )

    recorded_results = [
        message
        for message in sample.messages
        if isinstance(message, ChatMessageTool) and message.function == "run_command"
    ]
    assert recorded_results, (
        "the sample recorded no run_command result -- canonical state lost; "
        f"roles recorded: {[message.role for message in sample.messages]}"
    )
    assert any(_FIXTURE_CONTENT in message.text for message in recorded_results), [
        message.text for message in recorded_results
    ]


# --- bridged tools over MCP -------------------------------------------------

# The CLI renames a bridged MCP tool rather than declaring it under the name
# the host registered: `echo` on server `inspect-tools` reaches the model as
# `mcp_inspect-tools_echo`. The server name is spliced in verbatim, hyphens and
# all -- read off the tool declarations of a real run rather than inferred.
# The hyphen is the point: the conventional name for the bridged server is
# hyphenated, so an underscore-only stand-in leaves the path every caller
# actually takes unexercised.
#
# The derived name is also the only name the CLI will execute. A call naming
# anything it has not declared comes back as
# `unknown tool: "<name>" -- check spelling`, delivered as a tool RESULT rather
# than as a failed run, so nothing errors and the agent simply never gets its
# answer. That is why the declared name is asserted rather than assumed.
_MCP_SERVER = "inspect-tools"
_ECHO_DECLARED_TOOL = "mcp_inspect-tools_echo"
# The dispatcher the CLI declares INSTEAD of the tools themselves when they are
# not exposed eagerly. With every host tool exposed, it has nothing left to
# route and is absent from the declaration entirely.
_LAZY_DISPATCHER = "call_mcp_tool"
_ECHO_PREFIX = "echoed: "
_ECHO_MESSAGE = "antigravity-mcp-round-trip"
_ECHO_FINAL_ANSWER = "Done."

# Every key the CLI's declaration of the bridged tool marks required: the
# tool's own `message` argument, plus the two advisory strings the CLI attaches
# to every tool it declares. The declaration sets additionalProperties=false.
_ECHO_ARGS: dict[str, Any] = {
    "message": _ECHO_MESSAGE,
    "toolSummary": "MCP echo call",
    "toolAction": "Calling MCP tool",
}


@tool
def echo() -> Tool:
    async def execute(message: str) -> str:
        """Echo a message back verbatim.

        Args:
            message: The message to echo.
        """
        return f"{_ECHO_PREFIX}{message}"

    return execute


class _McpRoundTrip:
    """Call the bridged echo tool once through the CLI, then finish.

    Scripted on conversation state exactly as `_ScriptedModel` is, and for the
    same reason. The declared names are recorded from the first request that
    carries any: the CLI's opening request was observed to declare no tools,
    and it issues auxiliary requests of its own, so no ordering beyond "the
    first non-empty declaration" is assumed.
    """

    def __init__(self) -> None:
        self.declared: list[str] = []

    def __call__(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if not self.declared:
            self.declared = [info.name for info in tools]

        if any(isinstance(message, ChatMessageTool) for message in input):
            return ModelOutput.from_content(model=_MODEL, content=_ECHO_FINAL_ANSWER)
        return ModelOutput.for_tool_call(
            model=_MODEL,
            tool_name=_ECHO_DECLARED_TOOL,
            tool_arguments=_ECHO_ARGS,
        )


@skip_if_no_docker
@pytest.mark.slow
def test_native_cli_calls_a_bridged_tool_under_its_declared_name() -> None:
    scripted = _McpRoundTrip()
    task = Task(
        dataset=[
            Sample(
                input=f"Echo {_ECHO_MESSAGE} using the {_MCP_SERVER} MCP server.",
                target=_ECHO_MESSAGE,
            )
        ],
        solver=antigravity_cli(
            bridged_tools=[BridgedToolsSpec(name=_MCP_SERVER, tools=[echo()])],
            cwd=_FIXTURE_DIR,
            version=_CLI_VERSION,
        ),
        sandbox="docker",
    )

    logs = eval(
        task,
        model=get_model(_MODEL, custom_outputs=scripted),
        limit=1,
        time_limit=300,
    )

    assert len(logs) == 1
    log = logs[0]
    # Reaching the model proves the loopback MCP endpoint really served
    # tools/list: with bridged tools present the agent gates its launch on
    # wait_for_mcp_endpoints(required=True), so a dead endpoint errors the run
    # before any generation happens.
    assert log.status == "success", f"CLI run failed: {log.error}"
    assert log.samples
    assert scripted.declared, "the CLI made no bridged model request"

    declared = sorted(scripted.declared)
    # Recorded unconditionally: the declared surface IS the subject of this
    # test, so a reader should never have to infer it from source or re-run to
    # see what the CLI actually offered.
    print(f"declared tools: {declared}")

    # The bridged tool is declared, under the CLI's server-qualified name...
    assert _ECHO_DECLARED_TOOL in declared, declared
    # ...and the lazy dispatcher is not declared at all. A model that called it
    # would get `unknown tool` back as a tool RESULT, not an error, so its
    # absence has to be asserted rather than discovered.
    assert _LAZY_DISPATCHER not in declared, declared

    # --- what the bridge carried (supplement) -----------------------------
    #
    # As in the other round trip: these show what the bridge served, so a
    # failure below separates "the round trip broke" from "the eval lost it".
    sample = log.samples[0]
    requests = _bridged_requests(sample)
    assert requests, "the eval recorded no bridged model requests"

    # The model reached the tool under the declared name, which is the only
    # name the CLI will execute.
    dispatched = {
        call.id: call
        for messages in requests
        for message in messages
        if isinstance(message, ChatMessageAssistant)
        for call in (message.tool_calls or [])
        if call.function == _ECHO_DECLARED_TOOL
    }
    assert dispatched, f"no {_ECHO_DECLARED_TOOL} call was recorded"
    assert any(
        call.arguments.get("message") == _ECHO_MESSAGE for call in dispatched.values()
    ), [call.arguments for call in dispatched.values()]

    # The host-side tool really ran, and its result is linked by id to that
    # call. This is its return value, produced in the eval process and carried
    # back into the sandbox over the bridge's MCP endpoint -- not text the CLI
    # could have synthesized.
    results = [
        message
        for messages in requests
        for message in messages
        if isinstance(message, ChatMessageTool)
        and message.function == _ECHO_DECLARED_TOOL
        and message.tool_call_id in dispatched
    ]
    assert results, f"no {_ECHO_DECLARED_TOOL} result was bridged back"
    assert any(
        f"{_ECHO_PREFIX}{_ECHO_MESSAGE}" in message.text for message in results
    ), [message.text for message in results]

    # ...and the eval recorded it, which is the only form a scorer can read.
    # Acceptance, as in the other round trip; a sample bound to `agy`'s own
    # title-generator conversation has none of this and fails here.
    recorded = [
        message
        for message in sample.messages
        if isinstance(message, ChatMessageTool)
        and message.function == _ECHO_DECLARED_TOOL
    ]
    assert recorded, (
        f"the sample recorded no {_ECHO_DECLARED_TOOL} result -- canonical "
        f"state lost; roles recorded: {[message.role for message in sample.messages]}"
    )
    assert any(
        f"{_ECHO_PREFIX}{_ECHO_MESSAGE}" in message.text for message in recorded
    ), [message.text for message in recorded]
    assert _ECHO_FINAL_ANSWER in sample.output.completion, (
        "the sample did not record the answer the bridge served -- canonical "
        f"state lost: {sample.output.completion!r}"
    )


# --- public download/cache API ----------------------------------------------


def test_cached_agent_binaries_recognizes_the_antigravity_cli(tmp_path: Path) -> None:
    """The shared binary tooling is this agent's only public install surface.

    `antigravity_cli` is installed through the same `AgentBinarySource` flow as
    `claude_code` and `codex_cli`, and its factory is module-private like
    theirs, so `download_agent_binary` / `cached_agent_binaries` are the only
    way a caller can pre-download or inspect its binaries.

    Only the cache directory is redirected here: the artifacts are created at
    the paths the source itself names, and found by the source's own listing.
    The name written, the listing that finds it and the parse that reads a
    version back out are one contract, and a mismatch anywhere in it leaves a
    downloaded binary invisible to every caller and silently re-downloaded on
    each run. It also pins the naming, which follows the binary
    (`agy-<version>-<platform>`) rather than the agent.
    """
    with patch.object(agentbinary, "package_cache_dir", return_value=tmp_path):
        source = agentbinary.antigravity_cli_binary_source()
        for version in ("1.1.20", "1.1.27"):
            source.cached_binary_path(version, "linux-x64").write_bytes(b"binary")

        cached = download_tool.cached_agent_binaries("antigravity_cli")

    # Newest first, and attributed to this agent.
    assert [binary.version for binary in cached] == ["1.1.27", "1.1.20"]
    assert {binary.agent for binary in cached} == {"antigravity_cli"}
    assert {binary.path.name for binary in cached} == {
        "agy-1.1.27-linux-x64",
        "agy-1.1.20-linux-x64",
    }
    assert all(binary.path.exists() for binary in cached)


# --- unattended vs centaur launch -------------------------------------------

# Print-mode controls, and they belong to the unattended launch only. The task
# prompt has to stay literal (one beginning with `/` would otherwise expand as
# a slash command) and tool calls have to proceed unprompted. In centaur mode
# the human at the terminal types their own prompts and is the approver, so
# both are withheld from the command they are handed.
_PRINT_MODE_FLAGS = ("--disable-slash-commands", "--dangerously-skip-permissions")


_LAUNCH_INSPECTED = "antigravity launch inspected; the CLI is not run here"


class _LaunchInspected(Exception):
    """Ends a run at the launch, because there is no CLI result to carry on with.

    The alternative is to hand back a result the CLI never produced, and a
    result nobody can attribute to a real run is exactly what this agent's own
    verification exists to reject. Rather than weaken that check for the tests
    that never intended to run anything, the inspection stops here and says so
    by name, so a run that failed for any other reason cannot be mistaken for
    one of these.
    """


class _CaptureLaunch:
    """The real sandbox, except the agent's launch is captured, not run.

    Everything the agent does to prepare -- installing the binary, probing
    `$HOME`, writing settings and the MCP registry -- goes through to the real
    sandbox. `write_file` is recorded on its way through, so the settings the
    CLI would read are the ones asserted on. Only `exec_remote`, the launch
    itself, is intercepted, so the captured command and environment are the
    ones the agent genuinely built.
    """

    def __init__(self, inner: SandboxEnvironment) -> None:
        self._inner = inner
        self.cmd: list[str] = []
        self.env: dict[str, str] = {}
        self.written: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def write_file(self, file: str, contents: str | bytes) -> None:
        if isinstance(contents, str):
            self.written[file] = contents
        await self._inner.write_file(file, contents)

    async def exec_remote(
        self,
        cmd: list[str],
        options: ExecRemoteAwaitableOptions | None = None,
        *,
        stream: Literal[False],
    ) -> ExecResult[str]:
        self.cmd = list(cmd)
        self.env = dict(getattr(options, "env", None) or {})
        raise _LaunchInspected(_LAUNCH_INSPECTED)


def _settings_written(launch: _CaptureLaunch) -> dict[str, Any]:
    """Read back the settings.json the agent wrote into the sandbox."""
    paths = [path for path in launch.written if path.endswith("settings.json")]
    assert len(paths) == 1, sorted(launch.written)
    parsed: dict[str, Any] = json.loads(launch.written[paths[0]])
    return parsed


_LAUNCH_PROMPT = "Say nothing."


def _launch_task(centaur: bool, effort: AntigravityEffort | None = "low") -> Task:
    return Task(
        dataset=[Sample(input=_LAUNCH_PROMPT, target="nothing")],
        solver=antigravity_cli(
            centaur=centaur, cwd=_FIXTURE_DIR, effort=effort, version=_CLI_VERSION
        ),
        sandbox="docker",
    )


def _unattended_launch(effort: AntigravityEffort | None) -> _CaptureLaunch:
    """Take the agent as far as building its unattended launch, and capture it.

    The run ends there, at the sentinel, rather than in a score: everything up
    to the launch is real, and nothing after it happened at all.
    """
    launches: list[_CaptureLaunch] = []

    def capture_sandbox(name: str | None = None) -> Any:
        captured = _CaptureLaunch(sandbox(name))
        launches.append(captured)
        return captured

    with patch.object(agy_module, "sandbox_env", capture_sandbox):
        logs = eval(
            _launch_task(centaur=False, effort=effort),
            model=_MODEL,
            limit=1,
            time_limit=300,
        )

    # The sentinel, not merely "it failed": a broken image, a failed install or
    # a bad settings file would also end the run, and none of those reached the
    # launch this inspection is about.
    error = logs[0].error
    assert logs[0].status == "error" and error is not None, logs[0].status
    reported = f"{error.message}\n{error.traceback}"
    assert _LAUNCH_INSPECTED in reported, reported
    assert launches, "the agent never resolved a sandbox"
    launch = launches[0]
    cmd = launch.cmd
    assert cmd, "the agent never launched the CLI"
    # The pinned binary was really installed and is what is being launched, so
    # every assertion below reads a complete command rather than a stub. The
    # version is matched as a substring: the artifact name ends in the sandbox
    # platform, which differs on an arm64 host.
    assert any(f"agy-{_CLI_VERSION}" in arg for arg in cmd), cmd

    # The image's own PATH is left alone. The CLI is a self-contained binary
    # launched by absolute path, so it needs nothing from PATH -- while every
    # command the agent goes on to run needs the image's toolchain, which a
    # replacement PATH would hide.
    assert "PATH" not in launch.env, sorted(launch.env)
    # Withheld, not "no environment was built": the bridge route is still set,
    # which is what keeps generation on loopback instead of Google's endpoint.
    assert launch.env["GOOGLE_GEMINI_BASE_URL"].startswith("http://localhost:")

    # Nobody is watching an unattended run, so both approval policies have to
    # be in the settings file the CLI reads.
    settings = _settings_written(launch)
    assert settings["toolPermission"] == "always-proceed"
    assert settings["artifactReviewPolicy"] == "always-proceed"
    return launch


@skip_if_no_docker
@pytest.mark.slow
def test_unattended_launch_carries_the_print_mode_flags() -> None:
    cmd = _unattended_launch(effort="low").cmd

    for flag in _PRINT_MODE_FLAGS:
        assert flag in cmd, cmd
    # The prompt is delivered by the agent rather than typed by a human, and it
    # is delivered as `--print`'s own argument. A `--print` whose prompt landed
    # anywhere else leaves the CLI reading a stdin that is closed here, so it
    # would answer nothing at all.
    assert "--print" in cmd, cmd
    assert _LAUNCH_PROMPT in cmd[cmd.index("--print") + 1], cmd
    # Reasoning effort is passed as asked rather than left to the CLI's own
    # default, which would otherwise decide what an eval measured.
    assert "--effort" in cmd, cmd
    assert cmd[cmd.index("--effort") + 1] == "low", cmd
    # The headless result is read as the CLI's own JSON: identity and status
    # come from that envelope, so `text` would leave nothing to verify the run
    # against before it is scored.
    assert "--output-format" in cmd, cmd
    assert cmd[cmd.index("--output-format") + 1] == "json", cmd


@skip_if_no_docker
@pytest.mark.slow
def test_disabling_effort_withholds_the_flag_entirely() -> None:
    """`effort=None` is the escape hatch for a model that rejects the flag.

    `agy --effort` is only adjustable on the model families the CLI defaults
    to; passing it for any other model exits 1 before the run starts. A default
    that leaked through would therefore break exactly the configuration the
    option exists to serve, at launch rather than as a visible rejection of the
    option itself.
    """
    cmd = _unattended_launch(effort=None).cmd

    assert "--effort" not in cmd, cmd
    # ...and not under some other spelling either.
    assert "low" not in cmd, cmd
    # Withheld, not "the command was never built".
    assert "--model" in cmd, cmd
    assert "--print" in cmd, cmd


@skip_if_no_docker
@pytest.mark.slow
def test_centaur_launch_withholds_the_print_mode_flags() -> None:
    handed: list[list[str]] = []
    handed_env: list[dict[str, str]] = []
    handed_cwd: list[str] = []
    launches: list[_CaptureLaunch] = []

    async def capture_centaur(
        options: Any,
        agy_cmd: list[str],
        agent_env: dict[str, str],
        state: Any,
        *,
        user: str | None,
        sandbox: str | None,
        cwd: str,
    ) -> None:
        handed.append(list(agy_cmd))
        handed_env.append(dict(agent_env))
        handed_cwd.append(cwd)

    def capture_sandbox(name: str | None = None) -> Any:
        captured = _CaptureLaunch(sandbox(name))
        launches.append(captured)
        return captured

    with (
        patch.object(agy_module, "sandbox_env", capture_sandbox),
        patch.object(agy_module, "_run_antigravity_cli_centaur", capture_centaur),
    ):
        logs = eval(_launch_task(centaur=True), model=_MODEL, limit=1, time_limit=300)

    assert logs[0].status == "success", f"CLI run failed: {logs[0].error}"
    assert handed, "the agent never dispatched to centaur"
    cmd = handed[0]

    # The human is the approver here, which is the whole reason the print-mode
    # flags below are withheld. The settings file is the other half of that: it
    # is written before the human ever sees a terminal, and one that
    # auto-approves everything would take the approval back from a file they
    # have no reason to read.
    settings = _settings_written(launches[0])
    assert "toolPermission" not in settings, sorted(settings)
    assert "artifactReviewPolicy" not in settings, sorted(settings)
    # Withheld, not "no settings were written".
    assert settings["modelProvider"] == "gemini"

    # The image's PATH survives into the human's shell too.
    assert "PATH" not in handed_env[0], sorted(handed_env[0])
    # And the working directory this agent resolved is handed over with it: the
    # human's session runs where the agent's own commands would have.
    assert handed_cwd == [_FIXTURE_DIR], handed_cwd

    for flag in _PRINT_MODE_FLAGS:
        assert flag not in cmd, cmd
    # No prompt is injected either -- the human supplies it. Both halves matter:
    # the flag is gone, and so is the task text it would have carried.
    assert "--print" not in cmd, cmd
    assert not any(_LAUNCH_PROMPT in arg for arg in cmd), cmd
    # The print-mode result format goes with them. The human reads the CLI's
    # ordinary terminal output, and there is no headless envelope to parse.
    assert "--output-format" not in cmd, cmd
    # Still the real launch command -- the pinned binary, at that -- so the
    # absences above mean "withheld", not "the command was never built".
    assert f"agy-{_CLI_VERSION}" in cmd[0], cmd
    assert "--model" in cmd, cmd


# --- canonical producer identity --------------------------------------------
#
# `agy` opens a second conversation of its own to generate a title, and that
# request can end up as the conversation the bridge binds -- see
# `_bridged_requests` above, written against a run where exactly that happened
# and the task's own tool result and final answer were absent from
# `sample.messages` entirely. For a task with a single model call whose title
# arrives last there is no continued main loop left to recover from, so the
# only fix is to identify the native conversation and keep canonical state on
# it.

_LATE_TITLE_ANSWER = "The single answer."
_TITLE_TEXT = "Fixture Task Title"
# Producer identity is asserted on the id the model actually emitted, not on
# matching text: text can coincide, and an id cannot be reconstructed by
# anything that rebuilt state from the CLI's own output.
_MAIN_MESSAGE_ID = "antigravity-main-answer"
_RETRY_ANSWER = "Answered again."


class _Handoff:
    """A happens-after between the state filter and the title request.

    The event is created on first touch rather than in the test body: an
    `anyio.Event` binds to the running backend, and both touches here happen
    inside the sample's own loop -- the filter is called synchronously from
    `_track_state`, the title request from the model callback.
    """

    def __init__(self) -> None:
        self._event: anyio.Event | None = None

    def _ensure(self) -> anyio.Event:
        if self._event is None:
            self._event = anyio.Event()
        return self._event

    def signal(self) -> None:
        self._ensure().set()

    async def wait(self) -> None:
        # Bounded so a broken ordering fails loudly rather than hanging until
        # the eval's own time limit reports an opaque stall.
        with anyio.fail_after(60):
            await self._ensure().wait()


class _AcceptanceSpy:
    """The real native filter, signalling once it accepts a conversation."""

    def __init__(self, inner: Any, handoff: _Handoff) -> None:
        self._inner = inner
        self._handoff = handoff

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def accept(self, messages: Sequence[ChatMessage]) -> bool:
        verdict: bool = self._inner.accept(messages)
        if verdict:
            self._handoff.signal()
        return verdict


class _LateTitleModel:
    """One primary call plus the CLI's title call, ordered title-last.

    The two are told apart by the CLI's own declaration, through the same
    parser the agent uses: a primary request carries a system
    `<user_information>` block naming its conversation, a title request carries
    none. Nothing here classifies on title text or on tool presence.
    """

    def __init__(self, handoff: _Handoff) -> None:
        self._handoff = handoff
        self.conversation_ids: list[str] = []
        self.title_requests = 0

    async def __call__(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        conversation_id = _native_conversation_id(input)
        if conversation_id is None:
            self.title_requests += 1
            await self._handoff.wait()
            # Tool-free request, answered with text. Returning a tool call here
            # would be traffic the CLI never asks this conversation for.
            return ModelOutput.from_content(model=_MODEL, content=_TITLE_TEXT)

        self.conversation_ids.append(conversation_id)
        return ModelOutput.from_message(
            ChatMessageAssistant(
                content=_LATE_TITLE_ANSWER, id=_MAIN_MESSAGE_ID, model=_MODEL
            )
        )


@skip_if_no_docker
@pytest.mark.slow
def test_a_late_title_does_not_displace_the_task_conversation() -> None:
    handoff = _Handoff()
    scripted = _LateTitleModel(handoff)
    real_conversation = agy_module._NativeConversation

    def spy(**kwargs: Any) -> Any:
        return _AcceptanceSpy(real_conversation(**kwargs), handoff)

    task = Task(
        dataset=[Sample(input="Answer in one line.", target=_LATE_TITLE_ANSWER)],
        solver=antigravity_cli(cwd=_FIXTURE_DIR, version=_CLI_VERSION),
        sandbox="docker",
    )

    with patch.object(agy_module, "_NativeConversation", spy):
        logs = eval(
            task,
            model=get_model(_MODEL, custom_outputs=scripted),
            limit=1,
            time_limit=300,
        )

    log = logs[0]
    assert log.status == "success", f"CLI run failed: {log.error}"
    assert log.samples
    sample = resolve_sample_attachments(log.samples[0], "full")

    # Both conversations really happened. Without the title request this test
    # proves nothing, so its absence is a failure rather than a skip.
    assert len(scripted.conversation_ids) == 1, scripted.conversation_ids
    assert scripted.title_requests >= 1

    # Excluding a conversation from canonical state does not suppress its
    # generation: both requests are still recorded as model events.
    assert len(_bridged_requests(sample)) >= 2

    # Canonical state is the task's conversation, by producer identity.
    assert sample.output.message.id == _MAIN_MESSAGE_ID
    assert sample.output.completion == _LATE_TITLE_ANSWER

    # ...and the title conversation is nowhere in it.
    assert all(_TITLE_TEXT not in message.text for message in sample.messages), [
        message.text for message in sample.messages
    ]


class _RecordLaunches:
    """The real sandbox, with every CLI launch recorded on its way through.

    Unlike `_CaptureLaunch` the launch is not short-circuited: the CLI really
    runs both attempts, so the recorded commands are the ones a scored retry
    actually issues and each result read back is the CLI's own.
    """

    def __init__(self, inner: SandboxEnvironment) -> None:
        self._inner = inner
        self.cmds: list[list[str]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def exec_remote(
        self,
        cmd: list[str],
        options: ExecRemoteAwaitableOptions | None = None,
        *,
        stream: Literal[False],
    ) -> ExecResult[str]:
        self.cmds.append(list(cmd))
        result = await self._inner.exec_remote(cmd=cmd, options=options, stream=stream)
        return result


@scorer(metrics=[])
def always_incorrect() -> Scorer:
    """Force the agent's retry path: no attempt is ever accepted."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=0, explanation="forcing a second attempt")

    return score


class _TwoAttemptModel:
    """Answer every primary request with text, and title requests likewise."""

    def __init__(self) -> None:
        self.conversation_ids: list[str] = []

    def __call__(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        conversation_id = _native_conversation_id(input)
        if conversation_id is None:
            return ModelOutput.from_content(model=_MODEL, content=_TITLE_TEXT)
        self.conversation_ids.append(conversation_id)
        return ModelOutput.from_content(model=_MODEL, content=_RETRY_ANSWER)


@skip_if_no_docker
@pytest.mark.slow
def test_a_scored_retry_resumes_the_bound_conversation() -> None:
    scripted = _TwoAttemptModel()
    recorders: list[_RecordLaunches] = []

    def record_sandbox(name: str | None = None) -> Any:
        recorder = _RecordLaunches(sandbox(name))
        recorders.append(recorder)
        return recorder

    task = Task(
        dataset=[Sample(input="Answer in one line.", target=_RETRY_ANSWER)],
        solver=antigravity_cli(attempts=2, cwd=_FIXTURE_DIR, version=_CLI_VERSION),
        scorer=always_incorrect(),
        sandbox="docker",
    )

    with patch.object(agy_module, "sandbox_env", record_sandbox):
        logs = eval(
            task,
            model=get_model(_MODEL, custom_outputs=scripted),
            limit=1,
            time_limit=600,
        )

    log = logs[0]
    assert log.status == "success", f"CLI run failed: {log.error}"
    assert recorders, "the agent never resolved a sandbox"

    cmds = recorders[0].cmds
    assert len(cmds) == 2, cmds
    first, retry = cmds
    assert scripted.conversation_ids, "no primary request reached the model"
    bound = scripted.conversation_ids[0]

    # The retry names the conversation the first attempt bound, explicitly.
    assert "--conversation" in retry, retry
    assert retry[retry.index("--conversation") + 1] == bound, (retry, bound)
    # Not `--continue`, not latest, not a fresh conversation: each of those
    # resumes something this run never verified it owns.
    assert "--continue" not in retry, retry
    assert "--continue" not in first, first
    # The first attempt cannot resume anything: the wrapper must not preassign
    # a UUID, because an unknown-UUID probe warns and creates a different one.
    assert "--conversation" not in first, first


_REENTRY_FOLLOW_UP = "And once more, in one line."


@solver
def _reenter_the_agent() -> Solver:
    """Invoke the agent twice within one sample, as a multi-call task does.

    The second invocation gets a fresh agent state object but the accumulated
    conversation, which is the only place the prior conversation's identity
    survives: the id lives in the system message the bridge tracked, and the
    agent reads that same `state.messages` to build its next prompt.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cli = antigravity_cli(cwd=_FIXTURE_DIR, version=_CLI_VERSION)
        agent_state = await run(cli, state.messages)
        agent_state.messages.append(ChatMessageUser(content=_REENTRY_FOLLOW_UP))
        agent_state = await run(cli, agent_state)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@skip_if_no_docker
@pytest.mark.slow
def test_reentering_the_agent_resumes_the_prior_conversation() -> None:
    """A second invocation continues the conversation the first one ran.

    `--continue` asks the CLI to pick a conversation for us. Naming the prior
    id instead keeps the choice with the run that verified it, and it is
    available without any new state: the id is in the canonical messages the
    first invocation tracked, which is the same input the agent already reads
    to decide there was a prior assistant response at all.
    """
    scripted = _TwoAttemptModel()
    recorders: list[_RecordLaunches] = []

    def record_sandbox(name: str | None = None) -> Any:
        recorder = _RecordLaunches(sandbox(name))
        recorders.append(recorder)
        return recorder

    task = Task(
        dataset=[Sample(input="Answer in one line.", target=_RETRY_ANSWER)],
        solver=_reenter_the_agent(),
        sandbox="docker",
    )

    with patch.object(agy_module, "sandbox_env", record_sandbox):
        logs = eval(
            task,
            model=get_model(_MODEL, custom_outputs=scripted),
            limit=1,
            time_limit=600,
        )

    log = logs[0]
    assert log.status == "success", f"CLI run failed: {log.error}"
    assert recorders, "the agent never resolved a sandbox"

    # One launch per invocation, each through its own sandbox resolution.
    cmds = [cmd for recorder in recorders for cmd in recorder.cmds]
    assert len(cmds) == 2, cmds
    first, second = cmds
    assert scripted.conversation_ids, "no primary request reached the model"
    prior = scripted.conversation_ids[0]

    # The second invocation names the conversation the first one ran...
    assert "--conversation" in second, second
    assert second[second.index("--conversation") + 1] == prior, (second, prior)
    # ...rather than handing the choice back to the CLI.
    assert "--continue" not in second, second
    # The first invocation has nothing to resume and must claim nothing.
    assert "--conversation" not in first, first
    assert "--continue" not in first, first


@solver
def _reenter_without_a_prior_conversation() -> Solver:
    """Hand the agent assistant turns that did not come from this CLI."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cli = antigravity_cli(cwd=_FIXTURE_DIR, version=_CLI_VERSION)
        transcript: list[ChatMessage] = [
            ChatMessageUser(content="What is 1+1?"),
            ChatMessageAssistant(content="2"),
            ChatMessageUser(content=_REENTRY_FOLLOW_UP),
        ]
        agent_state = await run(cli, transcript)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@skip_if_no_docker
@pytest.mark.slow
def test_reentry_without_a_prior_conversation_fails_loudly() -> None:
    """Neither guess nor start over when the prior identity is missing.

    A caller can hand the agent assistant turns this CLI never produced -- a
    synthesised or replayed transcript -- and then there is no native id to
    resume. Both silent options are wrong. `--continue` resumes whatever
    conversation the CLI happens to have cached, which may be unrelated to
    this run; starting fresh silently drops the context those turns carry,
    because the prompt the agent builds is only the user turns AFTER the last
    assistant message. So it fails, while the cause is still visible.
    """
    recorders: list[_RecordLaunches] = []

    def record_sandbox(name: str | None = None) -> Any:
        recorder = _RecordLaunches(sandbox(name))
        recorders.append(recorder)
        return recorder

    task = Task(
        dataset=[Sample(input="Answer in one line.", target=_RETRY_ANSWER)],
        solver=_reenter_without_a_prior_conversation(),
        sandbox="docker",
    )

    with patch.object(agy_module, "sandbox_env", record_sandbox):
        logs = eval(
            task,
            model=_MODEL,
            limit=1,
            time_limit=300,
        )

    log = logs[0]
    assert log.status == "error", f"expected a loud failure, got: {log.status}"
    assert "native conversation id" in str(log.error), log.error

    # And it failed BEFORE launching: a run that launched and then complained
    # would already have started a conversation nobody asked for.
    assert recorders, "the agent never resolved a sandbox"
    assert all(not recorder.cmds for recorder in recorders), [
        recorder.cmds for recorder in recorders
    ]


# --- centaur accumulation ---------------------------------------------------
#
# Centaur mode's contract is that the human may start or resume several
# conversations in one session, and the existing accumulation contract keeps
# all of them in canonical state. A centaur state therefore carries more than
# one native system block legitimately. That is not an ambiguous declaration
# to resolve: centaur never resumes by id, because its filtering is per
# request and every conversation in the session is the human's.

_CENTAUR_CID_A = "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
_CENTAUR_CID_B = "c9bf9e57-1685-4c89-bafb-ff5af830be8a"


def _native_system_message(conversation_id: str) -> ChatMessageSystem:
    """A primary request's system message, in the shape the bridge tracks."""
    return ChatMessageSystem(
        content="\n".join(
            [
                "<identity>You are a coding assistant.</identity>",
                "<user_information>",
                f"Conversation ID: {conversation_id}",
                "</user_information>",
            ]
        )
    )


@solver
def _resume_two_native_conversations() -> Solver:
    """Run the agent on a state that accumulated two native conversations."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cli = antigravity_cli(centaur=True, cwd=_FIXTURE_DIR, version=_CLI_VERSION)
        accumulated: list[ChatMessage] = [
            _native_system_message(_CENTAUR_CID_A),
            ChatMessageUser(content="What is 1+1?"),
            ChatMessageAssistant(content="2"),
            _native_system_message(_CENTAUR_CID_B),
            ChatMessageUser(content=_REENTRY_FOLLOW_UP),
        ]
        agent_state = await run(cli, accumulated)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@skip_if_no_docker
@pytest.mark.slow
def test_centaur_launches_on_an_accumulated_multi_conversation_state() -> None:
    """Two native conversations in a centaur state is normal, not ambiguous.

    Deriving one prior conversation id to resume makes sense only where a run
    resumes by id, which centaur does not. Reading an accumulated centaur state
    as a single declaration turns the documented accumulation contract into a
    launch failure before the human is ever handed a terminal.
    """
    handed: list[list[str]] = []

    async def capture_centaur(
        options: Any,
        agy_cmd: list[str],
        agent_env: dict[str, str],
        state: Any,
        *,
        user: str | None,
        sandbox: str | None,
        cwd: str,
    ) -> None:
        handed.append(list(agy_cmd))

    task = Task(
        dataset=[Sample(input="Answer in one line.", target=_RETRY_ANSWER)],
        solver=_resume_two_native_conversations(),
        sandbox="docker",
    )

    with patch.object(agy_module, "_run_antigravity_cli_centaur", capture_centaur):
        logs = eval(task, model=_MODEL, limit=1, time_limit=300)

    log = logs[0]
    assert log.status == "success", f"centaur run failed: {log.error}"
    assert handed, "the agent never dispatched to centaur"

    # ...and it resumes nothing by id: the human's own command decides.
    cmd = handed[0]
    assert "--conversation" not in cmd, cmd


# --- which sandbox the human is handed --------------------------------------
#
# A task can define several containers, and this agent takes a `sandbox` name
# to say which one is its own. Unattended, every step names it. A centaur
# session cannot: `human_cli` installs the task tools and offers the login
# through an unnamed lookup, so it lands in whichever container is the default
# -- and the human ends up in a terminal that is not where the agent installed
# the CLI, wrote its settings, or would have run.
#
# The compose file gives the run two identical containers precisely so that
# "which one" has a wrong answer to catch.

_TARGET_SANDBOX = "target"
_DECOY_SANDBOX = "default"
_TWO_SANDBOXES = str(Path(__file__).parent / "agy_two_sandboxes_compose.yaml")
_HUMAN_WAS_HERE = "/tmp/agy-centaur-session.marker"
_HUMAN_BASHRC = "/tmp/agy-centaur-session.bashrc"


class _RecordedHumanSession:
    """Stands in for the human's terminal, and looks around from inside it."""

    def __init__(self) -> None:
        self.handed: list[dict[str, Any]] = []
        self.cwd: list[str] = []

    def __call__(self, **kwargs: Any) -> Agent:
        self.handed.append(dict(kwargs))
        bashrc = str(kwargs["bashrc"])

        @agent
        def session() -> Agent:
            async def execute(state: AgentState) -> AgentState:
                # An unnamed lookup is what `human_cli` itself does to install
                # the task tools and to offer the login, so whichever container
                # answers here is the one the human would be working in.
                sbox = sandbox()
                await sbox.write_file(_HUMAN_WAS_HERE, "the human was here")

                # Where the human lands is a fact about running their shell,
                # not about the text of it: the file is sourced in the
                # container the session was given, from a directory that is
                # deliberately not the answer, and the shell is then asked
                # where it is. Reading the bashrc for a `cd` instead would pin
                # one spelling of the command and fail on every other.
                await sbox.write_file(_HUMAN_BASHRC, bashrc)
                landed = await sbox.exec(
                    ["bash", "-c", f". {_HUMAN_BASHRC}; pwd"], cwd="/"
                )
                self.cwd.append(landed.stdout.strip())
                return state

            return execute

        return session()


@solver
def _centaur_in_the_named_sandbox(session: _RecordedHumanSession) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cli = antigravity_cli(
            centaur=True,
            cwd=_FIXTURE_DIR,
            sandbox=_TARGET_SANDBOX,
            version=_CLI_VERSION,
        )
        with patch.object(centaur_module, "human_cli", session):
            await run(cli, [ChatMessageUser(content="Look around.")])

        # Read both containers while they are still up: which one was touched
        # is a fact to check rather than infer from the launch.
        for name in (_TARGET_SANDBOX, _DECOY_SANDBOX):
            visited = await sandbox(name).exec(["test", "-f", _HUMAN_WAS_HERE])
            installed = await sandbox(name).exec(
                ["sh", "-c", f"ls -d {SANDBOX_INSTALL_DIR}/agy-* 2>/dev/null"]
            )
            state.metadata[f"{name}_visited"] = visited.success
            state.metadata[f"{name}_installed"] = installed.success
        state.metadata["human_shell_cwd"] = session.cwd[-1] if session.cwd else ""
        return state

    return solve


@skip_if_no_docker
@pytest.mark.slow
def test_centaur_runs_the_human_session_in_the_named_sandbox() -> None:
    """The human works where the agent installed, not wherever is default."""
    session = _RecordedHumanSession()
    task = Task(
        dataset=[Sample(input="Look around.", target="nothing")],
        solver=_centaur_in_the_named_sandbox(session),
        sandbox=("docker", _TWO_SANDBOXES),
    )

    logs = eval(task, model=_MODEL, limit=1, time_limit=600)
    log = logs[0]
    assert log.status == "success", f"centaur run failed: {log.error}"
    assert log.samples
    found = log.samples[0].metadata

    # The named container is the one the agent installed the CLI into...
    assert found[f"{_TARGET_SANDBOX}_installed"] is True, found
    # ...and the one the human's own session reached without naming it.
    assert found[f"{_TARGET_SANDBOX}_visited"] is True, found
    # The decoy is untouched. Both halves matter: the install proves the agent
    # never strayed, and the visit proves the session followed it.
    assert found[f"{_DECOY_SANDBOX}_installed"] is False, found
    assert found[f"{_DECOY_SANDBOX}_visited"] is False, found

    # And the shell the human is handed starts where the agent's own commands
    # would have run: sourced in the selected container from `/`, it leaves the
    # shell in the resolved working directory. The alias alone is not enough --
    # `agy` resolves relative paths, reads its project state, and writes its
    # artifacts against the working directory.
    assert session.handed, "the human session was never created"
    assert found["human_shell_cwd"] == _FIXTURE_DIR, found
