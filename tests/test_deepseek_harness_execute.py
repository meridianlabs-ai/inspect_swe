"""`deepseek_harness()` driven end to end against a faked sandbox and bridge.

The sandbox plumbing is faked at the module level so the tests drive
`execute()` itself and see exactly what it writes into the sandbox, what it
runs there, and what it hands the bridge.
"""

import importlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from inspect_ai.agent import AgentBridgeContext, AgentState, BridgedToolsSpec
from inspect_ai.agent._bridge.context import (
    bridged_request_scope,
    current_agent_bridge_context,
)
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import (
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
    Tool,
    ToolDef,
    tool,
)
from inspect_swe._deepseek_harness._events.consumer import DshConsumer

from tests.dsh_shapes import (
    assistant,
    child_task,
    conversation,
    delegation_call,
    root_task,
)

# the package re-exports the agent function under the module's name
dsh_module = importlib.import_module("inspect_swe._deepseek_harness.deepseek_harness")

# embedded double quotes, blank lines, a leading dash, no trailing newline
PROMPT = '--help me write the opposite of the word "left".\n\nAnswer with one word'

SESSION_EVENT = json.dumps({"type": "session", "sessionId": "session-1234"})


class FakeResult:
    def __init__(
        self, success: bool = True, stdout: str = "", stderr: str = ""
    ) -> None:
        self.success = success
        self.returncode = 0 if success else 1
        self.stdout = stdout
        self.stderr = stderr


class FakeBridge:
    port = 3201

    def __init__(self, state: AgentState, kwargs: dict[str, Any]) -> None:
        self.state = state
        self.kwargs = kwargs
        self.mcp_server_configs = [
            MCPServerConfigHTTP(
                name=spec.name,
                type="http",
                url=f"http://localhost:{self.port}/mcp/{spec.name}",
            )
            for spec in kwargs.get("bridged_tools") or []
        ]

    @property
    def consumer(self) -> DshConsumer:
        consumer = self.kwargs["model_event_sink"]
        assert isinstance(consumer, DshConsumer)
        return consumer


class FakeSandbox:
    """Records what the agent writes and runs; `on_run` plays dsh's part."""

    def __init__(self) -> None:
        self.exec_remote_calls: list[dict[str, Any]] = []
        self.execs: list[list[str]] = []
        self.written: dict[str, str] = {}
        self.bridge: FakeBridge | None = None
        self.on_run: Callable[[FakeBridge], FakeResult | None] | None = None

    async def exec(self, cmd: list[str], **kwargs: Any) -> FakeResult:
        self.execs.append(cmd)
        if cmd[:2] == ["sh", "-c"] and "HOME" in cmd[2]:
            return FakeResult(stdout="/root\n")
        if cmd[:2] == ["sh", "-c"] and "head -n 1" in cmd[2]:
            return FakeResult(stdout=SESSION_EVENT + "\n")
        return FakeResult()

    async def write_file(self, path: str, contents: str) -> None:
        self.written[path] = contents

    async def exec_remote(
        self, cmd: list[str], options: Any, stream: bool
    ) -> FakeResult:
        self.exec_remote_calls.append({"cmd": cmd, "options": options})
        assert self.bridge is not None
        result = self.on_run(self.bridge) if self.on_run is not None else None
        return result or FakeResult()


class FakeStore:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


class Run:
    """One or more invocations of the agent against the same sandbox."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sbox = FakeSandbox()
        self.store = FakeStore()
        self.bridges: list[FakeBridge] = []
        sbox = self.sbox

        @asynccontextmanager
        async def fake_bridge(
            state: AgentState, **kwargs: Any
        ) -> AsyncIterator[FakeBridge]:
            bridge = FakeBridge(state, kwargs)
            self.bridges.append(bridge)
            sbox.bridge = bridge
            yield bridge

        async def fake_cwd(*_: Any) -> str:
            return "/root"

        async def fake_install(*_: Any, **__: Any) -> str:
            return "/opt/dsh/dsh"

        async def fake_wait(*_: Any, **__: Any) -> None:
            return None

        monkeypatch.setattr(dsh_module, "sandbox_env", lambda *_: sbox)
        monkeypatch.setattr(dsh_module, "sandbox_agent_bridge", fake_bridge)
        monkeypatch.setattr(dsh_module, "resolve_agent_cwd", fake_cwd)
        monkeypatch.setattr(dsh_module, "ensure_agent_binary_installed", fake_install)
        monkeypatch.setattr(dsh_module, "wait_for_mcp_endpoints", fake_wait)
        monkeypatch.setattr(dsh_module, "store", lambda: self.store)

    def __call__(self, messages: list[ChatMessage], **kwargs: Any) -> AgentState:
        agent = dsh_module.deepseek_harness(
            model="mockllm/model", context_window=100_000, **kwargs
        )
        state: AgentState = anyio.run(agent, AgentState(messages=messages))
        return state

    @property
    def dsh_argv(self) -> list[str]:
        """Dsh's own argv on the last launch (after the `bash -c ... bash` wrapper)."""
        cmd = self.sbox.exec_remote_calls[-1]["cmd"]
        assert cmd[:2] == ["bash", "-c"] and cmd[3] == "bash"
        return list(cmd[4:])

    @property
    def patch(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = json.loads(
            self.sbox.written["/root/.dsh-inspect/inspect.patch.yml"]
        )
        return rows


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> Run:
    init_transcript(Transcript())
    return Run(monkeypatch)


def _generation(
    consumer: DshConsumer,
    input: list[ChatMessage],
    reply: ChatMessageAssistant,
    stop_reason: str = "stop",
) -> ModelOutput:
    """A generation the bridge would have run for dsh while it executed."""
    output = ModelOutput.from_message(reply)
    output.choices[0].stop_reason = stop_reason  # type: ignore[assignment]
    event = ModelEvent(
        model="mockllm/model",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=output,
    )
    consumer.classify(get_model("mockllm/model"), input, [])
    event.span_id = consumer.on_start(input)
    consumer.on_pending(event)
    consumer.on_complete(event)
    return output


# ---------------------------------------------------------------------------
# launch: prompt, argv, resume
# ---------------------------------------------------------------------------


def test_task_reaches_dsh_byte_for_byte(run: Run) -> None:
    """The bridge anchors the main thread on the task input, so nothing may alter it."""
    run([ChatMessageUser(content=PROMPT)])

    assert run.sbox.written["/root/.dsh-inspect/prompt.txt"] == PROMPT
    argv = run.dsh_argv
    assert argv == [
        "/opt/dsh/dsh",
        "--profile",
        "headless",
        "--patch",
        "/root/.dsh-inspect/inspect.patch.yml",
        "--json",
    ]
    wrapper = run.sbox.exec_remote_calls[-1]["cmd"][2]
    assert "< /root/.dsh-inspect/prompt.txt" in wrapper
    assert not any(PROMPT in arg for arg in run.sbox.exec_remote_calls[-1]["cmd"])


def test_system_messages_are_sent_once_and_the_session_is_resumed(run: Run) -> None:
    system = ChatMessageSystem(content="Be terse.")
    run([system, ChatMessageUser(content=PROMPT)], system_prompt="Use tabs.")
    assert run.sbox.written["/root/.dsh-inspect/prompt.txt"] == (
        f"Be terse.\n\nUse tabs.\n\n{PROMPT}"
    )
    assert "--session-id" not in run.dsh_argv

    # a later turn on the same sample continues dsh's session, which already
    # holds the system text: sending it again would double it
    run(
        [
            system,
            ChatMessageUser(content=PROMPT),
            ChatMessageAssistant(content="right"),
            ChatMessageUser(content="And of up?"),
        ],
        system_prompt="Use tabs.",
    )
    assert run.sbox.written["/root/.dsh-inspect/prompt.txt"] == "And of up?"
    assert run.dsh_argv[-2:] == ["--session-id", "session-1234"]


# ---------------------------------------------------------------------------
# credential boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server",
    [
        MCPServerConfigHTTP(
            name="secrets",
            type="http",
            url="https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer sk-live-secret"},
        ),
        MCPServerConfigHTTP(
            name="secrets",
            type="http",
            url="https://user:sk-live-secret@mcp.example.com/mcp",
        ),
        MCPServerConfigHTTP(
            name="secrets", type="http", url="https://sk-live-secret@mcp.example.com/"
        ),
    ],
    ids=["headers", "url-user-and-password", "url-user"],
)
def test_mcp_credentials_never_enter_the_sandbox(server: MCPServerConfigHTTP) -> None:
    """Dsh's bash tool can read anything dsh can, including /proc/<pid>/fd.

    A credential given to dsh is therefore one the model can print, however
    it is delivered: the agent refuses it instead, before anything runs.
    """
    with pytest.raises(ValueError, match="does not accept MCP servers with headers"):
        dsh_module.deepseek_harness(mcp_servers=[server])


def test_nothing_placed_in_the_sandbox_carries_a_credential(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the host's real provider keys must not follow dsh into the sandbox
    for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "sk-live-host-secret")

    @tool
    def lookup() -> Tool:
        async def execute(query: str) -> str:
            """Look something up.

            Args:
                query: What to look up.
            """
            return query

        return execute

    run(
        [ChatMessageUser(content=PROMPT)],
        mcp_servers=[
            MCPServerConfigHTTP(name="docs", type="http", url="http://docs:8080/mcp"),
            MCPServerConfigStdio(name="memory", command="memory-server"),
        ],
        bridged_tools=[lookup()],
    )

    call = run.sbox.exec_remote_calls[-1]
    env = call["options"].env
    # only the bridge's dummy key
    assert {k: v for k, v in env.items() if k.endswith("_API_KEY")} == {
        "DEEPSEEK_API_KEY": "sk-inspect-bridge"
    }
    assert env["DEEPSEEK_BASE_URL"] == "http://localhost:3201"

    placed = json.dumps(
        {
            "env": env,
            "argv": call["cmd"],
            "files": run.sbox.written,
            "execs": run.sbox.execs,
        }
    )
    assert "sk-live" not in placed
    # nor is the profile patch smuggled in through a descriptor: one dsh
    # inherits stays readable at /proc/<pid>/fd for as long as dsh runs
    assert "/dev/fd" not in placed

    # the profile patch is an ordinary file: it has nothing to hide
    servers = {
        row["config"]["serverName"]: row["config"]
        for row in next(row["insert"] for row in run.patch if "insert" in row)
    }
    assert set(servers) == {"docs", "memory", "tools"}
    assert all("headers" not in config for config in servers.values())
    assert servers["tools"]["url"] == "http://localhost:3201/mcp/tools"


# ---------------------------------------------------------------------------
# bridge wiring
# ---------------------------------------------------------------------------


def test_plain_tools_become_one_bridged_server(run: Run) -> None:
    @tool
    def lookup() -> Tool:
        async def execute(query: str) -> str:
            """Look something up.

            Args:
                query: What to look up.
            """
            return query

        return execute

    spec = BridgedToolsSpec(name="extra", tools=[lookup()])
    run([ChatMessageUser(content=PROMPT)], bridged_tools=[lookup(), spec])

    specs = run.bridges[-1].kwargs["bridged_tools"]
    assert [s.name for s in specs] == ["extra", "tools"]
    assert [ToolDef(t).name for t in specs[1].tools] == ["lookup"]

    with pytest.raises(TypeError, match="Tool or BridgedToolsSpec"):
        dsh_module.deepseek_harness(bridged_tools=["lookup"])
    with pytest.raises(ValueError, match="server names"):
        dsh_module.deepseek_harness(
            bridged_tools=[BridgedToolsSpec(name="my.tools", tools=[lookup()])]
        )


def test_bridge_is_given_the_classifying_filter_and_the_sink(run: Run) -> None:
    """The filter classifies even when the caller passed none."""
    run([ChatMessageUser(content=PROMPT)])
    kwargs = run.bridges[-1].kwargs
    assert isinstance(kwargs["model_event_sink"], DshConsumer)
    assert kwargs["retry_refusals"] is None
    # every model name dsh sends reaches the agent's model: its own, and the
    # one the request behind dsh's web_search tool names
    aliases = kwargs["model_aliases"]
    assert len(aliases) == 2
    served = aliases["deepseek-v4-flash"]
    assert all(model is served for model in aliases.values())

    async def stamp(messages: list[ChatMessage]) -> AgentBridgeContext | None:
        with bridged_request_scope("mockllm/model"):
            await kwargs["filter"](
                get_model("mockllm/model"), messages, [], None, GenerateConfig()
            )
            return current_agent_bridge_context()

    assert anyio.run(stamp, conversation(root_task(PROMPT))) == AgentBridgeContext(
        "root"
    )
    assert anyio.run(
        stamp, conversation(child_task("do it", "agent-1"))
    ) == AgentBridgeContext("subagent", subagent_id="agent-1")


# ---------------------------------------------------------------------------
# exits
# ---------------------------------------------------------------------------


def test_refusal_exit_is_scored_not_raised(run: Run) -> None:
    """Dsh exits 1 after a refused turn the bridge has already recorded."""
    refusal: list[ModelOutput] = []

    def dsh(bridge: FakeBridge) -> FakeResult:
        refusal.append(
            _generation(
                bridge.consumer,
                conversation(root_task(PROMPT)),
                assistant("I can't help with that."),
                stop_reason="content_filter",
            )
        )
        return FakeResult(success=False, stderr="dsh: turn ended: refusal")

    run.sbox.on_run = dsh
    state = run([ChatMessageUser(content=PROMPT)])
    assert state.output is refusal[0]
    assert state.output.stop_reason == "content_filter"


def test_failed_exit_raises_and_closes_open_spans(run: Run) -> None:
    def dsh(bridge: FakeBridge) -> FakeResult:
        _generation(
            bridge.consumer,
            conversation(root_task(PROMPT)),
            assistant("", [delegation_call("c1", "do it", "alpha")]),
        )
        return FakeResult(success=False, stderr="dsh: TRANSPORT: connection reset")

    run.sbox.on_run = dsh
    with pytest.raises(RuntimeError, match="connection reset"):
        run([ChatMessageUser(content=PROMPT)])

    events = transcript().events
    assert [e.id for e in events if isinstance(e, SpanBeginEvent)] == ["agent-c1"]
    assert [e.id for e in events if isinstance(e, SpanEndEvent)] == ["agent-c1"]


def test_a_childs_refusal_does_not_excuse_a_failed_exit(run: Run) -> None:
    def dsh(bridge: FakeBridge) -> FakeResult:
        consumer = bridge.consumer
        _generation(
            consumer,
            conversation(root_task(PROMPT)),
            assistant("", [delegation_call("c1", "do it", "alpha")]),
        )
        _generation(
            consumer,
            conversation(child_task("do it", "alpha")),
            assistant("No."),
            stop_reason="content_filter",
        )
        return FakeResult(success=False, stderr="dsh: crashed")

    run.sbox.on_run = dsh
    with pytest.raises(RuntimeError, match="crashed"):
        run([ChatMessageUser(content=PROMPT)])


# ---------------------------------------------------------------------------
# the sample's answer is the root agent's
# ---------------------------------------------------------------------------


def test_root_answer_replaces_a_tracked_child_thread(run: Run) -> None:
    """The bridge can be left tracking a sub-agent's thread (see test_..._trajectory)."""
    answers: dict[str, ModelOutput] = {}

    def dsh(bridge: FakeBridge) -> None:
        consumer = bridge.consumer
        root = conversation(root_task(PROMPT))
        start = assistant("", [delegation_call("c1", "do it", "alpha")])
        _generation(consumer, root, start)
        child = conversation(child_task("do it", "alpha"))
        answers["child"] = _generation(consumer, child, assistant("child answer"))
        answers["root"] = _generation(
            consumer, [*root, start], assistant("root answer")
        )
        # what the bridge's own tracking would have reported
        bridge.state.messages = [*child, answers["child"].message]
        bridge.state.output = answers["child"]

    run.sbox.on_run = dsh
    state = run([ChatMessageUser(content=PROMPT)])
    assert state.output is answers["root"]
    assert state.messages[-1].text == "root answer"
    assert state.messages[1].text.startswith(PROMPT)
