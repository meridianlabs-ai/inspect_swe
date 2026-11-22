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
mcp_module = importlib.import_module("inspect_swe._deepseek_harness.mcp")

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


@tool
def lookup() -> Tool:
    async def execute(query: str) -> str:
        """Look something up.

        Args:
            query: What to look up.
        """
        return query

    return execute


@tool
def purge() -> Tool:
    async def execute() -> str:
        """Delete everything."""
        return "gone"

    return execute


class FakeHostServer:
    """Stands in for inspect_ai's host-side MCP client (the `mcp` package)."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.created_with: dict[str, Any] = {}
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "FakeHostServer":
        self.entered += 1
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.exited += 1

    async def tools(self) -> list[Tool]:
        return [lookup(), purge()]


def _host_servers(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None = None
) -> dict[str, FakeHostServer]:
    """Replace the host-side client factories; servers created, by transport."""
    created: dict[str, FakeHostServer] = {}

    def factory(transport: str) -> Callable[..., FakeHostServer]:
        def create(**kwargs: Any) -> FakeHostServer:
            server = FakeHostServer(error)
            server.created_with = kwargs
            created[transport] = server
            return server

        return create

    monkeypatch.setattr(mcp_module, "mcp_server_http", factory("http"))
    monkeypatch.setattr(mcp_module, "mcp_server_sse", factory("sse"))
    return created


SECRET = "sk-live-secret-0123456789"

# every way a credential rides in an HTTP MCP server's configuration
CREDENTIALED_SERVERS = {
    "headers": MCPServerConfigHTTP(
        name="vault",
        type="http",
        url="https://mcp.example.com/mcp",
        headers={"Authorization": f"Bearer {SECRET}"},
    ),
    "url-user-and-password": MCPServerConfigHTTP(
        name="vault", type="http", url=f"https://user:{SECRET}@mcp.example.com/mcp"
    ),
    "url-query-token": MCPServerConfigHTTP(
        name="vault", type="http", url=f"https://mcp.example.com/mcp?api_key={SECRET}"
    ),
    "url-query-any-name": MCPServerConfigHTTP(
        name="vault", type="http", url=f"https://mcp.example.com/mcp?x={SECRET}"
    ),
}


def _placed_in_sandbox(run: Run) -> str:
    """Everything the agent wrote into, or ran in, the sandbox."""
    call = run.sbox.exec_remote_calls[-1]
    return json.dumps(
        {
            "env": call["options"].env,
            "argv": call["cmd"],
            "files": run.sbox.written,
            "execs": run.sbox.execs,
        }
    )


@pytest.mark.parametrize("form", sorted(CREDENTIALED_SERVERS))
def test_credentialed_http_server_is_connected_from_the_host(
    run: Run, monkeypatch: pytest.MonkeyPatch, form: str
) -> None:
    """Dsh's bash tool can read anything dsh can, including /proc/<pid>/fd.

    A credential handed to dsh is one the model can print, however it is
    delivered. So a server whose configuration carries one is never given to
    dsh: the host connects to it and dsh gets its tools through the bridge.
    """
    created = _host_servers(monkeypatch)
    server = CREDENTIALED_SERVERS[form]
    run([ChatMessageUser(content=PROMPT)], mcp_servers=[server])

    # the host-side client got the configuration as given
    host = created["http"]
    assert host.created_with == {
        "name": "vault",
        "url": server.url,
        "headers": server.headers,
    }
    # one connection for the run, closed after it
    assert (host.entered, host.exited) == (1, 1)

    # dsh sees a bridge endpoint with the server's name and tools
    [spec] = run.bridges[-1].kwargs["bridged_tools"]
    assert spec.name == "vault"
    assert [ToolDef(t).name for t in spec.tools] == ["lookup", "purge"]
    [row] = next(row["insert"] for row in run.patch if "insert" in row)
    assert row["config"] == {
        "serverName": "vault",
        "transport": "streamable-http",
        "url": "http://localhost:3201/mcp/vault",
        "failOnStartupError": True,
    }

    # and nothing of the server's own configuration is in the sandbox
    placed = _placed_in_sandbox(run)
    assert SECRET not in placed
    assert "mcp.example.com" not in placed
    assert "authorization" not in placed.lower()


def test_host_side_server_keeps_its_allowlist_and_transport(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect's client filters tools and speaks SSE, which dsh's own can't."""
    created = _host_servers(monkeypatch)
    run(
        [ChatMessageUser(content=PROMPT)],
        mcp_servers=[
            MCPServerConfigHTTP(
                name="vault",
                type="sse",
                url="https://mcp.example.com/sse",
                headers={"X-Api-Key": SECRET},
                tools=["look*"],
            )
        ],
    )
    assert list(created) == ["sse"]
    [spec] = run.bridges[-1].kwargs["bridged_tools"]
    assert [ToolDef(t).name for t in spec.tools] == ["lookup"]


def test_host_side_connection_failure_says_where_it_was_tried(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server only the sandbox can reach can't be connected from the host."""
    created = _host_servers(monkeypatch, error=ConnectionError("no route to host"))
    with pytest.raises(RuntimeError, match="from the Inspect host") as error:
        run(
            [ChatMessageUser(content=PROMPT)],
            mcp_servers=[CREDENTIALED_SERVERS["headers"]],
        )
    assert "no route to host" in str(error.value)
    assert "headers" in str(error.value)
    assert SECRET not in str(error.value)
    assert created["http"].entered == 1
    # nothing was launched
    assert run.sbox.exec_remote_calls == []


@pytest.mark.parametrize("form", ["url-user-and-password", "url-query-token"])
@pytest.mark.parametrize("grouped", [False, True], ids=["plain", "in-a-group"])
def test_host_side_connection_failure_keeps_url_credentials_out(
    run: Run, monkeypatch: pytest.MonkeyPatch, form: str, grouped: bool
) -> None:
    """An HTTP client's error names the URL it requested, credentials and all."""
    server = CREDENTIALED_SERVERS[form]
    cause: Exception = RuntimeError(
        f"Client error '401 Unauthorized' for url '{server.url}'"
    )
    if grouped:
        cause = ExceptionGroup("unhandled errors in a TaskGroup", [cause])
    _host_servers(monkeypatch, error=cause)
    with pytest.raises(RuntimeError, match="401 Unauthorized") as error:
        run([ChatMessageUser(content=PROMPT)], mcp_servers=[server])
    assert SECRET not in str(error.value)
    assert "mcp.example.com" in str(error.value)
    # and not in the traceback the eval log keeps, through the error's cause
    assert error.value.__cause__ is None and error.value.__suppress_context__


@pytest.mark.parametrize(
    "server",
    [
        MCPServerConfigStdio(
            name="github", command="gh-mcp", env={"GITHUB_TOKEN": SECRET}
        ),
        MCPServerConfigStdio(name="github", command="gh-mcp", env={"FOO": SECRET}),
        MCPServerConfigStdio(
            name="github", command="gh-mcp", args=["--api-key", SECRET]
        ),
        MCPServerConfigStdio(
            name="github", command="gh-mcp", args=[f"--token={SECRET}"]
        ),
        MCPServerConfigStdio(
            name="github", command="env", args=[f"GITHUB_TOKEN={SECRET}", "gh-mcp"]
        ),
        MCPServerConfigStdio(
            name="github",
            command="mcp-remote",
            args=["https://x/mcp", "--header", f"Authorization: Bearer {SECRET}"],
        ),
        MCPServerConfigStdio(name="github", command="gh-mcp", args=[SECRET]),
    ],
    ids=[
        "env-named-token",
        "env-key-shaped-value",
        "flag-then-value",
        "flag-with-value",
        "assignment-argument",
        "header-argument",
        "key-shaped-argument",
    ],
)
def test_stdio_server_credentials_are_refused(server: MCPServerConfigStdio) -> None:
    """A stdio server runs in the sandbox: its env and command line are readable there.

    There is nowhere in the sandbox to hide a credential for it, so a server
    configured with one is refused before anything is written.
    """
    with pytest.raises(ValueError, match="does not pass credentials") as error:
        dsh_module.deepseek_harness(mcp_servers=[server])
    assert SECRET not in str(error.value)


PLAIN_SECRET = "hunter2hunter2"


@pytest.mark.parametrize(
    ("server", "named"),
    [
        (
            MCPServerConfigStdio(
                name="db", command="pg-mcp", env={"PGPASSWORD": PLAIN_SECRET}
            ),
            "env PGPASSWORD",
        ),
        (
            MCPServerConfigStdio(
                name="db", command="mysql-mcp", env={"MYSQL_PWD": PLAIN_SECRET}
            ),
            "env MYSQL_PWD",
        ),
        (
            MCPServerConfigStdio(
                name="db", command="ngrok-mcp", env={"AUTHTOKEN": PLAIN_SECRET}
            ),
            "env AUTHTOKEN",
        ),
        (
            MCPServerConfigStdio(
                name="gh",
                command="docker",
                args=["run", f"--env=GITHUB_TOKEN={PLAIN_SECRET}", "gh-mcp"],
            ),
            "argument --env",
        ),
        (
            MCPServerConfigStdio(
                name="remote",
                command="mcp-remote",
                args=[f"https://mcp.example.com/mcp?api_key={PLAIN_SECRET}"],
            ),
            "an argument that looks like a credential",
        ),
        (
            MCPServerConfigStdio(
                name="remote",
                command="curl-mcp",
                args=[f"-HAuthorization: Bearer {PLAIN_SECRET}"],
            ),
            "argument -H",
        ),
        (
            MCPServerConfigStdio(
                name="remote", command="srv", args=[f"--token:{PLAIN_SECRET}"]
            ),
            "argument --token",
        ),
    ],
    ids=[
        "run-together-password",
        "pwd-suffix",
        "run-together-token",
        "assignment-in-a-flag",
        "url-query-argument",
        "short-flag-with-header",
        "flag-with-colon",
    ],
)
def test_stdio_credentials_under_other_names_and_forms(
    server: MCPServerConfigStdio, named: str
) -> None:
    """Values that aren't key-shaped, so found by name or form, never echoed."""
    with pytest.raises(ValueError, match="does not pass credentials") as error:
        dsh_module.deepseek_harness(mcp_servers=[server])
    assert f"({named})" in str(error.value)
    assert PLAIN_SECRET not in str(error.value)


def test_stdio_settings_that_only_look_like_credentials_pass() -> None:
    dsh_module.deepseek_harness(
        mcp_servers=[
            MCPServerConfigStdio(
                name="files",
                command="files-mcp",
                args=["--root", "/srv", "https://docs.example.com/mcp?project=acme"],
                env={"PWD": "/srv", "MAX_TOKENS": "1000", "AUTHOR": "me"},
            )
        ]
    )


def test_mcp_server_names_must_be_unique_wherever_they_are_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host_servers(monkeypatch)
    with pytest.raises(ValueError, match="must be unique.*: vault"):
        dsh_module.deepseek_harness(
            mcp_servers=[
                CREDENTIALED_SERVERS["headers"],
                MCPServerConfigStdio(name="vault", command="vault-mcp"),
            ]
        )
    with pytest.raises(ValueError, match="must be unique.*: tools"):
        dsh_module.deepseek_harness(
            mcp_servers=[MCPServerConfigStdio(name="tools", command="srv")],
            bridged_tools=[lookup()],
        )


def test_nothing_placed_in_the_sandbox_carries_a_credential(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the host's real provider keys must not follow dsh into the sandbox
    for name in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "sk-live-host-secret")

    run(
        [ChatMessageUser(content=PROMPT)],
        mcp_servers=[
            MCPServerConfigHTTP(name="docs", type="http", url="http://docs:8080/mcp"),
            MCPServerConfigStdio(
                name="memory",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                env={"MEMORY_FILE_PATH": "/tmp/memory.json", "MAX_TOKENS": "1000"},
            ),
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
