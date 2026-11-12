"""Agents inside an inspect_ai sandbox identity scope (inspect_swarm's users isolation).

`claude_code()` and `codex_cli()` run as the scope's user, write their
configuration under that user's passwd home and start in the scope's working
directory; their centaur mode and every other agent refuse to run in a scope.
A stub sandbox behind the sample's real proxy records each call and the user
it ran as.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple, overload

import pytest
from inspect_ai.agent import Agent, AgentState
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.model import ChatMessageUser
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_ai.util import ExecResult, SandboxEnvironment
from inspect_ai.util._sandbox._framework_operation import (
    in_framework_sandbox_operation,
)
from inspect_ai.util._sandbox.context import (
    sandbox_default_context_var,
    sandbox_environments_context_var,
)
from inspect_ai.util._sandbox.environment import SandboxEnvironmentConfigType
from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy
from inspect_ai.util._sandbox.identity import SandboxIdentityError, sandbox_identity
from inspect_swe import (
    antigravity,
    claude_code,
    codex_cli,
    gemini_cli,
    kimi_code,
    mini_swe_agent,
    opencode,
)
from inspect_swe._claude_code import claude_code as claude_code_module
from inspect_swe._codex_cli import codex_cli as codex_cli_module
from inspect_swe._util import agentbinary
from inspect_swe._util.mcp_ready import wait_for_mcp_endpoints
from inspect_swe.acp import agent as acp_agent_module
from inspect_swe.acp._agents.claude_code.claude_code import ClaudeCode
from inspect_swe.acp._agents.codex_cli.codex_cli import CodexCli
from inspect_swe.acp._agents.gemini_cli.gemini_cli import GeminiCli

MEMBER = "swarm-w-1"
MEMBER_HOME = "/scratch/w-1"
INHERITED_HOME = "/root"
"""The `HOME` the image sets, which a member must not use."""


class Exec(NamedTuple):
    cmd: list[str]
    user: str | None
    cwd: str | None
    env: dict[str, str] | None
    framework: bool
    """Whether the call was one of Inspect's own (framework) operations."""


class MemberSandbox(SandboxEnvironment):
    """Records every call; answers the few commands the agents read output from."""

    def __init__(self, workdir: str = "/") -> None:
        super().__init__()
        self.workdir = workdir
        self.execs: list[Exec] = []
        self.writes: list[tuple[str, str | None]] = []

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        self.execs.append(Exec(cmd, user, cwd, env, in_framework_sandbox_operation()))
        script = cmd[-1]
        if any("getent passwd" in arg for arg in cmd):
            stdout = f"{MEMBER_HOME}\n"
        elif script.startswith("which "):
            stdout = f"/usr/local/bin/{script.split()[-1]}\n"
        elif script == "pwd":
            stdout = f"{cwd or self.workdir}\n"
        elif script.startswith("cd ~"):
            stdout = f"{(env or {}).get('HOME', INHERITED_HOME)}\n"
        elif script.endswith("--version"):
            stdout = "codex-cli 0.150.0\n"
        elif script.startswith("curl") or (cmd and cmd[0] == "curl"):
            stdout = '{"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "t"}]}}'
        else:
            stdout = ""
        return ExecResult(True, 0, stdout, "")

    async def write_file(
        self, file: str, contents: str | bytes, *, user: str | None = None
    ) -> None:
        self.writes.append((file, user))

    @overload
    async def read_file(
        self, file: str, text: Literal[True] = True, *, user: str | None = None
    ) -> str: ...

    @overload
    async def read_file(
        self, file: str, text: Literal[False], *, user: str | None = None
    ) -> bytes: ...

    async def read_file(
        self, file: str, text: bool = True, *, user: str | None = None
    ) -> str | bytes:
        raise FileNotFoundError(file)

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        pass

    def member_execs(self) -> list[Exec]:
        """Calls the agent made itself (not Inspect's own operations)."""
        return [e for e in self.execs if not e.framework]


class Launched(Exception):
    """Raised by the stub `exec_remote()` in place of launching the CLI."""

    def __init__(self, options: Any) -> None:
        self.options = options


class SampleSandbox(SandboxEnvironmentProxy):
    """The sample's proxy, with a launch that records its options and stops."""

    async def exec_remote(  # type: ignore[override]
        self, cmd: list[str], options: Any, **kwargs: Any
    ) -> Any:
        raise Launched(options)


@pytest.fixture(autouse=True)
def transcript() -> None:
    init_transcript(Transcript())


@contextmanager
def sample(workdir: str = "/") -> Iterator[MemberSandbox]:
    """Make a stub sandbox the sample's only environment, behind its proxy."""
    stub = MemberSandbox(workdir)
    envs = sandbox_environments_context_var.set({"default": SampleSandbox(stub)})
    default = sandbox_default_context_var.set("default")
    try:
        yield stub
    finally:
        sandbox_default_context_var.reset(default)
        sandbox_environments_context_var.reset(envs)


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the bridge and checkpointer of both agents; records bridge starts."""
    started: list[dict[str, Any]] = []

    @asynccontextmanager
    async def fake_checkpointer() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            attempt="first", track=lambda name, getter, default: default
        )

    @asynccontextmanager
    async def fake_bridge(
        state: AgentState, **kwargs: Any
    ) -> AsyncIterator[SimpleNamespace]:
        started.append(kwargs)
        yield SimpleNamespace(
            state=state, port=kwargs["port"], mcp_server_configs=[], bridged_tools={}
        )

    for module in (claude_code_module, codex_cli_module):
        monkeypatch.setattr(module, "checkpointer", fake_checkpointer)
        monkeypatch.setattr(module, "sandbox_agent_bridge", fake_bridge)

    async def codex_model(*args: Any) -> str:
        return "gpt-5"

    monkeypatch.setattr(codex_cli_module, "_resolve_codex_model_checked", codex_model)
    monkeypatch.setattr(agentbinary, "trace", lambda message: None)
    return started


def _state() -> AgentState:
    return AgentState(messages=[ChatMessageUser(content="go")])


async def _launch(agent: Agent) -> Any:
    with pytest.raises(Launched) as launched:
        await agent(_state())
    return launched.value.options


@pytest.mark.parametrize("scope_cwd", [MEMBER_HOME, None])
async def test_claude_code_runs_as_the_scope_user(
    bridge: list[dict[str, Any]], scope_cwd: str | None
) -> None:
    # without a scope cwd, the sandbox's '/' working directory falls back to
    # the member's passwd home, not the image's HOME
    agent = claude_code(model="mockllm/model", version="auto")
    with sample() as stub, sandbox_identity(MEMBER, cwd=scope_cwd):
        options = await _launch(agent)

    assert (options.user, options.cwd) == (MEMBER, MEMBER_HOME)
    assert options.env["HOME"] == MEMBER_HOME
    [seed] = [e for e in stub.execs if ".claude/settings.json" in e.cmd[-1]]
    assert (seed.user, seed.cwd, seed.env) == (
        MEMBER,
        MEMBER_HOME,
        {"HOME": MEMBER_HOME},
    )
    assert {e.user for e in stub.member_execs()} == {MEMBER}


@pytest.mark.parametrize("scope_cwd", [MEMBER_HOME, None])
async def test_codex_cli_runs_as_the_scope_user(
    bridge: list[dict[str, Any]], scope_cwd: str | None
) -> None:
    agent = codex_cli(model="mockllm/model", version="auto", system_prompt="be brief")
    with sample() as stub, sandbox_identity(MEMBER, cwd=scope_cwd):
        options = await _launch(agent)

    codex_home = f"{MEMBER_HOME}/.codex"
    assert (options.user, options.cwd) == (MEMBER, MEMBER_HOME)
    assert options.env["HOME"] == MEMBER_HOME
    assert options.env["CODEX_HOME"] == codex_home
    # config files written without a user take the scope's user at the proxy
    assert stub.writes == [
        (f"{MEMBER_HOME}/AGENTS.md", MEMBER),
        (f"{codex_home}/config.toml", MEMBER),
    ]
    assert {e.user for e in stub.member_execs()} == {MEMBER}


async def test_home_dir_expands_against_the_member_home(
    bridge: list[dict[str, Any]],
) -> None:
    agent = codex_cli(model="mockllm/model", version="auto", home_dir="~/.codex")
    with sample() as stub, sandbox_identity(MEMBER, cwd=MEMBER_HOME):
        await _launch(agent)

    [expand] = [e for e in stub.execs if e.cmd[-1].startswith("eval echo")]
    assert (expand.user, expand.env) == (MEMBER, {"HOME": MEMBER_HOME})


@pytest.mark.parametrize("make_agent", [claude_code, codex_cli])
async def test_another_user_inside_a_scope_raises(
    bridge: list[dict[str, Any]], make_agent: Any
) -> None:
    agent = make_agent(model="mockllm/model", user="root")
    with sample() as stub, sandbox_identity(MEMBER, cwd=MEMBER_HOME):
        with pytest.raises(SandboxIdentityError, match="'root'"):
            await agent(_state())

    assert stub.member_execs() == []
    assert bridge == []


@pytest.mark.parametrize("make_agent", [claude_code, codex_cli])
async def test_centaur_inside_a_scope_raises_before_any_setup(
    bridge: list[dict[str, Any]], make_agent: Any
) -> None:
    agent = make_agent(model="mockllm/model", centaur=True)
    with sample() as stub, sandbox_identity(MEMBER, cwd=MEMBER_HOME):
        with pytest.raises(SandboxIdentityError, match=r"\(centaur=\.\.\.\)"):
            await agent(_state())

    assert stub.execs == []
    assert bridge == []


class _BridgeStarted(Exception):
    pass


@pytest.mark.parametrize(
    "module,make_agent",
    [(claude_code_module, claude_code), (codex_cli_module, codex_cli)],
)
async def test_centaur_outside_a_scope_starts_as_today(
    bridge: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    make_agent: Any,
) -> None:
    def fake_bridge(*args: Any, **kwargs: Any) -> Any:
        raise _BridgeStarted()

    monkeypatch.setattr(module, "sandbox_agent_bridge", fake_bridge)
    agent = make_agent(model="mockllm/model", centaur=True)
    with sample() as stub, pytest.raises(_BridgeStarted):
        await agent(_state())

    assert stub.execs == []


@pytest.mark.parametrize("make_agent", [claude_code, codex_cli])
async def test_outside_a_scope_nothing_changes(
    bridge: list[dict[str, Any]], make_agent: Any
) -> None:
    agent = make_agent(model="mockllm/model", version="auto")
    with sample() as stub:
        options = await _launch(agent)

    # no passwd lookup, no user, and the image's HOME
    assert options.user is None
    assert options.cwd == INHERITED_HOME
    assert "HOME" not in options.env
    assert all(e.user is None and not e.env for e in stub.execs)
    assert all(user is None for _, user in stub.writes)


async def test_readiness_probe_connects_as_the_member() -> None:
    # the probe names the member itself rather than relying on the sample
    # proxy's default, so it is given the provider object, which has none
    config = MCPServerConfigHTTP(
        type="http", name="srv", url="http://localhost:3001/mcp/srv"
    )
    bridge = SimpleNamespace(bridged_tools={"srv": {"t": object()}})
    with sample() as stub:
        with sandbox_identity(MEMBER, cwd=MEMBER_HOME):
            await wait_for_mcp_endpoints([config], bridge=bridge, sandbox=stub)  # type: ignore[arg-type]
        await wait_for_mcp_endpoints([config], bridge=bridge, sandbox=stub)  # type: ignore[arg-type]

    assert [(e.cmd[0], e.user) for e in stub.execs] == [
        ("curl", MEMBER),
        ("curl", None),
    ]


@pytest.mark.parametrize(
    "make_agent,name",
    [
        (gemini_cli, "gemini_cli()"),
        (opencode, "opencode()"),
        (kimi_code, "kimi_code()"),
        (mini_swe_agent, "mini_swe_agent()"),
        (antigravity, "antigravity()"),
    ],
)
async def test_other_agents_raise_inside_a_scope(make_agent: Any, name: str) -> None:
    agent = make_agent(model="mockllm/model")
    with sample() as stub, sandbox_identity(MEMBER, cwd=MEMBER_HOME):
        with pytest.raises(SandboxIdentityError) as error:
            await agent(_state())

    assert str(error.value).startswith(f"{name} does not support")
    assert "claude_code() and codex_cli()" in str(error.value)
    assert stub.execs == []


@pytest.mark.parametrize("acp_class", [ClaudeCode, CodexCli, GeminiCli])
async def test_acp_adapters_raise_inside_a_scope(
    monkeypatch: pytest.MonkeyPatch, acp_class: Any
) -> None:
    monkeypatch.setattr(acp_agent_module, "sample_active", lambda: object())
    with sample() as stub:
        agent = acp_class(model="mockllm/model")
        with sandbox_identity(MEMBER, cwd=MEMBER_HOME):
            with pytest.raises(SandboxIdentityError) as error:
                await agent(_state())

    assert acp_class.__name__ in str(error.value)
    assert "claude_code() and codex_cli()" in str(error.value)
    assert stub.execs == []


async def test_a_scope_on_another_sandbox_does_not_refuse() -> None:
    from inspect_swe._util.identity import refuse_sandbox_identity

    stub, other = MemberSandbox(), MemberSandbox()
    envs = sandbox_environments_context_var.set(
        {"main": SampleSandbox(stub), "other": SampleSandbox(other)}
    )
    default = sandbox_default_context_var.set("main")
    try:
        with sandbox_identity(MEMBER, sandbox="other"):
            refuse_sandbox_identity("gemini_cli()", "main")
            with pytest.raises(SandboxIdentityError):
                refuse_sandbox_identity("gemini_cli()", "other")
    finally:
        sandbox_default_context_var.reset(default)
        sandbox_environments_context_var.reset(envs)
