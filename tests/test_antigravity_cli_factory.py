"""Focused contracts for the Antigravity CLI factory."""

import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Literal, cast, overload
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import ExecResult, SandboxEnvironment
from inspect_ai.util._sandbox import (
    ExecRemoteAwaitableOptions,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    SandboxEnvironmentConfigType,
)
from inspect_swe._util.centaur import CentaurOptions, CentaurSession, CommandsFilter

_CID = "eccac0fd-d2b5-4b39-9888-175170faece0"
_OTHER_CID = "16fd2706-8baf-433b-82eb-8c7fada847da"
_NATIVE_RESULT = (
    '{"conversation_id":"eccac0fd-d2b5-4b39-9888-175170faece0",'
    '"status":"SUCCESS","response":"tool call for tool run_command\\n'
    'FINAL_NATIVE_STORE_JSON\\n","duration_seconds":4.712014882,"num_turns":1,'
    '"usage":{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,'
    '"cache_read_tokens":0,"total_tokens":0}}'
)


class _Sandbox(SandboxEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.files: dict[str, str | bytes] = {}
        self.remote_calls: list[tuple[list[str], ExecRemoteAwaitableOptions, bool]] = []
        self.state_filter: Callable[[Sequence[ChatMessage]], bool] | None = None

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
        if cmd == ["sh", "-c", "echo $HOME"]:
            return ExecResult(
                success=True, returncode=0, stdout="/home/agent\n", stderr=""
            )
        assert cmd == [
            "mkdir",
            "-p",
            "/home/agent/.gemini/antigravity-cli",
            # The onboarding cache. Provisioning creates it up front because the CLI
            # writes onboarding.json into it on first launch, and a missing parent is
            # what stalled a real hosted Drive session at the onboarding prompt.
            "/home/agent/.gemini/antigravity-cli/cache",
            "/home/agent/.gemini/config",
        ]
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    async def write_file(self, file: str, contents: str | bytes) -> None:
        self.files[file] = contents

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        contents = self.files[file]
        if text:
            return contents if isinstance(contents, str) else contents.decode()
        return contents if isinstance(contents, bytes) else contents.encode()

    @overload
    async def exec_remote(
        self,
        cmd: list[str],
        options: ExecRemoteStreamingOptions | None = None,
        *,
        stream: Literal[True] = True,
    ) -> ExecRemoteProcess: ...

    @overload
    async def exec_remote(
        self,
        cmd: list[str],
        options: ExecRemoteAwaitableOptions | None = None,
        *,
        stream: Literal[False],
    ) -> ExecResult[str]: ...

    async def exec_remote(
        self,
        cmd: list[str],
        options: ExecRemoteStreamingOptions | ExecRemoteAwaitableOptions | None = None,
        *,
        stream: bool = True,
    ) -> ExecRemoteProcess | ExecResult[str]:
        assert stream is False
        assert isinstance(options, ExecRemoteAwaitableOptions)
        self.remote_calls.append((cmd, options, stream))
        assert self.state_filter is not None
        assert self.state_filter(_primary())
        return ExecResult(success=True, returncode=0, stdout=_NATIVE_RESULT, stderr="")

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        raise AssertionError("factory contract double must not clean a sample")


def _primary(cid: str = _CID) -> list[ChatMessage]:
    return [
        ChatMessageSystem(
            content=(f"<user_information>\nConversation ID: {cid}\n</user_information>")
        ),
        ChatMessageUser(content="Write files."),
    ]


def _auxiliary() -> list[ChatMessage]:
    return [
        ChatMessageSystem(content="You are a conversation title generator"),
        ChatMessageUser(content="Write files."),
    ]


class _Store:
    def get(self, key: str, default: int) -> int:
        assert key == "antigravity_cli_model_port"
        return default

    def set(self, key: str, value: int) -> None:
        assert key == "antigravity_cli_model_port"
        assert value == 3001


def test_unattended_factory_passes_all_bridge_contracts_and_verifies_result() -> None:
    module = importlib.import_module("inspect_swe._antigravity_cli.antigravity_cli")
    state = AgentState(messages=[])
    sbox = _Sandbox()
    bridge_options: dict[str, object] = {}
    resolver = MagicMock()
    request_filter = MagicMock()

    @asynccontextmanager
    async def bridge_context(
        *_args: object, **kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        bridge_options.update(kwargs)
        sbox.state_filter = cast(
            Callable[[Sequence[ChatMessage]], bool], kwargs["state_filter"]
        )
        yield SimpleNamespace(
            port=8901,
            mcp_server_configs=[],
            bridged_tools={},
            state=state,
        )

    with (
        patch.object(module, "sandbox_env", return_value=sbox),
        patch.object(module, "store", return_value=_Store()),
        patch.object(module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")),
        patch.object(
            module, "ensure_agent_binary_installed", AsyncMock(return_value="/opt/agy")
        ),
        patch.object(module, "sandbox_agent_bridge", bridge_context),
        patch.object(module, "build_user_prompt", return_value=("write files", False)),
    ):
        assert (
            asyncio.run(
                module.antigravity_cli(
                    version="1.1.27",
                    model_resolver=resolver,
                    filter=request_filter,
                    accumulate_conversations=True,
                )(state)
            )
            is state
        )

    assert bridge_options["model_resolver"] is resolver
    assert bridge_options["filter"] is request_filter
    assert bridge_options["accumulate_conversations"] is True
    state_filter = cast(
        Callable[[Sequence[ChatMessage]], bool], bridge_options["state_filter"]
    )
    assert state_filter(_auxiliary()) is False
    assert state_filter(_primary(_OTHER_CID)) is False

    assert len(sbox.remote_calls) == 1
    command, options, stream = sbox.remote_calls[0]
    assert command == [
        "bash",
        "-c",
        'exec 0</dev/null; "$@"',
        "bash",
        "/opt/agy",
        "--model",
        "gemini-3.6-flash",
        "--effort",
        "low",
        "--disable-slash-commands",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
        "--print",
        "write files",
    ]
    assert options.cwd == "/workspace"
    assert options.user is None
    assert options.concurrency is False
    assert options.env is not None
    assert "PATH" not in options.env
    assert stream is False


def test_centaur_factory_preserves_session_and_scopes_the_named_sandbox() -> None:
    module = importlib.import_module("inspect_swe._antigravity_cli.antigravity_cli")
    state = AgentState(messages=[*_primary(), *_primary(_OTHER_CID)])
    sbox = _Sandbox()
    bridge_options: dict[str, object] = {}
    handed: dict[str, object] = {}
    commands_filter = MagicMock()
    endpoint = MCPServerConfigHTTP(
        type="http", name="inspect-tools", url="http://localhost:3001/mcp"
    )
    bridge = SimpleNamespace(
        port=8901,
        mcp_server_configs=[endpoint],
        bridged_tools={"inspect-tools": ["echo"]},
        state=state,
    )

    @asynccontextmanager
    async def bridge_context(
        *_args: object, **kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        bridge_options.update(kwargs)
        yield bridge

    async def capture_centaur(
        options: CentaurOptions,
        agy_cmd: list[str],
        agent_env: dict[str, str],
        session: CentaurSession,
        commands_filter: CommandsFilter | None = None,
    ) -> AgentState:
        handed.update(
            options=options,
            command=agy_cmd,
            environment=agent_env,
            session=session,
            commands_filter=commands_filter,
        )
        state_filter = cast(
            Callable[[Sequence[ChatMessage]], bool], bridge_options["state_filter"]
        )
        assert state_filter(_primary()) is True
        assert state_filter(_primary(_OTHER_CID)) is True
        assert state_filter(_auxiliary()) is False
        return session.state

    wait_for_mcp_endpoints = AsyncMock()
    options = CentaurOptions()
    with (
        patch.object(module, "sandbox_env", return_value=sbox) as sandbox_env,
        patch.object(module, "store", return_value=_Store()),
        patch.object(module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")),
        patch.object(
            module, "ensure_agent_binary_installed", AsyncMock(return_value="/opt/agy")
        ),
        patch.object(module, "sandbox_agent_bridge", bridge_context),
        patch.object(module, "build_user_prompt", return_value=("write files", False)),
        patch.object(module, "wait_for_mcp_endpoints", wait_for_mcp_endpoints),
        patch.object(module, "_run_antigravity_cli_centaur", capture_centaur),
    ):
        assert (
            asyncio.run(
                module.antigravity_cli(
                    centaur=options,
                    version="1.1.27",
                    sandbox="target",
                    cwd="/workspace",
                    user="agent",
                    commands_filter=commands_filter,
                )(state)
            )
            is state
        )

    sandbox_env.assert_called_once_with("target")
    wait_for_mcp_endpoints.assert_awaited_once_with(
        [endpoint],
        bridge,
        sandbox="target",
        timeout=module.DEFAULT_MCP_READY_TIMEOUT,
        required=True,
    )
    session = cast(CentaurSession, handed["session"])
    assert session.state is state
    assert session.sandbox is sbox
    assert session.sandbox_name == "target"
    assert session.user == "agent"
    assert session.cwd == "/workspace"
    assert handed["commands_filter"] is commands_filter
    assert handed["command"] == [
        "/opt/agy",
        "--model",
        "gemini-3.6-flash",
        "--effort",
        "low",
    ]
    environment = cast(dict[str, str], handed["environment"])
    assert "PATH" not in environment
    assert "toolPermission" not in json.loads(
        next(
            contents
            for path, contents in sbox.files.items()
            if path.endswith("settings.json")
        )
    )


def test_unattended_reentry_resumes_only_the_seeded_native_conversation() -> None:
    module = importlib.import_module("inspect_swe._antigravity_cli.antigravity_cli")
    state = AgentState(
        messages=[
            *_primary(),
            ChatMessageAssistant(content="The first turn completed."),
        ]
    )
    sbox = _Sandbox()

    @asynccontextmanager
    async def bridge_context(
        *_args: object, **kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        sbox.state_filter = cast(
            Callable[[Sequence[ChatMessage]], bool], kwargs["state_filter"]
        )
        yield SimpleNamespace(
            port=8901,
            mcp_server_configs=[],
            bridged_tools={},
            state=state,
        )

    with (
        patch.object(module, "sandbox_env", return_value=sbox),
        patch.object(module, "store", return_value=_Store()),
        patch.object(module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")),
        patch.object(
            module, "ensure_agent_binary_installed", AsyncMock(return_value="/opt/agy")
        ),
        patch.object(module, "sandbox_agent_bridge", bridge_context),
        patch.object(module, "build_user_prompt", return_value=("retry", True)),
    ):
        assert asyncio.run(module.antigravity_cli(version="1.1.27")(state)) is state

    assert len(sbox.remote_calls) == 1
    command = sbox.remote_calls[0][0]
    assert command[command.index("--conversation") + 1] == _CID
    assert "--continue" not in command
