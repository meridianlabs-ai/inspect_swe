from __future__ import annotations

import importlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Protocol

import anyio
import pytest
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageUser
from inspect_ai.tool import MCPServerConfigStdio
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP

from inspect_swe._antigravity_cli.agentbinary import (
    ensure_antigravity_cli_setup,
    provision_antigravity_home,
)
from inspect_swe._antigravity_cli.interceptor import (
    MITMDUMP_PATH,
    MITMPROXY_ARCHIVE_PATH,
    MITMPROXY_INSTALL_DIR,
    ensure_mitmproxy,
)
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR


@dataclass(frozen=True, slots=True)
class ExecResult:
    success: bool
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class RecordedCommand:
    command: tuple[str, ...]
    user: str | None


class RecordingSandbox:
    def __init__(self) -> None:
        self.commands: list[RecordedCommand] = []
        self.executable_paths: set[str] = set()
        self.writes: list[tuple[str, bytes]] = []

    async def exec(self, cmd: list[str], user: str | None = None) -> ExecResult:
        self.commands.append(RecordedCommand(command=tuple(cmd), user=user))
        shell_command = cmd[-1]
        if shell_command.startswith("test -x "):
            path = shell_command.removeprefix("test -x ")
            return ExecResult(success=path in self.executable_paths)
        if shell_command.startswith("chmod 0755 "):
            self.executable_paths.add(shell_command.removeprefix("chmod 0755 "))
        return ExecResult(success=True)

    async def write_file(self, path: str, contents: bytes) -> None:
        self.writes.append((path, contents))


def test_ensure_antigravity_cli_setup_writes_an_executable_once(
    tmp_path: Path,
) -> None:
    async def provision() -> None:
        binary_source = tmp_path / "antigravity"
        binary_data = b"agy-binary"
        binary_source.write_bytes(binary_data)
        sandbox = RecordingSandbox()
        agy_path = f"{SANDBOX_INSTALL_DIR}/antigravity-cli/agy"

        result = await ensure_antigravity_cli_setup(
            sandbox,
            binary_source=str(binary_source),
            user="root",
        )

        assert result == agy_path
        assert sandbox.writes == [(agy_path, binary_data)]
        assert RecordedCommand(
            command=("bash", "-c", f"chmod 0755 {agy_path}"),
            user="root",
        ) in sandbox.commands

        result = await ensure_antigravity_cli_setup(
            sandbox,
            binary_source=str(binary_source),
            user="root",
        )

        assert result == agy_path
        assert sandbox.writes == [(agy_path, binary_data)]

    anyio.run(provision)


def test_provision_antigravity_home_archives_only_runtime_tokens(tmp_path: Path) -> None:
    async def provision() -> None:
        token_source = tmp_path / "antigravity-cli"
        token_source.mkdir()
        (token_source / "antigravity-oauth-token").write_text(
            "token", encoding="utf-8"
        )
        for excluded_directory in (
            "log",
            "cache",
            "crashes",
            "conversations",
            "scratch",
        ):
            directory = token_source / excluded_directory
            directory.mkdir()
            (directory / "ignored.txt").write_text("ignored", encoding="utf-8")

        sandbox = RecordingSandbox()
        sandbox_home = "/root"

        result = await provision_antigravity_home(
            sandbox,
            sandbox_home=sandbox_home,
            token_source=str(token_source),
            user="root",
        )

        archive_path, archive_data = sandbox.writes[0]
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
            archive_names = archive.getnames()

        assert result == "/root/.gemini/antigravity-cli"
        assert "antigravity-cli/antigravity-oauth-token" in archive_names
        assert not any(
            excluded_directory in Path(name).parts
            for name in archive_names
            for excluded_directory in (
                "log",
                "cache",
                "crashes",
                "conversations",
                "scratch",
            )
        )
        assert RecordedCommand(
            command=("bash", "-c", "mkdir -p /root/.gemini"),
            user="root",
        ) in sandbox.commands
        assert RecordedCommand(
            command=(
                "bash",
                "-c",
                f"tar -xzf {archive_path} -C /root/.gemini",
            ),
            user="root",
        ) in sandbox.commands

    anyio.run(provision)


def test_ensure_mitmproxy_extracts_flat_archive_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MitmproxySandbox(RecordingSandbox):
        async def exec(self, cmd: list[str], user: str | None = None) -> ExecResult:
            if cmd[-1].startswith("tar -xzf "):
                self.executable_paths.add(MITMDUMP_PATH)
            return await super().exec(cmd, user=user)

    async def download_file(_: str) -> bytes:
        return b"mitmproxy-archive"

    async def provision() -> None:
        monkeypatch.setattr(
            "inspect_swe._antigravity_cli.interceptor.download_file", download_file
        )
        sandbox = MitmproxySandbox()

        result = await ensure_mitmproxy(sandbox, "linux-x64", "root")

        assert result == MITMDUMP_PATH
        assert sandbox.writes == [(MITMPROXY_ARCHIVE_PATH, b"mitmproxy-archive")]
        assert RecordedCommand(
            command=(
                "bash",
                "-c",
                f"tar -xzf {MITMPROXY_ARCHIVE_PATH} -C {MITMPROXY_INSTALL_DIR} && rm -f {MITMPROXY_ARCHIVE_PATH}",
            ),
            user="root",
        ) in sandbox.commands
        assert not any(
            "--strip-components" in recorded.command[-1]
            for recorded in sandbox.commands
        )

        result = await ensure_mitmproxy(sandbox, "linux-x64", "root")

        assert result == MITMDUMP_PATH
        assert len(sandbox.writes) == 1
        assert sandbox.commands[-1] == RecordedCommand(
            command=("bash", "-c", f"test -x {MITMDUMP_PATH}"), user="root"
        )

    anyio.run(provision)


class _FlowResponse(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, str]


def test_addon_rewrites_only_aiplatform(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code: int = 200
        content: bytes = b"data: {}\n\n"
        headers: dict[str, str] = {"content-type": "text/event-stream"}

    class FakeResponse:
        status_code: int = 200
        content: bytes = b"data: {}\n\n"
        headers: dict[str, str] = {"content-type": "text/event-stream"}

    @dataclass(frozen=True, slots=True)
    class FakeRequest:
        pretty_host: str
        path: str
        content: bytes

    @dataclass(slots=True)
    class FakeFlow:
        request: FakeRequest
        response: _FlowResponse | None = None

    posted: list[tuple[str, bytes, dict[str, str]]] = []

    def fake_post(
        url: str, *, data: bytes, headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        posted.append((url, data, headers))
        return FakeResponse()

    async def exercise() -> None:
        monkeypatch.setenv("ANTIGRAVITY_BRIDGE_PORT", "24680")
        addon = importlib.reload(
            importlib.import_module("inspect_swe._antigravity_cli._addon")
        )
        monkeypatch.setattr(addon.requests, "post", fake_post)

        non_aiplatform = FakeFlow(
            request=FakeRequest(
                pretty_host="oauth2.googleapis.com",
                path="/token",
                content=b"{}",
            )
        )
        await addon.request(non_aiplatform)

        aiplatform = FakeFlow(
            request=FakeRequest(
                pretty_host="us-central1-aiplatform.googleapis.com",
                path=(
                    "/v1/projects/project/locations/us-central1/publishers/google/"
                    "models/gemini-2.5-flash:streamGenerateContent?alt=sse"
                ),
                content=b'{"contents": []}',
            )
        )
        await addon.request(aiplatform)

        assert non_aiplatform.response is None
        assert posted == [
            (
                "http://127.0.0.1:24680/v1beta/models/"
                "gemini-2.5-flash:streamGenerateContent?alt=sse",
                b'{"contents": []}',
                {"content-type": "application/json"},
            )
        ]
        assert aiplatform.response is not None
        assert aiplatform.response.status_code == 200
        assert aiplatform.response.content == b"data: {}\n\n"
        assert aiplatform.response.headers["content-type"] == "text/event-stream"

    anyio.run(exercise)


def test_addon_lowercases_vertex_schema_types_for_the_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status_code: int = 200
        content: bytes = b"data: {}\n\n"
        headers: dict[str, str] = {"content-type": "text/event-stream"}

    @dataclass(frozen=True, slots=True)
    class FakeRequest:
        pretty_host: str
        path: str
        content: bytes

    @dataclass(slots=True)  # noqa: MUTABLE_OK -- addon.request() mutates .response
    class FakeFlow:
        request: FakeRequest
        response: _FlowResponse | None = None

    posted: list[bytes] = []

    def fake_post(
        url: str, *, data: bytes, headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        posted.append(data)
        return FakeResponse()

    async def exercise() -> None:
        monkeypatch.setenv("ANTIGRAVITY_BRIDGE_PORT", "24680")
        addon = importlib.reload(
            importlib.import_module("inspect_swe._antigravity_cli._addon")
        )
        monkeypatch.setattr(addon.requests, "post", fake_post)

        vertex_body = {
            "contents": [{"type": "OBJECT"}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "browser",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {
                                    "action": {"type": "STRING"},
                                    "count": {
                                        "anyOf": [
                                            {"type": "NUMBER"},
                                            {"type": "INTEGER"},
                                        ]
                                    },
                                    "options": {
                                        "type": "ARRAY",
                                        "items": {"type": ["STRING", "NULL"]},
                                    },
                                    "enabled": {"allOf": [{"type": "BOOLEAN"}]},
                                    "target": {
                                        "oneOf": [
                                            {
                                                "type": "OBJECT",
                                                "properties": {"id": {"type": "STRING"}},
                                            }
                                        ]
                                    },
                                },
                            },
                        }
                    ]
                }
            ],
        }
        aiplatform = FakeFlow(
            request=FakeRequest(
                pretty_host="us-central1-aiplatform.googleapis.com",
                path=(
                    "/v1/projects/project/locations/us-central1/publishers/google/"
                    "models/gemini-2.5-flash:streamGenerateContent?alt=sse"
                ),
                content=json.dumps(vertex_body).encode(),
            )
        )

        await addon.request(aiplatform)

        forwarded = json.loads(posted[0])
        schema = forwarded["tools"][0]["functionDeclarations"][0]["parameters"]
        assert schema["type"] == "object"
        assert schema["properties"]["action"]["type"] == "string"
        assert schema["properties"]["count"]["anyOf"][0]["type"] == "number"
        assert schema["properties"]["count"]["anyOf"][1]["type"] == "integer"
        assert schema["properties"]["options"]["type"] == "array"
        assert schema["properties"]["options"]["items"]["type"] == ["string", "null"]
        assert schema["properties"]["enabled"]["allOf"][0]["type"] == "boolean"
        target_schema = schema["properties"]["target"]["oneOf"][0]
        assert target_schema["type"] == "object"
        assert target_schema["properties"]["id"]["type"] == "string"
        assert forwarded["contents"] == [{"type": "OBJECT"}]

    anyio.run(exercise)


def test_addon_relabels_model_role_function_response_to_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status_code: int = 200
        content: bytes = b"data: {}\n\n"
        headers: dict[str, str] = {"content-type": "text/event-stream"}

    @dataclass(frozen=True, slots=True)
    class FakeRequest:
        pretty_host: str
        path: str
        content: bytes

    @dataclass(slots=True)  # noqa: MUTABLE_OK -- addon.request() mutates .response
    class FakeFlow:
        request: FakeRequest
        response: _FlowResponse | None = None

    posted: list[bytes] = []

    def fake_post(
        url: str, *, data: bytes, headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        posted.append(data)
        return FakeResponse()

    async def exercise() -> None:
        monkeypatch.setenv("ANTIGRAVITY_BRIDGE_PORT", "24680")
        addon = importlib.reload(
            importlib.import_module("inspect_swe._antigravity_cli._addon")
        )
        monkeypatch.setattr(addon.requests, "post", fake_post)

        # agy sends functionResponse under role="model" instead of the
        # standard role="user" -- a pure tool-result content, plus one mixed
        # with a functionCall alongside a functionResponse.
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "list the schemas"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "call_mcp_tool",
                                "args": {"tool": "postgres_schema"},
                            }
                        }
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "call_mcp_tool",
                                "response": {"output": "ok"},
                            }
                        }
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "mixed turn"},
                        {
                            "functionResponse": {
                                "name": "call_mcp_tool",
                                "response": {"output": "mixed"},
                            }
                        },
                    ],
                },
            ],
        }
        aiplatform = FakeFlow(
            request=FakeRequest(
                pretty_host="us-central1-aiplatform.googleapis.com",
                path=(
                    "/v1/projects/project/locations/us-central1/publishers/google/"
                    "models/gemini-2.5-flash:streamGenerateContent?alt=sse"
                ),
                content=json.dumps(body).encode(),
            )
        )

        await addon.request(aiplatform)

        forwarded = json.loads(posted[0])
        contents = forwarded["contents"]

        assert contents[0] == {"role": "user", "parts": [{"text": "list the schemas"}]}
        assert contents[1]["role"] == "model"
        assert "functionCall" in contents[1]["parts"][0]

        # pure functionResponse content: relabeled in place, not split
        assert contents[2] == {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "call_mcp_tool",
                        "response": {"output": "ok"},
                    }
                }
            ],
        }

        # mixed content: split into a model turn (remaining parts) followed
        # by a user turn (the functionResponse)
        assert contents[3] == {"role": "model", "parts": [{"text": "mixed turn"}]}
        assert contents[4] == {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "call_mcp_tool",
                        "response": {"output": "mixed"},
                    }
                }
            ],
        }
        assert len(contents) == 5

    anyio.run(exercise)


def _antigravity_module() -> ModuleType:
    try:
        return importlib.import_module("inspect_swe._antigravity_cli.antigravity_cli")
    except ModuleNotFoundError:
        pytest.fail("the Antigravity CLI agent module has not been implemented")


def test_resolve_mcp_servers_stdio() -> None:
    agent_module = _antigravity_module()
    server = MCPServerConfigStdio(
        name="taiga-mcp",
        command="/opt/venv/bin/browser_injections",
        args=["mcp"],
    )

    result = agent_module.resolve_mcp_servers_antigravity([server])

    assert result == (
        '{\n'
        '  "mcpServers": {\n'
        '    "taiga-mcp": {\n'
        '      "command": "/opt/venv/bin/browser_injections",\n'
        '      "args": [\n'
        '        "mcp"\n'
        '      ]\n'
        '    }\n'
        '  }\n'
        '}'
    )


def test_resolve_mcp_servers_http_uses_server_url() -> None:
    agent_module = _antigravity_module()
    server = MCPServerConfigHTTP(
        name="remote", type="http", url="https://mcp.example.test"
    )

    result = agent_module.resolve_mcp_servers_antigravity([server])

    assert result == (
        '{\n'
        '  "mcpServers": {\n'
        '    "remote": {\n'
        '      "serverUrl": "https://mcp.example.test"\n'
        '    }\n'
        '  }\n'
        '}'
    )


def test_agent_env_no_base_url_dynamic_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exercise() -> None:
        agent_module = _antigravity_module()
        captured: dict[str, int | str | bool] = {}

        class FakeStore:
            def get(self, _: str, default: int) -> int:
                return default

            def set(self, _: str, value: int) -> None:
                captured["stored_bridge_port"] = value

        class FakeBridge:
            port = 24680

            def __init__(self, state: AgentState) -> None:
                self.state = state

            async def __aenter__(self) -> FakeBridge:
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

        class FakeProcess:
            async def kill(self) -> None:
                captured["proxy_killed"] = True

        class FakeMonitor:
            def cancel(self) -> None:
                captured["monitor_cancelled"] = True

        class AgentSandbox(RecordingSandbox):
            async def exec_remote(
                self, *, cmd: list[str], options: object, stream: bool
            ) -> ExecResult:
                captured["https_proxy"] = options.env["HTTPS_PROXY"]
                captured["has_base_url"] = "GOOGLE_GEMINI_BASE_URL" in options.env
                return ExecResult(success=True)

        sandbox = AgentSandbox()

        def fake_bridge(state: AgentState, **_: object) -> FakeBridge:
            return FakeBridge(state)

        async def fake_ensure_cli(*_: object, **__: object) -> str:
            return "/usr/local/bin/agy"

        async def fake_provision_home(*_: object, **__: object) -> str:
            return "/root/.gemini/antigravity-cli"

        async def fake_ensure_mitmproxy(*_: object, **__: object) -> str:
            return "/usr/local/bin/mitmdump"

        async def fake_detect_sandbox_platform(_: RecordingSandbox) -> str:
            return "linux-x64"

        async def fake_start_interceptor(
            _: RecordingSandbox,
            *,
            listen_port: int,
            bridge_port: int,
            confdir: str,
            user: str | None,
        ) -> tuple[FakeProcess, str, FakeMonitor]:
            captured["intercept_port"] = listen_port
            captured["bridge_port"] = bridge_port
            captured["confdir"] = confdir
            captured["interceptor_user"] = user or ""
            return FakeProcess(), "/root/mitmproxy-ca-cert.pem", FakeMonitor()

        monkeypatch.setattr(agent_module, "store", lambda: FakeStore())
        monkeypatch.setattr(agent_module, "sandbox_agent_bridge", fake_bridge)
        monkeypatch.setattr(agent_module, "sandbox_env", lambda _: sandbox)
        monkeypatch.setattr(
            agent_module, "detect_sandbox_platform", fake_detect_sandbox_platform
        )
        monkeypatch.setattr(agent_module, "ensure_antigravity_cli_setup", fake_ensure_cli)
        monkeypatch.setattr(agent_module, "provision_antigravity_home", fake_provision_home)
        monkeypatch.setattr(agent_module, "ensure_mitmproxy", fake_ensure_mitmproxy)
        monkeypatch.setattr(agent_module, "start_interceptor", fake_start_interceptor)
        monkeypatch.setattr(agent_module, "trace", lambda _: None)

        agent = agent_module.antigravity_cli()
        await agent(AgentState(messages=[ChatMessageUser(content="Solve the task")]))

        assert captured["stored_bridge_port"] == 3001
        assert captured["intercept_port"] == 8001
        assert captured["bridge_port"] == 24680
        assert captured["https_proxy"] == "http://127.0.0.1:8001"
        assert captured["has_base_url"] is False
        assert captured["monitor_cancelled"] is True
        assert captured["proxy_killed"] is True

    anyio.run(exercise)


def test_antigravity_cli_returns_an_agent() -> None:
    agent_module = _antigravity_module()

    result = agent_module.antigravity_cli()

    assert isinstance(result, Agent)


def test_start_interceptor_waits_for_listening_socket_before_returning() -> None:
    async def exercise() -> None:
        interceptor_module = importlib.import_module(
            "inspect_swe._antigravity_cli.interceptor"
        )

        class FakeProcess:
            def __aiter__(self) -> "FakeProcess":
                return self

            async def __anext__(self) -> object:
                await anyio.sleep_forever()
                raise StopAsyncIteration

        class PortProbeSandbox(RecordingSandbox):
            def __init__(self) -> None:
                super().__init__()
                self.port_probe_attempts = 0

            async def exec_remote(
                self, *, cmd: list[str], options: object
            ) -> FakeProcess:
                return FakeProcess()

            async def exec(
                self, cmd: list[str], user: str | None = None
            ) -> ExecResult:
                shell_command = cmd[-1]
                if "/dev/tcp/" in shell_command:
                    self.port_probe_attempts += 1
                    return ExecResult(success=self.port_probe_attempts >= 3)
                return await super().exec(cmd, user=user)

        sandbox = PortProbeSandbox()

        process, ca_cert_path, monitor_task = await interceptor_module.start_interceptor(
            sandbox,
            listen_port=8001,
            bridge_port=3001,
            confdir="/root/.mitmproxy-antigravity",
            user="root",
        )

        assert isinstance(process, FakeProcess)
        assert ca_cert_path == "/root/.mitmproxy-antigravity/mitmproxy-ca-cert.pem"
        assert sandbox.port_probe_attempts == 3
        monitor_task.cancel()

    anyio.run(exercise)
