"""Unit tests for the MiniMax Code adapter."""

import shlex
import subprocess
import tarfile
from contextlib import asynccontextmanager
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, cast
from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest
import yaml
from inspect_ai.agent import AgentAttempts, AgentState
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelInfo,
)
from inspect_ai.scorer import Score
from inspect_ai.util import SandboxEnvironment, Store
from inspect_swe._minimax_code import agentbinary
from inspect_swe._minimax_code.minimax_code import (
    _config_yaml,
    _resolve_max_context_size,
)

_MINIMAX_MODULE = import_module("inspect_swe._minimax_code.minimax_code")


def test_config_yaml_routes_openai_responses_to_inspect_bridge() -> None:
    config = yaml.safe_load(
        _config_yaml(port=3123, model="inspect", max_context_size=131072)
    )

    provider = config["custom_provider"]["inspect"]
    assert provider["api"] == "openai-responses"
    assert provider["options"] == {
        "apiKey": "api-key",
        "baseURL": "http://127.0.0.1:3123/v1",
        "authMode": "api-key",
    }
    assert provider["models"]["inspect"]["limit"] == {"context": 131072}
    assert config["defaultModel"] == "custom_provider:inspect/inspect"


def test_config_yaml_quotes_model_ids_with_special_characters() -> None:
    config = yaml.safe_load(
        _config_yaml(port=3123, model="provider/model:latest", max_context_size=8192)
    )

    assert config["custom_provider"]["inspect"]["models"]["provider/model:latest"]
    assert config["defaultModel"] == "custom_provider:inspect/provider/model:latest"


def test_resolve_max_context_size_uses_explicit_override() -> None:
    with patch.object(_MINIMAX_MODULE, "resolve_inspect_model") as resolve_model:
        assert (
            _resolve_max_context_size(model=None, model_aliases=None, override=4096)
            == 4096
        )
    resolve_model.assert_not_called()


def test_resolve_max_context_size_uses_model_metadata() -> None:
    model = object()
    with (
        patch.object(_MINIMAX_MODULE, "resolve_inspect_model", return_value=model),
        patch.object(
            _MINIMAX_MODULE,
            "get_model_info",
            return_value=ModelInfo(context_length=32768),
        ),
    ):
        assert (
            _resolve_max_context_size(
                model="fast", model_aliases={"fast": "provider/model"}, override=None
            )
            == 32768
        )


def test_minimax_setup_installs_node_and_npm_package() -> None:
    fake_sandbox = cast(SandboxEnvironment, Mock())
    fake_sandbox.exec = AsyncMock(  # type: ignore[method-assign]
        side_effect=[Mock(success=False), Mock(success=True)]
    )
    with (
        patch.object(
            agentbinary, "detect_sandbox_platform", AsyncMock(return_value="linux-x64")
        ),
        patch.object(
            agentbinary,
            "ensure_node_available",
            AsyncMock(return_value="/opt/node/bin/node"),
        ) as ensure_node,
        patch.object(
            agentbinary,
            "resolve_npm_package_version",
            return_value="0.4.12",
        ),
        patch.object(
            agentbinary, "create_npm_bundle", return_value=b"bundle"
        ) as create_bundle,
        patch.object(
            agentbinary,
            "install_npm_bundle",
            AsyncMock(return_value="/opt/mcode/node_modules/.bin/mcode"),
        ) as install_bundle,
        patch.object(agentbinary, "_install_sqlite", AsyncMock()) as install_sqlite,
    ):
        result = anyio.run(
            agentbinary.ensure_minimax_code_setup, fake_sandbox, "0.4.12", None
        )

    ensure_node.assert_awaited_once_with(
        fake_sandbox,
        "linux-x64",
        None,
        node_version=agentbinary.MINIMAX_CODE_NODE_VERSION,
    )
    install_bundle.assert_awaited_once()
    create_bundle.assert_called_once_with(
        package=agentbinary.MINIMAX_CODE_PACKAGE,
        version="0.4.12",
        platform="linux-x64",
        cache_name="minimax-code-bundles",
        ignore_scripts=True,
        include_optional=True,
    )
    install_sqlite.assert_awaited_once()
    assert result == ("/opt/mcode/node_modules/.bin/mcode", "/opt/node/bin/node")


def test_minimax_setup_rejects_musl() -> None:
    fake_sandbox = cast(SandboxEnvironment, Mock())
    with (
        patch.object(
            agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value="linux-x64-musl"),
        ),
        patch.object(agentbinary, "ensure_node_available", AsyncMock()) as ensure_node,
    ):
        with pytest.raises(RuntimeError, match="does not support musl"):
            anyio.run(
                agentbinary.ensure_minimax_code_setup, fake_sandbox, "0.4.12", None
            )

    ensure_node.assert_not_awaited()


@pytest.mark.parametrize("version", ["auto", "sandbox"])
def test_preinstalled_cli_does_not_resolve_or_install(version: str) -> None:
    sbox = Mock(spec=SandboxEnvironment)
    sbox.exec = AsyncMock(
        side_effect=[
            Mock(success=True, stdout="/usr/local/bin/mcode\n"),
            Mock(success=True, stdout="/usr/local/bin/node\n"),
        ]
    )
    with patch.object(agentbinary, "detect_sandbox_platform", AsyncMock()) as detect:
        assert anyio.run(
            agentbinary.ensure_minimax_code_setup, sbox, version, "alice"
        ) == ("/usr/local/bin/mcode", "/usr/local/bin/node")
    detect.assert_not_awaited()
    assert all(call.kwargs["user"] == "alice" for call in sbox.exec.call_args_list)


def test_sandbox_version_fails_without_downloading() -> None:
    sbox = Mock(spec=SandboxEnvironment)
    sbox.exec = AsyncMock(return_value=Mock(success=False))
    with patch.object(agentbinary, "ensure_node_available", AsyncMock()) as install:
        with pytest.raises(RuntimeError, match="requires mcode"):
            anyio.run(agentbinary.ensure_minimax_code_setup, sbox, "sandbox", None)
    install.assert_not_awaited()


def test_version_resolution_is_shared_and_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from inspect_swe._util import versioncache

    monkeypatch.setattr(versioncache, "_resolved_versions", {})
    monkeypatch.setattr(versioncache, "_failed_resolutions", {})
    main_thread = threading.get_ident()

    def resolve(package: str) -> str:
        assert threading.get_ident() != main_thread
        return "0.4.12"

    async def run() -> None:
        async with anyio.create_task_group() as group:
            for _ in range(3):
                group.start_soon(
                    versioncache.cached_version_resolution,
                    "minimax-code",
                    agentbinary._latest_version,
                )

    with patch.object(
        agentbinary, "resolve_npm_package_version", side_effect=resolve
    ) as resolve_mock:
        anyio.run(run)
    resolve_mock.assert_called_once_with("@minimax-ai/code")


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    sbox = Mock(spec=SandboxEnvironment)
    sbox.exec = AsyncMock(return_value=Mock(success=True, stdout="/home/alice\n"))
    sbox.exec_remote = AsyncMock(
        return_value=Mock(success=True, stdout="out", stderr="err", returncode=0)
    )
    sbox.write_file = AsyncMock()
    centaur = AsyncMock()

    @asynccontextmanager
    async def bridge(state: AgentState, **kwargs: Any) -> AsyncIterator[Mock]:
        yield Mock(state=state, port=3123)

    monkeypatch.setattr(_MINIMAX_MODULE, "sandbox_agent_bridge", bridge)
    monkeypatch.setattr(_MINIMAX_MODULE, "sandbox_env", lambda _: sbox)
    monkeypatch.setattr(_MINIMAX_MODULE, "store", lambda: Store())
    monkeypatch.setattr(
        _MINIMAX_MODULE,
        "ensure_minimax_code_setup",
        AsyncMock(return_value=("/opt/agent's bin/mcode", "/opt/node/bin/node")),
    )
    monkeypatch.setattr(
        _MINIMAX_MODULE, "resolve_agent_cwd", AsyncMock(return_value="/tmp")
    )
    monkeypatch.setattr(
        _MINIMAX_MODULE, "score", AsyncMock(return_value=[Score(value=0)])
    )
    monkeypatch.setattr(_MINIMAX_MODULE, "run_centaur", centaur)
    return sbox, centaur


def run_agent(messages: list[ChatMessage], **kwargs: Any) -> AgentState:
    agent = _MINIMAX_MODULE.minimax_code(max_context_size=32768, **kwargs)
    return cast(AgentState, anyio.run(agent, AgentState(messages=messages)))


@pytest.mark.parametrize(
    "prompt", ["--help", "review", 'Fix "quotes"\n\nand spaces', "a" * 150000]
)
def test_execution_delivers_literal_prompt_on_stdin(
    execution: tuple[Mock, Mock], prompt: str
) -> None:
    sbox, _ = execution
    run_agent([ChatMessageUser(content=prompt)], env={"CUSTOM": "value"}, user="alice")
    call = sbox.exec_remote.call_args
    assert call.kwargs["cmd"] == [
        "/opt/agent's bin/mcode",
        "exec",
        "--permission",
        "full",
        "--input",
        "-",
    ]
    options = call.kwargs["options"]
    assert options.input == prompt
    assert options.env["CUSTOM"] == "value"
    assert options.cwd == "/tmp"
    assert options.user == "alice"
    config = yaml.safe_load(sbox.write_file.call_args.args[1])
    assert config["custom_provider"]["inspect"]["options"]["apiKey"] == "api-key"


@pytest.mark.parametrize("resumed", [False, True])
def test_retry_continues_exactly_once(
    execution: tuple[Mock, Mock], resumed: bool
) -> None:
    sbox, _ = execution
    messages: list[ChatMessage] = [ChatMessageUser(content="first")]
    if resumed:
        messages.extend(
            [ChatMessageAssistant(content="answer"), ChatMessageUser(content="next")]
        )
    run_agent(
        messages, attempts=AgentAttempts(attempts=2, incorrect_message="Try again")
    )
    first, second = sbox.exec_remote.call_args_list
    assert first.kwargs["cmd"].count("--continue") == int(resumed)
    assert second.kwargs["cmd"].count("--continue") == 1
    assert second.kwargs["options"].input == "Try again"


def test_successful_score_stops_retry(
    execution: tuple[Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    sbox, _ = execution
    monkeypatch.setattr(
        _MINIMAX_MODULE, "score", AsyncMock(return_value=[Score(value=1)])
    )
    run_agent([ChatMessageUser(content="task")], attempts=3)
    sbox.exec_remote.assert_awaited_once()


def test_failed_execution_is_traced_before_raising(
    execution: tuple[Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    sbox, _ = execution
    sbox.exec_remote.return_value = Mock(
        success=False, stdout="out", stderr="failure", returncode=2
    )
    trace = Mock()
    monkeypatch.setattr(_MINIMAX_MODULE, "trace", trace)
    with pytest.raises(RuntimeError, match="failure"):
        run_agent([ChatMessageUser(content="task")], attempts=3, debug=True)
    trace.assert_called_once()
    assert "failure" in trace.call_args.args[0]
    sbox.exec_remote.assert_awaited_once()


def test_centaur_preserves_env_and_shell_quotes(execution: tuple[Mock, Mock]) -> None:
    sbox, centaur = execution
    value = "a 'quote' \"double\" $HOME $(false) `false`\nend"
    run_agent(
        [ChatMessageUser(content="task")],
        centaur=True,
        env={"CUSTOM": value},
        user="alice",
    )
    bashrc = centaur.call_args.args[2]
    assert centaur.call_args.kwargs["user"] == "alice"
    # Execute the generated shell initialization: substitutions must remain
    # literal, and the alias must retain the complete path including spaces.
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", bashrc + '\nprintf "%s" "$CUSTOM"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == value
    alias_line = bashrc.splitlines()[-1]
    assert shlex.split(shlex.split(alias_line)[1].removeprefix("mcode=")) == [
        "/opt/agent's bin/mcode"
    ]
    sbox.exec_remote.assert_not_awaited()


def test_system_prompt_is_sent_with_task(execution: tuple[Mock, Mock]) -> None:
    sbox, _ = execution
    run_agent(
        [ChatMessageSystem(content="System"), ChatMessageUser(content="Task")],
        system_prompt="Extra",
    )
    assert (
        sbox.exec_remote.call_args.kwargs["options"].input == "System\n\nExtra\n\nTask"
    )


def test_resumption_does_not_duplicate_captured_system_prompt(
    execution: tuple[Mock, Mock],
) -> None:
    sbox, _ = execution
    run_agent(
        [
            ChatMessageSystem(content="Captured CLI system prompt"),
            ChatMessageUser(content="first"),
            ChatMessageAssistant(content="answer"),
            ChatMessageUser(content="next"),
        ],
        system_prompt="Original instruction",
    )
    assert sbox.exec_remote.call_args.kwargs["options"].input == "next"


def test_async_retry_feedback(execution: tuple[Mock, Mock]) -> None:
    sbox, _ = execution

    async def feedback(state: AgentState, scores: list[Score]) -> str:
        return "Feedback from scorer"

    run_agent(
        [ChatMessageUser(content="task")],
        attempts=AgentAttempts(attempts=2, incorrect_message=feedback),
    )
    assert sbox.exec_remote.call_args.kwargs["options"].input == "Feedback from scorer"


@pytest.mark.parametrize("platform", ["linux-x64", "linux-arm64"])
@pytest.mark.parametrize("empty_cache", [False, True])
def test_sqlite_download_matches_runtime_and_is_cached(
    tmp_path: Path, platform: str, empty_cache: bool
) -> None:
    from inspect_swe._util.sandbox import SandboxPlatform

    buffer = BytesIO()
    cache_path = tmp_path / f"better-sqlite3-v12.11.1-node-v127-{platform}"
    if empty_cache:
        cache_path.write_bytes(b"")
    replace = Path.replace

    def publish(staged: Path, target: Path) -> Path:
        assert staged.read_bytes() == b"native"
        assert not target.exists() or target.read_bytes() == b""
        return replace(staged, target)

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        member = tarfile.TarInfo("build/Release/better_sqlite3.node")
        member.size = len(b"native")
        tar.addfile(member, BytesIO(b"native"))
    with (
        patch.object(agentbinary, "package_cache_dir", return_value=tmp_path),
        patch.object(
            Path, "replace", autospec=True, side_effect=publish
        ) as replace_mock,
        patch.object(
            agentbinary, "download_file", AsyncMock(return_value=buffer.getvalue())
        ) as download,
    ):
        for _ in range(2):
            assert (
                anyio.run(
                    agentbinary._sqlite_binary,
                    "12.11.1",
                    cast(SandboxPlatform, platform),
                )
                == b"native"
            )
    download.assert_awaited_once_with(
        f"https://github.com/WiseLibs/better-sqlite3/releases/download/v12.11.1/better-sqlite3-v12.11.1-node-v127-{platform}.tar.gz"
    )
    replace_mock.assert_called_once()


def test_sqlite_install_resolves_nested_dependency() -> None:
    sbox = Mock(spec=SandboxEnvironment)
    package_dir = "/opt/mcode/node_modules/@minimax-ai/code/node_modules/better-sqlite3"
    sbox.exec = AsyncMock(
        return_value=Mock(success=True, stdout=package_dir + "/package.json\n")
    )
    sbox.read_file = AsyncMock(return_value='{"version":"12.11.1"}')
    sbox.write_file = AsyncMock()
    with patch.object(
        agentbinary, "_sqlite_binary", AsyncMock(return_value=b"native")
    ) as binary:
        anyio.run(
            agentbinary._install_sqlite,
            sbox,
            "/opt/mcode",
            "/opt/node/bin/node",
            "linux-arm64",
        )
    binary.assert_awaited_once_with("12.11.1", "linux-arm64")
    sbox.write_file.assert_awaited_once_with(
        package_dir + "/build/Release/better_sqlite3.node", b"native"
    )


@pytest.mark.parametrize(
    "version", ["../../outside", "12.11.1/../../../outside", "https://other-host"]
)
def test_sqlite_version_from_sandbox_cannot_escape_host_cache(version: str) -> None:
    with patch.object(agentbinary, "package_cache_dir") as cache:
        with pytest.raises(ValueError, match="Invalid better-sqlite3 version"):
            anyio.run(agentbinary._sqlite_binary, version, "linux-x64")
    cache.assert_not_called()


@pytest.mark.parametrize("override", [0, -1])
def test_context_override_must_be_positive(override: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        _resolve_max_context_size(model=None, model_aliases=None, override=override)


@pytest.mark.parametrize("healthy", [True, False])
def test_existing_installation_checks_sqlite_and_repairs_if_needed(
    healthy: bool,
) -> None:
    sbox = Mock(spec=SandboxEnvironment)
    sbox.exec = AsyncMock(
        side_effect=[Mock(success=True), Mock(success=True, stdout="0.4.12")]
    )
    with (
        patch.object(
            agentbinary, "detect_sandbox_platform", AsyncMock(return_value="linux-x64")
        ),
        patch.object(
            agentbinary, "ensure_node_available", AsyncMock(return_value="/node")
        ),
        patch.object(
            agentbinary,
            "_sqlite_error",
            AsyncMock(side_effect=[None if healthy else "missing SQLite", None]),
        ) as check,
        patch.object(
            agentbinary, "create_npm_bundle", return_value=b"bundle"
        ) as bundle,
        patch.object(
            agentbinary, "install_npm_bundle", AsyncMock(return_value="/mcode")
        ),
        patch.object(agentbinary, "_install_sqlite", AsyncMock()) as repair,
    ):
        anyio.run(agentbinary.ensure_minimax_code_setup, sbox, "0.4.12", None)
    assert check.await_count == (1 if healthy else 2)
    assert repair.await_count == (0 if healthy else 1)
    assert bundle.call_count == (0 if healthy else 1)


def test_centaur_rejects_named_sandbox_before_setup() -> None:
    with pytest.raises(ValueError, match="default sandbox"):
        _MINIMAX_MODULE.minimax_code(centaur=True, sandbox="secondary")


def test_centaur_helper_forwards_user_to_human_cli() -> None:
    from inspect_swe._util.centaur import CentaurOptions, run_centaur

    state = AgentState(messages=[ChatMessageUser(content="task")])
    with (
        patch("inspect_swe._util.centaur.human_cli") as human,
        patch("inspect_swe._util.centaur.run", AsyncMock()) as run,
    ):
        anyio.run(
            run_centaur, CentaurOptions(), "instructions", "bashrc", state, "alice"
        )
    assert human.call_args.kwargs["user"] == "alice"
    run.assert_awaited_once_with(human.return_value, state)
