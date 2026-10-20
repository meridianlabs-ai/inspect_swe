"""Unit tests for the DeepSeek Harness agent binary source and config helpers."""

import io
import json
import tarfile
import zipfile
from importlib import import_module
from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelInfo,
    ModelOutput,
)
from inspect_ai.tool import MCPServerConfigHTTP, MCPServerConfigStdio, ToolCall
from inspect_swe._deepseek_harness import agentbinary
from inspect_swe._deepseek_harness.deepseek_harness import (
    _parse_events,
    _profile_patch,
    _reasoning_unset,
    _resolve_context_window,
    _session_id,
    _turn_end_reason,
    _validate_mcp_server,
)
from inspect_swe._deepseek_harness.reasoning import ReasoningRestorer

_DSH_MODULE = import_module("inspect_swe._deepseek_harness.deepseek_harness")
_REASONING_MODULE = import_module("inspect_swe._deepseek_harness.reasoning")


def _release_json(version: str) -> str:
    def wheel(tag: str, digest: str) -> dict[str, object]:
        filename = f"deepseek_harness_runtime_bin-{version}-py3-none-{tag}.whl"
        return {
            "filename": filename,
            "packagetype": "bdist_wheel",
            "url": f"https://files.example/{filename}",
            "digests": {"sha256": digest},
        }

    return json.dumps(
        {
            "info": {"version": version},
            "urls": [
                wheel("macosx_14_0_arm64", "00" * 32),
                wheel("manylinux_2_28_x86_64", "aa" * 32),
                wheel("manylinux_2_28_aarch64", "bb" * 32),
            ],
        }
    )


def test_resolve_version_latest_reads_project_json() -> None:
    source = agentbinary.deepseek_harness_binary_source()
    with patch.object(
        agentbinary,
        "download_text_file",
        AsyncMock(return_value=_release_json("0.2.1rc2")),
    ) as mock_download:
        resolved = anyio.run(source.resolve_version, "latest", "linux-x64")
    assert resolved.version == "0.2.1rc2"
    assert resolved.expected_checksum == "aa" * 32
    assert resolved.download_url.endswith("-manylinux_2_28_x86_64.whl")
    assert resolved.package is True
    mock_download.assert_awaited_once_with(
        "https://pypi.org/pypi/deepseek-harness-runtime-bin/json"
    )


def test_resolve_version_specific_reads_release_json() -> None:
    source = agentbinary.deepseek_harness_binary_source()
    with patch.object(
        agentbinary,
        "download_text_file",
        AsyncMock(return_value=_release_json("0.1.5rc1")),
    ) as mock_download:
        resolved = anyio.run(source.resolve_version, "0.1.5rc1", "linux-arm64")
    assert resolved.version == "0.1.5rc1"
    assert resolved.expected_checksum == "bb" * 32
    assert resolved.download_url.endswith("-manylinux_2_28_aarch64.whl")
    mock_download.assert_awaited_once_with(
        "https://pypi.org/pypi/deepseek-harness-runtime-bin/0.1.5rc1/json"
    )


def test_resolve_version_musl_raises() -> None:
    source = agentbinary.deepseek_harness_binary_source()
    with (
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(return_value=_release_json("0.2.1rc2")),
        ),
        pytest.raises(RuntimeError, match="glibc-only"),
    ):
        anyio.run(source.resolve_version, "latest", "linux-x64-musl")


def test_resolve_version_missing_wheel_raises() -> None:
    release = json.loads(_release_json("0.2.1rc2"))
    release["urls"] = release["urls"][:1]
    source = agentbinary.deepseek_harness_binary_source()
    with (
        patch.object(
            agentbinary,
            "download_text_file",
            AsyncMock(return_value=json.dumps(release)),
        ),
        pytest.raises(RuntimeError, match="No DeepSeek Harness runtime wheel"),
    ):
        anyio.run(source.resolve_version, "latest", "linux-x64")


def _fake_wheel() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as wheel:

        def add(name: str, data: bytes, mode: int) -> None:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            wheel.writestr(info, data)

        runtime = "deepseek_harness_runtime/runtime"
        add("deepseek_harness_runtime/__init__.py", b"", 0o644)
        add(f"{runtime}/deepseek-harness-sdk-runtime-linux-x64", b"exe", 0o644)
        add(f"{runtime}/deepseek-harness-sdk-runtime-linux-x64-rg", b"rg", 0o644)
        add(f"{runtime}/linux-x64/downloads.json", b"{}", 0o644)
        add("deepseek_harness_runtime_bin-0.2.1rc2.dist-info/RECORD", b"", 0o644)
    return output.getvalue()


def test_wheel_to_tar_gz_installs_runtime_with_launcher() -> None:
    converted = agentbinary._wheel_to_tar_gz(_fake_wheel())
    with tarfile.open(fileobj=io.BytesIO(converted), mode="r:gz") as tar:
        members = {member.name: member for member in tar.getmembers()}
        launcher = tar.extractfile("dsh")
        assert launcher is not None
        launcher_text = launcher.read().decode()

    assert set(members) == {
        "dsh",
        "runtime/deepseek-harness-sdk-runtime-linux-x64",
        "runtime/deepseek-harness-sdk-runtime-linux-x64-rg",
        "runtime/linux-x64/downloads.json",
    }
    # executables are made executable even when the wheel carries no mode bits
    assert members["runtime/deepseek-harness-sdk-runtime-linux-x64"].mode == 0o755
    assert members["runtime/deepseek-harness-sdk-runtime-linux-x64-rg"].mode == 0o755
    assert members["runtime/linux-x64/downloads.json"].mode == 0o644
    assert members["dsh"].mode == 0o755
    assert '/runtime/deepseek-harness-sdk-runtime-linux-x64" "$@"' in launcher_text


def test_wheel_to_tar_gz_is_memoized() -> None:
    wheel = _fake_wheel()
    assert agentbinary._wheel_to_tar_gz(wheel) is agentbinary._wheel_to_tar_gz(wheel)


def test_wheel_without_runtime_raises() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as wheel:
        wheel.writestr("deepseek_harness_runtime/__init__.py", b"")
    with pytest.raises(RuntimeError, match="no runtime executable"):
        agentbinary._wheel_to_tar_gz(output.getvalue())


def test_profile_patch_routes_model_and_disables_upstream_reporting() -> None:
    rows = json.loads(
        _profile_patch(model="deepseek/v4", context_window=128_000, mcp_servers=[])
    )
    by_id = {row["id"]: row for row in rows if "id" in row}
    assert by_id["agent-default-model"]["config"] == {
        "provider": "deepseek-official",
        "model": "deepseek/v4",
    }
    assert by_id["llm-deepseek"]["config"] == {
        "models": [{"id": "deepseek/v4", "contextWindow": 128_000}]
    }
    for row_id in [
        "session-log-deepseek",
        "plugin-package-inventory-deepseek",
        "session-title-llm",
        "session-telemetry-otel",
    ]:
        assert by_id[row_id] == {"id": row_id, "disabled": True}
    assert not any("insert" in row for row in rows)


def test_profile_patch_omits_unknown_context_window() -> None:
    rows = json.loads(
        _profile_patch(model="inspect", context_window=None, mcp_servers=[])
    )
    by_id = {row["id"]: row for row in rows if "id" in row}
    assert by_id["llm-deepseek"]["config"] == {"models": [{"id": "inspect"}]}


def test_profile_patch_inserts_mcp_servers() -> None:
    rows = json.loads(
        _profile_patch(
            model="inspect",
            context_window=None,
            mcp_servers=[
                MCPServerConfigStdio(
                    name="memory",
                    command="npx",
                    args=["@modelcontextprotocol/server-memory"],
                    env={"A": "1"},
                    cwd="/work",
                ),
                MCPServerConfigHTTP(
                    name="secrets",
                    type="http",
                    url="http://localhost:9/mcp",
                    headers={"Authorization": "Bearer x"},
                ),
            ],
        )
    )
    inserted = next(row["insert"] for row in rows if "insert" in row)
    assert inserted == [
        {
            "id": "mcp-memory",
            "name": "@deepseek-ai/dsh-mcp-client",
            "config": {
                "serverName": "memory",
                "transport": "stdio",
                "command": "npx",
                "args": ["@modelcontextprotocol/server-memory"],
                "env": {"A": "1"},
                "cwd": "/work",
                "failOnStartupError": True,
            },
        },
        {
            "id": "mcp-secrets",
            "name": "@deepseek-ai/dsh-mcp-client",
            "config": {
                "serverName": "secrets",
                "transport": "streamable-http",
                "url": "http://localhost:9/mcp",
                "headers": {"Authorization": "Bearer x"},
                "failOnStartupError": True,
            },
        },
    ]


def test_validate_mcp_server_rejects_tool_filter_and_sse() -> None:
    _validate_mcp_server(MCPServerConfigStdio(name="ok", command="srv"))
    with pytest.raises(ValueError, match="server names"):
        _validate_mcp_server(MCPServerConfigStdio(name="my.tools", command="srv"))
    with pytest.raises(ValueError, match="server names"):
        _validate_mcp_server(MCPServerConfigStdio(name="x" * 33, command="srv"))
    with pytest.raises(ValueError, match="tool filtering"):
        _validate_mcp_server(
            MCPServerConfigStdio(name="filtered", command="srv", tools=["a"])
        )
    with pytest.raises(ValueError, match="not SSE"):
        _validate_mcp_server(
            MCPServerConfigHTTP(name="legacy", type="sse", url="http://x/sse")
        )


def test_resolve_context_window_prefers_explicit_value() -> None:
    model = Mock()
    model.name = "m"
    assert _resolve_context_window(model=model, context_window=4096) == 4096
    with pytest.raises(ValueError, match="positive"):
        _resolve_context_window(model=model, context_window=0)


def test_resolve_context_window_uses_metadata_or_dsh_default() -> None:
    model = Mock()
    model.name = "m"
    with patch.object(
        _DSH_MODULE,
        "get_model_info",
        return_value=ModelInfo(model="m", context_length=200_000),
    ):
        assert _resolve_context_window(model=model, context_window=None) == 200_000
    with patch.object(_DSH_MODULE, "get_model_info", return_value=None):
        assert _resolve_context_window(model=model, context_window=None) is None


def test_event_stream_parsing() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "session", "sessionId": "session-1", "cwd": "/w"}),
            "not json",
            json.dumps({"type": "status", "phase": "turn_start", "turn": 1}),
            json.dumps(
                {"type": "status", "phase": "turn_end", "turn": 1, "reason": "aborted"}
            ),
            json.dumps({"type": "final", "text": "done"}),
            "",
        ]
    )
    events = _parse_events(stdout)
    assert len(events) == 4
    assert _session_id(events) == "session-1"
    assert _turn_end_reason(events) == "aborted"
    assert _session_id([]) is None
    assert _turn_end_reason([{"type": "error", "message": "boom"}]) == "error: boom"


def _model_event(content: list[ContentReasoning | ContentText]) -> ModelEvent:
    return ModelEvent(
        model="m",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(ChatMessageAssistant(content=list(content))),
    )


def _restore(
    messages: list[ChatMessage], events: list[ModelEvent]
) -> tuple[list[ChatMessage], bool]:
    fake_transcript = Mock()
    fake_transcript.events = events
    with patch.object(_REASONING_MODULE, "transcript", return_value=fake_transcript):
        changed = ReasoningRestorer().restore(messages)
    return messages, changed


def test_restore_reasoning_recovers_original_block_and_drops_placeholder() -> None:
    original = ContentReasoning(reasoning="think hard", signature="sig-payload")
    messages: list[ChatMessage] = [
        ChatMessageUser(content="hi"),
        ChatMessageAssistant(
            content=[
                ContentText(text="<think>think hard</think>"),
                ContentText(text="(no content)"),
            ],
            tool_calls=[ToolCall(id="1", function="bash", arguments={})],
        ),
    ]
    restored, changed = _restore(messages, [_model_event([original])])
    assert changed
    content = restored[1].content
    assert isinstance(content, list)
    assert content == [original]


def test_restore_reasoning_leaves_unmatched_text_alone() -> None:
    messages: list[ChatMessage] = [
        ChatMessageAssistant(
            content=[
                ContentText(text="<think>no original</think>"),
                ContentText(text="(no content)"),
            ],
            tool_calls=[ToolCall(id="1", function="bash", arguments={})],
        ),
    ]
    restored, changed = _restore(messages, [])
    assert not changed
    assert restored[0].content == [
        ContentText(text="<think>no original</think>"),
        ContentText(text="(no content)"),
    ]


def test_restore_reasoning_matches_redacted_block_by_flattened_text() -> None:
    original = ContentReasoning(
        reasoning="ENCRYPTED", redacted=True, summary="short summary"
    )
    messages: list[ChatMessage] = [
        ChatMessageAssistant(
            content=[
                ContentText(text="<think>short summary</think>"),
                ContentText(text="ok"),
            ]
        ),
    ]
    restored, changed = _restore(messages, [_model_event([original])])
    assert changed
    assert restored[0].content == [original, ContentText(text="ok")]


def test_restore_reasoning_skips_ambiguous_text() -> None:
    first = ContentReasoning(reasoning="ENC-1", redacted=True)
    second = ContentReasoning(reasoning="ENC-2", redacted=True)
    messages: list[ChatMessage] = [
        ChatMessageAssistant(
            content=[ContentText(text="<think></think>"), ContentText(text="ok")]
        ),
    ]
    restored, changed = _restore(
        messages, [_model_event([first]), _model_event([second])]
    )
    assert not changed
    assert restored[0].content == [
        ContentText(text="<think></think>"),
        ContentText(text="ok"),
    ]


def test_restore_reasoning_rescans_pending_model_events() -> None:
    original = ContentReasoning(reasoning="late", signature="sig")
    pending = _model_event([])
    pending.pending = True
    events = [pending]
    fake_transcript = Mock()
    fake_transcript.events = events
    restorer = ReasoningRestorer()

    def messages() -> list[ChatMessage]:
        return [
            ChatMessageAssistant(
                content=[ContentText(text="<think>late</think>"), ContentText(text="x")]
            )
        ]

    with patch.object(_REASONING_MODULE, "transcript", return_value=fake_transcript):
        assert not restorer.restore(messages())
        # the pending call completes; a later scan must still see it
        events[0] = _model_event([original])
        restored = messages()
        assert restorer.restore(restored)
    assert restored[0].content == [original, ContentText(text="x")]


def test_restore_reasoning_keeps_sole_placeholder_and_plain_text() -> None:
    original = ContentReasoning(reasoning="r", signature="s")
    messages: list[ChatMessage] = [
        ChatMessageUser(content="<think>r</think>"),
        ChatMessageAssistant(content=[ContentText(text="(no content)")]),
        ChatMessageAssistant(content=[ContentText(text="no <think>r</think>")]),
    ]
    restored, changed = _restore(messages, [_model_event([original])])
    assert not changed
    assert restored[0].content == "<think>r</think>"
    assert restored[1].content == [ContentText(text="(no content)")]
    assert restored[2].content == [ContentText(text="no <think>r</think>")]


def test_reasoning_default_applies_only_when_unset() -> None:
    model = Mock()
    model.config = GenerateConfig()
    with patch.object(
        _DSH_MODULE, "active_generate_config", return_value=GenerateConfig()
    ):
        assert _reasoning_unset(model, GenerateConfig())
        assert not _reasoning_unset(model, GenerateConfig(reasoning_tokens=1024))
        model.config = GenerateConfig(reasoning_effort="low")
        assert not _reasoning_unset(model, GenerateConfig())
    model.config = GenerateConfig()
    with patch.object(
        _DSH_MODULE,
        "active_generate_config",
        return_value=GenerateConfig(reasoning_effort="medium"),
    ):
        assert not _reasoning_unset(model, GenerateConfig())
