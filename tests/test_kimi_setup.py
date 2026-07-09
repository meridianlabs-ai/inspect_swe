"""Unit tests for the Kimi Code agent binary source and config helpers."""

import json
import re
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import MCPServerConfigHTTP, MCPServerConfigStdio, ToolCall
from inspect_swe._kimi_code import agentbinary
from inspect_swe._kimi_code.kimi_code import (
    _config_toml,
    _dedupe_tool_call_ids,
    _mcp_json,
    _strip_repeat_reminders,
)

from tests.conftest import skip_if_github_action

_LATEST_JSON = json.dumps({"schemaVersion": 1, "version": "0.21.1"})
_MANIFEST_JSON = json.dumps(
    {
        "version": "0.21.1",
        "platforms": {
            "linux-x64": {"filename": "kimi-code-linux-x64", "checksum": "aa" * 32},
            "linux-arm64": {"filename": "kimi-code-linux-arm64", "checksum": "bb" * 32},
        },
    }
)


def test_resolve_version_latest_fetches_pointer_then_manifest() -> None:
    source = agentbinary.kimi_code_binary_source()
    with patch.object(
        agentbinary,
        "download_text_file",
        AsyncMock(side_effect=[_LATEST_JSON, _MANIFEST_JSON]),
    ) as mock_download:
        resolved = anyio.run(source.resolve_version, "latest", "linux-x64")
    assert resolved.version == "0.21.1"
    assert resolved.expected_checksum == "aa" * 32
    assert resolved.download_url == (
        "https://code.kimi.com/kimi-code/binaries/0.21.1/kimi-code-linux-x64"
    )
    assert mock_download.await_count == 2


def test_resolve_version_specific_skips_pointer() -> None:
    source = agentbinary.kimi_code_binary_source()
    with patch.object(
        agentbinary, "download_text_file", AsyncMock(return_value=_MANIFEST_JSON)
    ) as mock_download:
        resolved = anyio.run(source.resolve_version, "0.21.1", "linux-arm64")
    assert resolved.version == "0.21.1"
    assert resolved.expected_checksum == "bb" * 32
    assert resolved.download_url.endswith("/0.21.1/kimi-code-linux-arm64")
    mock_download.assert_awaited_once()


def test_platform_musl_collapses_to_glibc_key() -> None:
    assert agentbinary._platform_to_kimi_platform("linux-x64-musl") == "linux-x64"
    assert agentbinary._platform_to_kimi_platform("linux-arm64-musl") == "linux-arm64"


def test_resolve_version_unknown_platform_raises() -> None:
    source = agentbinary.kimi_code_binary_source()
    with patch.object(
        agentbinary, "download_text_file", AsyncMock(return_value=_MANIFEST_JSON)
    ):
        # manifest omits win32; requesting a linux platform absent from a manifest
        manifest_no_linux = json.dumps({"version": "0.21.1", "platforms": {}})
        with patch.object(
            agentbinary, "download_text_file", AsyncMock(return_value=manifest_no_linux)
        ):
            with pytest.raises(RuntimeError, match="No Kimi Code binary"):
                anyio.run(source.resolve_version, "0.21.1", "linux-x64")


def test_config_toml_routes_bridge_and_wires_rules() -> None:
    toml = _config_toml(
        port=3123,
        mcp_servers=[MCPServerConfigStdio(name="engine", command="serve", args=[])],
        disallowed_tools=["WebFetch"],
        extra_skill_dirs=["/root/.kimi-code/skills"],
    )
    assert 'base_url = "http://localhost:3123/v1"' in toml
    assert 'default_model = "bridge"' in toml
    assert 'extra_skill_dirs = ["/root/.kimi-code/skills"]' in toml
    assert 'pattern = "mcp__engine__*"' in toml
    assert 'decision = "allow"' in toml
    assert 'pattern = "WebFetch"' in toml
    assert 'decision = "deny"' in toml
    assert "skip_afk_prompt_injection" not in toml

    toml = _config_toml(
        port=3123,
        mcp_servers=[],
        disallowed_tools=[],
        extra_skill_dirs=[],
        skip_afk_prompt_injection=True,
    )
    assert "[system]" in toml
    assert "skip_afk_prompt_injection = true" in toml


def test_mcp_json_shape_and_rejects_non_stdio() -> None:
    parsed = json.loads(
        _mcp_json(
            [MCPServerConfigStdio(name="engine", command="serve", args=["--flag"])]
        )
    )
    assert parsed == {
        "mcpServers": {"engine": {"command": "serve", "args": ["--flag"]}}
    }
    with pytest.raises(ValueError, match="only supports stdio"):
        _mcp_json([MCPServerConfigHTTP(name="remote", type="http", url="http://x")])


def test_strip_repeat_reminders_removes_nag_and_keeps_pairing() -> None:
    reminder = (
        "<system-reminder>Repeated tool call detected: do not repeat.</system-reminder>"
    )
    messages: list[ChatMessage] = [
        ChatMessageUser(content=f"check status\n\n{reminder}"),
        ChatMessageAssistant(
            content="polling",
            tool_calls=[ToolCall(id="functions_Bash_0", function="Bash", arguments={})],
        ),
        ChatMessageTool(content=f"pending {reminder}", tool_call_id="functions_Bash_0"),
    ]
    cleaned, changed = _strip_repeat_reminders(messages)
    assert changed is True
    assert all("Repeated tool call detected" not in m.text for m in cleaned)
    # tool/user messages preserved (pairing intact), user text retained
    assert len(cleaned) == 3
    assert cleaned[0].text == "check status"


def test_strip_repeat_reminders_noop_when_absent() -> None:
    messages: list[ChatMessage] = [ChatMessageUser(content="hello")]
    cleaned, changed = _strip_repeat_reminders(messages)
    assert changed is False
    assert cleaned is messages


def test_dedupe_tool_call_ids_makes_ids_unique_preserving_pairing() -> None:
    messages: list[ChatMessage] = [
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="functions_Bash_0", function="Bash", arguments={})],
        ),
        ChatMessageTool(content="a", tool_call_id="functions_Bash_0"),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="functions_Bash_0", function="Bash", arguments={})],
        ),
        ChatMessageTool(content="b", tool_call_id="functions_Bash_0"),
    ]
    _dedupe_tool_call_ids(messages)
    call_ids = [
        tc.id
        for m in messages
        if isinstance(m, ChatMessageAssistant) and m.tool_calls
        for tc in m.tool_calls
    ]
    result_ids = [m.tool_call_id for m in messages if isinstance(m, ChatMessageTool)]
    assert len(set(call_ids)) == 2
    # each tool result still threads to its originating call (FIFO)
    assert result_ids == call_ids


@skip_if_github_action
def test_live_distribution_manifest_shape() -> None:
    """Drift check: Kimi's version pointer + manifest still expose linux binaries.

    The binary source depends on ``latest.json`` carrying ``version`` and each
    ``binaries/<version>/manifest.json`` carrying per-platform ``filename`` and
    64-hex ``checksum``. This fetches the live endpoints and fails if that shape
    changes. Skips when offline; github-action-gated so an upstream hiccup can't
    block unrelated CI.
    """
    from inspect_swe._util.download import download_text_file

    try:
        version = anyio.run(agentbinary._fetch_latest_version)
        manifest = anyio.run(agentbinary._fetch_manifest, version)
    except Exception as ex:
        pytest.skip(f"live Kimi distribution unavailable: {ex}")

    platforms = manifest["platforms"]
    for key in ("linux-x64", "linux-arm64"):
        assert key in platforms, f"manifest missing platform {key}"
        entry = platforms[key]
        assert entry["filename"], "manifest entry missing filename"
        assert re.fullmatch(r"[0-9a-f]{64}", entry["checksum"]), (
            f"manifest checksum for {key} is not sha256 hex: {entry['checksum']!r}"
        )
    assert callable(download_text_file)
