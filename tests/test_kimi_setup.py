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
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import (
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
    ToolCall,
    ToolChoice,
    ToolInfo,
)
from inspect_swe._kimi_code import agentbinary
from inspect_swe._kimi_code.kimi_code import (
    _config_toml,
    _dedupe_tool_call_ids,
    _is_legacy_str_filter,
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


def test_platform_musl_raises() -> None:
    # kimi publishes glibc-only linux builds; musl must fail at resolution time
    for platform in ("linux-x64-musl", "linux-arm64-musl"):
        with pytest.raises(RuntimeError, match="glibc-only"):
            agentbinary._platform_to_kimi_platform(platform)


def test_resolve_version_unknown_platform_raises() -> None:
    source = agentbinary.kimi_code_binary_source()
    # requesting a linux platform absent from a manifest
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

    # user-supplied strings are escaped rather than breaking the TOML
    toml = _config_toml(
        port=3123,
        mcp_servers=[],
        disallowed_tools=['Web"Fetch\\x'],
        extra_skill_dirs=[],
    )
    assert 'pattern = "Web\\"Fetch\\\\x"' in toml

    toml = _config_toml(
        port=3123,
        mcp_servers=[],
        disallowed_tools=[],
        extra_skill_dirs=[],
        skip_afk_prompt_injection=True,
    )
    assert "[system]" in toml
    assert "skip_afk_prompt_injection = true" in toml


def test_mcp_json_stdio_shape() -> None:
    parsed = json.loads(
        _mcp_json(
            [
                MCPServerConfigStdio(
                    name="engine",
                    command="serve",
                    args=["--flag"],
                    cwd="/srv/engine",
                    tools=["query", "status"],
                )
            ]
        )
    )
    assert parsed == {
        "mcpServers": {
            "engine": {
                "transport": "stdio",
                "command": "serve",
                "args": ["--flag"],
                "cwd": "/srv/engine",
                "enabledTools": ["query", "status"],
            }
        }
    }


def test_mcp_json_http_shape() -> None:
    # bridged tools arrive as MCPServerConfigHTTP (the bridge's own MCP
    # endpoint); kimi supports remote servers natively via transport http/sse
    parsed = json.loads(
        _mcp_json(
            [
                MCPServerConfigHTTP(
                    name="bridged",
                    type="http",
                    url="http://localhost:3101/mcp/bridged",
                    headers={"Authorization": "Bearer tok"},
                )
            ]
        )
    )
    assert parsed == {
        "mcpServers": {
            "bridged": {
                "transport": "http",
                "url": "http://localhost:3101/mcp/bridged",
                "headers": {"Authorization": "Bearer tok"},
            }
        }
    }


# reminder wordings by kimi-code era (rewritten in 0.23.4) and escalation tier
_REMINDERS_STRIPPED = [
    # >= 0.23.4, tiers 1-2
    "<system-reminder>\nThe same tool call has been repeated several times in a "
    "row. Before your next call, write one sentence stating what new information "
    "you expect it to produce.\n</system-reminder>",
    "<system-reminder>\nThe same tool call has now been issued 5 times in a row. "
    "Choose exactly one of the following and state your choice before acting.\n"
    "</system-reminder>",
    # < 0.23.4, tiers 1-2
    "<system-reminder>\nYou are repeating the exact same tool call with identical "
    "parameters. Please carefully analyze the previous result.\n</system-reminder>",
    "<system-reminder>\nYou have repeatedly called the same tool with identical "
    "parameters many times.\nRepeated tool call detected:\n- tool: Bash\n"
    "</system-reminder>",
]
# tier 3 precedes kimi's forced turn stop (streak >= 12) and must be preserved
_REMINDERS_KEPT = [
    "<system-reminder>\nWrite your final response now, without any further tool "
    "calls.\n</system-reminder>",
    "<system-reminder>\nYou are stuck in a dead end and have repeatedly made the "
    "same function call without progress.\nStop all function calls immediately.\n"
    "</system-reminder>",
]


def test_strip_repeat_reminders_removes_nag_and_keeps_pairing() -> None:
    for reminder in _REMINDERS_STRIPPED:
        messages: list[ChatMessage] = [
            ChatMessageUser(content=f"check status\n\n{reminder}"),
            ChatMessageAssistant(
                content="polling",
                tool_calls=[
                    ToolCall(id="functions_Bash_0", function="Bash", arguments={})
                ],
            ),
            ChatMessageTool(
                content=f"pending {reminder}", tool_call_id="functions_Bash_0"
            ),
        ]
        cleaned, changed = _strip_repeat_reminders(messages)
        assert changed is True
        assert all("<system-reminder>" not in m.text for m in cleaned)
        # tool/user messages preserved (pairing intact), user text retained
        assert len(cleaned) == 3
        assert cleaned[0].text == "check status"


def test_strip_repeat_reminders_keeps_final_response_tier() -> None:
    for reminder in _REMINDERS_KEPT:
        messages: list[ChatMessage] = [
            ChatMessageTool(
                content=f"pending {reminder}", tool_call_id="functions_Bash_0"
            ),
        ]
        cleaned, changed = _strip_repeat_reminders(messages)
        assert changed is False
        assert cleaned is messages


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

    # idempotent: a second pass (bridge retry) must not grow suffixes
    _dedupe_tool_call_ids(messages)
    assert [
        tc.id
        for m in messages
        if isinstance(m, ChatMessageAssistant) and m.tool_calls
        for tc in m.tool_calls
    ] == call_ids
    assert [
        m.tool_call_id for m in messages if isinstance(m, ChatMessageTool)
    ] == result_ids


def test_is_legacy_str_filter_dispatch() -> None:
    async def legacy(
        model: str,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        return None

    async def modern(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        return None

    assert _is_legacy_str_filter(legacy) is True
    assert _is_legacy_str_filter(modern) is False


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
