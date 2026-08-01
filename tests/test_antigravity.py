"""Host-safe unit tests for the antigravity agent (no google.antigravity import)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP, MCPServerConfigStdio
from inspect_swe._antigravity import agentbinary
from inspect_swe._antigravity.antigravity import (
    _SANDBOX_DUMMY_API_KEY,
    _mcp_server_entries,
    _reported_conversation_id,
    sdk_execution_spec,
)
from inspect_swe._antigravity.sdk_runner import load_payload
from pydantic import ValidationError


def _spec_kwargs() -> dict[str, str | None]:
    return {
        "python": "/opt/venv/bin/python",
        "runner_path": "/home/model/.antigravity/runner.py",
        "config_path": "/home/model/.antigravity/request.json",
        "cwd": "/home/model",
        "user": "model",
    }


def test_execution_spec_runs_the_resolved_python_on_the_runner() -> None:
    spec = sdk_execution_spec(**_spec_kwargs())  # type: ignore[arg-type]
    assert spec.command == [
        "/opt/venv/bin/python",
        "/home/model/.antigravity/runner.py",
        "--config",
        "/home/model/.antigravity/request.json",
    ]
    assert spec.cwd == "/home/model"
    assert spec.user == "model"


def test_execution_spec_keeps_real_credentials_out_of_the_sandbox() -> None:
    spec = sdk_execution_spec(**_spec_kwargs())  # type: ignore[arg-type]
    assert spec.env["GEMINI_API_KEY"] == _SANDBOX_DUMMY_API_KEY
    assert _SANDBOX_DUMMY_API_KEY
    assert spec.env["PYTHONNOUSERSITE"] == "1"
    assert spec.env["NO_PROXY"] == "127.0.0.1,localhost"


def test_mcp_server_entries_passes_all_http_servers_through() -> None:
    entries = _mcp_server_entries(
        [
            MCPServerConfigHTTP(name="taiga-mcp", type="http", url="http://x/mcp/t"),
            MCPServerConfigHTTP(name="secrets", type="http", url="http://x/mcp/s"),
        ]
    )
    assert entries == [
        {"name": "taiga-mcp", "url": "http://x/mcp/t"},
        {"name": "secrets", "url": "http://x/mcp/s"},
    ]


def test_mcp_server_entries_allows_no_servers() -> None:
    assert _mcp_server_entries([]) == []


def test_mcp_server_entries_rejects_non_http_servers() -> None:
    with pytest.raises(ValueError, match="Stdio"):
        _mcp_server_entries(
            [MCPServerConfigStdio(name="local", type="stdio", command="server")]
        )


def test_reported_conversation_id_reads_the_result_line() -> None:
    stdout = "\n".join(
        (
            "harness noise",
            json.dumps({"conversation_id": "conv-123", "final_text": "done"}),
        )
    )
    assert _reported_conversation_id(stdout) == "conv-123"


def test_reported_conversation_id_tolerates_missing_or_bad_output() -> None:
    assert _reported_conversation_id("") is None
    assert _reported_conversation_id("no json here") is None
    assert (
        _reported_conversation_id(json.dumps({"conversation_id": None, "x": 1})) is None
    )


def test_load_payload_round_trips_the_host_payload(tmp_path: Path) -> None:
    payload = {
        "prompt": "do the thing",
        "system_instructions": "be safe",
        "bridge_base_url": "http://127.0.0.1:3001",
        "endpoint_model": "gemini-3.6-flash",
        "api_key": _SANDBOX_DUMMY_API_KEY,
        "mcp_servers": [{"name": "taiga-mcp", "url": "http://x/mcp/t"}],
        "app_data_dir": "/home/model/.antigravity",
        "save_dir": "/home/model/.antigravity/session",
        "conversation_id": None,
    }
    config = tmp_path / "request.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    assert load_payload(config) == payload


def test_load_payload_rejects_missing_fields(tmp_path: Path) -> None:
    config = tmp_path / "request.json"
    config.write_text(json.dumps({"prompt": "only"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_payload(config)


def _sdk_probe_sandbox(success: bool) -> tuple[SimpleNamespace, AsyncMock]:
    exec_mock = AsyncMock(
        return_value=SimpleNamespace(success=success, stdout="", stderr="")
    )
    return SimpleNamespace(exec=exec_mock), exec_mock


def test_sdk_presence_probe_runs_under_runner_import_isolation() -> None:
    """A user-site-only install must not pass the probe (PYTHONNOUSERSITE=1)."""
    sandbox, exec_mock = _sdk_probe_sandbox(success=True)
    python = anyio.run(agentbinary._python_with_sdk, sandbox, "model")
    assert python is not None
    for call in exec_mock.await_args_list:
        assert call.kwargs["env"] == {"PYTHONNOUSERSITE": "1"}
