"""Host-safe unit tests for the antigravity agent (no google.antigravity import)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast
from unittest.mock import AsyncMock

import anyio
import pytest
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP, MCPServerConfigStdio
from inspect_ai.util import SandboxEnvironment
from inspect_swe._antigravity import agentbinary
from inspect_swe._antigravity.antigravity import (
    _SANDBOX_DUMMY_API_KEY,
    _mcp_server_entries,
    _reported_conversation_id,
    sdk_execution_spec,
)
from inspect_swe._antigravity.sdk_runner import _confine_to_dir, load_payload
from pydantic import ValidationError


class _SpecKwargs(TypedDict):
    python: str
    runner_path: str
    config_path: str
    cwd: str
    home: str
    user: str | None


def _spec_kwargs() -> _SpecKwargs:
    return {
        "python": "/opt/venv/bin/python",
        "runner_path": "/home/model/.antigravity/runner.py",
        "config_path": "/home/model/.antigravity/request.json",
        "cwd": "/workspace/repo",
        "home": "/home/model",
        "user": "model",
    }


def test_execution_spec_runs_the_resolved_python_on_the_runner() -> None:
    spec = sdk_execution_spec(**_spec_kwargs())
    assert spec.command == [
        "/opt/venv/bin/python",
        "/home/model/.antigravity/runner.py",
        "--config",
        "/home/model/.antigravity/request.json",
    ]
    assert spec.cwd == "/workspace/repo"
    assert spec.user == "model"


def test_execution_spec_points_home_at_the_sandbox_home_not_the_workspace() -> None:
    """SDK state must not land in (or present as HOME) the evaluated repo."""
    spec = sdk_execution_spec(**_spec_kwargs())
    assert spec.env["HOME"] == "/home/model"
    assert spec.cwd == "/workspace/repo"


def test_execution_spec_keeps_real_credentials_out_of_the_sandbox() -> None:
    spec = sdk_execution_spec(**_spec_kwargs())
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
        {"name": "taiga-mcp", "url": "http://x/mcp/t", "headers": None, "tools": None},
        {"name": "secrets", "url": "http://x/mcp/s", "headers": None, "tools": None},
    ]


def test_mcp_server_entries_carries_tool_allowlist_and_headers() -> None:
    """A per-server tools allowlist and auth headers must survive the payload."""
    entries = _mcp_server_entries(
        [
            MCPServerConfigHTTP(
                name="secrets",
                type="http",
                url="http://x/mcp/s",
                tools=["read_only"],
                headers={"Authorization": "Bearer abc"},
            ),
        ]
    )
    assert entries == [
        {
            "name": "secrets",
            "url": "http://x/mcp/s",
            "headers": {"Authorization": "Bearer abc"},
            "tools": ["read_only"],
        }
    ]


def test_mcp_server_entries_allows_no_servers() -> None:
    assert _mcp_server_entries([]) == []


def test_mcp_server_entries_rejects_non_http_servers() -> None:
    with pytest.raises(ValueError, match="Stdio"):
        _mcp_server_entries(
            [MCPServerConfigStdio(name="local", type="stdio", command="server")]
        )


def test_mcp_server_entries_rejects_sse_servers() -> None:
    """The SDK has no SSE client; sse configs must not silently become http."""
    with pytest.raises(ValueError, match="SSE"):
        _mcp_server_entries(
            [MCPServerConfigHTTP(name="events", type="sse", url="http://x/sse")]
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
        "mcp_servers": [
            {
                "name": "taiga-mcp",
                "url": "http://x/mcp/t",
                "headers": {"Authorization": "Bearer abc"},
                "tools": ["browser"],
            }
        ],
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


def _sdk_probe_sandbox(success: bool) -> tuple[SandboxEnvironment, AsyncMock]:
    exec_mock = AsyncMock(
        return_value=SimpleNamespace(success=success, stdout="", stderr="")
    )
    # A structural fake: only .exec is consulted by the probe helpers.
    return cast(SandboxEnvironment, SimpleNamespace(exec=exec_mock)), exec_mock


def test_sdk_presence_probe_runs_under_runner_import_isolation() -> None:
    """A user-site-only install must not pass the probe (PYTHONNOUSERSITE=1)."""
    sandbox, exec_mock = _sdk_probe_sandbox(success=True)
    python = anyio.run(agentbinary._python_with_sdk, sandbox, "model", "0.1.7")
    assert python is not None
    for call in exec_mock.await_args_list:
        assert call.kwargs["env"] == {"PYTHONNOUSERSITE": "1"}


class _FakeEgressSandbox:
    """Structural sandbox fake: no baked SDK, venv provisioning succeeds."""

    def __init__(self) -> None:
        self.installed_version: str | None = None
        self.install_count = 0

    async def exec(
        self,
        cmd: list[str],
        user: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SimpleNamespace:
        program, code = cmd[0], cmd[-1]
        if program == "bash":  # the venv-create + pip-install step
            self.install_count += 1
            self.installed_version = code.split("==")[-1].rstrip('"')
            return SimpleNamespace(success=True, stdout="", stderr="")
        if "ensurepip" in code:  # _base_python capability probe
            success = program == "/usr/bin/python3"
            return SimpleNamespace(success=success, stdout="", stderr="")
        if program == agentbinary._SDK_VENV_PYTHON:  # version-checked venv probe
            success = (
                self.installed_version is not None
                and f"== {self.installed_version!r}" in code
            )
            return SimpleNamespace(success=success, stdout="", stderr="")
        # baked-interpreter probes: no SDK in the image
        return SimpleNamespace(success=False, stdout="", stderr="")


def test_ensure_sdk_provisions_once_and_reuses_the_venv() -> None:
    """A second ensure call must reuse the provisioned venv, not reinstall."""
    fake = _FakeEgressSandbox()
    sandbox = cast(SandboxEnvironment, fake)

    async def _run() -> tuple[str, str]:
        first = await agentbinary.ensure_antigravity_sdk(sandbox, "model")
        second = await agentbinary.ensure_antigravity_sdk(sandbox, "model")
        return first, second

    first, second = anyio.run(_run)
    assert first == second == agentbinary._SDK_VENV_PYTHON
    assert fake.install_count == 1


def test_ensure_sdk_does_not_reuse_a_stale_venv_version() -> None:
    """A version bump must reinstall rather than silently serve the old pin."""
    fake = _FakeEgressSandbox()
    sandbox = cast(SandboxEnvironment, fake)

    async def _run() -> None:
        await agentbinary.ensure_antigravity_sdk(sandbox, "model", version="0.1.7")
        await agentbinary.ensure_antigravity_sdk(sandbox, "model", version="0.1.8")

    anyio.run(_run)
    assert fake.install_count == 2
    assert fake.installed_version == "0.1.8"


def test_view_file_predicate_confines_to_the_data_dir() -> None:
    """view_file may only read localharness's offloaded results, nothing else."""
    within = _confine_to_dir("/home/model/.antigravity")
    assert within({"AbsolutePath": "/home/model/.antigravity/brain/x/out.txt"})
    assert within({"AbsolutePath": "/home/model/.antigravity"})
    assert within({"AbsolutePath": "/home/model/.antigravity/a/../b"})
    assert not within({"AbsolutePath": "/home/model/.antigravity/../secrets.txt"})
    assert not within({"AbsolutePath": "/home/model/.antigravity-evil/out.txt"})
    assert not within({"AbsolutePath": "/workspace/repo/grading/regular.py"})
    assert not within({"AbsolutePath": ""})
    assert not within({"AbsolutePath": 42})
    assert not within({})
