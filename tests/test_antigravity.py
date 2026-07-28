"""Host-safe unit tests for the native Antigravity SDK agent.

These exercise only host-side logic that does not import ``google.antigravity``
(which lives in the sandbox): the execution spec (credential boundary), the
taiga-mcp resolver, and constructor validation.
"""

from __future__ import annotations

import pytest
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_swe._antigravity.antigravity import (
    _SANDBOX_DUMMY_API_KEY,
    _taiga_mcp_config,
    antigravity,
    sdk_execution_spec,
)

_RUNNER = "/home/model/.antigravity/runner.py"
_CONFIG = "/home/model/.antigravity/request.json"


def test_execution_spec_runs_the_resolved_python_on_the_runner() -> None:
    spec = sdk_execution_spec(
        python="/opt/venv/bin/python", runner_path=_RUNNER, config_path=_CONFIG
    )
    assert spec.command == ["/opt/venv/bin/python", _RUNNER, "--config", _CONFIG]
    assert spec.user == "model"
    assert spec.cwd == "/home/model"


def test_execution_spec_keeps_real_credentials_out_of_the_sandbox() -> None:
    # The only GEMINI_API_KEY entering the sandbox must be the dummy; inference is
    # routed to the loopback bridge via the endpoint base_url, so the value is
    # never used for auth. This upholds the only-dummy-creds-in-sandbox invariant.
    spec = sdk_execution_spec(
        python="/opt/venv/bin/python", runner_path=_RUNNER, config_path=_CONFIG
    )
    assert spec.env["GEMINI_API_KEY"] == _SANDBOX_DUMMY_API_KEY
    assert _SANDBOX_DUMMY_API_KEY
    # no other credential-shaped env leaks in
    assert set(spec.env) == {
        "HOME",
        "NO_PROXY",
        "PYTHONNOUSERSITE",
        "no_proxy",
        "GEMINI_API_KEY",
    }


def test_taiga_mcp_config_selects_the_single_http_server() -> None:
    server = MCPServerConfigHTTP(
        name="taiga-mcp", type="http", url="http://127.0.0.1:3001/mcp/taiga-mcp"
    )
    assert _taiga_mcp_config([server]) is server


def test_taiga_mcp_config_requires_the_server() -> None:
    with pytest.raises(RuntimeError, match="one taiga-mcp bridge server"):
        _taiga_mcp_config([])


def test_taiga_mcp_config_rejects_empty_url() -> None:
    server = MCPServerConfigHTTP(name="taiga-mcp", type="http", url="")
    with pytest.raises(RuntimeError, match="nonempty URL"):
        _taiga_mcp_config([server])


def test_constructor_rejects_non_model_user() -> None:
    with pytest.raises(ValueError, match="model"):
        antigravity(user="root")


def test_constructor_rejects_non_home_cwd() -> None:
    with pytest.raises(ValueError, match="/home/model"):
        antigravity(cwd="/workspace")
