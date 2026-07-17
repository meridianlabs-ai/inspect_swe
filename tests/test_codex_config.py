import pytest
from inspect_swe._codex_cli.config import (
    codex_cli_config_overrides,
    codex_config_options,
    codex_mcp_server_toml,
    resolve_codex_deprecated_args,
    resolve_codex_web_search,
)
from inspect_swe._codex_cli.config import (
    codex_cli_config_overrides,
    codex_config_options,
    resolve_codex_deprecated_args,
    resolve_codex_web_search,
)
from inspect_swe._util.toml import to_toml


def test_codex_config_defaults() -> None:
    config = codex_config_options("live", True)

    assert config["web_search"] == "live"
    assert config["features.goals"] is True
    toml = to_toml(config)
    assert 'web_search = "live"' in toml
    assert "features.goals = true" in toml


@pytest.mark.parametrize("web_search", ["live", "cached", "disabled"])
def test_resolve_codex_web_search_modes(web_search: str) -> None:
    assert resolve_codex_web_search(web_search) == web_search


def test_resolve_codex_web_search_invalid_mode() -> None:
    with pytest.raises(ValueError, match="web_search must be one of"):
        resolve_codex_web_search("offline")


def test_deprecated_disallowed_tools_disable_web_search() -> None:
    disallowed_tools = resolve_codex_deprecated_args(
        {"disallowed_tools": ["web_search"]}
    )

    assert resolve_codex_web_search("live", disallowed_tools) == "disabled"


def test_deprecated_disallowed_tools_reject_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unsupported Codex disallowed_tools"):
        resolve_codex_deprecated_args({"disallowed_tools": ["bash"]})


def test_deprecated_args_reject_unexpected_keyword() -> None:
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        resolve_codex_deprecated_args({"unexpected": True})


def test_codex_cli_config_overrides_format_values_for_cli() -> None:
    assert codex_cli_config_overrides("cached", False) == {
        "web_search": '"cached"',
        "features.goals": "false",
    }


def test_codex_mcp_server_toml_sets_approve_when_never() -> None:
    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    result = codex_mcp_server_toml(dump, "never")
    assert result == {
        "type": "http",
        "url": "http://localhost:8901/mcp/taiga-mcp",
        "default_tools_approval_mode": "approve",
    }


def test_codex_mcp_server_toml_leaves_other_policies_untouched() -> None:
    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    for policy in ("untrusted", "on-request"):
        assert codex_mcp_server_toml(dump, policy) == dump


def test_codex_mcp_server_toml_does_not_mutate_input() -> None:
    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    codex_mcp_server_toml(dump, "never")
    assert "default_tools_approval_mode" not in dump
