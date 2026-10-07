"""Unit tests for the Codex code-mode check on bridged tools (no Docker)."""

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import Tool, ToolDef
from inspect_ai.util import SandboxEnvironment
from inspect_swe._codex_cli import agentbinary
from inspect_swe._codex_cli import codex_cli as codex_cli_module
from inspect_swe._codex_cli.code_mode import (
    check_codex_code_mode_bridged_tools,
    codex_config_overrides_tree,
    codex_model_catalog_json,
    read_codex_model_catalog,
)

CATALOG: dict[str, Any] = {
    "models": [
        {"slug": "gpt-5.6-sol", "tool_mode": "code_mode_only"},
        {"slug": "gpt-5.6-hybrid", "tool_mode": "code_mode"},
        {"slug": "gpt-5.6-direct", "tool_mode": "direct"},
        {"slug": "gpt-5.5"},
    ]
}

DIRECT_ONLY = "features.code_mode.direct_only_tool_namespaces"


def _spec(
    name: str, require_proposal: bool = True, tools: list[Tool] | None = None
) -> BridgedToolsSpec:
    return BridgedToolsSpec(
        name=name, tools=tools or [], require_proposal=require_proposal
    )


def _check(
    model: str = "gpt-5.6-sol",
    overrides: dict[str, str] | None = None,
    specs: list[BridgedToolsSpec] | None = None,
    catalog: dict[str, Any] | None = CATALOG,
    other_servers: list[str] | None = None,
) -> None:
    check_codex_code_mode_bridged_tools(
        model,
        catalog,
        overrides,
        [_spec("host_tools")] if specs is None else specs,
        other_servers or [],
    )


def test_code_mode_raises_for_servers_that_require_proposal() -> None:
    with pytest.raises(ValueError) as ex:
        _check(specs=[_spec("host_tools"), _spec("search"), _spec("open", False)])
    message = str(ex.value)
    assert "'gpt-5.6-sol'" in message
    assert 'tool_mode = "code_mode_only"' in message
    assert "'host_tools', 'search'" in message
    assert "'open'" not in message
    assert "require_proposal=False on the existing BridgedToolsSpec" in message


def test_code_mode_passes_when_every_server_opts_out() -> None:
    _check(specs=[_spec("host_tools", False), _spec("search", False)])


def test_code_mode_passes_without_bridged_tools() -> None:
    _check(specs=[])


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.6-hybrid", "gpt-5.6-direct"])
def test_non_code_mode_models_do_not_raise(model: str) -> None:
    _check(model)


def test_unknown_catalog_does_not_raise() -> None:
    """Without the installed binary's own catalog the mode is not established."""
    _check(catalog=None)
    _check(catalog=None, overrides={"features.code_mode_only": "true"})


def test_custom_catalog_decides_the_mode() -> None:
    """A caller's model_catalog_json replaces the release catalog in Codex."""
    direct = {"models": [{"slug": "gpt-5.6-sol", "tool_mode": "direct"}]}
    _check("gpt-5.6-sol", catalog=direct)
    code_mode = {"models": [{"slug": "gpt-5.5", "tool_mode": "code_mode_only"}]}
    with pytest.raises(ValueError, match="in code mode"):
        _check("gpt-5.5", catalog=code_mode)


@pytest.mark.parametrize(
    "overrides",
    [
        {"features.code_mode_only": "true"},
        {"features.code_mode_only": "true # enable code mode"},
        {"features": "{code_mode_only = true}"},
        {"features.code_mode_only": "false", "features": "{code_mode_only=true}"},
    ],
)
def test_code_mode_only_feature_raises(overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="features.code_mode_only is enabled"):
        _check("gpt-5.5", overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"features.code_mode_only": "false"},
        # a TOML string is not a boolean (Codex rejects the config)
        {"features.code_mode_only": '"true"'},
        {"features.code_mode_only": "true", "features": "{goals = true}"},
    ],
)
def test_code_mode_only_feature_off_does_not_raise(overrides: dict[str, str]) -> None:
    _check("gpt-5.5", overrides)


def test_catalog_tool_mode_wins_over_code_mode_only_feature() -> None:
    _check("gpt-5.6-direct", {"features.code_mode_only": "true"})


def test_selected_profile_does_not_raise() -> None:
    """A profile can change any setting the check reads."""
    _check(overrides={"profile": '"eval"'})


@pytest.mark.parametrize(
    "overrides",
    [
        {DIRECT_ONLY: '["mcp__host_tools"]'},
        {DIRECT_ONLY: "['mcp__host_tools']"},
        {DIRECT_ONLY: '["other", "mcp__host_tools",]'},
        {DIRECT_ONLY: '["mcp__host\\u005ftools"]'},
        {"features.code_mode": '{direct_only_tool_namespaces=["mcp__host_tools"]}'},
        {"features": '{code_mode = {direct_only_tool_namespaces=["mcp__host_tools"]}}'},
        # a boolean after the table only sets `enabled` on it
        {
            "features.code_mode": '{direct_only_tool_namespaces=["mcp__host_tools"]}',
            "features.code_mode.enabled": "true",
        },
    ],
)
def test_direct_only_namespace_skips_server(overrides: dict[str, str]) -> None:
    _check(overrides=overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        # Codex names the namespace mcp__host_tools by default
        {DIRECT_ONLY: '["host_tools"]'},
        # the comment is not part of the array
        {DIRECT_ONLY: '["other"] # "mcp__host_tools"]'},
        # a later parent table replaces the earlier dotted key
        {DIRECT_ONLY: '["mcp__host_tools"]', "features": "{code_mode_only = false}"},
    ],
)
def test_wrong_direct_only_namespace_raises(overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="'host_tools'"):
        _check(overrides=overrides)


def test_direct_only_namespace_is_sanitized() -> None:
    _check(overrides={DIRECT_ONLY: '["mcp__host_tools"]'}, specs=[_spec("host-tools")])


def test_direct_only_namespace_without_prefixes() -> None:
    non_prefixed = {"features.non_prefixed_mcp_tool_names": "true"}
    _check(overrides={**non_prefixed, DIRECT_ONLY: '["host_tools"]'})
    with pytest.raises(ValueError, match="'host_tools'"):
        _check(overrides={**non_prefixed, DIRECT_ONLY: '["mcp__host_tools"]'})


def test_direct_only_namespace_with_unprefixed_server_list() -> None:
    """With `server_names`, only the listed servers lose the `mcp__` prefix."""
    non_prefixed = {
        "features.non_prefixed_mcp_tool_names": '{enabled = true, server_names = ["host_tools"]}'
    }
    specs = [_spec("host_tools"), _spec("search")]
    _check(
        overrides={**non_prefixed, DIRECT_ONLY: '["host_tools", "mcp__search"]'},
        specs=specs,
    )
    with pytest.raises(ValueError, match="'host_tools', 'search'"):
        _check(
            overrides={**non_prefixed, DIRECT_ONLY: '["mcp__host_tools", "search"]'},
            specs=specs,
        )


@pytest.mark.parametrize(
    "other_servers,overrides",
    [
        # host-tools and host_tools both sanitize to mcp__host_tools
        (["host-tools"], {}),
        ([], {"mcp_servers.host-tools.command": '"server"'}),
    ],
)
def test_colliding_namespace_is_not_established(
    other_servers: list[str], overrides: dict[str, str]
) -> None:
    """Codex hashes colliding namespaces, so a listed name may not match."""
    _check(
        overrides={**overrides, DIRECT_ONLY: '["mcp__host_tools"]'},
        other_servers=other_servers,
    )
    # without a direct-only list there is nothing to match, so it still raises
    with pytest.raises(ValueError, match="'host_tools'"):
        _check(overrides=overrides, other_servers=other_servers)


def test_overlong_namespace_is_not_established() -> None:
    """Codex hashes a namespace whose tool names would exceed 128 characters."""

    async def execute() -> str:
        """Look something up."""
        return ""

    tool = ToolDef(execute, name="t" * 40).as_tool()
    name = "s" * 85  # mcp__ + 85 + __ + 40 = 132
    _check(
        overrides={DIRECT_ONLY: f'["mcp__{name}"]'},
        specs=[_spec(name, tools=[tool])],
    )
    with pytest.raises(ValueError):
        _check(specs=[_spec(name, tools=[tool])])


def test_code_mode_ignores_specs_without_require_proposal() -> None:
    """inspect_ai releases before require_proposal have no proposal check."""

    @dataclass
    class LegacySpec:
        name: str

    _check(specs=[cast(BridgedToolsSpec, LegacySpec(name="host_tools"))])


def test_config_overrides_tree_follows_codex() -> None:
    assert codex_config_overrides_tree(
        {
            "features.code_mode": "true",
            "features.code_mode.direct_only_tool_namespaces": '["a"]',
            "model": "gpt-5.5",
            "web_search": '"live"',
        }
    ) == {
        "features": {
            "code_mode": {"enabled": True, "direct_only_tool_namespaces": ["a"]}
        },
        # an unparsable value falls back to the raw string
        "model": "gpt-5.5",
        "web_search": "live",
    }


class _Sandbox:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.reads = 0

    async def read_file(self, file: str) -> str:
        self.reads += 1
        if file not in self.files:
            raise FileNotFoundError(file)
        return self.files[file]


def _read_catalog(path: Any, files: dict[str, str]) -> dict[str, Any] | None:
    sandbox = cast(SandboxEnvironment, _Sandbox(files))
    return anyio.run(read_codex_model_catalog, sandbox, path)


def test_model_catalog_json_from_config_overrides() -> None:
    assert codex_model_catalog_json(None) is None
    assert codex_model_catalog_json({"model": '"gpt-5.5"'}) is None
    overrides = {"model_catalog_json": '"/tmp/catalog.json"'}
    assert codex_model_catalog_json(overrides) == "/tmp/catalog.json"


def test_read_model_catalog() -> None:
    custom = '{"models": [{"slug": "gpt-5.6-sol", "tool_mode": "direct"}]}'
    assert _read_catalog("/tmp/catalog.json", {"/tmp/catalog.json": custom}) == {
        "models": [{"slug": "gpt-5.6-sol", "tool_mode": "direct"}]
    }


@pytest.mark.parametrize(
    "path,files",
    [
        ("catalog.json", {"catalog.json": "{}"}),
        (1, {}),
        ("/tmp/missing.json", {}),
        ("/tmp/catalog.json", {"/tmp/catalog.json": "not json"}),
        ("/tmp/catalog.json", {"/tmp/catalog.json": "[]"}),
    ],
)
def test_read_model_catalog_unknown_when_unreadable(
    path: Any, files: dict[str, str]
) -> None:
    assert _read_catalog(path, files) is None


@pytest.mark.parametrize(
    "specs",
    [None, [_spec("host_tools", False)], [_spec("host_tools")]],
    ids=["no-bridge", "opted-out", "proposal-required"],
)
@pytest.mark.parametrize(
    "fetched", [None, CATALOG], ids=["fetch-fails", "cache-unwritable"]
)
def test_launch_fetches_the_release_catalog_once(
    specs: list[BridgedToolsSpec] | None, fetched: dict[str, Any] | None
) -> None:
    """Alignment and the check share one fetch; nothing is cached in between."""
    fetch = AsyncMock(return_value=fetched)
    sandbox = _Sandbox({})
    with (
        patch.object(agentbinary, "_read_cached_catalog", return_value=None),
        patch.object(agentbinary, "_fetch_models_catalog", fetch),
        patch.object(codex_cli_module, "trace"),
    ):

        async def launch() -> str:
            return await codex_cli_module._resolve_codex_model_checked(
                cast(SandboxEnvironment, sandbox),
                "mockllm/model",
                "gpt-5.6-sol",
                "0.160.1",
                None,
                specs,
                None,
            )

        if fetched is not None and specs and specs[0].require_proposal:
            with pytest.raises(ValueError, match="in code mode"):
                anyio.run(launch)
        else:
            assert anyio.run(launch) == "gpt-5.6-sol"
    fetch.assert_awaited_once_with("0.160.1")
    assert sandbox.reads == 0


@pytest.mark.parametrize(
    "specs,reads",
    [(None, 0), ([_spec("host_tools", False)], 0), ([_spec("host_tools")], 1)],
)
def test_launch_reads_a_custom_catalog_only_when_checking(
    specs: list[BridgedToolsSpec] | None, reads: int
) -> None:
    custom = '{"models": [{"slug": "gpt-5.6-sol", "tool_mode": "direct"}]}'
    sandbox = _Sandbox({"/tmp/catalog.json": custom})
    with (
        patch.object(agentbinary, "_read_cached_catalog", return_value=CATALOG),
        patch.object(codex_cli_module, "trace"),
    ):
        slug = anyio.run(
            codex_cli_module._resolve_codex_model_checked,
            cast(SandboxEnvironment, sandbox),
            "mockllm/model",
            "gpt-5.6-sol",
            "0.160.1",
            {"model_catalog_json": '"/tmp/catalog.json"'},
            specs,
            None,
        )
    assert slug == "gpt-5.6-sol"
    assert sandbox.reads == reads


RELEASE_CATALOG: dict[str, Any] = {"models": [{"slug": "gpt-5.5", "priority": 0}]}
CUSTOM_CATALOG = '{"models": [{"slug": "gpt-5.9", "priority": 0}]}'


def _align(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    overrides: dict[str, str] | None,
    sandbox: _Sandbox,
    specs: list[BridgedToolsSpec] | None = None,
) -> str:
    """The `--model` slug a launch aligns an OpenAI model to."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with (
        patch.object(agentbinary, "_read_cached_catalog", return_value=RELEASE_CATALOG),
        patch.object(codex_cli_module, "trace"),
    ):
        slug = anyio.run(
            codex_cli_module._resolve_codex_model_checked,
            cast(SandboxEnvironment, sandbox),
            f"openai/{model}",
            None,
            "0.160.1",
            overrides,
            specs,
            None,
        )
    return slug


@pytest.mark.parametrize(
    "model,slug",
    [
        # only the custom catalog has it: its own entry
        ("gpt-5.9", "gpt-5.9"),
        # only the release catalog has it: aliased to the custom catalog's latest
        ("gpt-5.5", "gpt-5.9"),
    ],
)
def test_alignment_uses_a_custom_catalog(
    monkeypatch: pytest.MonkeyPatch, model: str, slug: str
) -> None:
    sandbox = _Sandbox({"/tmp/catalog.json": CUSTOM_CATALOG})
    overrides = {"model_catalog_json": '"/tmp/catalog.json"'}
    assert _align(monkeypatch, model, overrides, sandbox) == slug
    assert sandbox.reads == 1


@pytest.mark.parametrize("model,slug", [("gpt-5.9", "gpt-5.5"), ("gpt-5.5", "gpt-5.5")])
def test_alignment_uses_the_release_catalog_without_a_custom_one(
    monkeypatch: pytest.MonkeyPatch, model: str, slug: str
) -> None:
    sandbox = _Sandbox({})
    assert _align(monkeypatch, model, None, sandbox) == slug
    assert sandbox.reads == 0


@pytest.mark.parametrize(
    "path,files",
    [
        ('"catalog.json"', {"catalog.json": CUSTOM_CATALOG}),
        ('"/tmp/missing.json"', {}),
        ('"/tmp/catalog.json"', {"/tmp/catalog.json": "not json"}),
    ],
)
def test_alignment_falls_back_to_the_release_catalog_when_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
    files: dict[str, str],
) -> None:
    overrides = {"model_catalog_json": path}
    assert _align(monkeypatch, "gpt-5.9", overrides, _Sandbox(files)) == "gpt-5.5"
    assert "could not read model_catalog_json" in caplog.text


def test_alignment_and_check_share_one_custom_catalog_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = '{"models": [{"slug": "gpt-5.9", "tool_mode": "code_mode_only"}]}'
    sandbox = _Sandbox({"/tmp/catalog.json": custom})
    overrides = {"model_catalog_json": '"/tmp/catalog.json"'}
    with pytest.raises(ValueError, match="'gpt-5.9' in code mode"):
        _align(monkeypatch, "gpt-5.9", overrides, sandbox, [_spec("host_tools")])
    assert sandbox.reads == 1
