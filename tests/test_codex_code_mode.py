"""Unit tests for the Codex code-mode check on bridged tools (no Docker)."""

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.agent import BridgedToolsSpec
from inspect_swe._codex_cli import agentbinary
from inspect_swe._codex_cli import codex_cli as codex_cli_module
from inspect_swe._codex_cli.code_mode import check_codex_code_mode_bridged_tools

CATALOG: dict[str, Any] = {
    "models": [
        {"slug": "gpt-5.6-sol", "tool_mode": "code_mode_only"},
        {"slug": "gpt-5.6-direct", "tool_mode": "direct"},
        {"slug": "gpt-5.5"},
    ]
}


def _spec(name: str, require_proposal: bool = True) -> BridgedToolsSpec:
    return BridgedToolsSpec(name=name, tools=[], require_proposal=require_proposal)


def _check(
    model: str = "gpt-5.6-sol",
    overrides: dict[str, str] | None = None,
    specs: list[BridgedToolsSpec] | None = None,
    catalog: dict[str, Any] | None = CATALOG,
) -> None:
    check_codex_code_mode_bridged_tools(
        model, catalog, overrides, [_spec("host_tools")] if specs is None else specs
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


def test_code_mode_raises_for_a_suffixed_slug() -> None:
    """Codex resolves a dated or suffixed slug by longest prefix."""
    with pytest.raises(ValueError, match="in code mode"):
        _check("gpt-5.6-sol-2026-09-01")


def test_code_mode_passes_when_every_server_opts_out() -> None:
    _check(specs=[_spec("host_tools", False), _spec("search", False)])


def test_code_mode_passes_without_bridged_tools() -> None:
    _check(specs=[])


@pytest.mark.parametrize("model", ["gpt-5.6-direct", "gpt-5.5", "claude-sonnet-5"])
def test_direct_or_unlisted_models_do_not_raise(model: str) -> None:
    _check(model)


def test_unknown_release_catalog_does_not_raise() -> None:
    _check(catalog=None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"features.code_mode.direct_only_tool_namespaces": '["mcp__host_tools"]'},
        {"features.code_mode_only": "false"},
        {"features": "{goals = true}"},
        {"model_catalog_json": '"/tmp/catalog.json"'},
        {"profile": '"eval"'},
        {"profile.x": '"eval"'},
    ],
)
def test_configured_codex_is_not_checked(overrides: dict[str, str]) -> None:
    """A caller who configured Codex's tool or catalog settings is not checked."""
    _check(overrides=overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": '"gpt-5.6-sol"'},
        {"featuresx": "true"},
        {"model_catalog_json_old": '"x"'},
    ],
)
def test_other_overrides_are_still_checked(overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="in code mode"):
        _check(overrides=overrides)


def test_code_mode_ignores_specs_without_require_proposal() -> None:
    """inspect_ai releases before require_proposal have no proposal check."""

    @dataclass
    class LegacySpec:
        name: str

    _check(specs=[cast(BridgedToolsSpec, LegacySpec(name="host_tools"))])


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
    with (
        patch.object(agentbinary, "_read_cached_catalog", return_value=None),
        patch.object(agentbinary, "_fetch_models_catalog", fetch),
        patch.object(codex_cli_module, "trace"),
    ):

        async def launch() -> str:
            return await codex_cli_module._resolve_codex_model_checked(
                "mockllm/model", "gpt-5.6-sol", "0.160.1", None, specs
            )

        if fetched is not None and specs and specs[0].require_proposal:
            with pytest.raises(ValueError, match="in code mode"):
                anyio.run(launch)
        else:
            assert anyio.run(launch) == "gpt-5.6-sol"
    fetch.assert_awaited_once_with("0.160.1")
