"""Unit tests for the Codex code-mode check on bridged tools (no Docker)."""

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import Tool, tool
from inspect_ai.util import SandboxEnvironment
from inspect_swe._codex_cli import agentbinary
from inspect_swe._codex_cli import codex_cli as codex_cli_module
from inspect_swe._codex_cli.code_mode import check_codex_code_mode_bridged_tools
from inspect_swe._util.bridged_tools import resolve_bridged_tools

CATALOG: dict[str, Any] = {
    "models": [
        {"slug": "gpt-5.6-sol", "tool_mode": "code_mode_only"},
        {"slug": "gpt-5.6-direct", "tool_mode": "direct"},
        {"slug": "gpt-5.5"},
    ]
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


def test_code_mode_raises_for_plain_tools() -> None:
    """Plain Tools resolve to the implicit `tools` server, which requires proposal."""

    @tool
    def lookup() -> Tool:
        async def execute(key: str) -> str:
            """Look up a value by key.

            Args:
                key: The key to look up.
            """
            return key

        return execute

    with pytest.raises(ValueError) as ex:
        _check(specs=resolve_bridged_tools([lookup()]))
    message = str(ex.value)
    assert "bridged server(s) 'tools'" in message
    assert "wrap them in a BridgedToolsSpec" in message


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
        {" model_catalog_json ": '"/tmp/catalog.json"'},
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
                cast(SandboxEnvironment, _Sandbox({})),
                "mockllm/model",
                "gpt-5.6-sol",
                "0.160.1",
                None,
                specs,
            )

        if fetched is not None and specs and specs[0].require_proposal:
            with pytest.raises(ValueError, match="in code mode"):
                anyio.run(launch)
        else:
            assert anyio.run(launch) == "gpt-5.6-sol"
    fetch.assert_awaited_once_with("0.160.1")


RELEASE_CATALOG: dict[str, Any] = {"models": [{"slug": "gpt-5.5", "priority": 0}]}
CUSTOM_CATALOG = '{"models": [{"slug": "gpt-5.9", "priority": 0}]}'


def _align(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    overrides: dict[str, str] | None,
    sandbox: _Sandbox,
    model_config: str | None = None,
) -> str:
    """The `--model` slug a launch aligns an OpenAI model to."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with (
        patch.object(agentbinary, "_read_cached_catalog", return_value=RELEASE_CATALOG),
        patch.object(codex_cli_module, "trace"),
    ):
        return anyio.run(
            codex_cli_module._resolve_codex_model_checked,
            cast(SandboxEnvironment, sandbox),
            f"openai/{model}",
            model_config,
            "0.160.1",
            overrides,
            None,
        )


@pytest.mark.parametrize(
    "model,slug",
    [
        # only the custom catalog has it: its own entry
        ("gpt-5.9", "gpt-5.9"),
        # only the release catalog has it: aliased to the custom catalog's latest
        ("gpt-5.5", "gpt-5.9"),
    ],
)
@pytest.mark.parametrize("path", ['"/tmp/catalog.json"', "/tmp/catalog.json"])
@pytest.mark.parametrize("key", ["model_catalog_json", " model_catalog_json "])
def test_alignment_uses_a_custom_catalog(
    monkeypatch: pytest.MonkeyPatch, model: str, slug: str, path: str, key: str
) -> None:
    sandbox = _Sandbox({"/tmp/catalog.json": CUSTOM_CATALOG})
    overrides = {key: path}
    assert _align(monkeypatch, model, overrides, sandbox) == slug
    assert sandbox.reads == 1


@pytest.mark.parametrize(
    "path,file",
    [
        ("/tmp/catalog.json", "/tmp/catalog.json"),
        ('"/tmp/catalog.json"', "/tmp/catalog.json"),
        ("'/tmp/catalog.json'", "/tmp/catalog.json"),
        (" /tmp/catalog.json ", "/tmp/catalog.json"),
        (' "/tmp/my catalog.json" ', "/tmp/my catalog.json"),
        # quotes keep the spaces inside them
        ('"/tmp/catalog.json "', "/tmp/catalog.json "),
    ],
)
def test_alignment_reads_the_file_codex_reads(
    monkeypatch: pytest.MonkeyPatch, path: str, file: str
) -> None:
    decoy = '{"models": [{"slug": "gpt-5.7", "priority": 0}]}'
    candidates = ["/tmp/catalog.json", "/tmp/catalog.json ", file, path.strip()]
    sandbox = _Sandbox(
        {name: CUSTOM_CATALOG if name == file else decoy for name in candidates}
    )
    overrides = {"model_catalog_json": path}
    assert _align(monkeypatch, "gpt-5.5", overrides, sandbox) == "gpt-5.9"
    assert sandbox.reads == 1


@pytest.mark.parametrize("model,slug", [("gpt-5.9", "gpt-5.5"), ("gpt-5.5", "gpt-5.5")])
def test_alignment_uses_the_release_catalog_without_a_custom_one(
    monkeypatch: pytest.MonkeyPatch, model: str, slug: str
) -> None:
    sandbox = _Sandbox({})
    assert _align(monkeypatch, model, None, sandbox) == slug
    assert sandbox.reads == 0


def test_alignment_skips_the_custom_catalog_with_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox({"/tmp/catalog.json": CUSTOM_CATALOG})
    overrides = {"model_catalog_json": '"/tmp/catalog.json"'}
    assert _align(monkeypatch, "gpt-5.9", overrides, sandbox, "gpt-5.5") == "gpt-5.5"
    assert sandbox.reads == 0


@pytest.mark.parametrize(
    "path,files",
    [
        ('"catalog.json"', {"catalog.json": CUSTOM_CATALOG}),
        ('"/tmp/missing.json"', {}),
        ('"/tmp/catalog.json"', {"/tmp/catalog.json": "not json"}),
        ('"/tmp/catalog.json"', {"/tmp/catalog.json": "[]"}),
        ('"/tmp/catalog.json"', {"/tmp/catalog.json": "{}"}),
        ('"/tmp/catalog.json"', {"/tmp/catalog.json": '{"models": "x"}'}),
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
    assert path.strip('"') in caplog.text


@pytest.mark.parametrize(
    "path",
    [
        # TOML values that are not strings
        "1",
        "true",
        '["/tmp/catalog.json"]',
        '{ path = "/tmp/catalog.json" }',
    ],
)
def test_alignment_falls_back_for_values_that_name_no_file(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, path: str
) -> None:
    """No file is guessed for these values, even if one they could name exists."""
    names = ["/tmp/catalog.json", path.strip("\"'"), path[1:-1]]
    sandbox = _Sandbox({name: CUSTOM_CATALOG for name in names})
    overrides = {"model_catalog_json": path}
    assert _align(monkeypatch, "gpt-5.9", overrides, sandbox) == "gpt-5.5"
    assert "could not read model_catalog_json" in caplog.text


@pytest.mark.parametrize(
    "path,decoded",
    [
        # escapes and backslashes
        (r'"/tmp/\u0063atalog.json"', "/tmp/catalog.json"),
        (r'"/tmp/\x63atalog.json"', "/tmp/catalog.json"),
        (r"'/tmp/c\d.json'", r"/tmp/c\d.json"),
        # quotes inside, unmatched or tripled
        ('"/tmp/a\'b.json"', "/tmp/a'b.json"),
        ("/tmp/catalog.json'", "/tmp/catalog.json"),
        ("\"/tmp/catalog.json'", "/tmp/catalog.json"),
        ('"/tmp/catalog.json" # catalog', "/tmp/catalog.json"),
        ("'''/tmp/catalog.json'''", "/tmp/catalog.json"),
        ('"""\n/tmp/catalog.json"""', "/tmp/catalog.json"),
        # control characters, which Codex and Python trim differently
        ('"/tmp/a\tb.json"', "/tmp/a\tb.json"),
        ("/tmp/catalog.json\x1c", "/tmp/catalog.json\x1c"),
        ("'/tmp/a\x7fb.json'", "/tmp/a\x7fb.json"),
        # whitespace Codex trims from a bare path
        ("/tmp/catalog.json ", "/tmp/catalog.json"),
    ],
)
def test_alignment_refuses_values_that_are_not_a_plain_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
    decoded: str,
) -> None:
    """No file is read for these, even one Codex would read."""
    names = ["/tmp/catalog.json", decoded, path.strip(), path.strip("\"'")]
    sandbox = _Sandbox({name: CUSTOM_CATALOG for name in names})
    overrides = {"model_catalog_json": path}
    assert _align(monkeypatch, "gpt-5.9", overrides, sandbox) == "gpt-5.5"
    assert "reads only a plain absolute path, optionally quoted" in caplog.text
    assert sandbox.reads == 0


def test_alignment_falls_back_for_a_dotted_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A dotted key sets a table, not a file path."""
    sandbox = _Sandbox({"/tmp/catalog.json": CUSTOM_CATALOG})
    overrides = {"model_catalog_json.path": '"/tmp/catalog.json"'}
    assert _align(monkeypatch, "gpt-5.9", overrides, sandbox) == "gpt-5.5"
    assert "could not read model_catalog_json" in caplog.text
    assert sandbox.reads == 0


def test_alignment_passes_an_empty_custom_catalog_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty models list is Codex's to reject; alignment does not replace it."""
    sandbox = _Sandbox({"/tmp/catalog.json": '{"models": []}'})
    overrides = {"model_catalog_json": '"/tmp/catalog.json"'}
    assert _align(monkeypatch, "gpt-5.9", overrides, sandbox) == "gpt-5.9"
