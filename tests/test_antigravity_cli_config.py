"""Unit tests for the Antigravity CLI agent's on-disk configuration.

Both builders encode behaviour that was measured against the real `agy` binary
(1.1.20) rather than read off the documentation, and in both cases getting it
wrong fails SILENTLY -- the CLI keeps running with the setting or the server
ignored. That is what these tests exist to catch. The eager-tool marking is
read off 1.1.27's own config struct and changelog rather than the (silent on
it) published schema; that the CLI then declares the tool natively is proven
by the CLI-level test, not here.

The binary source is covered here too -- which release archive is installed,
what it is verified against, and under what name it lands. None of that needs
Docker or the network: the GitHub release lookup is the only thing stubbed.
"""

import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.tool import Tool, ToolDef, tool
from inspect_ai.tool._mcp._config import (
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
)
from inspect_swe._antigravity_cli import agentbinary
from inspect_swe._antigravity_cli.antigravity_cli import (
    _native_conversation_id,
    _NativeConversation,
    build_antigravity_agent_env,
    build_antigravity_mcp_config,
    build_antigravity_settings,
)
from inspect_swe._util.agentbinary import AgentBinaryVersion
from inspect_swe._util.sandbox import SandboxPlatform


def _settings(unattended: bool = True) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(
        build_antigravity_settings(unattended=unattended)
    )
    return parsed


def test_settings_select_the_direct_gemini_api_route() -> None:
    # the load-bearing key: without it the CLI blocks on OAuth sign-in and
    # never reads GEMINI_API_KEY / GOOGLE_GEMINI_BASE_URL at all.
    assert _settings()["modelProvider"] == "gemini"


def test_centaur_settings_withhold_the_approval_policies() -> None:
    # The human at the terminal is the approver in centaur mode, which is the
    # whole reason --dangerously-skip-permissions is withheld from the command
    # they are handed. A persisted settings file that auto-approves everything
    # would take that back silently, from a file they never see.
    centaur = _settings(unattended=False)

    assert "toolPermission" not in centaur
    assert "artifactReviewPolicy" not in centaur
    # Withheld, not "no settings were written": the direct-Gemini route and the
    # rest of the headless hygiene still apply to whatever the human launches,
    # so the CLI still reaches the bridge rather than a sign-in page.
    assert centaur["modelProvider"] == "gemini"
    assert centaur["enableTelemetry"] is False
    assert centaur["altScreenMode"] == "never"

    # ...and those two keys are the ONLY difference between the modes. An
    # unattended run has nobody to answer a prompt, so it keeps both.
    unattended = _settings()
    assert set(unattended) - set(centaur) == {
        "toolPermission",
        "artifactReviewPolicy",
    }
    assert all(centaur[key] == unattended[key] for key in centaur)


@pytest.mark.parametrize(
    "key",
    ["enableTelemetry", "showTips", "showFeedbackSurvey", "enableTerminalSandbox"],
)
def test_boolean_settings_are_json_booleans(key: str) -> None:
    # These four are documented with "on"/"off" wording, but the CLI only
    # accepts real booleans: a string is dropped on load with no error and
    # without rewriting settings.json, so the file reads "off" while /config
    # still reports the default (telemetry ON).
    value = _settings()[key]
    assert isinstance(value, bool), f"{key} must be a JSON boolean, got {value!r}"
    assert value is False


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("toolPermission", "always-proceed"),
        ("artifactReviewPolicy", "always-proceed"),
        ("altScreenMode", "never"),
    ],
)
def test_enum_settings_use_the_cli_vocabulary(key: str, expected: str) -> None:
    assert _settings()[key] == expected


def test_agent_env_defaults_have_no_caller_env() -> None:
    # Byte-identical baseline: with no caller `env`, the merged environment is
    # exactly the five wrapper-owned keys.
    result = build_antigravity_agent_env(bridge_port=3000, sandbox_home="/root")
    assert result == {
        "GOOGLE_GEMINI_BASE_URL": "http://localhost:3000",
        "GEMINI_API_KEY": "api-key",
        "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
        "AGY_CLI_HIDE_LOGO": "1",
        "HOME": "/root",
    }


def test_caller_env_cannot_redirect_the_bridge_or_inject_a_credential() -> None:
    # Credential boundary: a caller passing a full environment snapshot (the
    # realistic `env=os.environ.copy()` case) must not be able to silently
    # point the CLI at Google's real endpoint or supply a real Google
    # credential in place of the bridge's placeholder.
    caller_env = {
        "GOOGLE_GEMINI_BASE_URL": "https://generativelanguage.googleapis.com",
        "GEMINI_API_KEY": "real-google-api-key",
    }
    result = build_antigravity_agent_env(
        bridge_port=3000, sandbox_home="/root", env=caller_env
    )
    assert result["GOOGLE_GEMINI_BASE_URL"] == "http://localhost:3000"
    assert result["GEMINI_API_KEY"] == "api-key"


def test_caller_env_still_overrides_the_cosmetic_defaults() -> None:
    # Everything that is NOT the credential boundary keeps the established
    # repo-wide "caller wins" contract (matches `claude_code_agent_env`):
    # a caller may still override HOME or opt back into the auto-updater.
    result = build_antigravity_agent_env(
        bridge_port=3000,
        sandbox_home="/root",
        env={"HOME": "/home/custom", "AGY_CLI_DISABLE_AUTO_UPDATE": "false"},
    )
    assert result["HOME"] == "/home/custom"
    assert result["AGY_CLI_DISABLE_AUTO_UPDATE"] == "false"


def test_http_mcp_servers_use_server_url() -> None:
    # The CLI's remote schema is `serverUrl`. `url`/`httpUrl` are accepted by the
    # JSON parser and then ignored, which presents as a configured server that
    # never connects.
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigHTTP(
                    type="http",
                    name="remote-tools",
                    url="http://localhost:8901/mcp",
                )
            ],
            eager_tools={},
        )
    )
    server = config["mcpServers"]["remote-tools"]
    assert server["serverUrl"] == "http://localhost:8901/mcp"
    assert "url" not in server
    assert "httpUrl" not in server


def test_stdio_mcp_servers_keep_command_and_args() -> None:
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigStdio(
                    type="stdio", name="local", command="server", args=["--flag"]
                )
            ],
            eager_tools={},
        )
    )
    server = config["mcpServers"]["local"]
    assert server["command"] == "server"
    assert server["args"] == ["--flag"]


def test_stdio_cwd_is_written_as_a_string() -> None:
    # `MCPServerConfigStdio.cwd` accepts a Path and `json.dumps` refuses one, so
    # an unconverted Path is not a bad registry but no registry at all: the
    # write raises and the run dies before the CLI is ever launched.
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigStdio(
                    type="stdio",
                    name="local",
                    command="server",
                    cwd=Path("/srv/mcp"),
                )
            ],
            eager_tools={},
        )
    )

    assert config["mcpServers"]["local"]["cwd"] == str(Path("/srv/mcp"))


def test_no_mcp_servers_still_writes_an_empty_registry() -> None:
    # The CLI reads the file unconditionally; an absent `mcpServers` object is a
    # parse error it reports as a broken configuration.
    empty = build_antigravity_mcp_config([], eager_tools={})
    assert json.loads(empty) == {"mcpServers": {}}


# --- eager bridged tools ----------------------------------------------------
#
# `agy` loads MCP tools LAZILY by default: it declares a single native
# dispatcher whose own description reads "Call a lazy-loaded MCP tool. Read the
# tool's schema file to understand the tool's arguments and usage", and routes
# every MCP tool through it. A host-bridged Inspect tool therefore never
# appears as a tool the model can call by name, which is what the rest of the
# CLI wrappers give it.
#
# The CLI's own per-server config carries the switch: a `tools` map whose
# values hold an `eager` boolean (1.1.27 config struct: `Eager *bool
# "json:\"eager,omitempty,omitzero\""` beside `Background`, inside the entry's
# `tools` map; named in the shipped changelog as "`tools.eager` from
# `mcp_config.json`"). An eager tool is "registered as native tools under the
# name `%s`. Call eager tools directly."
#
# The map key is the tool name the SERVER serves -- the namespace the
# documented `disabledTools` list and the `mcp(server/tool)` permission syntax
# also use. The model-facing `mcp_<server>_<tool>` name (`mcp_secrets_
# secret_lookup` for gemini_cli, `mcp_chrome_devtools_new_page` in agy's own
# browser server) is derived by the CLI, so a key written in that spelling
# marks nothing and silently leaves the tool lazy.
#
# Bridged servers are the only ones marked: a user's own MCP server keeps the
# CLI's native lazy behaviour, since Inspect's MCPServerConfig has no way to
# ask for anything else.

_BRIDGE_URL = "http://localhost:3001/mcp/inspect-tools"


def _servers(config: str) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(config)["mcpServers"]
    return parsed


@tool
def secret_lookup() -> Tool:
    async def execute(name: str) -> str:
        """Look up a secret.

        Args:
            name: Secret to look up.
        """
        return name

    return execute


def test_bridged_tools_are_marked_eager_under_their_own_server() -> None:
    servers = _servers(
        build_antigravity_mcp_config(
            [MCPServerConfigHTTP(type="http", name="inspect-tools", url=_BRIDGE_URL)],
            eager_tools={"inspect-tools": ["bash", "submit"]},
        )
    )

    assert servers["inspect-tools"]["tools"] == {
        "bash": {"eager": True},
        "submit": {"eager": True},
    }


def test_eager_marking_accepts_the_bridge_registry_verbatim() -> None:
    # `SandboxAgentBridge.bridged_tools` is dict[server, dict[tool, Tool]], and
    # its keys are exactly the names the bridge's own `tools/list` serves
    # (`ToolDef(tool).name`), so it is passed through with no transformation.
    registry = {"inspect-tools": {ToolDef(secret_lookup()).name: secret_lookup()}}

    servers = _servers(
        build_antigravity_mcp_config(
            [MCPServerConfigHTTP(type="http", name="inspect-tools", url=_BRIDGE_URL)],
            eager_tools=registry,
        )
    )

    assert servers["inspect-tools"]["tools"] == {"secret_lookup": {"eager": True}}


def test_eager_tool_names_are_written_verbatim() -> None:
    # No prefixing and no rewriting: the CLI matches these against what the
    # server serves, and derives the native `mcp_<server>_<tool>` name itself.
    servers = _servers(
        build_antigravity_mcp_config(
            [MCPServerConfigHTTP(type="http", name="inspect-tools", url=_BRIDGE_URL)],
            eager_tools={"inspect-tools": ["read-file", "web.search", "run_command"]},
        )
    )

    assert set(servers["inspect-tools"]["tools"]) == {
        "read-file",
        "web.search",
        "run_command",
    }


def test_authenticated_http_server_is_rejected() -> None:
    """Credential boundary.

    `MCPServerConfigHTTP.headers` (e.g. an Authorization bearer token) would
    otherwise be written verbatim into `$HOME/.gemini/config/mcp_config.json`
    inside the sandbox -- a file the evaluated CLI, and any of its own tools,
    can read back out at any point during the run.
    `build_antigravity_mcp_config` must refuse a caller-supplied server
    carrying headers rather than persist a real credential where the
    sandboxed agent can read it.
    """
    with pytest.raises(ValueError, match="bridged_tools"):
        build_antigravity_mcp_config(
            [
                MCPServerConfigHTTP(
                    type="http",
                    name="inspect-tools",
                    url=_BRIDGE_URL,
                    headers={"Authorization": "Bearer token"},
                )
            ],
            eager_tools={"inspect-tools": ["submit"]},
        )


def test_authenticated_stdio_server_is_rejected() -> None:
    """Same credential boundary as the HTTP case above, for stdio transport.

    `MCPServerConfigStdio.env` would land verbatim in the same
    sandbox-readable `mcp_config.json`, so a real credential passed through
    it must be refused rather than persisted where the sandboxed agent can
    read it back out.
    """
    with pytest.raises(ValueError, match="bridged_tools"):
        build_antigravity_mcp_config(
            [
                MCPServerConfigStdio(
                    type="stdio",
                    name="local",
                    command="server",
                    env={"API_TOKEN": "secret"},
                )
            ],
            eager_tools={},
        )


def test_user_supplied_servers_keep_the_cli_default() -> None:
    # Nothing the caller passes in `mcp_servers` is bridged, so nothing there
    # is marked: those servers behave exactly as the CLI configures them.
    servers = _servers(
        build_antigravity_mcp_config(
            [
                MCPServerConfigStdio(
                    type="stdio", name="local", command="server", args=["--flag"]
                ),
                MCPServerConfigHTTP(
                    type="http", name="remote", url="https://mcp.example.com/mcp"
                ),
                MCPServerConfigHTTP(type="http", name="inspect-tools", url=_BRIDGE_URL),
            ],
            eager_tools={"inspect-tools": ["submit"]},
        )
    )

    assert "tools" not in servers["local"]
    assert "tools" not in servers["remote"]
    assert servers["local"]["command"] == "server"
    assert servers["remote"]["serverUrl"] == "https://mcp.example.com/mcp"
    assert servers["inspect-tools"]["tools"] == {"submit": {"eager": True}}


def test_inspect_tool_filter_never_becomes_the_cli_tools_map() -> None:
    # Inspect's own `tools` field is a filter (`"all"` or a list). The CLI's
    # `tools` is a map of per-tool options, so a list landing there is a schema
    # mismatch the CLI drops on load, taking the eager marking with it.
    servers = _servers(
        build_antigravity_mcp_config(
            [
                MCPServerConfigHTTP(
                    type="http",
                    name="remote",
                    url="https://mcp.example.com/mcp",
                    tools=["lookup"],
                )
            ],
            eager_tools={},
        )
    )

    assert "tools" not in servers["remote"]


def test_bridged_server_with_no_tools_gets_no_tools_key() -> None:
    # A zero-tool bridged server is a valid configuration (the readiness gate
    # skips probing one); an empty `tools` map would claim otherwise.
    servers = _servers(
        build_antigravity_mcp_config(
            [MCPServerConfigHTTP(type="http", name="inspect-tools", url=_BRIDGE_URL)],
            eager_tools={"inspect-tools": []},
        )
    )

    assert "tools" not in servers["inspect-tools"]


# --- binary source ----------------------------------------------------------


def test_binary_source_installs_under_the_agy_name() -> None:
    source = agentbinary.antigravity_cli_binary_source()
    # The release tarball's single member is named `antigravity`; the sandbox
    # binary (and the `version="sandbox"` probe) is `agy`.
    assert source.binary == "agy"


def test_the_downloaded_archive_is_unpacked_to_its_single_member() -> None:
    # The published asset is a tar.gz whose one member is the executable, so
    # something has to unpack it on the way in: installed as downloaded, the
    # sandbox writes an archive to the binary path and chmods it, and every
    # launch dies with an exec-format error. Asserted as behaviour -- these
    # bytes in, that member's bytes out -- rather than as which helper happens
    # to be wired up, since a replacement that unpacks correctly is fine and a
    # helper that stops unpacking is not.
    executable = b"\x7fELF\x02\x01\x01" + b"agy" * 64
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo(name="antigravity")
        member.size = len(executable)
        tar.addfile(member, BytesIO(executable))
    packed = archive.getvalue()

    source = agentbinary.antigravity_cli_binary_source()

    assert source.post_download is not None, "the archive would install as-is"
    assert source.post_download(packed) == executable


def test_binary_source_rejects_musl_platforms() -> None:
    # Only glibc assets are published. Installing the glibc binary on a musl
    # image fails at exec time with an opaque loader error, so refuse up front.
    source = agentbinary.antigravity_cli_binary_source()
    with pytest.raises(ValueError, match="Unsupported platform"):
        anyio.run(source.resolve_version, "1.1.20", "linux-x64-musl")


# --- release resolution -----------------------------------------------------
#
# `resolve_version` decides which archive is installed and what it is checked
# against, and every mistake it can make lands somewhere other than here: the
# wrong architecture dies at exec time inside the sandbox, an unverified digest
# is a download nothing ever checks, and a floating version left unresolved
# names a cache entry no later run can hit. The release lookup is stubbed;
# nothing else is.

_RELEASE: dict[str, Any] = {
    "assets": [
        {
            "name": "agy_cli_linux_x64.tar.gz",
            "digest": "sha256:1111",
            "browser_download_url": "https://example.com/agy_cli_linux_x64.tar.gz",
        },
        {
            "name": "agy_cli_linux_arm64.tar.gz",
            "digest": "sha256:2222",
            "browser_download_url": "https://example.com/agy_cli_linux_arm64.tar.gz",
        },
    ]
}


def _resolve(
    platform: SandboxPlatform,
    release: dict[str, Any] | None = None,
    version: str = "1.1.27",
) -> AgentBinaryVersion:
    source = agentbinary.antigravity_cli_binary_source()
    with patch.object(
        agentbinary, "_fetch_release", AsyncMock(return_value=release or _RELEASE)
    ):
        return anyio.run(source.resolve_version, version, platform)


@pytest.mark.parametrize(
    ("platform", "asset", "checksum"),
    [
        ("linux-x64", "agy_cli_linux_x64.tar.gz", "1111"),
        ("linux-arm64", "agy_cli_linux_arm64.tar.gz", "2222"),
    ],
)
def test_each_architecture_resolves_to_its_own_asset(
    platform: SandboxPlatform, asset: str, checksum: str
) -> None:
    resolved = _resolve(platform)

    assert resolved.download_url.endswith(asset)
    # The digest of THAT asset, with GitHub's `sha256:` prefix stripped: the
    # installer compares this against the archive's own sha256, so a prefix
    # left on matches nothing, and a digest read off the wrong asset condemns
    # one of the two architectures to a checksum failure on every download.
    assert resolved.expected_checksum == checksum
    assert resolved.version == "1.1.27"
    # A single binary, never a package archive: this source defines no
    # package_entrypoint, and claiming one raises during install.
    assert resolved.package is False


def test_a_release_missing_the_platform_asset_is_rejected() -> None:
    # e.g. a release that shipped x64 only. Naming the missing asset is the
    # point: the alternative is an install that reports a missing binary from
    # inside the sandbox, one layer away from the release that lacks it.
    x64_only: dict[str, Any] = {"assets": [_RELEASE["assets"][0]]}

    with pytest.raises(RuntimeError, match="No asset agy_cli_linux_arm64.tar.gz"):
        _resolve("linux-arm64", release=x64_only)


@pytest.mark.parametrize("digest", ["", "md5:1111", "1111"])
def test_a_release_without_a_sha256_digest_is_rejected(digest: str) -> None:
    # Older releases carry no `digest` field at all. Accepting one would install
    # the archive unverified -- a checksum check that passes by having nothing
    # to compare is worse than none, because it still reads as verified.
    release: dict[str, Any] = {"assets": [{**_RELEASE["assets"][0], "digest": digest}]}

    with pytest.raises(RuntimeError, match="Invalid digest"):
        _resolve("linux-x64", release=release)


@pytest.mark.parametrize("requested", ["stable", "latest"])
def test_a_floating_version_resolves_to_a_concrete_release(requested: str) -> None:
    # The concrete version is what the release lookup, the cache filename and
    # the sandbox install path are all built from. Left as "latest" it asks
    # GitHub for a release tagged "latest" and caches under a name no pinned
    # run can ever match.
    source = agentbinary.antigravity_cli_binary_source()
    with (
        patch.object(
            agentbinary, "_fetch_latest_version", AsyncMock(return_value="1.1.27")
        ),
        patch.object(
            agentbinary, "_fetch_release", AsyncMock(return_value=_RELEASE)
        ) as fetch_release,
    ):
        resolved = anyio.run(source.resolve_version, requested, "linux-x64")

    assert resolved.version == "1.1.27"
    fetch_release.assert_awaited_once_with("1.1.27")


# --- native conversation identity -------------------------------------------
#
# A title-generation request can displace the native task's canonical
# `AgentState`, and for a single-call task whose title arrives last there is no
# continued main loop left to recover from. The CLI's own declaration is what
# separates them: official 1.1.27/1.1.28 primary requests carry exactly one
# system `<user_information>` block containing `Conversation ID: <UUID>`, and
# title requests carry no such block at all.
#
# This is deliberately NOT a title-text or tool-presence classifier. Those
# guess; this reads the identity the CLI states. And because the wrapper cannot
# preassign a UUID -- an unknown-UUID `--conversation` probe warns and creates
# a different one -- the parser must never invent or normalise an ID: anything
# it cannot read unambiguously raises instead.

_CID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
_OTHER_CID = "16fd2706-8baf-433b-82eb-8c7fada847da"


def _block(body: str) -> str:
    return f"<user_information>\n{body}\n</user_information>"


def _native_system(cid: str = _CID) -> ChatMessageSystem:
    """A primary request's system message, block embedded in other content."""
    return ChatMessageSystem(
        content="\n".join(
            [
                "<identity>You are a coding assistant.</identity>",
                _block(f"Conversation ID: {cid}"),
                "<rules>Be brief.</rules>",
            ]
        )
    )


def _primary(cid: str = _CID) -> list[ChatMessage]:
    return [_native_system(cid), ChatMessageUser(content="Do the task.")]


def _auxiliary() -> list[ChatMessage]:
    """The title generator's request: its own system prompt, and no block."""
    return [
        ChatMessageSystem(content="You are a conversation title generator"),
        ChatMessageUser(content="Do the task."),
    ]


def test_a_primary_request_yields_its_conversation_id() -> None:
    assert _native_conversation_id(_primary()) == _CID


def test_the_id_is_read_from_the_block_not_the_whole_message() -> None:
    # The block sits inside a much larger system prompt, and a bare UUID can
    # appear elsewhere in it (a task prompt, a path, a pasted log). Only the
    # block's own line counts.
    messages: list[ChatMessage] = [
        _native_system(),
        ChatMessageUser(content=f"Compare against run {_OTHER_CID} please."),
    ]
    assert _native_conversation_id(messages) == _CID


def test_an_auxiliary_request_has_no_conversation_id() -> None:
    # No block is the positive signal for "not the native task", not an error.
    assert _native_conversation_id(_auxiliary()) is None


def test_a_user_message_block_is_not_a_native_declaration() -> None:
    # Only SYSTEM messages carry the CLI's declaration. Honouring one from
    # user content would let task text nominate the canonical conversation.
    messages: list[ChatMessage] = [
        ChatMessageSystem(content="You are a conversation title generator"),
        ChatMessageUser(content=_block(f"Conversation ID: {_CID}")),
    ]
    assert _native_conversation_id(messages) is None


def test_a_malformed_uuid_raises_rather_than_being_accepted() -> None:
    # Accepting a non-UUID would bind the run to an identity the CLI's final
    # result can never match, turning a parse bug into a scoring failure two
    # layers away.
    with pytest.raises(ValueError):
        _native_conversation_id(_primary("not-a-uuid"))


def test_a_native_block_without_an_id_raises() -> None:
    # The block is present, so this IS a primary request -- treating it as
    # auxiliary would silently exclude the real task from canonical state.
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=_block("Workspace: /repo")),
        ChatMessageUser(content="Do the task."),
    ]
    with pytest.raises(ValueError):
        _native_conversation_id(messages)


def test_two_blocks_in_one_message_are_ambiguous_and_raise() -> None:
    content = "\n".join(
        [_block(f"Conversation ID: {_CID}"), _block(f"Conversation ID: {_OTHER_CID}")]
    )
    with pytest.raises(ValueError):
        _native_conversation_id([ChatMessageSystem(content=content)])


def test_two_native_system_messages_are_ambiguous_and_raise() -> None:
    messages: list[ChatMessage] = [_native_system(_CID), _native_system(_OTHER_CID)]
    with pytest.raises(ValueError):
        _native_conversation_id(messages)


# --- binding the native conversation ----------------------------------------
#
# `_NativeConversation.accept` is what the agent hands to the bridge as
# `state_filter`, so its signature is the core contract's:
# `Callable[[Sequence[ChatMessage]], bool]`.


def test_unattended_binds_the_first_native_conversation() -> None:
    conversation = _NativeConversation(unattended=True)
    assert conversation.bound_id is None

    assert conversation.accept(_primary()) is True
    assert conversation.bound_id == _CID
    # The same conversation keeps being canonical across its whole invocation.
    assert conversation.accept(_primary()) is True


def test_unattended_rejects_a_second_native_conversation() -> None:
    # One CLI invocation, one canonical conversation. A second native UUID is
    # not the task the run is scoring, and the binding must not follow it.
    conversation = _NativeConversation(unattended=True)
    assert conversation.accept(_primary(_CID)) is True
    assert conversation.accept(_primary(_OTHER_CID)) is False
    assert conversation.bound_id == _CID


def test_an_auxiliary_request_never_binds_and_is_never_canonical() -> None:
    # The defect in one line: the title request arrives first, and must neither
    # become canonical nor consume the binding the primary request needs.
    conversation = _NativeConversation(unattended=True)
    assert conversation.accept(_auxiliary()) is False
    assert conversation.bound_id is None

    assert conversation.accept(_primary()) is True
    assert conversation.bound_id == _CID


def test_centaur_accepts_every_native_conversation() -> None:
    # The human may start or resume several conversations in one session, and
    # the existing internal accumulation contract preserves them all.
    conversation = _NativeConversation(unattended=False)
    assert conversation.accept(_primary(_CID)) is True
    assert conversation.accept(_primary(_OTHER_CID)) is True


def test_centaur_still_excludes_auxiliary_requests() -> None:
    conversation = _NativeConversation(unattended=False)
    assert conversation.accept(_auxiliary()) is False


@pytest.mark.parametrize("unattended", [True, False])
def test_an_unreadable_declaration_propagates_out_of_the_filter(
    unattended: bool,
) -> None:
    # The core contract propagates a callback exception rather than swallowing
    # it, so a run against a CLI whose declaration has drifted fails loudly
    # instead of quietly scoring the wrong conversation.
    conversation = _NativeConversation(unattended=unattended)
    with pytest.raises(ValueError):
        conversation.accept(_primary("not-a-uuid"))


def test_messages_with_no_system_message_have_no_conversation_id() -> None:
    # The shape a re-entry carries when the prior turns did not come from this
    # CLI at all -- a synthesised or replayed transcript. Distinct from the
    # title request, which does have a system message.
    messages: list[ChatMessage] = [
        ChatMessageUser(content="What is 1+1?"),
        ChatMessageAssistant(content="2"),
        ChatMessageUser(content="And 2+2?"),
    ]
    assert _native_conversation_id(messages) is None


# --- resuming a prior conversation ------------------------------------------
#
# On re-entry the conversation to keep canonical is the one the PREVIOUS
# invocation ran, and its id is recoverable from the canonical system block the
# previous invocation tracked. Binding it at construction rather than on the
# first accepted request matters: the CLI's title conversation can be the first
# request to arrive, and a tracker that binds whatever comes first would make
# it canonical before the resumed conversation ever speaks.


def test_a_prior_conversation_can_be_bound_before_any_request() -> None:
    conversation = _NativeConversation(unattended=True, bound_id=_CID)
    assert conversation.bound_id == _CID

    # Another conversation arriving first does not take the binding...
    assert conversation.accept(_primary(_OTHER_CID)) is False
    assert conversation.bound_id == _CID
    # ...and the resumed one is canonical whenever it arrives.
    assert conversation.accept(_primary(_CID)) is True


def test_a_prior_binding_still_excludes_auxiliary_requests() -> None:
    conversation = _NativeConversation(unattended=True, bound_id=_CID)
    assert conversation.accept(_auxiliary()) is False
    assert conversation.bound_id == _CID


def test_centaur_keeps_accepting_every_conversation_when_resuming() -> None:
    # Resuming does not narrow centaur mode: the human still owns whatever
    # conversations they start or resume in that session.
    conversation = _NativeConversation(unattended=False, bound_id=_CID)
    assert conversation.accept(_primary(_OTHER_CID)) is True
    assert conversation.accept(_primary(_CID)) is True


# --- native declaration framing ---------------------------------------------
#
# The block is found by matching an opener to a closer, so text that opens a
# declaration without closing it -- or closes one without opening it -- matches
# nothing at all. Both silent outcomes are wrong, and in opposite ways:
#
#   * no match at all reads as "no block", which is the positive signal for the
#     CLI's auxiliary title request. A malformed primary request would then be
#     excluded from canonical state, and the run would bind to whatever
#     conversation came next -- or score nothing at all.
#   * a match that stops early reads as "exactly one block", so a second,
#     unterminated declaration is never counted and the ambiguity rule that
#     exists to catch exactly that never fires.
#
# So the delimiters have to be counted and ordered before their contents are
# read: whatever the framing fault is, it is not an auxiliary request and it is
# not an unambiguous declaration.


def test_an_unclosed_block_is_not_read_as_an_auxiliary_request() -> None:
    # The dangerous direction. This request declares an identity; dropping it
    # for want of a closing delimiter hands the task's conversation to the
    # filter as though it were the CLI's own title traffic.
    unclosed = ChatMessageSystem(
        content="\n".join(
            [
                "<identity>You are a coding assistant.</identity>",
                "<user_information>",
                f"Conversation ID: {_CID}",
                "<rules>Be brief.</rules>",
            ]
        )
    )
    with pytest.raises(ValueError):
        _native_conversation_id([unclosed, ChatMessageUser(content="Do the task.")])


def test_a_complete_block_followed_by_an_unclosed_one_raises() -> None:
    # Two declarations, one of them truncated. Reading the first and ignoring
    # the remainder is a guess: the id that survives may not be the one the
    # CLI's own result reports, and a mismatch scores an unattributable run.
    truncated_second = ChatMessageSystem(
        content="\n".join(
            [
                _block(f"Conversation ID: {_CID}"),
                "<user_information>",
                f"Conversation ID: {_OTHER_CID}",
            ]
        )
    )
    with pytest.raises(ValueError):
        _native_conversation_id(
            [truncated_second, ChatMessageUser(content="Do the task.")]
        )


def test_a_stray_closing_delimiter_is_not_read_as_an_auxiliary_request() -> None:
    stray = ChatMessageSystem(
        content=f"Conversation ID: {_CID}\n</user_information>",
    )
    with pytest.raises(ValueError):
        _native_conversation_id([stray, ChatMessageUser(content="Do the task.")])


def test_a_repeated_opening_delimiter_raises() -> None:
    # One closer for two openers: the match runs from the first opener, so the
    # second is swallowed into the block's own body and the count still reads
    # as one. Order and count of delimiters is what makes a block complete.
    nested = ChatMessageSystem(
        content="\n".join(
            [
                "<user_information>",
                "<user_information>",
                f"Conversation ID: {_CID}",
                "</user_information>",
            ]
        )
    )
    with pytest.raises(ValueError):
        _native_conversation_id([nested, ChatMessageUser(content="Do the task.")])


def test_an_unclosed_block_before_a_complete_one_raises() -> None:
    # This one already raises, by way of the two ids the swallowing match ends
    # up carrying. It is asserted so that framing validation, once it counts
    # the delimiters properly, keeps rejecting it rather than resolving it to
    # the one complete block.
    swallowed = ChatMessageSystem(
        content="\n".join(
            [
                "<user_information>",
                f"Conversation ID: {_OTHER_CID}",
                _block(f"Conversation ID: {_CID}"),
            ]
        )
    )
    with pytest.raises(ValueError):
        _native_conversation_id([swallowed, ChatMessageUser(content="Do the task.")])
