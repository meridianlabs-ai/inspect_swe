"""MCP servers for dsh: what is wired into the sandbox, and what stays on the host.

dsh's own tools run as dsh's user, so the model can read anything dsh can: its
profile, its environment, the files and descriptors it has open (through
`/proc`), and the command line and environment of the processes it starts.
Nothing given to dsh inside the sandbox can be kept from the model. So a
credential for an MCP server never goes there:

- An HTTP server whose configuration carries one (`headers`, or a user,
  password or query string in its URL) is connected from the Inspect host, and
  its tools are served to dsh through the model bridge like `bridged_tools`.
  Nothing of its configuration is written into the sandbox.
- A stdio server runs inside the sandbox, with its environment and arguments
  there too. One whose `env` or `args` look like they carry a credential is
  rejected.

Everything else about a server wired into the sandbox (its URL, command,
arguments, environment) is written to dsh's profile and is model-visible.
"""

import contextlib
import re
from collections.abc import AsyncIterator, Sequence
from fnmatch import fnmatch
from typing import NamedTuple
from urllib.parse import parse_qsl, urlsplit

from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import (
    MCPServer,
    MCPServerConfig,
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
    Tool,
    ToolDef,
    mcp_server_http,
    mcp_server_sse,
)

# server names dsh's mcp-client accepts
_MCP_SERVER_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")

# words that make a name a credential's (`GITHUB_TOKEN`, `--api-key`,
# `client-secret`); `MAX_TOKENS` and `TOKENIZERS_PARALLELISM` are not
_CREDENTIAL_WORDS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "passphrase",
        "passwd",
        "password",
        "pat",
        "secret",
        "secrets",
        "token",
    }
)

# endings that make a run-together word a credential's name (`PGPASSWORD`,
# `AUTHTOKEN`, `OPENAIAPIKEY`)
_CREDENTIAL_ENDINGS = ("apikey", "passwd", "password", "secret", "token")

# an argument's flag, for an error message: a long one up to anything run on
# after its name (`--token:abc`), a short one's letter alone (`-pabc`)
_FLAG_RE = re.compile(r"--[A-Za-z0-9_.-]*|-[A-Za-z0-9]?")

# what a credential looks like whatever it is called: an Authorization value
# or header line, a URL with a password, and the well-known key formats
_CREDENTIAL_VALUE_RE = re.compile(
    r"^\s*(?:bearer|basic)\s+\S+"
    r"|\b(?:proxy-)?authorization\s*:\s*\S+"
    r"|\b(?:x-api-key|api-key|cookie)\s*:\s*\S+"
    r"|://[^/\s:@]+:[^/\s@]+@"
    r"|\bsk-[a-z0-9_-]{16,}"
    r"|\bgh[pousr]_[a-z0-9]{20,}"
    r"|\bgithub_pat_[a-z0-9_]{20,}"
    r"|\bglpat-[a-z0-9_-]{16,}"
    r"|\bxox[abprs]-[a-z0-9-]{10,}"
    r"|\bakia[0-9a-z]{16}\b"
    r"|\baiza[0-9a-z_-]{30,}"
    r"|\bhf_[a-z0-9]{30,}"
    r"|\beyj[a-z0-9_-]{8,}\.[a-z0-9_-]{8,}\.[a-z0-9_-]{8,}",
    re.IGNORECASE,
)

_ASSIGNMENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.DOTALL)


class HostMCPServer(NamedTuple):
    """An MCP server connected from the Inspect host."""

    config: MCPServerConfigHTTP
    client: MCPServer


def validate_mcp_server_name(name: str) -> None:
    if not _MCP_SERVER_NAME_RE.fullmatch(name):
        raise ValueError(
            f"MCP server {name!r}: deepseek_harness requires server names "
            "of 1-32 letters, digits, '_' or '-'."
        )


def is_host_side(server: MCPServerConfig) -> bool:
    """Whether a server's configuration carries a credential, so stays on the host."""
    return isinstance(server, MCPServerConfigHTTP) and (
        bool(server.headers) or _url_has_credentials(server.url)
    )


def validate_sandbox_mcp_server(server: MCPServerConfig) -> None:
    """Check a server that dsh itself will connect to, from inside the sandbox."""
    validate_mcp_server_name(server.name)
    if server.tools != "all":
        raise ValueError(
            f"MCP server {server.name!r}: deepseek_harness does not support "
            "per-server tool filtering (dsh exposes every tool a server lists)."
        )
    if isinstance(server, MCPServerConfigHTTP) and server.type != "http":
        raise ValueError(
            f"MCP server {server.name!r}: deepseek_harness supports only stdio "
            "and Streamable HTTP MCP servers (not SSE)."
        )
    if isinstance(server, MCPServerConfigStdio):
        found = stdio_credentials(server)
        if found:
            raise ValueError(
                f"MCP server {server.name!r}: deepseek_harness does not pass "
                f"credentials to a stdio MCP server ({', '.join(found)}). The "
                "server runs inside the sandbox, where dsh's own tools can "
                "read its environment and command line, so the model could "
                "print them. Serve tools that need a credential from the "
                "host with bridged_tools instead."
            )


def stdio_credentials(server: MCPServerConfigStdio) -> list[str]:
    """The `env` entries and arguments of a stdio server that look like credentials.

    Named for an error message, never with their values. This is a screen for
    the ways credentials are usually passed, not proof that none is: a secret
    under an innocent name, in a shape of its own, gets through.
    """
    found = [
        f"env {name}"
        for name, value in (server.env or {}).items()
        if _names_credential(name) or _looks_like_credential(value)
    ]
    for argument in server.args:
        if argument.startswith("-"):
            flag, _, value = argument.partition("=")
            shown = _FLAG_RE.match(flag)
            shown_flag = shown.group(0) if shown is not None else "-"
            if _names_credential(flag) or _carries_credential(value):
                found.append(f"argument {shown_flag}")
            continue
        assignment = _ASSIGNMENT_RE.fullmatch(argument)
        if assignment is not None and _names_credential(assignment.group(1)):
            found.append(f"argument {assignment.group(1)}=...")
        elif _carries_credential(argument):
            found.append("an argument that looks like a credential")
    return found


def host_mcp_server(config: MCPServerConfigHTTP) -> HostMCPServer:
    """A client for a server that is connected from the host.

    Raises:
        PrerequisiteError: The `mcp` package is not installed.
    """
    validate_mcp_server_name(config.name)
    create = mcp_server_http if config.type == "http" else mcp_server_sse
    return HostMCPServer(
        config, create(name=config.name, url=config.url, headers=config.headers)
    )


@contextlib.asynccontextmanager
async def host_mcp_tools(
    servers: Sequence[HostMCPServer],
) -> AsyncIterator[list[BridgedToolsSpec]]:
    """Connect to host-side servers and offer their tools to the model bridge.

    Each server becomes a bridged MCP server of the same name, so dsh sees
    its tools under the names it would have had (`mcp__<server>__<tool>`),
    with the server's `tools` allowlist applied. One connection per server is
    held for the block.
    """
    async with contextlib.AsyncExitStack() as stack:
        specs: list[BridgedToolsSpec] = []
        for server in servers:
            name = server.config.name
            try:
                await stack.enter_async_context(server.client)
                tools = _allowed(await server.client.tools(), server.config)
            except Exception as ex:
                # not chained: the error's own text (an HTTP error names the
                # URL it requested) would carry the credentials into the log
                raise RuntimeError(
                    f"MCP server {name!r} may carry credentials "
                    f"({_host_side_reason(server.config)}), so deepseek_harness "
                    "connects to it from the Inspect host (its configuration "
                    "never enters the sandbox). That failed: "
                    f"{_redacted_error(ex, server.config)}"
                ) from None
            if not tools:
                raise RuntimeError(
                    f"MCP server {name!r} (connected from the Inspect host) "
                    "offers no tools"
                    + ("." if server.config.tools == "all" else " its allowlist keeps.")
                )
            specs.append(BridgedToolsSpec(name=name, tools=tools))
        yield specs


def _allowed(tools: list[Tool], config: MCPServerConfigHTTP) -> list[Tool]:
    if config.tools == "all":
        return tools
    return [
        tool
        for tool in tools
        if any(fnmatch(ToolDef(tool).name, pattern) for pattern in config.tools)
    ]


def _host_side_reason(config: MCPServerConfigHTTP) -> str:
    try:
        parts = urlsplit(config.url)
    except ValueError:
        parts = None
    reasons = [
        reason
        for reason, present in (
            ("headers", bool(config.headers)),
            (
                "a user or password in its URL",
                bool(parts and (parts.username or parts.password)),
            ),
            ("a query string in its URL", bool(parts and parts.query)),
        )
        if present
    ]
    return " and ".join(reasons)


def _redacted_error(error: BaseException, config: MCPServerConfigHTTP) -> str:
    """An error's type and text, with the server's credentials taken out.

    An exception group (the MCP client's task groups raise them) is described
    by its first leaf, which says what actually went wrong.
    """
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    text = str(error)
    for secret in sorted(_secrets(config), key=len, reverse=True):
        text = text.replace(secret, "***")
    # the same, however the client re-encoded the URL
    text = re.sub(r"(://)[^/\s'\"@]+@", r"\1***@", text)
    text = re.sub(r"(://[^\s'\"?]*\?)[^\s'\"#]+", r"\1***", text)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _secrets(config: MCPServerConfigHTTP) -> set[str]:
    """Every form a server's credentials could take in an error's text.

    Whole header values and each word of them (`Bearer <token>` quoted as
    the token alone), the URL's user, password, query and each query value,
    each also as Python would escape it (a trailing newline quoted as a
    backslash and `n` in an HTTP client's complaint about a header value).
    """
    pieces: list[str] = []
    for value in (config.headers or {}).values():
        pieces += [value, value.strip(), *value.split()]
    try:
        parts = urlsplit(config.url)
    except ValueError:
        parts = None
    if parts is not None:
        pieces += [parts.username or "", parts.password or "", parts.query]
        for name_value in parts.query.split("&"):
            pieces += name_value.split("=", 1)[1:]
        pieces += [value for _, value in parse_qsl(parts.query)]
    secrets: set[str] = set()
    for piece in pieces:
        # too short to be a credential, and would garble the text
        if len(piece) >= 4 and piece.lower() not in _AUTH_SCHEMES:
            secrets |= {piece, piece.encode("unicode_escape").decode("ascii")}
    return secrets


# words in an Authorization value that aren't the credential
_AUTH_SCHEMES = frozenset({"basic", "bearer", "digest", "token"})


def _url_has_credentials(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return bool(parts.username or parts.password or parts.query)


def _names_credential(name: str) -> bool:
    # GITHUB_TOKEN, --api-key and clientSecret alike
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    words = [word for word in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if word]
    if not words:
        return False
    return (
        # `*_KEY` is how keys are named (`OPENAI_API_KEY`, `--api-key`)
        words[-1] == "key"
        or any(word in _CREDENTIAL_WORDS for word in words)
        or any(word.endswith(_CREDENTIAL_ENDINGS) for word in words)
        # `MYSQL_PWD`, but not `PWD` (the working directory)
        or "pwd" in words[1:]
    )


def _looks_like_credential(value: str) -> bool:
    return bool(value) and _CREDENTIAL_VALUE_RE.search(value) is not None


def _carries_credential(value: str) -> bool:
    """A value that is a credential, sets one (`GITHUB_TOKEN=...`), or is a URL holding one."""
    if _looks_like_credential(value):
        return True
    assignment = _ASSIGNMENT_RE.fullmatch(value)
    if assignment is not None and _names_credential(assignment.group(1)):
        return True
    return _url_query_names_credential(value)


def _url_query_names_credential(value: str) -> bool:
    """Whether a URL's query has a parameter named like a credential (`?api_key=`)."""
    if "://" not in value or "?" not in value:
        return False
    try:
        query = urlsplit(value.strip()).query
    except ValueError:
        return False
    return any(
        _names_credential(name) for name, _ in parse_qsl(query, keep_blank_values=True)
    )
