"""Which MCP server configurations `deepseek_harness()` lets into the sandbox.

dsh's own tools can read anything dsh can, so a credential handed to dsh is
one the model can print. A server that carries one is either connected from
the host (HTTP) or refused (stdio); see `_deepseek_harness/mcp.py`. The
end-to-end checks are in `tests/test_deepseek_harness_execute.py`.
"""

import pytest
from inspect_ai.tool import MCPServerConfigHTTP, MCPServerConfigStdio
from inspect_swe._deepseek_harness.mcp import (
    _redacted_error,
    is_host_side,
    stdio_credentials,
    validate_sandbox_mcp_server,
)


def _stdio(
    args: list[str] | None = None, env: dict[str, str] | None = None
) -> MCPServerConfigStdio:
    return MCPServerConfigStdio(name="srv", command="srv", args=args or [], env=env)


@pytest.mark.parametrize(
    "url,headers,host_side",
    [
        ("http://mcp:8080/mcp", None, False),
        ("https://mcp.example.com/v1/mcp", None, False),
        ("https://mcp.example.com/mcp", {}, False),
        ("https://mcp.example.com/mcp", {"Authorization": "Bearer t"}, True),
        ("https://mcp.example.com/mcp", {"X-Tenant": "acme"}, True),
        ("https://user:pass@mcp.example.com/mcp", None, True),
        ("https://token@mcp.example.com/mcp", None, True),
        ("https://mcp.example.com/mcp?api_key=t", None, True),
        ("https://mcp.example.com/mcp?project=acme", None, True),
    ],
)
def test_http_server_with_any_credential_form_stays_on_the_host(
    url: str, headers: dict[str, str] | None, host_side: bool
) -> None:
    """Headers, URL user info and query strings never reach the sandbox.

    Whatever they hold: a header or query parameter can't be told from a
    credential by its name, so none is written into dsh's profile.
    """
    server = MCPServerConfigHTTP(name="srv", type="http", url=url, headers=headers)
    assert is_host_side(server) is host_side
    if not host_side:
        validate_sandbox_mcp_server(server)


@pytest.mark.parametrize(
    "env,named",
    [
        ({"GITHUB_TOKEN": "x"}, "env GITHUB_TOKEN"),
        ({"OPENAI_API_KEY": "x"}, "env OPENAI_API_KEY"),
        ({"AWS_SECRET_ACCESS_KEY": "x"}, "env AWS_SECRET_ACCESS_KEY"),
        ({"DB_PASSWORD": "x"}, "env DB_PASSWORD"),
        ({"BASIC_AUTH": "user:pw"}, "env BASIC_AUTH"),
        ({"GITHUB_PAT": "x"}, "env GITHUB_PAT"),
        ({"client_secret": "x"}, "env client_secret"),
        ({"clientSecret": "x"}, "env clientSecret"),
        ({"apiKey": "x"}, "env apiKey"),
        # an innocent name with a key-shaped value
        ({"FOO": "ghp_0123456789abcdefghijABCDEFGHIJ"}, "env FOO"),
        ({"DATABASE_URL": "postgres://app:hunter2@db/app"}, "env DATABASE_URL"),
        ({"HEADERS": "Authorization: Bearer abc.def"}, "env HEADERS"),
    ],
)
def test_env_that_looks_like_a_credential_is_named(
    env: dict[str, str], named: str
) -> None:
    assert named in stdio_credentials(_stdio(env=env))


@pytest.mark.parametrize(
    "args,named",
    [
        (["--api-key", "x"], "argument --api-key"),
        (["--token=x"], "argument --token"),
        (["--client-secret", "x"], "argument --client-secret"),
        (["-p", "8080", "--password", "x"], "argument --password"),
        (["API_KEY=x", "serve"], "argument API_KEY=..."),
        (["--header", "X-Api-Key: abc"], "an argument that looks like a credential"),
        (["Bearer abc.def.ghi"], "an argument that looks like a credential"),
        (
            ["--url=https://app:hunter2@db.example.com/x"],
            "argument --url",
        ),
        (
            ["sk-ant-0123456789abcdefghij"],
            "an argument that looks like a credential",
        ),
    ],
)
def test_argument_that_looks_like_a_credential_is_named(
    args: list[str], named: str
) -> None:
    assert named in stdio_credentials(_stdio(args=args))


def test_ordinary_configuration_is_let_through() -> None:
    """The screen is by name and shape; plain settings must not trip it."""
    server = _stdio(
        args=[
            "-y",
            "@modelcontextprotocol/server-memory",
            "--port=8080",
            "--max-tokens",
            "4096",
            "--keyboard-layout",
            "us",
            "MEMORY_FILE_PATH=/tmp/memory.json",
            "https://docs.example.com/search?q=mcp",
            "/workspace/data",
        ],
        env={
            "MEMORY_FILE_PATH": "/tmp/memory.json",
            "MAX_TOKENS": "1000",
            "TOKENIZERS_PARALLELISM": "false",
            "PATH": "/usr/local/bin:/usr/bin",
            "PWD": "/workspace",
            "KEY_FILE": "/run/keys/server.pem",
            "LOG_LEVEL": "debug",
        },
    )
    assert stdio_credentials(server) == []
    validate_sandbox_mcp_server(server)


def test_refusal_names_the_field_but_never_the_value() -> None:
    secret = "ghp_0123456789abcdefghijABCDEFGHIJ"
    server = MCPServerConfigStdio(
        name="github",
        command="gh-mcp",
        args=["--token", secret],
        env={"GITHUB_TOKEN": secret},
    )
    with pytest.raises(ValueError) as error:
        validate_sandbox_mcp_server(server)
    message = str(error.value)
    assert "env GITHUB_TOKEN" in message and "argument --token" in message
    assert "bridged_tools" in message
    assert secret not in message


@pytest.mark.parametrize(
    "url,headers,error",
    [
        (
            "https://mcp.example.com/mcp",
            {"Authorization": "Bearer sk-live-abcdef0123456789\n"},
            ValueError("Illegal header value b'Bearer sk-live-abcdef0123456789\n'"),
        ),
        (
            "https://mcp.example.com/mcp?api_key=sk-live-abcdef0123456789",
            None,
            RuntimeError("Invalid API key sk-live-abcdef0123456789 for this project"),
        ),
        (
            "https://sk-live-abcdef0123456789@mcp.example.com/mcp",
            None,
            RuntimeError("no account for sk-live-abcdef0123456789"),
        ),
        (
            "https://user:sk-live-abcdef0123456789@mcp.example.com/mcp?x=1",
            None,
            ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [
                    RuntimeError(
                        "Client error '401 Unauthorized' for url "
                        "'https://user:sk-live-abcdef0123456789@mcp.example.com/mcp/?x=1'"
                    )
                ],
            ),
        ),
    ],
    ids=["escaped-header-value", "echoed-query-value", "url-user", "grouped-url"],
)
def test_host_connection_errors_are_redacted(
    url: str, headers: dict[str, str] | None, error: Exception
) -> None:
    """What the host-side connection error says, without the credential in any form."""
    server = MCPServerConfigHTTP(name="vault", type="http", url=url, headers=headers)
    text = _redacted_error(error, server)
    assert "sk-live-abcdef0123456789" not in text
    assert text.startswith(("ValueError: ", "RuntimeError: "))
