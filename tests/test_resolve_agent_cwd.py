from typing import Any

import anyio
from inspect_ai.util import ExecResult
from inspect_swe._util.sandbox import expand_sandbox_home, resolve_agent_cwd

PWD_CMD = "pwd"
HOME_CMD = 'cd ~ 2>/dev/null && pwd || echo "/"'
PRINT_HOME_CMD = 'printf "%s" "$HOME"'


class StubSandbox:
    """Sandbox stub answering bash -c scripts from a canned mapping."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.commands: list[tuple[str, str | None]] = []

    async def exec(
        self, cmd: list[str], cwd: str | None = None, **kwargs: Any
    ) -> ExecResult[str]:
        script = cmd[-1]
        self.commands.append((script, cwd))
        return ExecResult(
            success=True, returncode=0, stdout=self.responses[script] + "\n", stderr=""
        )


def resolve(
    responses: dict[str, str], cwd: str | None
) -> tuple[str, list[tuple[str, str | None]]]:
    sandbox = StubSandbox(responses)
    result = anyio.run(resolve_agent_cwd, sandbox, None, cwd)  # type: ignore[arg-type]
    return result, sandbox.commands


def test_explicit_absolute_cwd_returned_verbatim() -> None:
    result, commands = resolve({}, "/workspace")
    assert result == "/workspace"
    assert commands == []


def test_explicit_relative_cwd_canonicalized_in_sandbox() -> None:
    result, commands = resolve({PWD_CMD: "/app/workspace"}, "workspace")
    assert result == "/app/workspace"
    assert commands == [(PWD_CMD, "workspace")]


def test_default_working_dir_used_when_not_root() -> None:
    result, commands = resolve({PWD_CMD: "/app"}, None)
    assert result == "/app"
    assert commands == [(PWD_CMD, None)]


def test_root_working_dir_falls_back_to_home() -> None:
    result, commands = resolve({PWD_CMD: "/", HOME_CMD: "/root"}, None)
    assert result == "/root"
    assert commands == [(PWD_CMD, None), (HOME_CMD, None)]


def test_root_home_stays_at_root() -> None:
    result, _ = resolve({PWD_CMD: "/", HOME_CMD: "/"}, None)
    assert result == "/"


def test_expand_sandbox_home_only_expands_leading_home_tokens() -> None:
    sandbox = StubSandbox({PRINT_HOME_CMD: "/root"})

    async def run() -> str:
        return await expand_sandbox_home(
            sandbox,  # type: ignore[arg-type]
            "$HOME/config dir",
        )

    result = anyio.run(run)
    assert result == "/root/config dir"
    assert sandbox.commands == [(PRINT_HOME_CMD, None)]


def test_expand_sandbox_home_does_not_evaluate_shell_syntax() -> None:
    sandbox = StubSandbox({})
    path = "/tmp/it's; touch injected; $(echo bad)"

    async def run() -> str:
        return await expand_sandbox_home(sandbox, path)  # type: ignore[arg-type]

    result = anyio.run(run)
    assert result == path
    assert sandbox.commands == []
