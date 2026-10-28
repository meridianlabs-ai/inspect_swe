"""ACP Codex creates its config directories under any cwd."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from inspect_swe.acp._agents.codex_cli.codex_cli import CodexCli


class _LocalSandbox:
    """Runs sandbox commands as local subprocesses."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    async def exec(
        self, cmd: list[str], user: str | None = None, cwd: str | None = None
    ) -> Any:
        p = subprocess.run(
            cmd, cwd=cwd or str(self.cwd), capture_output=True, text=True
        )
        return SimpleNamespace(
            success=p.returncode == 0, stdout=p.stdout, stderr=p.stderr
        )


@pytest.mark.anyio
async def test_config_dir_with_spaces(tmp_path: Path) -> None:
    workdir = tmp_path / "agent workspace"
    workdir.mkdir()
    # a file named like the first word: an unquoted `mkdir -p` fails on it
    (tmp_path / "agent").write_text("")
    agent = object.__new__(CodexCli)
    agent.cwd = str(workdir)
    agent.user = None
    agent._home_dir = None
    config = await agent._config_toml_path(
        _LocalSandbox(workdir),  # type: ignore[arg-type]
        str(workdir / ".codex"),
    )
    assert Path(config) == workdir / ".codex" / "config.toml"
    assert Path(config).parent.is_dir()
