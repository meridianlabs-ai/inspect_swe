"""Node runtime and native npm bundle regression tests."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest
from inspect_ai.util import SandboxEnvironment
from inspect_swe._util import node
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR


@pytest.mark.parametrize(
    "installed",
    [
        "v22.19.0",
        "v22.20.0",
        "v24.2.0",
        "v25.0.0",
        "v27.0.0",
        "v22.19.0-rc.1",
        "invalid",
    ],
)
def test_pinned_node_requires_exact_abi_runtime(installed: str) -> None:
    sbox = Mock(spec=SandboxEnvironment)
    sbox.exec = AsyncMock(
        side_effect=[
            Mock(success=False),
            Mock(success=True, stdout="/usr/bin/node\n"),
            Mock(success=True, stdout=installed + "\n"),
        ]
    )
    with patch.object(
        node, "_download_and_install_node", AsyncMock(return_value="/opt/pinned/node")
    ) as download:
        result = anyio.run(
            node.ensure_node_available, sbox, "linux-x64", "alice", "22.19.0"
        )
    if installed == "v22.19.0":
        assert result == "/usr/bin/node"
        download.assert_not_awaited()
    else:
        assert result == "/opt/pinned/node"
        download.assert_awaited_once_with(
            sbox, "linux-x64", "22.19.0", f"{SANDBOX_INSTALL_DIR}/node-22.19.0"
        )


def test_legacy_node_call_still_accepts_system_runtime() -> None:
    sbox = Mock(spec=SandboxEnvironment)
    sbox.exec = AsyncMock(
        side_effect=[Mock(success=False), Mock(success=True, stdout="/usr/bin/node\n")]
    )
    assert anyio.run(node.ensure_node_available, sbox, "linux-x64") == "/usr/bin/node"
    assert sbox.exec.await_count == 2


def test_native_bundle_targets_sandbox_arch_and_node(tmp_path: Path) -> None:
    def install(cmd: list[str], **kwargs: object) -> Mock:
        Path(str(kwargs["cwd"]), "node_modules").mkdir()
        return Mock(returncode=0)

    with (
        patch.object(node, "package_cache_dir", return_value=tmp_path),
        patch.object(shutil, "which", return_value="/usr/bin/npm"),
        patch.object(subprocess, "run", side_effect=install) as npm,
    ):
        first = node.create_npm_bundle(
            "@minimax-ai/code",
            "0.4.12",
            "linux-arm64",
            "minimax-test",
            ignore_scripts=True,
            include_optional=True,
        )
        second = node.create_npm_bundle(
            "@minimax-ai/code",
            "0.4.12",
            "linux-arm64",
            "minimax-test",
            ignore_scripts=True,
            include_optional=True,
        )
    assert first == second
    npm.assert_called_once()
    cmd = npm.call_args.args[0]
    assert cmd[cmd.index("--cpu") + 1] == "arm64"
    assert "--ignore-scripts" in cmd
    assert "--include=optional" in cmd
    assert not any(arg.startswith("--allow-scripts") for arg in cmd)
    assert next(tmp_path.iterdir()).name.endswith("-noscripts-optional.tar.gz")
