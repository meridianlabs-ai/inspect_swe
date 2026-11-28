"""Unit tests for gemini cli version selection in the sandbox."""

from contextlib import ExitStack
from functools import partial
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from inspect_ai.util import ExecResult, SandboxEnvironment
from inspect_swe._gemini_cli import agentbinary

SANDBOX_GEMINI = "/usr/local/bin/gemini"


class _FakeSandbox:
    """exec() answers `which gemini` per `has_gemini`; everything else fails."""

    def __init__(self, has_gemini: bool) -> None:
        self.has_gemini = has_gemini

    async def exec(self, cmd: list[str], **kwargs: object) -> ExecResult[str]:
        if cmd[-1] == "which gemini" and self.has_gemini:
            return ExecResult(True, 0, f"{SANDBOX_GEMINI}\n", "")
        return ExecResult(False, 1, "", "")


def _setup(
    version: str,
    sandbox: _FakeSandbox,
    ensure_node_available: AsyncMock,
    create_npm_bundle: MagicMock,
) -> tuple[str, str]:
    """Run ensure_gemini_cli_setup with node setup and downloads mocked out."""
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(agentbinary, "trace", lambda msg: None, create=True)
        )
        stack.enter_context(
            patch.object(
                agentbinary,
                "detect_sandbox_platform",
                AsyncMock(return_value="linux-x64"),
            )
        )
        stack.enter_context(
            patch.object(agentbinary, "ensure_node_available", ensure_node_available)
        )
        stack.enter_context(
            patch.object(
                agentbinary,
                "cached_version_resolution",
                AsyncMock(return_value="9.9.9"),
            )
        )
        stack.enter_context(
            patch.object(agentbinary, "create_npm_bundle", create_npm_bundle)
        )
        stack.enter_context(
            patch.object(
                agentbinary,
                "install_npm_bundle",
                AsyncMock(return_value="/installed/gemini"),
            )
        )
        return anyio.run(
            partial(
                agentbinary.ensure_gemini_cli_setup,
                cast(SandboxEnvironment, sandbox),
                version,
                None,
            )
        )


@pytest.mark.parametrize("version", ["auto", "sandbox"])
def test_gemini_uses_sandbox_binary(version: str) -> None:
    create_npm_bundle = MagicMock(return_value=b"bundle")
    gemini_binary, node_binary = _setup(
        version,
        _FakeSandbox(has_gemini=True),
        AsyncMock(return_value="/usr/bin/node"),
        create_npm_bundle,
    )
    assert gemini_binary == SANDBOX_GEMINI
    assert node_binary == "/usr/bin/node"
    create_npm_bundle.assert_not_called()


def test_gemini_sandbox_version_missing_raises() -> None:
    ensure_node_available = AsyncMock(return_value="/usr/bin/node")
    create_npm_bundle = MagicMock(return_value=b"bundle")
    with pytest.raises(RuntimeError, match="unable to locate gemini cli in sandbox"):
        _setup(
            "sandbox",
            _FakeSandbox(has_gemini=False),
            ensure_node_available,
            create_npm_bundle,
        )
    # fails before setting up node or building a bundle
    ensure_node_available.assert_not_awaited()
    create_npm_bundle.assert_not_called()
