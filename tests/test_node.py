"""Tests for host-side Node and npm helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_swe._util import node


def test_npm_before_libc_support_cannot_seed_bundle_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "package_cache_dir", lambda name: tmp_path)
    monkeypatch.setattr(
        "inspect_swe._util.node.shutil.which", lambda name: "/usr/bin/npm"
    )
    monkeypatch.setattr(
        "inspect_swe._util.node.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="11.5.0\n", stderr=""
        ),
    )
    stale = tmp_path / "claude-1.0.0-linux-x64-v2.tar.gz"
    stale.write_bytes(b"bundle built after --libc was ignored")

    with pytest.raises(RuntimeError, match=r"npm >= 11\.5\.1"):
        node.create_npm_bundle(
            "@zed-industries/claude-agent-acp",
            "1.0.0",
            "linux-x64",
            "claude",
        )

    assert not (tmp_path / "claude-1.0.0-linux-x64-v3.tar.gz").exists()


def test_validated_v3_bundle_cache_does_not_require_host_npm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "package_cache_dir", lambda name: tmp_path)
    monkeypatch.setattr("inspect_swe._util.node.shutil.which", lambda name: None)
    cache = tmp_path / "claude-1.0.0-linux-x64-v3.tar.gz"
    cache.write_bytes(b"validated bundle")

    assert (
        node.create_npm_bundle(
            "@zed-industries/claude-agent-acp",
            "1.0.0",
            "linux-x64",
            "claude",
        )
        == b"validated bundle"
    )
