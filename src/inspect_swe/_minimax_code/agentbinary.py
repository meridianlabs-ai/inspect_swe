import json
import re
import tarfile
import tempfile
from functools import partial
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal

import anyio
from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.appdirs import package_cache_dir
from .._util.download import download_file
from .._util.node import (
    create_npm_bundle,
    ensure_node_available,
    install_npm_bundle,
    resolve_npm_package_version,
)
from .._util.sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
)
from .._util.versioncache import cached_version_resolution

MINIMAX_CODE_PACKAGE = "@minimax-ai/code"
# Pin the runtime as well as the bundle's native module ABI. Node 22 is
# supported by MiniMax Code and by older glibc-based evaluation images.
MINIMAX_CODE_NODE_VERSION = "22.19.0"
MINIMAX_CODE_NODE_ABI = "127"


async def _latest_version() -> str:
    return await anyio.to_thread.run_sync(
        resolve_npm_package_version, MINIMAX_CODE_PACKAGE
    )


async def ensure_minimax_code_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, str]:
    """Install MiniMax Code and a compatible Node.js runtime in the sandbox."""
    if version in ("auto", "sandbox"):
        binary = await sandbox.exec(bash_command("command -v mcode"), user=user)
        if binary.success:
            node = await sandbox.exec(bash_command("command -v node"), user=user)
            if not node.success:
                raise RuntimeError("Pre-installed mcode requires Node.js on PATH")
            return binary.stdout.strip(), node.stdout.strip()
        if version == "sandbox":
            raise RuntimeError("version='sandbox' requires mcode on the sandbox PATH")

    platform = await detect_sandbox_platform(sandbox)
    if platform.endswith("-musl"):
        raise RuntimeError(
            "MiniMax Code requires glibc and does not support musl/Alpine sandboxes"
        )
    node_binary = await ensure_node_available(
        sandbox, platform, user, node_version=MINIMAX_CODE_NODE_VERSION
    )
    package_version = (
        await cached_version_resolution("minimax-code", _latest_version)
        if version in ["auto", "stable", "latest"]
        else version
    )
    install_dir = f"{SANDBOX_INSTALL_DIR}/minimax-code"
    minimax_binary = f"{install_dir}/node_modules/.bin/mcode"

    result = await sandbox.exec(bash_command(f"test -x {minimax_binary}"), user=user)
    if result.success:
        result = await sandbox.exec(
            cmd=[node_binary, minimax_binary, "--version"], user=user
        )
        if result.success and result.stdout.strip() == package_version:
            if await _sqlite_error(sandbox, install_dir, node_binary, user) is None:
                return minimax_binary, node_binary

    async with concurrency("minimax-code-install", 1, visible=False):
        bundle_data = await anyio.to_thread.run_sync(
            partial(
                create_npm_bundle,
                package=MINIMAX_CODE_PACKAGE,
                version=package_version,
                platform=platform,
                cache_name="minimax-code-bundles",
                ignore_scripts=True,
                include_optional=True,
            )
        )
        minimax_binary = await install_npm_bundle(
            sandbox=sandbox,
            bundle_data=bundle_data,
            install_dir=install_dir,
            binary_name="mcode",
            user=user,
        )
        await _install_sqlite(sandbox, install_dir, node_binary, platform)

        # SQLite is an optional npm dependency but mandatory at runtime. npm
        # can otherwise report success after silently dropping a failed build.
        error = await _sqlite_error(sandbox, install_dir, node_binary, user)
        if error is not None:
            raise RuntimeError(
                "MiniMax Code's native SQLite dependency is unavailable: " + error
            )
    return minimax_binary, node_binary


async def _sqlite_error(
    sandbox: SandboxEnvironment,
    install_dir: str,
    node_binary: str,
    user: str | None,
) -> str | None:
    result = await sandbox.exec(
        [
            node_binary,
            "-e",
            "const {createRequire}=require('module');"
            "const r=createRequire(process.cwd() + "
            "'/node_modules/@minimax-ai/code/package.json');"
            "new (r('better-sqlite3'))(':memory:').close()",
        ],
        cwd=install_dir,
        user=user,
    )
    return None if result.success else result.stderr


async def _install_sqlite(
    sandbox: SandboxEnvironment,
    install_dir: str,
    node_binary: str,
    platform: SandboxPlatform,
) -> None:
    # Resolve from MiniMax's package so npm's hoisting layout is immaterial.
    result = await sandbox.exec(
        [
            node_binary,
            "-p",
            "require.resolve('better-sqlite3/package.json', "
            "{paths: [process.argv[1]]})",
            f"{install_dir}/node_modules/@minimax-ai/code",
        ],
        user="root",
    )
    if not result.success:
        raise RuntimeError(
            "MiniMax Code requires the optional better-sqlite3 package: "
            + result.stderr
        )
    package_path = result.stdout.strip()
    sqlite_version = str(json.loads(await sandbox.read_file(package_path))["version"])
    binary = await _sqlite_binary(sqlite_version, platform)
    binary_dir = str(PurePosixPath(package_path).parent / "build" / "Release")
    await sandbox.exec(["mkdir", "-p", binary_dir], user="root")
    await sandbox.write_file(f"{binary_dir}/better_sqlite3.node", binary)


async def _sqlite_binary(version: str, platform: SandboxPlatform) -> bytes:
    # The version came from a sandbox file. Never allow it to select an
    # arbitrary host cache path or download URL.
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("Invalid better-sqlite3 version in sandbox package metadata")
    name = f"better-sqlite3-v{version}-node-v{MINIMAX_CODE_NODE_ABI}-{platform}"
    cache_path = package_cache_dir("minimax-code-sqlite") / name
    if cache_path.exists():
        cached = cache_path.read_bytes()
        if cached:
            return cached
    archive = await download_file(
        f"https://github.com/WiseLibs/better-sqlite3/releases/download/v{version}/{name}.tar.gz"
    )
    with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as tar:
        native = tar.extractfile("build/Release/better_sqlite3.node")
        assert native is not None
        binary = native.read()
    # Other evaluator processes must never see a partially written binary.
    with tempfile.TemporaryDirectory(dir=cache_path.parent) as staging:
        staged = Path(staging) / "binary"
        staged.write_bytes(binary)
        staged.replace(cache_path)
    return binary
