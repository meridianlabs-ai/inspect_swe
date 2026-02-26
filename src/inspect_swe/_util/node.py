"""Node.js and npm utilities for sandbox environments.

Shared infrastructure for agent implementations that need Node.js
and/or npm packages installed in sandboxes.
"""

import json
import lzma
import os
import shutil
import subprocess
import tarfile
import tempfile
from io import BytesIO

from inspect_ai.util import SandboxEnvironment, concurrency

from .appdirs import package_cache_dir
from .download import download_file
from .sandbox import SANDBOX_INSTALL_DIR, SandboxPlatform, bash_command, sandbox_exec

NODE_VERSION = "20.11.0"


async def ensure_node_available(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Ensure Node.js is available in the sandbox.

    Returns the path to the node binary.
    """
    node_path = f"{SANDBOX_INSTALL_DIR}/node/bin/node"

    result = await sandbox.exec(bash_command(f"test -x {node_path}"), user=user)
    if result.success:
        return node_path

    # Check if node is available system-wide
    result = await sandbox.exec(bash_command("which node"), user=user)
    if result.success:
        return result.stdout.strip()

    async with concurrency("node-npm-install", 1, visible=False):
        return await _download_and_install_node(sandbox, platform)


def _platform_to_node_arch(platform: SandboxPlatform) -> str:
    """Map SandboxPlatform to Node.js architecture string.

    Node.js doesn't have musl-specific builds, so musl platforms
    use the standard glibc builds.
    """
    platform_map = {
        "linux-x64": "linux-x64",
        "linux-x64-musl": "linux-x64",
        "linux-arm64": "linux-arm64",
        "linux-arm64-musl": "linux-arm64",
    }
    if platform not in platform_map:
        raise ValueError(f"Unsupported platform: {platform}")
    return platform_map[platform]


async def _download_and_install_node(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
) -> str:
    """Download Node.js and install the node binary to the sandbox."""
    node_arch = _platform_to_node_arch(platform)
    archive_name = f"node-v{NODE_VERSION}-{node_arch}.tar.xz"
    download_url = f"https://nodejs.org/dist/v{NODE_VERSION}/{archive_name}"

    cache_dir = package_cache_dir("node-binary-downloads")
    cache_path = cache_dir / f"node-v{NODE_VERSION}-{node_arch}"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            node_binary_data = f.read()
    else:
        archive_data = await download_file(download_url)
        tar_data = lzma.decompress(archive_data)

        with tarfile.open(fileobj=BytesIO(tar_data)) as tar:
            member = tar.getmember(f"node-v{NODE_VERSION}-{node_arch}/bin/node")
            extracted = tar.extractfile(member)
            assert extracted is not None
            node_binary_data = extracted.read()

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as cache_file:
            cache_file.write(node_binary_data)

    node_path = f"{SANDBOX_INSTALL_DIR}/node/bin/node"

    await sandbox_exec(sandbox, f"mkdir -p {SANDBOX_INSTALL_DIR}/node/bin", user="root")
    await sandbox.write_file(node_path, node_binary_data)
    await sandbox_exec(sandbox, f"chmod +x {node_path}", user="root")

    return node_path


def resolve_npm_package_version(package: str) -> str:
    """Get the latest version of an npm package from the registry.

    Requires ``npm`` to be available on the host.
    """
    if not shutil.which("npm"):
        raise RuntimeError(
            f"npm is required on the host to resolve the version of {package}. "
            "Please install Node.js (https://nodejs.org/) and ensure npm is on "
            "your PATH."
        )
    result = subprocess.run(
        ["npm", "view", package, "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to resolve version of {package}: {result.stderr}")
    return result.stdout.strip()


def create_npm_bundle(
    package: str,
    version: str,
    platform: SandboxPlatform,
    cache_name: str,
) -> bytes:
    """Create an npm package bundle with dependencies for the target platform.

    Runs ``npm install`` on the host and bundles the ``node_modules``
    directory into a tarball.  Uses ``--os`` and ``--cpu`` to get native
    modules for the sandbox architecture.  Results are cached locally.

    Args:
        package: npm package name (e.g. ``"@google/gemini-cli"``).
        version: Exact semver version string.
        platform: Target sandbox platform.
        cache_name: Subdirectory name inside the package cache for
            storing the tarball (also used as the cache-key prefix).
    """
    cpu = platform.split("-")[1]

    cache_dir = package_cache_dir(cache_name)
    cache_path = cache_dir / f"{cache_name}-{version}-{platform}.tar.gz"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return f.read()

    if not shutil.which("npm"):
        raise RuntimeError(
            f"npm is required on the host to bundle {package} for the sandbox. "
            "Please install Node.js (https://nodejs.org/) and ensure npm is on "
            "your PATH."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        package_json = {
            "name": cache_name,
            "version": "1.0.0",
            "private": True,
            "dependencies": {
                package: version,
            },
        }
        package_json_path = f"{tmpdir}/package.json"
        with open(package_json_path, "w") as f:
            json.dump(package_json, f)

        result = subprocess.run(
            [
                "npm",
                "install",
                "--no-audit",
                "--no-fund",
                "--os",
                "linux",
                "--cpu",
                cpu,
            ],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to run npm install for {package}@{version}:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(f"{tmpdir}/node_modules", arcname="node_modules")
            tar.add(package_json_path, arcname="package.json")
            lock_path = f"{tmpdir}/package-lock.json"
            if os.path.exists(lock_path):
                tar.add(lock_path, arcname="package-lock.json")

        bundle_data = buffer.getvalue()

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(bundle_data)

    return bundle_data


async def install_npm_bundle(
    sandbox: SandboxEnvironment,
    bundle_data: bytes,
    install_dir: str,
    binary_name: str,
    user: str | None = None,
) -> str:
    """Extract an npm bundle tarball into the sandbox.

    Returns the path to the package's binary at
    ``{install_dir}/node_modules/.bin/{binary_name}``.
    """
    await sandbox_exec(sandbox, f"mkdir -p {install_dir}", user="root")

    bundle_path = f"{install_dir}/bundle.tar.gz"
    await sandbox.write_file(bundle_path, bundle_data)

    result = await sandbox.exec(
        bash_command(f"tar -xzf {bundle_path} -C {install_dir} && rm -f {bundle_path}"),
        user="root",
        timeout=60,
    )
    if not result.success:
        raise RuntimeError(
            f"Failed to extract npm bundle:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    binary_path = f"{install_dir}/node_modules/.bin/{binary_name}"

    result = await sandbox.exec(bash_command(f"test -x {binary_path}"), user=user)
    if not result.success:
        raise RuntimeError(
            f"{binary_name} binary not found after installation at {binary_path}"
        )

    return binary_path
