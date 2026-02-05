import asyncio
import json
import lzma
import os
import subprocess
import tarfile
import tempfile
from io import BytesIO
from typing import Any, Literal

from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.appdirs import package_cache_dir
from .._util.download import download_file, download_text_file
from .._util.sandbox import SandboxPlatform, bash_command, sandbox_exec

# Node.js version to download if not available in sandbox
NODE_VERSION = "20.11.0"

# Installation paths in sandbox
SANDBOX_INSTALL_DIR = "/var/tmp/.5c95f967ca830048"


async def resolve_gemini_version(
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
) -> str:
    """Resolve version string to an actual semver version."""
    if version in ["auto", "sandbox", "stable", "latest"]:
        release = await _fetch_latest_release()
        return str(release["tag_name"]).lstrip("v")

    return version


async def ensure_node_and_npm_available(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
    user: str | None = None,
) -> tuple[str, str]:
    """Ensure Node.js and npm are available in the sandbox.

    Returns tuple of (node_path, npm_path).
    """
    node_path = f"{SANDBOX_INSTALL_DIR}/node/bin/node"
    npm_path = f"{SANDBOX_INSTALL_DIR}/node/bin/npm"

    result = await sandbox.exec(
        bash_command(f"test -x {node_path} && test -x {npm_path}"), user=user
    )
    if result.success:
        return node_path, npm_path

    # Check if node is available system-wide
    result = await sandbox.exec(bash_command("which node && which npm"), user=user)
    if result.success:
        paths = result.stdout.strip().split("\n")
        if len(paths) == 2:
            return paths[0], paths[1]

    async with concurrency("node-npm-install", 1, visible=False):
        return await _download_and_install_node(sandbox, platform)


async def ensure_gemini_cli_installed(
    sandbox: SandboxEnvironment,
    node_path: str,
    version: str,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install Gemini CLI via npm and return path to the gemini binary.

    This installs the full @google/gemini-cli package including all policy files,
    ensuring YOLO mode and other features work correctly.
    """
    gemini_install_dir = f"{SANDBOX_INSTALL_DIR}/gemini-cli"
    gemini_binary = f"{gemini_install_dir}/node_modules/.bin/gemini"

    result = await sandbox.exec(
        bash_command(f"test -x {gemini_binary}"),
        user=user,
    )
    if result.success:
        result = await sandbox.exec(
            cmd=[node_path, gemini_binary, "--version"],
            user=user,
        )
        if result.success:
            installed_version = result.stdout.strip()
            if installed_version == version:
                return gemini_binary

    async with concurrency("gemini-cli-install", 1, visible=False):
        return await _install_bundle(
            sandbox, version, gemini_install_dir, platform, user
        )


async def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest release from GitHub."""
    releases_url = "https://api.github.com/repos/google-gemini/gemini-cli/releases"
    release_json = await download_text_file(f"{releases_url}/latest")
    return dict(json.loads(release_json))


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
) -> tuple[str, str]:
    """Download and install full Node.js distribution including npm."""
    node_arch = _platform_to_node_arch(platform)
    archive_name = f"node-v{NODE_VERSION}-{node_arch}.tar.xz"
    download_url = f"https://nodejs.org/dist/v{NODE_VERSION}/{archive_name}"

    cache_dir = package_cache_dir("node-full-downloads")
    cache_path = cache_dir / f"node-v{NODE_VERSION}-{node_arch}.tar"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            tar_data = f.read()
    else:
        archive_data = await download_file(download_url)

        # Decompress xz to tar (keep as tar for caching)
        tar_data = lzma.decompress(archive_data)

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(tar_data)

    # Extract to sandbox by writing tar and extracting with tar command
    # This is MUCH faster than extracting file-by-file through the sandbox interface
    install_dir = f"{SANDBOX_INSTALL_DIR}/node"
    tar_path = f"{SANDBOX_INSTALL_DIR}/node.tar"

    await sandbox_exec(sandbox, f"mkdir -p {SANDBOX_INSTALL_DIR}", user="root")
    await sandbox.write_file(tar_path, tar_data)

    result = await sandbox.exec(
        bash_command(
            f"mkdir -p {install_dir} && "
            f"tar -xf {tar_path} -C {install_dir} --strip-components=1 && "
            f"rm -f {tar_path}"
        ),
        user="root",
        timeout=60,
    )
    if not result.success:
        raise RuntimeError(
            f"Failed to extract Node.js tar:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    node_path = f"{install_dir}/bin/node"
    npm_path = f"{install_dir}/bin/npm"

    return node_path, npm_path


def _create_bundle(version: str, platform: SandboxPlatform) -> bytes:
    """Create a gemini-cli bundle with dependencies for a specific platform.

    Runs npm install on the host (where we have network access) and bundles
    the entire node_modules directory into a tarball for transfer to sandbox.
    Uses --os and --cpu flags to install native modules for the target platform.
    """
    # Extract cpu from platform (e.g., "linux-x64" -> "x64", "linux-arm64" -> "arm64")
    # Note: platform format is "linux-{cpu}" or "linux-{cpu}-musl", so index [1] gets the cpu
    cpu = platform.split("-")[1]

    cache_dir = package_cache_dir("gemini-cli-bundles")
    cache_path = cache_dir / f"gemini-cli-bundle-{version}-{platform}.tar.gz"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return f.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create package.json - just the main dependency
        # npm will install correct native modules via --os/--cpu flags
        package_json = {
            "name": "gemini-cli-bundle",
            "version": "1.0.0",
            "private": True,
            "dependencies": {
                "@google/gemini-cli": version,
            },
        }
        package_json_path = f"{tmpdir}/package.json"
        with open(package_json_path, "w") as f:
            json.dump(package_json, f)

        # Run npm install with --os/--cpu to get correct native modules for sandbox
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
                f"Failed to run npm install for gemini-cli@{version}:\n"
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


async def _install_bundle(
    sandbox: SandboxEnvironment,
    version: str,
    install_dir: str,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install @google/gemini-cli from pre-built bundle.

    Creates a bundle with dependencies for the target platform on the host
    (where we have network access), then transfers and extracts it in the sandbox.
    """
    # Create bundle on host (runs npm install with network access)
    # Run in thread pool since it's blocking I/O
    loop = asyncio.get_running_loop()
    bundle_data = await loop.run_in_executor(
        None, lambda: _create_bundle(version, platform)
    )

    await sandbox_exec(sandbox, f"mkdir -p {install_dir}", user="root")

    bundle_path = f"{install_dir}/gemini-bundle.tar.gz"
    await sandbox.write_file(bundle_path, bundle_data)

    result = await sandbox.exec(
        bash_command(f"tar -xzf {bundle_path} -C {install_dir} && rm -f {bundle_path}"),
        user="root",
        timeout=60,
    )

    if not result.success:
        raise RuntimeError(
            f"Failed to extract gemini-cli bundle:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # The gemini binary is at node_modules/.bin/gemini
    gemini_binary = f"{install_dir}/node_modules/.bin/gemini"

    result = await sandbox.exec(
        bash_command(f"test -x {gemini_binary}"),
        user=user,
    )
    if not result.success:
        raise RuntimeError(
            f"Gemini CLI binary not found after installation at {gemini_binary}"
        )

    return gemini_binary
