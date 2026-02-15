import json
import lzma
import os
import shutil
import subprocess
import tarfile
import tempfile
from io import BytesIO
from typing import Any, Literal

from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.appdirs import package_cache_dir
from .._util.download import download_file, download_text_file
from .._util.sandbox import SandboxPlatform, bash_command, detect_sandbox_platform, sandbox_exec

# Node.js version to download if not available in sandbox
NODE_VERSION = "20.11.0"

# Installation paths in sandbox
SANDBOX_INSTALL_DIR = "/var/tmp/.5c95f967ca830048"


async def ensure_gemini_cli_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, str]:
    """Install node and gemini-cli in the sandbox.

    Returns (gemini_binary, node_binary) paths.
    """
    platform = await detect_sandbox_platform(sandbox)
    node_binary = await ensure_node_available(sandbox, platform, user)
    gemini_version = await resolve_gemini_version(version)
    gemini_binary = await ensure_gemini_cli_installed(
        sandbox, node_binary, gemini_version, platform, user
    )
    return gemini_binary, node_binary


async def resolve_gemini_version(
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
) -> str:
    """Resolve version string to an actual semver version."""
    if version in ["auto", "sandbox", "stable", "latest"]:
        release = await _fetch_latest_release()
        return str(release["tag_name"]).lstrip("v")

    return version


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

        # Extract the node binary from the tar
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

    if not shutil.which("npm"):
        raise RuntimeError(
            "npm is required on the host to bundle gemini-cli for the sandbox. "
            "Please install Node.js (https://nodejs.org/) and ensure npm is on your PATH."
        )

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
    bundle_data = _create_bundle(version, platform)

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
