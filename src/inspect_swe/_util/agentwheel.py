import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Literal

from inspect_ai.util import SandboxEnvironment, concurrency
from inspect_ai.util import sandbox as sandbox_env

from .appdirs import package_cache_dir
from .sandbox import (
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
    sandbox_exec,
)
from .trace import trace


@dataclass
class AgentWheelSource:
    agent: Literal["mini-swe-agent"]  # Human-readable
    package: Literal["mini-swe-agent"]  # PyPI package
    binary: Literal["mini"]  # CLI entrypoint
    default_version: Literal["1.17.4"]  # Default stable version


async def ensure_agent_wheel_installed(
    source: AgentWheelSource,
    version: Literal["stable", "sandbox", "latest"] | str = "stable",
    user: str | None = None,
    sandbox: SandboxEnvironment | None = None,
) -> str:
    """Ensure a Python package agent is installed in the sandbox.

    Args:
        source: Agent wheel source configuration
        version: Version to install. Options:
            - "stable": Download and install the default version
            - "sandbox": Use only what's in sandbox, error if not found
            - "latest": Download and install latest version from PyPI
            - Specific version string (e.g., "1.17.4")
        user: User to run commands as in sandbox
        sandbox: Sandbox environment (uses default if not provided)

    Returns:
        Path to the agent binary in the sandbox
    """
    # resolve sandbox
    sandbox = sandbox or sandbox_env()

    # "sandbox" means only use what's already installed
    if version == "sandbox":
        result = await sandbox.exec(bash_command(f"which {source.binary}"), user=user)
        if result.success:
            binary_path = result.stdout.strip()
            trace(f"Using {source.agent} installed in sandbox: {binary_path}")
            return binary_path
        raise RuntimeError(f"unable to locate {source.agent} in sandbox")

    # "stable" means use default pinned version
    if version == "stable":
        version = source.default_version

    # "latest" means fetch latest from PyPI (version=None in download)
    if version == "latest":
        version = None  # type: ignore[assignment]

    # detect the sandbox target platform and python version
    platform = await detect_sandbox_platform(sandbox)
    python_version = await detect_python_version(sandbox, user)

    # use concurrency so multiple samples don't attempt the same download
    async with concurrency(f"{source.binary}-install", 1, visible=False):
        # download wheels (caching handled internally by download_wheels_tarball)
        tarball, resolved_version = download_wheels_tarball(
            package_name=source.package,
            version=version,
            platform=platform,
            python_version=python_version,
        )
        trace(f"Using {source.agent} wheels: {resolved_version} ({platform})")

        # write tarball to sandbox
        sandbox_tarball_path = f"/var/tmp/.{source.package}-{resolved_version}.tar.gz"
        await sandbox.write_file(sandbox_tarball_path, tarball)

        # extract and install (paths quoted for safety)
        install_cmd = f"""
set -e
mkdir -p "/var/tmp/{source.package}-wheels"
tar -xzf "{sandbox_tarball_path}" -C "/var/tmp/{source.package}-wheels"
pip install --no-index --find-links "/var/tmp/{source.package}-wheels" "{source.package}"
rm -rf "/var/tmp/{source.package}-wheels" "{sandbox_tarball_path}"
"""
        await sandbox_exec(sandbox, install_cmd, user=user)

        # verify installation and return binary path
        result = await sandbox.exec(bash_command(f"which {source.binary}"), user=user)
        if not result.success:
            raise RuntimeError(f"{source.agent} not found after installation")

        binary_path = result.stdout.strip()
        trace(f"Installed {source.agent}: {resolved_version} -> {binary_path}")
        return binary_path


#  Platform Mapping: SandboxPlatform -> pip platform strings
# https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/
PIP_PLATFORMS: dict[SandboxPlatform, str] = {
    "linux-x64": "manylinux2014_x86_64",
    "linux-arm64": "manylinux2014_aarch64",
    "linux-x64-musl": "musllinux_1_1_x86_64",
    "linux-arm64-musl": "musllinux_1_1_aarch64",
}


def platform_to_pip_platform(platform: SandboxPlatform) -> str:
    if platform not in PIP_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")
    return PIP_PLATFORMS[platform]


async def detect_python_version(
    sandbox: SandboxEnvironment,
    user: str | None = None,
) -> str:
    # Detects Python version in sandbox.
    result = await sandbox.exec(bash_command("python3 --version"), user=user)
    if not result.success:
        raise RuntimeError("Python 3 not found in sandbox (required for agent)")

    # Parse "Python 3.12.0" -> "312"
    match = re.search(r"Python (\S+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse Python version: {result.stdout}")
    return match.group(1)


# mirror of cleanup_cached_binaries in _util/agentbinary.py
def read_cached_wheels(cache_path: Path) -> bytes | None:
    if not cache_path.exists():
        return None
    cache_path.touch()  # Update access time for LRU cleanup
    return cache_path.read_bytes()


# mirror of read_cached_binary in _util/agentbinary.py
def write_cached_wheels(
    cache_path: Path,
    tarball: bytes,
    list_cached_fn: Callable[[], list[Path]],
    keep_count: int = 5,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(tarball)
    cleanup_wheel_cache(list_cached_fn, keep_count)


# mirror of _cleanup_binary_cache in _util/agentbinary.py
def cleanup_wheel_cache(
    list_cached_fn: Callable[[], list[Path]],
    keep_count: int = 5,
) -> None:
    cache_files = list_cached_fn()
    if len(cache_files) <= keep_count:
        return
    cache_files.sort(key=lambda f: f.stat().st_atime)
    for file_path in cache_files[:-keep_count]:
        file_path.unlink(missing_ok=True)


def _wheels_cache_dir(package_name: str) -> Path:
    # Normalize package name for filesystem
    safe_name = package_name.replace("-", "_").replace(".", "_")
    return package_cache_dir(f"{safe_name}-wheels")


def _wheels_cache_key(
    package_name: str, version: str, platform: SandboxPlatform, python_version: str
) -> str:
    """Generate cache filename for a version/platform/python combo."""
    return f"{package_name}-{version}-{platform}-py{python_version}.tar.gz"


def _list_cached_wheels(package_name: str) -> list[Path]:
    """List all cached wheel tarballs for a package."""
    return list(_wheels_cache_dir(package_name).glob(f"{package_name}-*.tar.gz"))


def download_wheels_tarball(
    package_name: str,
    version: str | None,
    platform: SandboxPlatform,
    python_version: str,
) -> tuple[bytes, str]:
    """Download all wheels for a package and its dependencies.

    Downloads wheels from PyPI for the specified platform and Python version,
    then bundles them into a tarball for offline installation in sandbox.
    Downloaded wheels are cached locally (retaining 5 most recent versions).

    Args:
        package_name: PyPI package name (e.g., "mini-swe-agent")
        version: Package version or None for latest
        platform: Target sandbox platform
        python_version: Python version without dots (e.g., "312")

    Returns:
        Tuple of (tarball_bytes, resolved_version)

    Raises:
        RuntimeError: If download fails or no wheels are found
    """
    # 1 check cache first
    if version is not None:
        cache_path = _wheels_cache_dir(package_name) / _wheels_cache_key(
            package_name, version, platform, python_version
        )
        cached = read_cached_wheels(cache_path)
        if cached is not None:
            return cached, version

    # 2 map platform and run pip download
    pip_platform = platform_to_pip_platform(platform)
    package_spec = f"{package_name}=={version}" if version else package_name

    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_dir = Path(tmpdir) / "wheels"
        wheel_dir.mkdir()
        # pip download docs https://pip.pypa.io/en/stable/cli/pip_download/
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            package_spec,
            "--dest",
            str(wheel_dir),
            "--platform",
            pip_platform,
            "--python-version",
            python_version,
            "--only-binary=:all:",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"pip download failed for {package_spec}: {result.stderr}"
            )
        # 3 collect downloaded wheels
        wheels = list(wheel_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"No wheels downloaded for {package_spec}")

        # 4 extract version from main package wheel filename
        # Wheel naming: {distribution}-{version}-{python}-{abi}-{platform}.whl
        # Distribution name uses underscores (PEP 427)
        resolved_version = version
        if not resolved_version:
            wheel_prefix = package_name.replace("-", "_")
            for wheel in wheels:
                if wheel.name.startswith(f"{wheel_prefix}-"):
                    resolved_version = wheel.name.split("-")[1]
                    break
            if not resolved_version:
                raise RuntimeError(
                    f"Could not determine package version from wheels "
                    f"(looking for prefix '{wheel_prefix}-')"
                )

        # 5 create tarball
        tarball_buffer = BytesIO()
        with tarfile.open(fileobj=tarball_buffer, mode="w:gz") as tar:
            for wheel_file in sorted(wheels):
                tar.add(wheel_file, arcname=wheel_file.name)

        tarball = tarball_buffer.getvalue()

        # 6 cache for future use (retains 5 most recent versions)
        cache_path = _wheels_cache_dir(package_name) / _wheels_cache_key(
            package_name, resolved_version, platform, python_version
        )
        write_cached_wheels(
            cache_path,
            tarball,
            lambda: _list_cached_wheels(package_name),
        )

        return tarball, resolved_version
