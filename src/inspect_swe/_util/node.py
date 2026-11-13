"""Node.js and npm utilities for sandbox environments.

Shared infrastructure for agent implementations that need Node.js
and/or npm packages installed in sandboxes.
"""

import json
import lzma
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

from inspect_ai.util import SandboxEnvironment, concurrency

from .appdirs import package_cache_dir
from .download import download_file
from .sandbox import SANDBOX_INSTALL_DIR, SandboxPlatform, bash_command, sandbox_exec

NODE_VERSION = "20.11.0"

NPM_RELEASE_COOLDOWN = timedelta(days=7)
"""How long an npm release must be public before it is installed on the host.

Applies to every package in the installed tree. A compromised or broken
release is usually found and removed or deprecated within this time.
"""

_STABLE_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


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


def npm_cooldown_cutoff() -> datetime:
    """Return the newest publish time an npm install may use (whole seconds)."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now - NPM_RELEASE_COOLDOWN


def resolve_npm_package_version(package: str) -> str:
    """Get the newest version of an npm package that is past the cooldown.

    Returns the ``latest`` version if it was published by
    ``npm_cooldown_cutoff()``, otherwise the highest plain ``X.Y.Z`` version
    below it that was. Unpublished versions keep their ``time`` entry but
    leave ``versions``, so they are skipped.

    This is not left to ``npm install {package}@latest --before``: when
    ``latest`` is inside the cooldown, npm prefers versions whose ``engines``
    accept the host's Node, which is not the sandbox's, so the result would
    depend on the host. ``npm view`` ignores ``--before``, so the choice is
    made here from one metadata request. Unlike npm's fallback, this does not
    skip deprecated versions.

    Requires ``npm`` to be available on the host.
    """
    if not shutil.which("npm"):
        raise RuntimeError(
            f"npm is required on the host to resolve the version of {package}. "
            "Please install Node.js (https://nodejs.org/) and ensure npm is on "
            "your PATH."
        )
    # an empty directory, as for create_npm_bundle, so a project .npmrc in the
    # caller's working directory applies to neither
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["npm", "view", package, "time", "versions", "dist-tags", "--json"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to resolve version of {package}: {result.stderr}")
    packument = json.loads(result.stdout)
    cutoff = npm_cooldown_cutoff()
    version = _latest_version_before(packument, cutoff)
    if version is None:
        raise RuntimeError(
            f"No release of {package} up to its latest version "
            f"({packument['dist-tags']['latest']}) was published before "
            f"{cutoff.isoformat()} (the {NPM_RELEASE_COOLDOWN.days}-day cooldown)."
        )
    return version


def _latest_version_before(packument: dict[str, Any], cutoff: datetime) -> str | None:
    latest = str(packument["dist-tags"]["latest"])
    versions = set(packument["versions"])
    published = {
        version: datetime.fromisoformat(time.replace("Z", "+00:00"))
        for version, time in packument["time"].items()
        if version in versions
    }
    if latest in published and published[latest] <= cutoff:
        return latest

    latest_key = _semver_key(re.split(r"[-+]", latest)[0])
    if latest_key is None:
        raise RuntimeError(f"Unexpected latest dist-tag version: {latest}")
    candidates = [
        (key, version)
        for version, time in published.items()
        if time <= cutoff
        and (key := _semver_key(version)) is not None
        and key < latest_key
    ]
    return max(candidates)[1] if candidates else None


def _semver_key(version: str) -> tuple[int, int, int] | None:
    """Return a sort key for a stable ``X.Y.Z`` version, or None otherwise."""
    match = _STABLE_SEMVER.match(version)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


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

    The install passes ``--before`` with ``npm_cooldown_cutoff()``, so every
    package in the tree is a release at least ``NPM_RELEASE_COOLDOWN`` old;
    a ``version`` published after the cutoff fails to install. It also passes ``--ignore-scripts``:
    install scripts would run on the host and build for the host's platform,
    not the sandbox's. The agent CLIs bundled with this need none
    (gemini-cli's optional ``node-pty`` and ``@github/keytar`` builds are
    unused; it loads the prebuilt ``@lydell/node-pty`` package instead).

    Args:
        package: npm package name (e.g. ``"@google/gemini-cli"``).
        version: Exact semver version string.
        platform: Target sandbox platform.
        cache_name: Subdirectory name inside the package cache for
            storing the tarball (also used as the cache-key prefix).
    """
    cpu = platform.split("-")[1]

    cache_dir = package_cache_dir(cache_name)
    # bundles cached before the cooldown and --ignore-scripts have no suffix,
    # so they are rebuilt rather than reused
    cache_path = cache_dir / f"{cache_name}-{version}-{platform}-noscripts.tar.gz"

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

        before = npm_cooldown_cutoff().isoformat()
        npm_cmd = [
            "npm",
            "install",
            "--no-audit",
            "--no-fund",
            "--os",
            "linux",
            "--cpu",
            cpu,
            "--ignore-scripts",
            "--before",
            before,
            # npm 11.17+ lets a user's min-release-age-exclude lift --before
            "--min-release-age-exclude=",
            # the host's Node is not the sandbox's, so its engines don't apply
            "--engine-strict=false",
        ]

        result = subprocess.run(
            npm_cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to run npm install for {package}@{version} with "
                f"releases published before {before} "
                f"(the {NPM_RELEASE_COOLDOWN.days}-day cooldown):\n"
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
