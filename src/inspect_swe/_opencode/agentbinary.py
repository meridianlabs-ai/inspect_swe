import json
import tarfile
from io import BytesIO
from typing import Any, Literal

from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.appdirs import package_cache_dir
from .._util.download import download_file, download_text_file
from .._util.node import (
    create_npm_bundle,
    ensure_node_available,
    install_npm_bundle,
)
from .._util.sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
)


async def ensure_opencode_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, list[str]]:
    """Install OpenCode and return its binary plus dependency bin directories."""
    platform = await detect_sandbox_platform(sandbox)

    node_binary = await ensure_node_available(sandbox, platform, user)
    dependency_bin_dirs = [
        node_binary.rsplit("/", 1)[0],
        await ensure_ripgrep_installed(sandbox, platform, user),
    ]

    opencode_version = await resolve_opencode_version(version)
    opencode_binary = await ensure_opencode_installed(
        sandbox, node_binary, opencode_version, platform, user
    )
    return opencode_binary, dependency_bin_dirs


async def resolve_opencode_version(
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
) -> str:
    """Resolve version string to an actual semver version."""
    if version in ["auto", "sandbox", "stable", "latest"]:
        release = await _fetch_latest_release()
        return str(release["tag_name"]).lstrip("v")

    return version


async def ensure_opencode_installed(
    sandbox: SandboxEnvironment,
    node_path: str,
    version: str,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install OpenCode via npm and return path to the opencode binary."""
    install_dir = f"{SANDBOX_INSTALL_DIR}/opencode"
    binary = f"{install_dir}/node_modules/.bin/opencode"

    result = await sandbox.exec(bash_command(f"test -x {binary}"), user=user)
    if result.success:
        result = await sandbox.exec(cmd=[node_path, binary, "--version"], user=user)
        if result.success and result.stdout.strip() == version:
            return binary

    async with concurrency("opencode-install", 1, visible=False):
        bundle_data = create_npm_bundle(
            package="opencode-ai",
            version=version,
            platform=platform,
            cache_name="opencode-bundles",
            ignore_scripts=True,
        )
        binary_path = await install_npm_bundle(
            sandbox=sandbox,
            bundle_data=bundle_data,
            install_dir=install_dir,
            binary_name="opencode",
            user=user,
        )
        await _run_opencode_postinstall(sandbox, node_path, install_dir, user)
        return binary_path


async def _run_opencode_postinstall(
    sandbox: SandboxEnvironment,
    node_path: str,
    install_dir: str,
    user: str | None,
) -> None:
    """Run the opencode-ai postinstall script inside the sandbox.

    The postinstall fetches the platform-specific opencode binary
    (e.g. ``opencode-linux-arm64``). We skip it on the host (host is
    typically darwin/arm64 and the linux-only bundle wouldn't contain
    a matching package) and run it here where the platform matches.
    """
    package_dir = f"{install_dir}/node_modules/opencode-ai"
    postinstall = f"{package_dir}/postinstall.mjs"

    exists = await sandbox.exec(bash_command(f"test -f {postinstall}"), user=user)
    if not exists.success:
        return

    result = await sandbox.exec(
        cmd=[node_path, "postinstall.mjs"],
        cwd=package_dir,
        user=user,
        timeout=120,
    )
    if not result.success:
        raise RuntimeError(
            f"opencode-ai postinstall failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


async def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest release from GitHub."""
    releases_url = "https://api.github.com/repos/anomalyco/opencode/releases"
    release_json = await download_text_file(f"{releases_url}/latest")
    return dict(json.loads(release_json))


RIPGREP_VERSION = "15.1.0"


def _ripgrep_target(platform: SandboxPlatform) -> str:
    match platform:
        case "linux-x64" | "linux-x64-musl":
            return "x86_64-unknown-linux-musl"
        case "linux-arm64" | "linux-arm64-musl":
            return "aarch64-unknown-linux-gnu"
    raise ValueError(f"Unsupported platform: {platform}")


async def ensure_ripgrep_installed(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install ripgrep and return its bin directory."""
    result = await sandbox.exec(bash_command("command -v rg"), user=user)
    if result.success:
        return result.stdout.strip().rsplit("/", 1)[0]

    install_dir = f"{SANDBOX_INSTALL_DIR}/ripgrep"
    bin_dir = f"{install_dir}/bin"
    binary = f"{bin_dir}/rg"

    result = await sandbox.exec(bash_command(f"test -x {binary}"), user=user)
    if result.success:
        return bin_dir

    async with concurrency("ripgrep-install", 1, visible=False):
        rg_data = await _ripgrep_binary(platform)
        await sandbox.exec(bash_command(f"mkdir -p {bin_dir}"), user="root")
        await sandbox.write_file(binary, rg_data)
        await sandbox.exec(bash_command(f"chmod +x {binary}"), user="root")
        return bin_dir


async def _ripgrep_binary(platform: SandboxPlatform) -> bytes:
    target = _ripgrep_target(platform)
    archive_name = f"ripgrep-{RIPGREP_VERSION}-{target}.tar.gz"
    cache_dir = package_cache_dir("ripgrep-binary-downloads")
    cache_path = cache_dir / f"ripgrep-{RIPGREP_VERSION}-{target}"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return f.read()

    url = (
        "https://github.com/BurntSushi/ripgrep/releases/download/"
        f"{RIPGREP_VERSION}/{archive_name}"
    )
    archive_data = await download_file(url)
    with tarfile.open(fileobj=BytesIO(archive_data), mode="r:gz") as tar:
        rg_member = next(m for m in tar.getmembers() if m.name.endswith("/rg"))
        extracted = tar.extractfile(rg_member)
        assert extracted is not None
        rg_data = extracted.read()

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(rg_data)
    return rg_data
