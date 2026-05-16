import tarfile
from io import BytesIO

from inspect_ai.util import SandboxEnvironment, concurrency

from .appdirs import package_cache_dir
from .download import download_file
from .sandbox import SANDBOX_INSTALL_DIR, SandboxPlatform, bash_command, sandbox_exec

RIPGREP_VERSION = "15.1.0"


def _ripgrep_target(platform: SandboxPlatform) -> str:
    match platform:
        case "linux-x64" | "linux-x64-musl":
            return "x86_64-unknown-linux-musl"
        case "linux-arm64" | "linux-arm64-musl":
            return "aarch64-unknown-linux-gnu"
    raise ValueError(f"Unsupported platform: {platform}")


async def ensure_ripgrep_available(
    sandbox: SandboxEnvironment,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Ensure ripgrep is available in the sandbox and return its bin directory."""
    result = await sandbox.exec(bash_command("command -v rg"), user=user)
    if result.success:
        return result.stdout.strip().rsplit("/", 1)[0]

    bin_dir = f"{SANDBOX_INSTALL_DIR}/ripgrep/bin"
    rg_path = f"{bin_dir}/rg"
    result = await sandbox.exec(bash_command(f"test -x {rg_path}"), user=user)
    if result.success:
        return bin_dir

    async with concurrency("ripgrep-install", 1, visible=False):
        rg_data = await _ripgrep_binary(platform)
        await sandbox_exec(sandbox, f"mkdir -p {bin_dir}", user="root")
        await sandbox.write_file(rg_path, rg_data)
        await sandbox_exec(sandbox, f"chmod +x {rg_path}", user="root")
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
