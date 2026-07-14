import io
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Final

from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.sandbox import SANDBOX_INSTALL_DIR, bash_command

DEFAULT_BINARY_SOURCE: Final = Path("/tmp/opencode/agy-bin/antigravity")
DEFAULT_TOKEN_SOURCE: Final = Path.home() / ".gemini" / "antigravity-cli"
EXCLUDED_TOKEN_DIRECTORIES: Final = frozenset(
    {"log", "cache", "crashes", "conversations", "scratch"}
)
TOKEN_ARCHIVE_PATH: Final = "/tmp/antigravity-cli-home.tgz"


def _include_token_member(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    member_path = PurePosixPath(member.name)
    if member.isdir() and member_path.name in EXCLUDED_TOKEN_DIRECTORIES:
        return None
    if any(part in EXCLUDED_TOKEN_DIRECTORIES for part in member_path.parts[:-1]):
        return None
    return member


async def ensure_antigravity_cli_setup(
    sandbox: SandboxEnvironment, *, binary_source: str | None, user: str | None
) -> str:
    """Install agy in the sandbox and return its absolute path."""
    configured_source = binary_source
    if configured_source is None:
        configured_source = os.environ.get("ANTIGRAVITY_CLI_BINARY")
    source = Path(configured_source) if configured_source is not None else DEFAULT_BINARY_SOURCE
    if not source.is_file():
        raise FileNotFoundError(f"Antigravity CLI binary source not found: {source}")

    install_dir = f"{SANDBOX_INSTALL_DIR}/antigravity-cli"
    agy_path = f"{install_dir}/agy"
    result = await sandbox.exec(bash_command(f"test -x {agy_path}"), user=user)
    if result.success:
        return agy_path

    async with concurrency("antigravity-cli-install", 1, visible=False):
        await sandbox.write_file(agy_path, source.read_bytes())
        # write_file() creates the binary owned by root, so the chmod must run as
        # root (the owner); running it as the unprivileged agent user fails with
        # EPERM. 0755 leaves it world-executable so the agent user can still exec it.
        result = await sandbox.exec(bash_command(f"chmod 0755 {agy_path}"))
        if not result.success:
            raise RuntimeError(f"Unable to make Antigravity CLI executable: {result.stderr}")

    return agy_path


async def provision_antigravity_home(
    sandbox: SandboxEnvironment,
    *,
    sandbox_home: str,
    token_source: str | None,
    user: str | None,
) -> str:
    """Copy the Antigravity CLI token home into the sandbox."""
    configured_source = token_source
    if configured_source is None:
        configured_source = os.environ.get("ANTIGRAVITY_CLI_HOME")
    source = Path(configured_source) if configured_source is not None else DEFAULT_TOKEN_SOURCE
    if not source.is_dir():
        raise FileNotFoundError(f"Antigravity CLI token source not found: {source}")

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz", encoding="utf-8") as archive:
        archive.add(
            source,
            arcname=source.name,
            filter=_include_token_member,
        )

    gemini_home = f"{sandbox_home}/.gemini"
    result = await sandbox.exec(bash_command(f"mkdir -p {gemini_home}"), user=user)
    if not result.success:
        raise RuntimeError(f"Unable to create Antigravity CLI home: {result.stderr}")
    await sandbox.write_file(TOKEN_ARCHIVE_PATH, archive_buffer.getvalue())
    result = await sandbox.exec(
        bash_command(f"tar -xzf {TOKEN_ARCHIVE_PATH} -C {gemini_home}"),
        user=user,
    )
    if not result.success:
        raise RuntimeError(f"Unable to extract Antigravity CLI tokens: {result.stderr}")

    return f"{gemini_home}/antigravity-cli"
