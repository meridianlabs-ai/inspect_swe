"""Install agent binaries that several sandbox users run.

Inside a sandbox identity scope (inspect_swarm's users isolation), each member
runs as its own user, but agent binaries are installed once at fixed paths
under `SANDBOX_INSTALL_DIR` and run by every member. A member that could place
or change such a file would run code as each peer that uses it. So inside a
scope these artifacts follow inspect_ai's verified-directory and
verify-before-use contracts (design/sandbox-trust-model.md in inspect_ai):

- they are provisioned as the sandbox's default user (`user=None`), through
  inspect_ai's framework-directory helpers, which mark their own sandbox calls
  as framework operations so the scope does not apply to them;
- the install directory is created or adopted only if that user owns it and
  no other user can write to it, and content is written into the verified
  directory from stdin, never staged at a path another user controls;
- an existing artifact is used only if every file and directory in it is
  owned by that user and not writable by group or others, and is refused
  otherwise rather than repaired.

Outside a scope the installer in `agentbinary.py` is unchanged.
"""

import posixpath

from inspect_ai.util import SandboxEnvironment
from inspect_ai.util._sandbox._framework_directory import (
    ensure_framework_directory,
    exec_in_framework_directory,
    write_file_in_framework_directory,
)

INSTALL_DIR_MODE = 0o755
"""Mode of install directories: others can read and run, only the owner writes."""

INSTALL_FILE_MODE = 0o755
"""Mode of a single-binary artifact."""

# Run in a verified directory. Prints "missing" when the executable $2 does
# not exist, otherwise checks that $2 is an executable regular file and that
# every entry under $1 is owned by the user running the check and (symbolic
# links aside, whose mode means nothing) is not writable by group or others,
# and prints "ok". A failed check is reported on stderr with exit status 3.
_CHECK = """
uid=$(id -u) || exit 1
if [ ! -e "$2" ] && [ ! -L "$2" ]; then echo missing; exit 0; fi
if [ -L "$2" ] || [ ! -f "$2" ] || [ ! -x "$2" ]; then
    echo "$2 is not an executable regular file" >&2
    exit 3
fi
bad=$(find "$1" \\( ! -user "$uid" -o \\( ! -type l \\( -perm -020 -o -perm -002 \\) \\) \\) -print) || exit 1
if [ -n "$bad" ]; then
    printf 'not owned by uid %s, or writable by group or others: %s\\n' \\
        "$uid" "$(printf '%s\\n' "$bad" | head -n 3)" >&2
    exit 3
fi
echo ok
"""

# Run in a verified package directory with the archive on stdin. Extracts it
# with the extracting user as owner (-o), then gives every entry but symbolic
# links the modes others need to read and run it, with no group or other
# write. The helper's umask 077 would otherwise leave a non-root user's
# extraction private.
_EXTRACT = "tar -x -z -o -f - && find . ! -type l -exec chmod u+rwX,go+rX,go-w {} +"


async def install_shared_artifact(
    sandbox: SandboxEnvironment,
    install_path: str,
    data: bytes,
    package_entrypoint: str | None,
) -> None:
    """Install an agent artifact at `install_path` for use inside a scope.

    A single binary (`package_entrypoint` is `None`) is published as the file
    `install_path`. A package archive is extracted into the directory
    `install_path`, whose `package_entrypoint` is the binary. An artifact
    already there is verified and reused, not rewritten.

    Args:
        sandbox: Sandbox to install into.
        install_path: Absolute path of the artifact, directly under an
            install directory whose ancestors are root-owned.
        data: The binary, or the package's gzipped tar archive.
        package_entrypoint: Path of the binary inside the package, relative
            to `install_path`; `None` for a single binary.

    Raises:
        RuntimeError: An existing artifact fails verification (it is left as
            it is), or the artifact cannot be written or extracted.
        FrameworkDirectoryError: An install directory fails the
            verified-directory contract.
    """
    install_dir, name = posixpath.split(install_path)
    await ensure_framework_directory(
        sandbox, install_dir, user=None, mode=INSTALL_DIR_MODE
    )

    if package_entrypoint is None:
        target = f"./{name}"
        if await _check(sandbox, install_dir, target, target) == "missing":
            await write_file_in_framework_directory(
                sandbox,
                install_dir,
                name,
                data,
                user=None,
                mode=INSTALL_DIR_MODE,
                file_mode=INSTALL_FILE_MODE,
            )
            await _check(sandbox, install_dir, target, target, required=True)
        return

    await ensure_framework_directory(
        sandbox, install_path, user=None, mode=INSTALL_DIR_MODE
    )
    entrypoint = f"./{package_entrypoint}"
    if await _check(sandbox, install_path, ".", entrypoint) == "missing":
        result = await exec_in_framework_directory(
            sandbox,
            install_path,
            ["sh", "-c", _EXTRACT],
            user=None,
            mode=INSTALL_DIR_MODE,
            input=data,
        )
        if not result.success:
            raise RuntimeError(
                f"Unable to extract {install_path}: {result.stderr.strip()}"
            )
        await _check(sandbox, install_path, ".", entrypoint, required=True)


async def _check(
    sandbox: SandboxEnvironment,
    directory: str,
    tree: str,
    executable: str,
    required: bool = False,
) -> str:
    """Verify `tree` and `executable` inside the verified `directory`.

    Returns "ok", or "missing" when `executable` does not exist (an error
    when `required`).
    """
    result = await exec_in_framework_directory(
        sandbox,
        directory,
        ["sh", "-c", _CHECK, "sh", tree, executable],
        user=None,
        mode=INSTALL_DIR_MODE,
    )
    path = posixpath.normpath(posixpath.join(directory, executable))
    status = result.stdout.strip()
    if not result.success or status not in ("ok", "missing"):
        raise RuntimeError(
            f"{path} cannot be used inside a sandbox identity scope: "
            f"{result.stderr.strip() or status}. Remove "
            f"{posixpath.normpath(posixpath.join(directory, tree))} so it can be "
            "installed again."
        )
    if status == "missing" and required:
        raise RuntimeError(f"{path} was not found after installation.")
    return status
