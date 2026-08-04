"""Provision the ``google-antigravity`` SDK into the sandbox at runtime.

Mirrors ``_gemini_cli/agentbinary.py`` (which provisions node + the Gemini CLI):
inspect_swe agents run against arbitrary sandbox images, so the agent runtime
must be ensured present. Here that runtime is the ``google-antigravity`` Python
SDK, which bundles the ``localharness`` engine that actually runs the agent.

If the SDK is already importable (e.g. baked into the image, as in agent-c's
images), provisioning is skipped. Otherwise it is installed into a dedicated
venv under ``/var/tmp`` (which requires the sandbox to have network egress;
no-egress images -- e.g. agent-c's network_mode:none sandboxes -- must bake the
SDK into the image instead). A venv is used rather than ``pip install --user``
because the runner executes with ``PYTHONNOUSERSITE=1``, which hides user-site
installs; the venv's own site-packages stays importable.
"""

from __future__ import annotations

from typing import Final

from inspect_ai.util import SandboxEnvironment, concurrency

# Candidate interpreters, most-specific first. agent-c images ship the SDK in
# ``/opt/venv``; generic images fall back to a system python.
_CANDIDATE_PYTHONS: tuple[str, ...] = (
    "/opt/venv/bin/python",
    "python3",
    "python",
)
_IMPORT_CHECK = "import google.antigravity"
# Runner processes execute with PYTHONNOUSERSITE=1 (sdk_execution_spec), so the
# presence probe and post-install verify must run under the same isolation: a
# user-site-only install would otherwise pass the probe here and then die with
# ModuleNotFoundError in the runner.
_RUNNER_IMPORT_ENV: Final = {"PYTHONNOUSERSITE": "1"}
# Default pin: the version validated end-to-end (3/3). Newer releases (e.g.
# 0.1.8) are not yet validated against the native bridge path.
_SDK_VERSION: Final = "0.1.7"
# Dedicated venv for the runtime-provisioned SDK on egress images. /var/tmp is
# world-writable so the unprivileged agent user can create it (mirrors the uv
# bundle path in _util/agentwheel.py).
_SDK_VENV_DIR: Final = "/var/tmp/.antigravity-sdk-venv"
_SDK_VENV_PYTHON: Final = f"{_SDK_VENV_DIR}/bin/python"


async def ensure_antigravity_sdk(
    sandbox: SandboxEnvironment,
    user: str | None = None,
    version: str = _SDK_VERSION,
) -> str:
    """Return the path to a sandbox python that can import ``google.antigravity``.

    Skips work when the SDK is already present; otherwise provisions it.
    """
    python = await _python_with_sdk(sandbox, user, version)
    if python is not None:
        return python
    return await _provision_antigravity_sdk(sandbox, user, version)


def _version_check_code(version: str) -> str:
    return (
        "import google.antigravity, importlib.metadata; "
        "assert importlib.metadata.version('google-antigravity') "
        f"== {version!r}"
    )


async def _python_with_sdk(
    sandbox: SandboxEnvironment,
    user: str | None,
    version: str,
) -> str | None:
    for python in _CANDIDATE_PYTHONS:
        result = await sandbox.exec(
            [python, "-c", _IMPORT_CHECK], user=user, env=_RUNNER_IMPORT_ENV
        )
        if result.success:
            return python
    # A venv provisioned by an earlier call (previous agent turn, concurrent
    # sample) is reused rather than reinstalled -- but only when it holds the
    # requested version, so a configurable `version` can never be silently
    # satisfied by a stale install; a mismatch falls through to provisioning,
    # which pip-installs the requested pin into the same venv.
    result = await sandbox.exec(
        [_SDK_VENV_PYTHON, "-c", _version_check_code(version)],
        user=user,
        env=_RUNNER_IMPORT_ENV,
    )
    if result.success:
        return _SDK_VENV_PYTHON
    return None


async def _provision_antigravity_sdk(
    sandbox: SandboxEnvironment,
    user: str | None,
    version: str,
) -> str:
    """Install the pinned ``google-antigravity`` into a dedicated venv (egress-only).

    Creates a venv under ``/var/tmp`` and installs the SDK into it, then returns
    that venv's python. A venv (rather than ``pip install --user``) is required
    because the runner sets ``PYTHONNOUSERSITE=1``, which would hide a user-site
    install. Requires network egress; no-egress images must bake the SDK in.
    """
    async with concurrency("antigravity-sdk-install", 1, visible=False):
        # Re-check under the lock in case a concurrent sample installed it.
        python = await _python_with_sdk(sandbox, user, version)
        if python is not None:
            return python

        base_python = await _base_python(sandbox, user)
        venv_python = _SDK_VENV_PYTHON
        install = await sandbox.exec(
            [
                "bash",
                "-c",
                f'set -e; "{base_python}" -m venv "{_SDK_VENV_DIR}"; '
                f'"{venv_python}" -m pip install --disable-pip-version-check '
                f'--no-input "google-antigravity=={version}"',
            ],
            user=user,
        )
        if not install.success:
            raise RuntimeError(
                "Failed to provision the google-antigravity SDK into the sandbox. "
                "Bake it into the image for no-egress sandboxes, or ensure the "
                f"sandbox has network egress. stderr:\n{install.stderr.strip()}"
            )
        # Version-checked verify: catches both a broken install and any drift
        # between the requested pin and what pip resolved.
        verify = await sandbox.exec(
            [venv_python, "-c", _version_check_code(version)],
            user=user,
            env=_RUNNER_IMPORT_ENV,
        )
        if not verify.success:
            raise RuntimeError(
                "google-antigravity installed but is not importable under the "
                f"runner's import isolation: {verify.stderr.strip()}"
            )
        return venv_python


async def _base_python(sandbox: SandboxEnvironment, user: str | None) -> str:
    # Need an interpreter that can build a venv with pip bootstrapped into it
    # (``python -m venv`` uses ensurepip), so check for both stdlib modules rather
    # than just ``--version``. Prefer a system interpreter over any active venv.
    for python in ("/usr/bin/python3", "python3", "python"):
        result = await sandbox.exec([python, "-c", "import ensurepip, venv"], user=user)
        if result.success:
            return python
    raise RuntimeError(
        "No python capable of creating a venv (needs the stdlib 'venv' and "
        "'ensurepip' modules) was found in the sandbox to install "
        "google-antigravity. Bake the SDK into the image for no-egress sandboxes."
    )
