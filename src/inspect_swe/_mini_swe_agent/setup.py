"""Sandbox setup for mini-swe-agent multi-turn support.

Handles writing the resumable agent to the sandbox, building the
mini CLI environment, and generating stable trajectory paths for
conversation state persistence.
"""

import re
import uuid
from importlib import resources

from inspect_ai.util import SandboxEnvironment, store

_MIN_MAJOR_VERSION = 2
_VERSION_KEYWORDS = {"stable", "sandbox", "latest"}
_VERSION_RE = re.compile(r"^(\d+)\.")


def validate_version(version: str) -> None:
    """Reject mini-swe-agent versions older than v2.

    v1.x used a different CLI interface and is not compatible with the
    resumable-agent.
    """
    if version in _VERSION_KEYWORDS:
        return
    m = _VERSION_RE.match(version)
    if not m:
        raise ValueError(
            f"Invalid mini-swe-agent version: {version!r}. "
            f"Use a valid version (>= {_MIN_MAJOR_VERSION}.0.0), or one of: "
            f"{', '.join(sorted(_VERSION_KEYWORDS))}."
        )
    if int(m.group(1)) < _MIN_MAJOR_VERSION:
        raise ValueError(
            f"mini-swe-agent version {version} is not supported. "
            f"This integration requires v{_MIN_MAJOR_VERSION}.0.0 or later (v2 CLI). "
            f"Use version='stable' for the default pinned version."
        )


RESUMABLE_AGENT_PATH = "/var/tmp/resumable_agent.py"
_TRAJECTORY_STORE_KEY = "mini_swe_agent_trajectory_path"


def _read_resumable_agent_source() -> str:
    """Read the resumable_agent.py source from this package."""
    return resources.files(__package__).joinpath("resumable_agent.py").read_text()


async def install_resumable_agent(sbox: SandboxEnvironment) -> None:
    """Write resumable_agent.py to sandbox for --agent-class loading."""
    source = _read_resumable_agent_source()
    await sbox.write_file(RESUMABLE_AGENT_PATH, source)


def get_trajectory_path() -> str:
    """Get or create a stable trajectory path for this sandbox session.

    The path is generated once per sample (stored in inspect's store())
    and reused across multi-turn calls to preserve conversation state.
    """
    path: str | None = store().get(_TRAJECTORY_STORE_KEY, None)
    if path is None:
        path = f"/var/tmp/.mini-swe-trajectory-{uuid.uuid4()}.json"
        store().set(_TRAJECTORY_STORE_KEY, path)
    return path
