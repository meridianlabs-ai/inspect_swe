"""Agents inside an inspect_ai sandbox identity scope.

inspect_ai's `sandbox_identity()` scope (entered by inspect_swarm's users
isolation) runs a member's sandbox calls as its own Linux user. `claude_code()`
and `codex_cli()` resolve their user, working directory and home through the
scope; every other agent refuses to run inside one.
"""

from typing import NamedTuple

from inspect_ai.util import SandboxEnvironment
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util._sandbox.identity import (
    SandboxIdentityError,
    sandbox_identity_active,
    sandbox_identity_cwd,
    sandbox_identity_user,
    sandbox_identity_vars,
)

SUPPORTED_AGENTS = "claude_code() and codex_cli()"
"""The agents that can run inside a sandbox identity scope."""


class AgentIdentity(NamedTuple):
    """The user an agent runs as, and where."""

    user: str | None
    """User to run the agent's commands as."""

    cwd: str | None
    """Working directory the caller or the scope asked for, if any."""

    home_env: dict[str, str] | None
    """`HOME` from the user's passwd entry inside a scope; `None` outside one,
    or when the user has no passwd entry."""

    scoped: bool
    """Whether a sandbox identity scope applies to the agent's sandbox."""


async def resolve_agent_identity(
    sandbox: str | None, user: str | None, cwd: str | None
) -> AgentIdentity:
    """Resolve the user, working directory and home an agent runs with.

    Outside a sandbox identity scope for the `sandbox` environment, `user` and
    `cwd` are returned unchanged. Inside one, `user` becomes the scope's user
    (an explicit other user raises `SandboxIdentityError`), an omitted `cwd`
    becomes the scope's, and `HOME` is looked up in the sandbox's passwd
    database rather than inherited from the image.
    """
    sbox = _scoped_sandbox(sandbox)
    if sbox is None:
        return AgentIdentity(user, cwd, None, False)
    return AgentIdentity(
        user=sandbox_identity_user(sbox, user, strict=True),
        cwd=sandbox_identity_cwd(sbox, cwd),
        home_env=await sandbox_identity_vars(sbox, None),
        scoped=True,
    )


def refuse_centaur_in_scope(agent: str, sandbox: str | None) -> None:
    """Raise if `agent`'s centaur mode is asked for inside a scope.

    Centaur mode runs the CLI under Inspect's human agent, which connects as
    the sandbox's default user rather than the scope's.
    """
    if _scoped_sandbox(sandbox) is not None:
        raise SandboxIdentityError(
            f"{agent}(centaur=...) does not support sandbox identity scopes "
            "(such as inspect_swarm's users isolation)."
        )


def refuse_sandbox_identity(agent: str, sandbox: str | None) -> None:
    """Raise if `agent` is run inside a scope for the `sandbox` environment."""
    if _scoped_sandbox(sandbox) is not None:
        raise SandboxIdentityError(
            f"{agent} does not support sandbox identity scopes (such as "
            f"inspect_swarm's users isolation). Only {SUPPORTED_AGENTS} can run "
            "inside one."
        )


def _scoped_sandbox(sandbox: str | None) -> SandboxEnvironment | None:
    """The `sandbox` environment, if a sandbox identity scope applies to it.

    The environment is looked up only while a scope is active, so outside
    one nothing changes, even where no sandbox is available.
    """
    if not sandbox_identity_active():
        return None
    sbox = sandbox_env(sandbox)
    return sbox if sandbox_identity_active(sbox) else None
