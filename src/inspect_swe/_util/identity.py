"""Agents inside an inspect_ai sandbox identity scope.

inspect_ai's `sandbox_identity()` scope (entered by inspect_swarm's users
isolation) runs a member's sandbox calls as its own Linux user. `claude_code()`
and `codex_cli()` resolve their user, working directory and home through the
scope; every other agent refuses to run inside one.
"""

from typing import NamedTuple

from inspect_ai.agent._bridge.sandbox import bridge as sandbox_bridge
from inspect_ai.tool._tools._skill import install as skill_install
from inspect_ai.util import SandboxEnvironment
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util._sandbox import identity as sandbox_identity_module
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

    Raises:
        SandboxIdentityError: Inside a scope, if `user` is another user, or if
            the installed inspect_ai is too old to run an agent there (see
            `inspect_ai_supports_scoped_agents()`).
    """
    sbox = _scoped_sandbox(sandbox)
    if sbox is None:
        return AgentIdentity(user, cwd, None, False)
    if not inspect_ai_supports_scoped_agents():
        raise SandboxIdentityError(
            "Running an agent inside a sandbox identity scope needs an inspect_ai "
            "release later than 0.3.290 (or main at commit ce19dd38 or later). "
            "Upgrade inspect_ai."
        )
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


def inspect_ai_supports_scoped_agents() -> bool:
    """Whether the installed inspect_ai can run an agent inside a scope.

    inspect_ai 0.3.289 has the sandbox identity scope, but its agent bridge
    serves any user in the sandbox; the peer check came later (the bridge
    passes the scope's user to its proxy under `_PEER_USER_ENV`). Releases up
    to 0.3.290 run a member's setup commands, and the skill installer's
    `chmod +x` for scripts, in the provider's default directory, which the
    member may not be able to enter (an image `WORKDIR` of `/root`). Later
    ones run them in the scope's working directory and from `/`, and say so
    with `SANDBOX_OPERATIONS_USE_SCOPE_CWD` and
    `INSTALL_SKILLS_AS_MEMBER_RUNS_FROM_ROOT`.
    """
    return (
        hasattr(sandbox_bridge, "_PEER_USER_ENV")
        and getattr(sandbox_identity_module, "SANDBOX_OPERATIONS_USE_SCOPE_CWD", False)
        is True
        and getattr(skill_install, "INSTALL_SKILLS_AS_MEMBER_RUNS_FROM_ROOT", False)
        is True
    )
