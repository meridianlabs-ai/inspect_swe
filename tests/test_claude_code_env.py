"""Unit tests for the Claude Code agent-process environment.

These exercise ``claude_code_agent_env`` directly rather than asserting on
source text, so they fail if the behavior changes rather than if the wording
does. No sandbox, no Docker, no API keys.

The MCP assertions are the load-bearing ones. Claude Code connects MCP servers
without blocking its first model call, which let 18.3% of samples reach the
model with only the ``WaitForMcpServers`` placeholder and produced silently
mis-scored toolless trajectories. The env below is what removes that window, and
``MCP_CONNECTION_NONBLOCKING`` has inverted polarity -- ``"1"`` leaves it
non-blocking -- so a test that merely asserts "the key is present" would pass
against a value that reintroduces the bug.
"""

from inspect_swe._claude_code.env import (
    BLOCKING_MCP_ENV,
    FALSY_ENV_VALUES,
    TRUTHY_ENV_VALUES,
    claude_code_agent_env,
)
from inspect_swe._claude_code.model import resolve_claude_code_models


def _env(**overrides: str) -> dict[str, str]:
    return claude_code_agent_env(
        bridge_port=13337,
        models=resolve_claude_code_models("mockllm/model", None),
        env=overrides or None,
    )


def test_bridge_port_is_wired_into_the_base_url() -> None:
    assert _env()["ANTHROPIC_BASE_URL"] == "http://localhost:13337"


def test_presented_model_identities_are_populated() -> None:
    env = _env()
    # every role resolves to the served model's name for a single-model run
    assert env["ANTHROPIC_MODEL"] == "model"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "model"
    # the subagent role is the exception: it never presents the primary slug
    # (see resolve_claude_code_models), so subagent traffic is distinguishable
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "model-subagent"


def test_mcp_connection_is_blocking_by_default() -> None:
    """The whole point: the agent must not start before its MCP tools exist."""
    value = _env()["MCP_CONNECTION_NONBLOCKING"]
    # Inverted polarity: only an explicitly falsy token makes connection block.
    # Asserting membership (not just presence) is what catches a regression to
    # "1"/"true", which would silently restore the non-blocking behavior.
    assert value in FALSY_ENV_VALUES, (
        f"MCP_CONNECTION_NONBLOCKING={value!r} does not disable non-blocking "
        f"connection; it must be one of {sorted(FALSY_ENV_VALUES)}"
    )


def test_mcp_startup_budgets_cover_a_slow_sandbox() -> None:
    """Blocking alone is not enough -- the wait is bounded.

    With only the blocking flag set, the toolless-start window went to 0% on
    EKS but remained 6.8% on Modal, whose MCP startup is far slower. Both
    budgets must exceed realistic sandbox setup time.
    """
    env = _env()
    for key in ("MCP_TIMEOUT", "MCP_CONNECT_TIMEOUT_MS"):
        assert int(env[key]) >= 120_000, (
            f"{key}={env[key]} is too small to cover a slow sandbox's MCP "
            "startup; the bounded wait would expire and fall through to a "
            "toolless start"
        )


def test_caller_env_overrides_defaults() -> None:
    env = _env(IS_SANDBOX="0", CUSTOM="x")
    assert env["IS_SANDBOX"] == "0"
    assert env["CUSTOM"] == "x"


def test_caller_can_override_the_mcp_defaults() -> None:
    """Escape hatch: the defaults are opinionated, not mandatory."""
    env = _env(MCP_CONNECTION_NONBLOCKING="1")
    assert env["MCP_CONNECTION_NONBLOCKING"] == "1"


def test_blocking_mcp_env_is_applied_verbatim() -> None:
    env = _env()
    assert {k: env[k] for k in BLOCKING_MCP_ENV} == dict(BLOCKING_MCP_ENV)


def test_auto_memory_is_disabled_by_default() -> None:
    """See ``DISABLE_AUTO_MEMORY_ENV`` for why a sandboxed agent wants this off."""
    value = _env()["CLAUDE_CODE_DISABLE_AUTO_MEMORY"]
    # Claude Code reads this with a truthy check: only these tokens disable.
    # Asserting membership (not just presence) catches a regression to a value
    # like "false" that would silently re-enable auto-memory.
    assert value in TRUTHY_ENV_VALUES, (
        f"CLAUDE_CODE_DISABLE_AUTO_MEMORY={value!r} does not disable "
        f"auto-memory; it must be one of {sorted(TRUTHY_ENV_VALUES)}"
    )


def test_caller_can_re_enable_auto_memory() -> None:
    """Escape hatch for tasks that deliberately study memory persistence.

    Claude Code treats an explicit falsy token as "enabled" (overriding even
    ``settings.json``), so a caller opts back in with ``"0"``.
    """
    env = _env(CLAUDE_CODE_DISABLE_AUTO_MEMORY="0")
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "0"


def test_acp_claude_code_disables_auto_memory_too() -> None:
    """The ACP adapter builds its own env; keep it from drifting."""
    from inspect_swe.acp._agents.claude_code.claude_code import _BRIDGE_SAFE_ENV

    assert _BRIDGE_SAFE_ENV["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] in TRUTHY_ENV_VALUES
