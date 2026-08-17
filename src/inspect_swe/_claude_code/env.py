"""Agent-process environment for the Claude Code agent.

Bundles the environment handed to the Claude Code subprocess so
``claude_code()`` stays readable, and so the MCP-startup defaults below are
testable without standing up a sandbox.

The MCP defaults exist because Claude Code connects its MCP servers *without*
blocking its first model call, so the agent can reach the model before its
bridged tools exist. Measured on a 300-sample run at 250-way concurrency: 55
samples (18.3%) had a first model call whose only tool was the
``WaitForMcpServers`` placeholder, with ``mcp_servers: [{..., status:
pending}]``. Most recovered by calling that placeholder, but some reasoned from
the *server name* that the server was irrelevant to the task, answered in a
single turn, and were scored as a normal trajectory -- a silent false negative,
and on an attack task a sample recorded as "the agent resisted" that never had
the means to act.

Setting the connection to blocking removes the precondition rather than
detecting the symptom: the first model call already carries the task tools, so
the pending window is never observable by the model. Verified 18.3% -> 0% on
336 EKS samples and 250 Modal samples.
"""

from typing import Final

from .model import ClaudeCodeModels

#: Values Claude Code treats as an explicit "false" for boolean env vars.
#:
#: ``MCP_CONNECTION_NONBLOCKING`` has INVERTED polarity, which is a trap worth
#: naming: in the CLI the gate reads (transcribed from the bundled source)
#: ``let nonBlocking = !isFalsy(process.env.MCP_CONNECTION_NONBLOCKING)``, where
#: ``isFalsy`` is true *only* for one of these literals. The variable it feeds is
#: the NON-blocking flag, so leaving it unset -- or setting it to ``"1"`` or
#: ``"true"`` -- leaves connection non-blocking. Only one of these tokens makes
#: it block.
FALSY_ENV_VALUES: Final = frozenset({"0", "false", "no", "off"})

#: Make MCP connection blocking, with a budget that covers a slow sandbox.
#:
#: The blocking wait is bounded, so blocking alone is not sufficient on every
#: backend: with only the flag set, the toolless-start window went 18.3% -> 0%
#: on EKS but only 18.3% -> 6.8% on Modal, whose MCP startup is far slower.
#: Both budgets are therefore raised above realistic sandbox setup time so the
#: wait actually covers it.
BLOCKING_MCP_ENV: Final = {
    "MCP_CONNECTION_NONBLOCKING": "false",
    "MCP_TIMEOUT": "300000",
    "MCP_CONNECT_TIMEOUT_MS": "300000",
}


def claude_code_agent_env(
    *,
    bridge_port: int,
    models: ClaudeCodeModels,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for the Claude Code subprocess.

    Args:
        bridge_port: Port of the in-sandbox bridge the agent's API calls go to.
        models: Resolved presented identities (cosmetic; the bridge routes to
            the real model).
        env: Caller overrides, applied last so any default here can be
            replaced -- including the MCP startup defaults.

    Returns:
        The merged environment, caller values winning on conflict.
    """
    return {
        "ANTHROPIC_BASE_URL": f"http://localhost:{bridge_port}",
        "ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-DOq5tyLPrk9M4hPE",
        "ANTHROPIC_MODEL": models.presented,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": models.opus,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": models.sonnet,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": models.haiku,
        "CLAUDE_CODE_SUBAGENT_MODEL": models.subagent,
        "ANTHROPIC_SMALL_FAST_MODEL": models.haiku,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "IS_SANDBOX": "1",
        **BLOCKING_MCP_ENV,
    } | (env or {})
