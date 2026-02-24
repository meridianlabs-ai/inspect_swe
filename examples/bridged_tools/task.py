from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.dataset import Sample
from inspect_ai.tool import Tool, tool
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli, gemini_cli


@tool
def secret_lookup() -> Tool:
    async def execute(key: str) -> str:
        """Look up a secret value by key.

        Args:
            key: The key to look up.
        """
        secrets = {
            "alpha": "ALPHA-SECRET-12345",
            "beta": "BETA-SECRET-67890",
            "gamma": "GAMMA-SECRET-ABCDE",
        }
        return secrets.get(key, f"Unknown key: {key}")

    return execute


@task
def bridged_tools_test(
    agent: Literal["claude_code", "codex_cli", "gemini_cli"] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    system_prompt = (
        "You have access to a secret_lookup tool via MCP. "
        "Use it to look up secret values when asked."
    )
    bridged_tools = [BridgedToolsSpec(name="secrets", tools=[secret_lookup()])]

    match agent:
        case "claude_code":
            solver = claude_code(
                system_prompt=system_prompt, bridged_tools=bridged_tools
            )
        case "codex_cli":
            solver = codex_cli(system_prompt=system_prompt, bridged_tools=bridged_tools)
        case "gemini_cli":
            solver = gemini_cli(
                system_prompt=system_prompt, bridged_tools=bridged_tools
            )

    return Task(
        dataset=[
            Sample(
                input="Use the secret_lookup tool to find the value for the key 'alpha'. Report the exact secret value you find.",
                target="ALPHA-SECRET-12345",
            )
        ],
        solver=solver,
        sandbox=sandbox,
    )
