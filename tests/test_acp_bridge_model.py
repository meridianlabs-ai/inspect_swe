"""The ACP Claude Code and Codex agents serve bridged requests with their own model.

The sandbox bridge serves a model name that is not an alias, a resolver result
or the active model with the eval's model (UKGovernmentBEIS/inspect_ai#5701).
These agents may run a model other than the eval's, so they pin the bridge to
their own model, as the ACP Gemini agent does. Each test stops at the bridge
call, then resolves the names the scaffold sends through the bridge's own
resolver with the options the agent passed. No sandbox, no Docker, no API keys.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import ModuleType
from typing import Any

import anyio
import pytest
from inspect_ai.agent import AgentState
from inspect_ai.agent._bridge.util import resolve_inspect_model
from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.model._model import init_active_model
from inspect_ai.util import Store
from inspect_ai.util._store import init_subtask_store
from inspect_swe._codex_cli.config import GUARDIAN_MODEL_SLUG
from inspect_swe.acp import ACPAgent
from inspect_swe.acp._agents.claude_code import claude_code as acp_claude_code
from inspect_swe.acp._agents.claude_code.claude_code import ClaudeCode
from inspect_swe.acp._agents.codex_cli import codex_cli as acp_codex_cli
from inspect_swe.acp._agents.codex_cli.codex_cli import CodexCli

EVAL_MODEL = "mockllm/eval"
AGENT_MODEL = "mockllm/agent"


class _BridgeReached(Exception):
    pass


def _bridge_options(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    make_agent: Any,
    names: list[str],
    eval_model: Model | None = None,
) -> tuple[dict[str, Any], dict[str, Model]]:
    """Start the agent with the eval on `eval_model` and stop at the bridge.

    `eval_model` defaults to EVAL_MODEL.

    Returns the options the agent passed to `sandbox_agent_bridge()` and the
    model the bridge serves for each of `names`.
    """
    import inspect_swe.acp.agent as acp_agent_mod

    monkeypatch.setattr(acp_agent_mod, "sample_active", lambda: object())
    monkeypatch.setattr(module, "sandbox_env", lambda sandbox=None: object())

    options: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_bridge(state: AgentState, **kwargs: Any) -> AsyncIterator[Any]:
        options.update(kwargs)
        raise _BridgeReached()
        yield

    monkeypatch.setattr(module, "sandbox_agent_bridge", fake_bridge)

    async def run() -> dict[str, Model]:
        init_active_model(eval_model or get_model(EVAL_MODEL), GenerateConfig())
        init_subtask_store(Store())
        agent: ACPAgent = make_agent()
        with pytest.raises(_BridgeReached):
            async with agent._start_agent(AgentState(messages=[])):
                pass
        return {
            name: resolve_inspect_model(
                name, options["model_aliases"], options["model"]
            )
            for name in names
        }

    served = anyio.run(run)
    return options, served


def test_acp_claude_code_serves_every_name_with_the_agent_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the presented name (ANTHROPIC_MODEL and the unset tiers) and a name
    # Claude Code might send that the agent never configured
    names = ["agent", "claude-haiku-4-5"]
    options, served = _bridge_options(
        monkeypatch,
        acp_claude_code,
        lambda: ClaudeCode(model=AGENT_MODEL),
        names,
    )
    assert options["model"] == AGENT_MODEL
    for name in names:
        assert str(served[name]) == AGENT_MODEL, name


def test_acp_claude_code_keeps_tier_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _, served = _bridge_options(
        monkeypatch,
        acp_claude_code,
        lambda: ClaudeCode(
            model=AGENT_MODEL,
            haiku_model="mockllm/haiku",
            subagent_model="mockllm/subagent",
        ),
        ["agent", "haiku", "subagent", "claude-sonnet-4-5"],
    )
    assert str(served["agent"]) == AGENT_MODEL
    assert str(served["haiku"]) == "mockllm/haiku"
    assert str(served["subagent"]) == "mockllm/subagent"
    assert str(served["claude-sonnet-4-5"]) == AGENT_MODEL


@pytest.mark.parametrize("auto_review", [False, True])
def test_acp_codex_serves_every_name_with_the_agent_model(
    monkeypatch: pytest.MonkeyPatch, auto_review: bool
) -> None:
    # the config.toml model, a name Codex might send that the agent never
    # configured, and the auto-review guardian slug
    names = ["agent", "gpt-5.1-codex-mini", GUARDIAN_MODEL_SLUG]
    options, served = _bridge_options(
        monkeypatch,
        acp_codex_cli,
        lambda: CodexCli(model=AGENT_MODEL, auto_review=auto_review),
        names,
    )
    assert options["model"] == AGENT_MODEL
    for name in names:
        assert str(served[name]) == AGENT_MODEL, name


def test_acp_codex_guardian_keeps_the_agent_model_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a Model instance carries config the model= pin (a name) does not, so
    # the guardian slug is bound to the instance itself
    agent_model = get_model(AGENT_MODEL, config=GenerateConfig(temperature=0.25))
    _, served = _bridge_options(
        monkeypatch,
        acp_codex_cli,
        lambda: CodexCli(model=agent_model, auto_review=True),
        ["agent", GUARDIAN_MODEL_SLUG],
    )
    assert served["agent"] is agent_model
    assert served[GUARDIAN_MODEL_SLUG] is agent_model


def _config_bearing_eval_model() -> Model:
    return get_model(EVAL_MODEL, config=GenerateConfig(temperature=0.25))


def test_acp_claude_code_on_the_eval_model_serves_the_eval_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # with no model= the agent runs the eval's model; an unaliased name must
    # reach the eval's own instance and its config, not a fresh get_model()
    eval_model = _config_bearing_eval_model()
    _, served = _bridge_options(
        monkeypatch,
        acp_claude_code,
        lambda: ClaudeCode(),
        ["eval", "claude-haiku-4-5"],
        eval_model=eval_model,
    )
    assert served["eval"] is eval_model
    assert served["claude-haiku-4-5"] is eval_model


def test_acp_codex_on_the_eval_model_serves_the_eval_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_model = _config_bearing_eval_model()
    _, served = _bridge_options(
        monkeypatch,
        acp_codex_cli,
        lambda: CodexCli(auto_review=True),
        ["eval", "gpt-5.1-codex-mini", GUARDIAN_MODEL_SLUG],
        eval_model=eval_model,
    )
    for name, model in served.items():
        assert model is eval_model, name
