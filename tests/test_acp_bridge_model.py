"""The ACP agents serve bridged requests with their own model.

The sandbox bridge serves a model name that is not an alias, a resolver result
or the active model with the eval's model (UKGovernmentBEIS/inspect_ai#5701).
These agents may run a model other than the eval's, so they pass a
`model_resolver` that serves every name outside their aliases with the agent's
own `Model` instance, keeping its config, base URL and model args. Each test
stops at the bridge call, then resolves the names the scaffold sends through
the bridge's own resolver with the options the agent passed. No sandbox, no
Docker, no API keys.
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
from inspect_swe import CodexAutoReview
from inspect_swe._codex_cli.config import GUARDIAN_MODEL_SLUG
from inspect_swe.acp import ACPAgent
from inspect_swe.acp._agents.claude_code import claude_code as acp_claude_code
from inspect_swe.acp._agents.claude_code.claude_code import ClaudeCode
from inspect_swe.acp._agents.codex_cli import codex_cli as acp_codex_cli
from inspect_swe.acp._agents.codex_cli.codex_cli import CodexCli
from inspect_swe.acp._agents.gemini_cli import gemini_cli as acp_gemini_cli
from inspect_swe.acp._agents.gemini_cli.gemini_cli import GeminiCli

EVAL_MODEL = "mockllm/eval"


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
                name,
                options["model_aliases"],
                options.get("model"),
                model_resolver=options.get("model_resolver"),
            )
            for name in names
        }

    served = anyio.run(run)
    return options, served


def _agent_model() -> Model:
    """An agent model that is not the eval's, with config, base URL and args."""
    return get_model(
        "mockllm/agent",
        config=GenerateConfig(temperature=0.25),
        base_url="http://agent.invalid/v1",
        custom_arg="x",
    )


def _eval_model() -> Model:
    return get_model(EVAL_MODEL, config=GenerateConfig(temperature=0.5))


def test_acp_claude_code_serves_every_name_with_the_agent_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the presented name (ANTHROPIC_MODEL and the unset tiers), a name Claude
    # Code might send that the agent never configured, and "inspect"
    agent_model = _agent_model()
    options, served = _bridge_options(
        monkeypatch,
        acp_claude_code,
        lambda: ClaudeCode(model=agent_model),
        ["agent", "claude-haiku-4-5", "inspect"],
    )
    for name, model in served.items():
        assert model is agent_model, name
    assert options.get("model") is None


def test_acp_claude_code_keeps_tier_models(monkeypatch: pytest.MonkeyPatch) -> None:
    agent_model = _agent_model()
    _, served = _bridge_options(
        monkeypatch,
        acp_claude_code,
        lambda: ClaudeCode(
            model=agent_model,
            haiku_model="mockllm/haiku",
            subagent_model="mockllm/subagent",
        ),
        ["agent", "haiku", "subagent", "claude-sonnet-4-5"],
    )
    assert served["agent"] is agent_model
    assert str(served["haiku"]) == "mockllm/haiku"
    assert str(served["subagent"]) == "mockllm/subagent"
    assert served["claude-sonnet-4-5"] is agent_model


@pytest.mark.parametrize("auto_review", [False, True])
def test_acp_codex_serves_every_name_with_the_agent_instance(
    monkeypatch: pytest.MonkeyPatch, auto_review: bool
) -> None:
    # the config.toml model, a name Codex might send that the agent never
    # configured, and the auto-review guardian slug
    agent_model = _agent_model()
    options, served = _bridge_options(
        monkeypatch,
        acp_codex_cli,
        lambda: CodexCli(model=agent_model, auto_review=auto_review),
        ["agent", "gpt-5.1-codex-mini", GUARDIAN_MODEL_SLUG],
    )
    for name, model in served.items():
        assert model is agent_model, name
    assert options.get("model") is None


def test_acp_codex_keeps_a_configured_guardian_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_model = _agent_model()
    _, served = _bridge_options(
        monkeypatch,
        acp_codex_cli,
        lambda: CodexCli(
            model=agent_model,
            auto_review=CodexAutoReview(model="mockllm/guardian"),
        ),
        ["gpt-5.1-codex-mini", GUARDIAN_MODEL_SLUG],
    )
    assert served["gpt-5.1-codex-mini"] is agent_model
    assert str(served[GUARDIAN_MODEL_SLUG]) == "mockllm/guardian"


def test_acp_gemini_serves_every_name_with_the_agent_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the --model id and one of Gemini CLI's hardcoded utility-model names
    agent_model = _agent_model()
    options, served = _bridge_options(
        monkeypatch,
        acp_gemini_cli,
        lambda: GeminiCli(model=agent_model),
        ["agent", "gemini-2.5-flash-lite"],
    )
    for name, model in served.items():
        assert model is agent_model, name
    assert options.get("model") is None


@pytest.mark.parametrize(
    "module,make_agent,names",
    [
        (acp_claude_code, ClaudeCode, ["eval", "claude-haiku-4-5"]),
        (
            acp_codex_cli,
            lambda: CodexCli(auto_review=True),
            ["eval", "gpt-5.1-codex-mini", GUARDIAN_MODEL_SLUG],
        ),
        (acp_gemini_cli, GeminiCli, ["eval", "gemini-2.5-flash-lite"]),
    ],
)
def test_acp_agent_on_the_eval_model_serves_the_eval_instance(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    make_agent: Any,
    names: list[str],
) -> None:
    # with no model= the agent runs the eval's model; every name must reach
    # the eval's own instance and its config, not a fresh get_model()
    eval_model = _eval_model()
    _, served = _bridge_options(
        monkeypatch, module, make_agent, names, eval_model=eval_model
    )
    for name, model in served.items():
        assert model is eval_model, name


def _classify(options: dict[str, Any], names: list[str]) -> dict[str, str]:
    """The agent-context kind the bridge filter the agent passed stamps per name."""
    from inspect_ai.agent import current_agent_bridge_context
    from inspect_ai.agent._bridge.context import bridged_request_scope
    from inspect_ai.model import ChatMessageUser

    async def run() -> dict[str, str]:
        kinds: dict[str, str] = {}
        for name in names:
            with bridged_request_scope(name):
                await options["filter"](
                    get_model("mockllm/model"),
                    [ChatMessageUser(content="hi")],
                    [],
                    None,
                    GenerateConfig(),
                )
                context = current_agent_bridge_context()
                assert context is not None, name
                kinds[name] = context.kind
        return kinds

    return anyio.run(run)


def test_acp_claude_code_default_subagent_slug_is_the_agent_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # with subagent_model unset Claude Code is given a distinct synthetic
    # sub-agent slug ("<presented>-subagent"); it is aliased to the agent's own
    # instance like the presented name, and every other name still falls back
    # to that instance through the resolver
    agent_model = _agent_model()
    names = ["agent", "agent-subagent", "claude-opus-4-1", "inspect"]
    options, served = _bridge_options(
        monkeypatch,
        acp_claude_code,
        lambda: ClaudeCode(model=agent_model),
        names,
    )
    assert "agent-subagent" in options["model_aliases"]
    for name, model in served.items():
        assert model is agent_model, name
    assert _classify(options, names[:3]) == {
        "agent": "root",
        "agent-subagent": "subagent",
        "claude-opus-4-1": "unknown",
    }


def test_acp_claude_code_routes_and_classifies_each_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # canonical tier and sub-agent names keep their own models and kinds; a
    # name Claude Code was never given is served by the agent's own instance
    agent_model = _agent_model()
    names = ["agent", "opus", "haiku", "subagent", "claude-sonnet-4-5"]
    options, served = _bridge_options(
        monkeypatch,
        acp_claude_code,
        lambda: ClaudeCode(
            model=agent_model,
            opus_model="mockllm/opus",
            haiku_model="mockllm/haiku",
            subagent_model="mockllm/subagent",
        ),
        names,
    )
    assert served["agent"] is agent_model
    assert str(served["opus"]) == "mockllm/opus"
    assert str(served["haiku"]) == "mockllm/haiku"
    assert str(served["subagent"]) == "mockllm/subagent"
    assert served["claude-sonnet-4-5"] is agent_model
    assert _classify(options, names) == {
        "agent": "root",
        "opus": "root",
        "haiku": "utility",
        "subagent": "subagent",
        "claude-sonnet-4-5": "unknown",
    }


def test_acp_claude_code_on_the_eval_model_routes_its_subagent_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # with no model= the presented and synthetic sub-agent names reach the
    # eval's own instance, as does any name outside the aliases
    eval_model = _eval_model()
    names = ["eval", "eval-subagent", "claude-haiku-4-5"]
    options, served = _bridge_options(
        monkeypatch, acp_claude_code, ClaudeCode, names, eval_model=eval_model
    )
    for name, model in served.items():
        assert model is eval_model, name
    assert _classify(options, names) == {
        "eval": "root",
        "eval-subagent": "subagent",
        "claude-haiku-4-5": "unknown",
    }
