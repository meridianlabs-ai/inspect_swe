import json
from typing import Any

import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_swe import pi
from inspect_swe._pi.config import pi_models_json, pi_result_error

from tests.conftest import get_available_sandboxes, run_example, skip_if_no_openai


def test_pi_registration() -> None:
    assert registry_info(pi()).name == "inspect_swe/pi"


@pytest.mark.parametrize("thinking", ["invalid", "", "HIGH"])
def test_pi_rejects_invalid_thinking(thinking: Any) -> None:
    with pytest.raises(ValueError, match="thinking"):
        pi(thinking=thinking)


@pytest.mark.parametrize("context_size", [0, -1])
def test_pi_rejects_invalid_context_size(context_size: int) -> None:
    with pytest.raises(ValueError, match="max_context_size"):
        pi(max_context_size=context_size)


@pytest.mark.parametrize("attempts", [0, -1])
def test_pi_rejects_invalid_attempts(attempts: int) -> None:
    with pytest.raises(ValueError, match="attempts"):
        pi(attempts=attempts)


@pytest.mark.parametrize("reasoning", [False, True])
def test_pi_model_config(reasoning: bool) -> None:
    config = json.loads(
        pi_models_json(
            port=3201,
            model="openai/gpt-5-mini",
            context_size=400000,
            max_tokens=128000,
            reasoning=reasoning,
        )
    )
    assert list(config["providers"]) == ["inspect"]
    provider = config["providers"]["inspect"]
    assert provider["baseUrl"] == "http://localhost:3201/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "inspect-bridge"
    assert provider["compat"]["supportsDeveloperRole"] is False
    model = provider["models"][0]
    assert model["id"] == "openai/gpt-5-mini"
    assert model["contextWindow"] == 400000
    assert model["maxTokens"] == 128000
    assert model["reasoning"] is reasoning


@pytest.mark.parametrize(
    "output", ["", "startup failure", "null\n[]\n42", '{"type":"agent_end"}']
)
def test_pi_incomplete_output_is_error(output: str) -> None:
    assert (
        pi_result_error(stdout=output) == "Pi exited without a completed assistant turn"
    )


@pytest.mark.parametrize("stop_reason", ["error", "aborted"])
def test_pi_assistant_failure_is_error(stop_reason: str) -> None:
    output = (
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": stop_reason,
                    "errorMessage": "Request failed",
                },
            }
        )
        + '\n{"type":"agent_end"}'
    )
    assert pi_result_error(stdout=output) == "Request failed"


@pytest.mark.parametrize("stop_reason", ["toolUse", "pending", None])
def test_pi_unfinished_assistant_is_error(stop_reason: str | None) -> None:
    output = (
        json.dumps(
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": stop_reason},
            }
        )
        + '\n{"type":"agent_end"}'
    )
    assert (
        pi_result_error(stdout=output) == "Pi exited without a completed assistant turn"
    )


def test_pi_complete_json_turn() -> None:
    output = (
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "content": [{"type": "text", "text": "First\u2028second"}],
                },
            }
        )
        + '\n{"type":"agent_end"}'
    )
    assert pi_result_error(stdout=output) is None
    assert (
        pi_result_error(stdout=output + '\n{"type":"agent_start"}')
        == "Pi exited without a completed assistant turn"
    )


@pytest.mark.slow
@pytest.mark.api
@skip_if_no_openai
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_pi_multi_call(sandbox: str) -> None:
    log = run_example(
        example="multi_call", agent="pi", model="openai/gpt-5-mini", sandbox=sandbox
    )[0]
    assert log.status == "success", log.error
    assert log.samples
    messages = log.samples[0].messages
    assert len([m for m in messages if isinstance(m, ChatMessageUser)]) == 4
    assert len([m for m in messages if isinstance(m, ChatMessageAssistant)]) >= 4
    assert "paris" in messages[-1].text.lower()
    system = messages[0].text
    assert system.count("Answer simple questions concisely") == 1


@pytest.mark.slow
@pytest.mark.api
@skip_if_no_openai
@pytest.mark.parametrize("example", ["skills", "multiple_attempts", "system_explorer"])
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_pi_examples(example: str, sandbox: str) -> None:
    log = run_example(
        example=example, agent="pi", model="openai/gpt-5-mini", sandbox=sandbox
    )[0]
    assert log.status == "success", log.error
    assert log.samples
    assert log.samples[0].error is None
    if example == "skills":
        content = "\n".join(message.text for message in log.samples[0].messages)
        assert "ALPHA-BRAVO-CHARLIE" in content
        assert "DELTA-ECHO-FOXTROT" in content
    elif example == "multiple_attempts":
        from inspect_ai.event import ScoreEvent

        assert (
            len(
                [
                    event
                    for event in log.samples[0].events
                    if isinstance(event, ScoreEvent)
                ]
            )
            == 2
        )
