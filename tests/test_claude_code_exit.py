import pytest
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput, StopReason
from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._claude_code.claude_code import _is_claude_code_refusal_exit


@pytest.mark.parametrize(
    ("exit_code", "stderr_data", "stop_reason", "expected"),
    [
        # Anthropic refusal -> Inspect maps it to content_filter
        (1, "", "content_filter", True),
        (1, "", "stop", False),
        (1, "", "tool_calls", False),
        (1, "", None, False),
        # genuine scaffold crash (stderr present) is not a refusal
        (1, "boom", "content_filter", False),
        # non-1 exit codes are never refusals
        (2, "", "content_filter", False),
    ],
)
def test_is_claude_code_refusal_exit(
    exit_code: int,
    stderr_data: str,
    stop_reason: StopReason | None,
    expected: bool,
) -> None:
    assert (
        _is_claude_code_refusal_exit(
            exit_code=exit_code,
            stderr_data=stderr_data,
            stop_reason=stop_reason,
        )
        is expected
    )


def _model_event(output: ModelOutput) -> ModelEvent:
    return ModelEvent(
        model="m",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=output,
    )


def test_last_stop_reason_tracks_latest_completed_event() -> None:
    consumer = LiveConsumer()
    assert consumer.last_stop_reason is None

    consumer.on_complete(_model_event(ModelOutput.from_content("m", "hi")))
    assert consumer.last_stop_reason == "stop"

    consumer.on_complete(
        _model_event(ModelOutput.from_content("m", "", stop_reason="content_filter"))
    )
    assert consumer.last_stop_reason == "content_filter"


def test_reset_clears_last_stop_reason() -> None:
    consumer = LiveConsumer()
    consumer.on_complete(
        _model_event(ModelOutput.from_content("m", "", stop_reason="content_filter"))
    )
    assert consumer.last_stop_reason == "content_filter"
    consumer.reset()
    assert consumer.last_stop_reason is None
