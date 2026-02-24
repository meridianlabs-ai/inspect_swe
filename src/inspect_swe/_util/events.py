from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from inspect_ai.event import Event, ModelEvent
from inspect_ai.log import transcript


@asynccontextmanager
async def capture_model_events() -> AsyncIterator[Callable[[ModelEvent], None]]:
    """Capture bridge ModelEvents and yield a function to apply them.

    Wraps the transcript's event logger to intercept completed
    ModelEvents emitted by model.generate(). Yields a callback
    that, given a JSONL-derived ModelEvent, finds the matching
    bridge event by message ID and copies span context onto it.

    Usage::

        with capture_model_events() as apply_bridge_event:
            async for event in claude_code_events(...):
                if isinstance(event, ModelEvent):
                    apply_bridge_event(event)
                else:
                    transcript()._event(event)
    """
    bridge_events: dict[str, ModelEvent] = {}
    original_logger = transcript()._event_logger

    def _capture(event: Event) -> None:
        if isinstance(event, ModelEvent) and event.pending is None:
            if event.output and event.output.choices:
                msg_id = event.output.choices[0].message.id
                if msg_id:
                    bridge_events[msg_id] = event
        if original_logger:
            original_logger(event)

    def _apply(jsonl_event: ModelEvent) -> None:
        """Replace bridge ModelEvent's span context with JSONL context."""
        msg_id = (
            jsonl_event.output.choices[0].message.id
            if jsonl_event.output and jsonl_event.output.choices
            else None
        )
        if msg_id and msg_id in bridge_events:
            bridge_event = bridge_events.pop(msg_id)
            bridge_event.span_id = jsonl_event.span_id
            bridge_event.timestamp = jsonl_event.timestamp

    transcript()._subscribe(_capture)
    try:
        yield _apply
    finally:
        if original_logger:
            transcript()._subscribe(original_logger)
        else:
            transcript()._event_logger = None
