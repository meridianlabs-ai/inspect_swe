"""Helpers shared by the Scout event converters."""

from inspect_ai.event import Event, ModelEvent


def sum_scout_tokens(events: list[Event]) -> int:
    """Sum total tokens from converted Scout ModelEvent objects.

    Counts tokens from all ModelEvents, including loaded subagent events.

    Args:
        events: List of Inspect AI events

    Returns:
        Total token count across all ModelEvents
    """
    total = 0
    for event in events:
        if isinstance(event, ModelEvent) and event.output and event.output.usage:
            total += event.output.usage.total_tokens
    return total
