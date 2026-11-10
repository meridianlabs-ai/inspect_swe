"""Per-request state a bridge `ModelEventSink` shares across its callbacks.

`started`: whether inspect_ai called the sink's `on_start` for the generate now
running. An inspect_ai with `ModelEventSink.on_start` calls it before each
generate's events; an older one never does, and the sink then places each event
in `on_pending`. The mark is a context variable set by `on_start` in the
generate's own task and read by its `on_pending` there, so concurrent requests,
each in its own task, cannot overwrite each other's mark.

`remember_span` / `remembered_span`: the span a sink's filter-time classifier
chose for the bridged request now being handled. The bridge runs its filter
(where the classifier runs) and then the generation in the same task, but a
user filter may return a rewritten `GenerateInput`, and the generation then
sees only the rewritten messages. Reading the decision back in `on_start`
keeps the transcript placement on the owner the filter was told about instead
of re-deriving it from messages that may no longer identify it. The decision is
keyed on the sink and on the bridge's own per-request `BridgeRequest` object,
so it never applies to another request, even one that shares the task.
"""

from contextvars import ContextVar

from inspect_ai.agent import current_bridge_request

_started: ContextVar[object | None] = ContextVar(
    "inspect_swe_model_event_sink_started", default=None
)

_decided: ContextVar[tuple[object, object, str | None] | None] = ContextVar(
    "inspect_swe_model_event_sink_decided", default=None
)


def mark_started(sink: object) -> None:
    _started.set(sink)


def started(sink: object) -> bool:
    return _started.get() is sink


def remember_span(sink: object, span_id: str | None) -> None:
    """Record `span_id` as `sink`'s placement for the current bridged request.

    Without a current bridged request (a direct call outside the bridge) there
    is nothing to key the decision on, so any earlier one is cleared instead.
    """
    request = current_bridge_request()
    _decided.set((sink, request, span_id) if request is not None else None)


def remembered_span(sink: object) -> tuple[bool, str | None]:
    """`(True, span_id)` if `sink` decided a placement for this bridged request."""
    decided = _decided.get()
    request = current_bridge_request()
    if (
        decided is None
        or request is None
        or decided[0] is not sink
        or decided[1] is not request
    ):
        return False, None
    return True, decided[2]
