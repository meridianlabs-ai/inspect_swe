"""Whether inspect_ai called a model event sink's `on_start` for the generate now running.

An inspect_ai with `ModelEventSink.on_start` calls it before each generate's events; an older
one never does, and the sink then places each event in `on_pending`. The mark is a context
variable set by `on_start` in the generate's own task and read by its `on_pending` there, so
concurrent requests, each in its own task, cannot overwrite each other's mark.
"""

from contextvars import ContextVar

_started: ContextVar[object | None] = ContextVar(
    "inspect_swe_model_event_sink_started", default=None
)


def mark_started(sink: object) -> None:
    _started.set(sink)


def started(sink: object) -> bool:
    return _started.get() is sink
