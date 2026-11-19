"""Per-request state a bridge `ModelEventSink` shares across its callbacks.

`started`: whether inspect_ai called the sink's `on_start` for the generate now
running. An inspect_ai with `ModelEventSink.on_start` calls it before each
generate's events; an older one never does, and the sink then places each event
in `on_pending`. The mark is a context variable set by `on_start` in the
generate's own task and read by its `on_pending` there, so concurrent requests,
each in its own task, cannot overwrite each other's mark.

`remember_decision` / `remembered_decision`: the placement a sink's filter-time
classifier decided for the bridged request now being handled (its span, and for
Codex the requesting agent's V2 path). The bridge runs its filter (where the
classifier runs) and then the generation in the same task, but a user filter
may return a rewritten `GenerateInput`, or edit the message list in place, and
the generation then sees only the changed messages. Reading the decision back
in `on_start`, `on_complete` and `on_filter_output` keeps the transcript
placement and delegation tracking on the owner the filter was told about,
instead of re-deriving it from messages that may no longer identify it. The
decision is keyed on the sink and on the bridge's own per-request
`BridgeRequest` object, so it never applies to another request, even one that
shares the task. A user filter may itself run a nested bridged request in the
same task; `preserve_decision` (around the user filter, in `classify_filter`)
restores the enclosing request's decision when that filter returns.

`filter_generation` / `in_filter_generation`: a user filter may generate on its
own; `classify_filter` routes that generation to the sink so it is placed like
the bridge's own, and marks the sink's `on_complete` for it. Its output reaches
the agent only if the filter returns it, so the sink leaves delegation tracking
to `on_filter_output` there rather than registering tool calls the agent may
never see.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from inspect_ai.agent import current_bridge_request

_started: ContextVar[object | None] = ContextVar(
    "inspect_swe_model_event_sink_started", default=None
)


@dataclass(frozen=True)
class Decision:
    """A sink's filter-time placement for one bridged request."""

    span_id: str | None
    """Span the request's generate runs under."""
    requester_path: str | None = None
    """Codex: the V2 path of the agent that made the request, when known."""


_filter_generation: ContextVar[object | None] = ContextVar(
    "inspect_swe_model_event_sink_filter_generation", default=None
)

_decided: ContextVar[tuple[object, object, Decision] | None] = ContextVar(
    "inspect_swe_model_event_sink_decided", default=None
)


def mark_started(sink: object) -> None:
    _started.set(sink)


def started(sink: object) -> bool:
    return _started.get() is sink


def remember_decision(sink: object, decision: Decision) -> None:
    """Record `decision` as `sink`'s placement for the current bridged request.

    Without a current bridged request (a direct call outside the bridge) there
    is nothing to key the decision on, so any earlier one is cleared instead.
    """
    request = current_bridge_request()
    _decided.set((sink, request, decision) if request is not None else None)


def remembered_decision(sink: object) -> Decision | None:
    """The placement `sink` decided for this bridged request, if it decided one."""
    decided = _decided.get()
    request = current_bridge_request()
    if (
        decided is None
        or request is None
        or decided[0] is not sink
        or decided[1] is not request
    ):
        return None
    return decided[2]


@contextmanager
def preserve_decision() -> Iterator[None]:
    """Restore the current decision on exit, whatever nested requests decided.

    A user filter that awaits another bridged request in the same task runs
    that request's classifier, which records its own decision; on exit (normal
    or not) the enclosing request's decision is back in place for its
    generation.
    """
    token = _decided.set(_decided.get())
    try:
        yield
    finally:
        _decided.reset(token)


@contextmanager
def reinstate_decision(sink: object, decision: Decision | None) -> Iterator[None]:
    """Make `decision` `sink`'s placement for the current request in the block.

    A user filter's own generation reads the decision of the request the filter
    is handling, which a nested bridged request it awaited earlier may have
    replaced (`preserve_decision` restores it only once the filter returns).
    The filter-generation sink captures the decision when the filter starts and
    reinstates it around each callback it forwards, which are synchronous.
    """
    if decision is None:
        yield
        return
    token = _decided.set((sink, current_bridge_request(), decision))
    try:
        yield
    finally:
        _decided.reset(token)


@contextmanager
def filter_generation(sink: object) -> Iterator[None]:
    """Mark `sink`'s callbacks in the block as for a user filter's own generation."""
    token = _filter_generation.set(sink)
    try:
        yield
    finally:
        _filter_generation.reset(token)


def in_filter_generation(sink: object) -> bool:
    """Whether `sink` is completing a generation the user's filter ran itself."""
    return _filter_generation.get() is sink
