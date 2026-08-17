"""In-process cache for upstream version resolution.

Agents that install from npm resolve their version on every sample, so
without caching a single multi-sample eval issues one ``api.github.com``
request per sample. Unauthenticated requests are limited to 60/hour per IP,
which one ordinary eval can exhaust on its own.

Agents installed via ``AgentBinarySource`` have an equivalent cache in
``agentbinary.download_agent_binary_async``.
"""

import threading
from typing import Awaitable, Callable

from inspect_ai.util import concurrency

# Guards the cache dicts. Held only across synchronous dict access, never an
# await.
_resolve_lock = threading.Lock()
_resolved_versions: dict[str, str] = {}
# per key: (number of failed resolutions so far, exception from the latest
# one). Lets callers already queued behind a failing resolution share its
# exception instead of each retrying in turn.
_failed_resolutions: dict[str, tuple[int, Exception]] = {}


async def cached_version_resolution(
    key: str, resolve: Callable[[], Awaitable[str]]
) -> str:
    """Resolve a version, reusing the result for the process lifetime.

    Concurrent callers for the same key share a single resolution rather than
    each issuing their own request — including its failure, so a burst of
    callers hitting a rate-limited API produces one error, not one per caller.
    Failures are not cached: a call arriving after a failed resolution has
    completed retries.

    Args:
        key: Cache key identifying what is being resolved (e.g. ``"opencode"``).
        resolve: Called to resolve the version on a cache miss.

    Returns:
        The resolved version string.
    """
    with _resolve_lock:
        cached = _resolved_versions.get(key)
        if cached is not None:
            return cached
        failure = _failed_resolutions.get(key)
        failures_at_arrival = failure[0] if failure is not None else 0

    # serialize per key so a burst of samples starting together makes one
    # request rather than one each
    async with concurrency(f"version-resolution-{key}", 1, visible=False):
        # another sample may have resolved (or failed) while we were waiting
        with _resolve_lock:
            cached = _resolved_versions.get(key)
            if cached is not None:
                return cached
            failure = _failed_resolutions.get(key)
        if failure is not None and failure[0] > failures_at_arrival:
            raise failure[1]

        try:
            version = await resolve()
        except Exception as ex:
            with _resolve_lock:
                _failed_resolutions[key] = (failures_at_arrival + 1, ex)
            raise
        with _resolve_lock:
            _resolved_versions[key] = version
        return version
