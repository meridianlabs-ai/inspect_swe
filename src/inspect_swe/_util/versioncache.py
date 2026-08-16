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

# Guards the cache dict. Held only across synchronous dict access, never an
# await.
_resolve_lock = threading.Lock()
_resolved_versions: dict[str, str] = {}


async def cached_version_resolution(
    key: str, resolve: Callable[[], Awaitable[str]]
) -> str:
    """Resolve a version, reusing the result for the process lifetime.

    Concurrent callers for the same key share a single resolution rather than
    each issuing their own request. Failures are not cached, so a later caller
    retries.

    Args:
        key: Cache key identifying what is being resolved (e.g. ``"opencode"``).
        resolve: Called to resolve the version on a cache miss.

    Returns:
        The resolved version string.
    """
    cached = _cached(key)
    if cached is not None:
        return cached

    # serialize per key so a burst of samples starting together makes one
    # request rather than one each
    async with concurrency(f"version-resolution-{key}", 1, visible=False):
        # another sample may have resolved while we were waiting
        cached = _cached(key)
        if cached is not None:
            return cached

        version = await resolve()
        with _resolve_lock:
            _resolved_versions[key] = version
        return version


def _cached(key: str) -> str | None:
    with _resolve_lock:
        return _resolved_versions.get(key)
