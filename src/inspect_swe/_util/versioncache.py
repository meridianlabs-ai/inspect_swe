"""In-process cache for upstream version resolution.

Agents that install from npm resolve their version on every sample (the
install itself is guarded by ``concurrency()``, but resolution runs ahead
of it), so without caching a single multi-sample eval issues one
``api.github.com`` request per sample. Unauthenticated requests are limited
to 60/hour per IP, which one ordinary eval can exhaust on its own.

Agents installed via ``AgentBinarySource`` have an equivalent cache in
``agentbinary.download_agent_binary_async``.

We use a ``threading.Lock`` (not ``anyio.Lock``) because it only guards
synchronous dict reads/writes — never held across an await — and avoids
issues with module-level anyio locks binding to a stale event loop across
multiple ``anyio.run()`` calls. No expiry: entries live for the process
lifetime. Two callers may race and both resolve, but this is benign (same
result) and merely costs one extra request.
"""

import threading
from typing import Awaitable, Callable

_resolve_lock = threading.Lock()
_resolved_versions: dict[str, str] = {}


async def cached_version_resolution(
    key: str, resolve: Callable[[], Awaitable[str]]
) -> str:
    """Resolve a version, reusing the result for the process lifetime.

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

    version = await resolve()

    with _resolve_lock:
        _resolved_versions[key] = version
    return version
