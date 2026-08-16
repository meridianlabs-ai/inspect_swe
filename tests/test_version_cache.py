"""Unit tests for in-process caching of upstream version resolution.

npm-installed agents resolve their version on every sample, so without
caching a single multi-sample eval issues one api.github.com request per
sample and can exhaust the 60/hour unauthenticated rate limit on its own.
"""

from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_swe._gemini_cli import agentbinary as gemini_agentbinary
from inspect_swe._opencode import agentbinary as opencode_agentbinary
from inspect_swe._util import versioncache
from inspect_swe._util.versioncache import cached_version_resolution


@pytest.fixture(autouse=True)
def clear_version_cache() -> Iterator[None]:
    versioncache._resolved_versions.clear()
    yield
    versioncache._resolved_versions.clear()


def test_cached_version_resolution_resolves_once() -> None:
    calls = 0

    async def resolve() -> str:
        nonlocal calls
        calls += 1
        return "1.2.3"

    async def run() -> None:
        assert await cached_version_resolution("agent", resolve) == "1.2.3"
        assert await cached_version_resolution("agent", resolve) == "1.2.3"

    anyio.run(run)
    assert calls == 1


def test_cached_version_resolution_keys_are_independent() -> None:
    async def run() -> None:
        assert await cached_version_resolution("a", _const("1.0.0")) == "1.0.0"
        assert await cached_version_resolution("b", _const("2.0.0")) == "2.0.0"
        # each key keeps its own entry
        assert await cached_version_resolution("a", _const("9.9.9")) == "1.0.0"

    anyio.run(run)


def _const(value: str) -> Any:
    async def resolve() -> str:
        return value

    return resolve


@pytest.mark.parametrize(
    "module,resolve_name,cache_key",
    [
        (gemini_agentbinary, "resolve_gemini_version", "gemini-cli"),
        (opencode_agentbinary, "resolve_opencode_version", "opencode"),
    ],
)
def test_latest_resolution_fetches_once_across_calls(
    module: Any, resolve_name: str, cache_key: str
) -> None:
    """Repeated 'latest' resolution (one per sample) hits the API once."""
    resolve = getattr(module, resolve_name)
    fetch = AsyncMock(return_value={"tag_name": "v1.2.3"})

    async def run() -> None:
        with patch.object(module, "_fetch_latest_release", fetch):
            for _ in range(5):
                assert await resolve("latest") == "1.2.3"

    anyio.run(run)
    fetch.assert_awaited_once()


@pytest.mark.parametrize(
    "module,resolve_name",
    [
        (gemini_agentbinary, "resolve_gemini_version"),
        (opencode_agentbinary, "resolve_opencode_version"),
    ],
)
def test_version_aliases_share_a_cache_entry(module: Any, resolve_name: str) -> None:
    """auto/sandbox/stable/latest all mean 'the latest release' -> one fetch."""
    resolve = getattr(module, resolve_name)
    fetch = AsyncMock(return_value={"tag_name": "v1.2.3"})

    async def run() -> None:
        with patch.object(module, "_fetch_latest_release", fetch):
            for alias in ("auto", "sandbox", "stable", "latest"):
                assert await resolve(alias) == "1.2.3"

    anyio.run(run)
    fetch.assert_awaited_once()


@pytest.mark.parametrize(
    "module,resolve_name",
    [
        (gemini_agentbinary, "resolve_gemini_version"),
        (opencode_agentbinary, "resolve_opencode_version"),
    ],
)
def test_explicit_version_never_fetches(module: Any, resolve_name: str) -> None:
    """An explicit semver is returned as-is and is not cached as 'latest'."""
    resolve = getattr(module, resolve_name)
    fetch = AsyncMock(return_value={"tag_name": "v9.9.9"})

    async def run() -> None:
        with patch.object(module, "_fetch_latest_release", fetch):
            assert await resolve("1.14.30") == "1.14.30"

    anyio.run(run)
    fetch.assert_not_awaited()
    assert versioncache._resolved_versions == {}


@pytest.mark.parametrize(
    "module,resolve_name",
    [
        (gemini_agentbinary, "resolve_gemini_version"),
        (opencode_agentbinary, "resolve_opencode_version"),
    ],
)
def test_concurrent_resolution_makes_one_request(
    module: Any, resolve_name: str
) -> None:
    """A cold-start burst of samples shares one request, not one each."""
    resolve = getattr(module, resolve_name)
    calls = 0
    results: list[str] = []

    async def fetch() -> dict[str, str]:
        nonlocal calls
        calls += 1
        # suspend so the other tasks reach the cache miss while this is
        # in flight — without single-flight they would each fetch too
        await anyio.sleep(0.05)
        return {"tag_name": "v1.2.3"}

    async def run() -> None:
        with patch.object(module, "_fetch_latest_release", fetch):

            async def one() -> None:
                results.append(await resolve("latest"))

            async with anyio.create_task_group() as tg:
                for _ in range(10):
                    tg.start_soon(one)

    anyio.run(run)
    assert results == ["1.2.3"] * 10
    assert calls == 1


@pytest.mark.parametrize(
    "module,resolve_name",
    [
        (gemini_agentbinary, "resolve_gemini_version"),
        (opencode_agentbinary, "resolve_opencode_version"),
    ],
)
def test_failed_resolution_is_not_cached(module: Any, resolve_name: str) -> None:
    """A failed fetch doesn't poison the cache; the next caller retries."""
    resolve = getattr(module, resolve_name)
    fetch = AsyncMock(side_effect=[RuntimeError("boom"), {"tag_name": "v1.2.3"}])

    async def run() -> None:
        with patch.object(module, "_fetch_latest_release", fetch):
            with pytest.raises(RuntimeError):
                await resolve("latest")
            assert await resolve("latest") == "1.2.3"

    anyio.run(run)
    assert fetch.await_count == 2
