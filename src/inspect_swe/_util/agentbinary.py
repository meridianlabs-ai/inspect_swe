import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal, NamedTuple, TypeAlias

import anyio
from inspect_ai.log import transcript
from inspect_ai.util import SandboxEnvironment, concurrency
from inspect_ai.util import sandbox as sandbox_env
from pydantic import BaseModel

from inspect_swe._util.trace import trace

from .checksum import ChecksumMismatchError, sha256_checksum, verify_checksum
from .download import download_file
from .sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
    sandbox_exec,
)

logger = logging.getLogger(__name__)

# source identifier for the InfoEvents this module writes to the transcript
AGENT_BINARY_EVENT_SOURCE = "inspect_swe"

AgentBinaryOrigin: TypeAlias = Literal[
    "download", "cache", "sandbox", "cache_unverified"
]
"""Where the agent binary a sample ran came from."""


class AgentBinaryInstall(BaseModel):
    """Provenance of the agent binary a sample actually ran.

    Recorded as the `data` of an `InfoEvent` (`source="inspect_swe"`) each time
    an agent resolves its binary, so a sample that invokes two agents (or the
    same agent twice) carries one record per invocation. The event exists
    because the installed version can differ from the one the task appears to
    request, and can differ between samples of a single eval:
    `"stable"`/`"latest"` resolve through a per-process cache with no expiry,
    so a resumed run, an `eval_set` retry, or a multi-process run can install a
    newer release for the remaining samples, and `"auto"` takes whatever the
    sandbox image happens to contain.
    """

    agent: str
    """Agent the binary belongs to (e.g. `"claude_code"`)."""

    requested: str
    """`version=` as the caller passed it: `"auto"`, `"sandbox"`, `"stable"`, `"latest"`, or an explicit version."""

    version: str | None
    """Concrete version installed. `None` when `origin="sandbox"`: the binary was already in the image, and its version is not knowable without running it."""

    platform: SandboxPlatform | None
    """Platform the artifact was resolved for. `None` when `origin="sandbox"`, where no platform-specific artifact is chosen and detection would add three sandbox execs to the default path."""

    checksum: str | None
    """SHA-256 of the host-side artifact this install resolved: the binary itself, or the package archive extracted into the sandbox. `None` when `origin="sandbox"`, where the host never reads the binary.

    Two caveats. It is not always the digest published upstream: an agent that transforms its download (`codex_cli` extracting a single binary from a tarball on pre-`rust-v0.133.0` releases) installs bytes the manifest never described, while a package archive is cached and extracted verbatim and so does match when a manifest was resolved at all.

    And it describes the artifact on the host, not a re-read of the sandbox. When a package for this version is already extracted in the sandbox the write and extract are skipped, so the digest is of the cached archive that install came from rather than of bytes this call placed there.
    """

    origin: AgentBinaryOrigin
    """How the binary got into the sandbox.

    - `"download"`: fetched over the network and verified against the resolved digest.
    - `"cache"`: served from the local cache. This is not a claim of verification. A pinned version read on the fast path is not checked against a digest because none is resolved for it, and neither is a cached artifact for an agent with a `post_download` transform, whose installed bytes cannot be compared against the download digest.
    - `"sandbox"`: already present in the image; nothing was installed.
    - `"cache_unverified"`: the offline fallback. Resolution failed, so no digest could be obtained to check the cached bytes against, and they were installed anyway.
    """


class AgentBinaryVersion(NamedTuple):
    version: str
    expected_checksum: str
    download_url: str
    # the artifact is a package archive (extracted in the sandbox) rather than
    # a single binary. requires the source to define package_entrypoint and
    # cached_package_path.
    package: bool = False


class AgentBinaryDownload(NamedTuple):
    """Result of resolving and fetching an agent binary."""

    data: bytes
    resolved: AgentBinaryVersion
    # the bytes came off the local cache rather than the network. the caller
    # cannot infer this: resolution has to run before the cache path is known,
    # so a successful resolve says nothing about whether anything was fetched.
    from_cache: bool


@dataclass
class AgentBinarySource:
    agent: str
    binary: str
    resolve_version: Callable[
        [Literal["stable", "latest"] | str, SandboxPlatform],
        Awaitable[AgentBinaryVersion],
    ]
    cached_binary_path: Callable[[str, SandboxPlatform], Path]
    list_cached_binaries: Callable[[], list[Path]]
    post_download: Callable[[bytes], bytes] | None
    post_install: str | None
    # package-archive support: relative path of the agent binary within the
    # extracted package (e.g. "bin/codex") and the cache location for package
    # archives. required when resolve_version can return package=True.
    package_entrypoint: str | None = None
    cached_package_path: Callable[[str, SandboxPlatform], Path] | None = None


def _agent_event_name(source: AgentBinarySource) -> str:
    """Registry name of the agent (`claude_code`) from its display name ("claude code")."""
    return source.agent.replace(" ", "_")


def record_agent_binary_install(
    binary_source: AgentBinarySource,
    *,
    requested: str,
    version: str | None,
    platform: SandboxPlatform | None,
    checksum: str | None,
    origin: AgentBinaryOrigin,
) -> None:
    """Record which agent binary this sample ran as a transcript `InfoEvent`."""
    install = AgentBinaryInstall(
        agent=_agent_event_name(binary_source),
        requested=requested,
        version=version,
        platform=platform,
        checksum=checksum,
        origin=origin,
    )
    transcript().info(install.model_dump(mode="json"), source=AGENT_BINARY_EVENT_SOURCE)


# In-process cache for version resolution results. When many samples run
# concurrently they all call resolve_version with the same arguments.
# Without caching, each call hits upstream APIs (e.g. GitHub, GCS),
# risking rate-limit exhaustion. We use a threading.Lock (not anyio.Lock)
# because it only guards synchronous dict reads/writes — never held across
# an await — and avoids issues with module-level anyio.Lock binding to a
# stale event loop across multiple anyio.run() calls. No expiry: entries
# live for the process lifetime.
_resolve_version_lock = threading.Lock()
_resolved_versions: dict[tuple[str, str, SandboxPlatform], AgentBinaryVersion] = {}
# per key: (number of failed resolutions so far, exception from the latest
# one). Lets callers already queued behind a failing resolution share its
# exception instead of each retrying in turn — mirrors
# versioncache.cached_version_resolution's idiom for npm-installed agents.
_failed_resolutions: dict[tuple[str, str, SandboxPlatform], tuple[int, Exception]] = {}


async def resolve_agent_binary_version(
    source: AgentBinarySource,
    version: Literal["stable", "latest"] | str,
    platform: SandboxPlatform,
) -> AgentBinaryVersion:
    """Resolve a binary's version, reusing the result for the process lifetime.

    Concurrent/queued callers for the same (binary, version, platform) key
    share a single resolution — including its failure, so a burst of samples
    hitting a rate-limited API produces one error, not one retry per queued
    sample. Failures are not cached: a call arriving after a failed
    resolution has completed retries.
    """
    cache_key = (source.binary, version, platform)
    with _resolve_version_lock:
        cached = _resolved_versions.get(cache_key)
        if cached is not None:
            return cached
        failure = _failed_resolutions.get(cache_key)
        failures_at_arrival = failure[0] if failure is not None else 0

    # serialize per key so a burst of samples starting together makes one
    # request rather than one each (in addition to, and independent of, the
    # per-binary install concurrency(1) lock in ensure_agent_binary_installed)
    async with concurrency(
        f"{source.binary}-version-resolution-{version}-{platform}", 1, visible=False
    ):
        # another sample may have resolved (or failed) while we were waiting
        with _resolve_version_lock:
            cached = _resolved_versions.get(cache_key)
            if cached is not None:
                return cached
            failure = _failed_resolutions.get(cache_key)
        if failure is not None and failure[0] > failures_at_arrival:
            raise failure[1]

        try:
            resolved = await source.resolve_version(version, platform)
        except Exception as ex:
            with _resolve_version_lock:
                _failed_resolutions[cache_key] = (failures_at_arrival + 1, ex)
            raise
        with _resolve_version_lock:
            _resolved_versions[cache_key] = resolved
        return resolved


async def ensure_agent_binary_installed(
    source: AgentBinarySource,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    user: str | None = None,
    sandbox: SandboxEnvironment | None = None,
) -> str:
    # resolve sandbox
    sandbox = sandbox or sandbox_env()

    # `version` is reassigned below ("auto" becomes "stable" when the sandbox
    # has no binary), and the provenance event reports what the caller asked
    # for rather than what it was rewritten to
    requested = version

    # look in the sandbox first if we need to
    if version == "auto" or version == "sandbox":
        result = await sandbox.exec(bash_command(f"which {source.binary}"), user=user)
        if result.success:
            binary_path = result.stdout.strip()
            trace(f"Using {source.agent} installed in sandbox: {binary_path}")
            # no version/checksum: the binary predates us and the host never
            # reads it. platform is left undetected because that costs three
            # sandbox execs and this is the default path.
            record_agent_binary_install(
                source,
                requested=requested,
                version=None,
                platform=None,
                checksum=None,
                origin="sandbox",
            )
            return binary_path

        # if version == "sandbox" and we don't find it that's an error
        if version == "sandbox":
            raise RuntimeError(f"unable to locate {source.agent} in sandbox")

        # otherwise set to "stable"
        version = "stable"

    # detect the sandbox target platform
    platform = await detect_sandbox_platform(sandbox)

    # use concurrency so multiple samples don't attempt the same download all at once
    async with concurrency(f"{source.binary}-install", 1, visible=False):
        # if a specific version is requested, first try to read it directly from
        # the cache. package-capable sources only honor a package cache hit
        # here: a single-binary entry may predate package support (cached by an
        # older inspect_swe), so it must not short-circuit resolution or the
        # package install would be silently defeated for exactly the users this
        # exists for. resolution below still prefers caches; the single-binary
        # cache remains the offline fallback.
        binary_bytes: bytes | None = None
        package = False
        origin: AgentBinaryOrigin = "download"
        if version not in ["stable", "latest"]:
            if source.cached_package_path is not None:
                binary_bytes = read_cached_file(
                    source.cached_package_path(version, platform), None
                )
                package = binary_bytes is not None
            else:
                binary_bytes = read_cached_binary(source, version, platform, None)
            if binary_bytes is not None:
                trace(f"Used {source.agent} binary from cache: {version} ({platform})")

        # download the binary
        if binary_bytes is None:
            try:
                downloaded = await download_agent_binary_async(
                    source, version, platform, trace
                )
                binary_bytes = downloaded.data
                resolved_version = downloaded.resolved.version
                package = downloaded.resolved.package
                # resolution succeeding does not mean anything was fetched:
                # "stable" resolves to a concrete version that is usually
                # already cached from an earlier eval on this host
                origin = "cache" if downloaded.from_cache else "download"
            except ChecksumMismatchError:
                # integrity failure: the freshly-downloaded (or shared,
                # already-failed) bytes did not match the expected digest.
                # never mask this by silently installing unverified bytes
                # from the legacy single-binary cache — a corrupted cache
                # or a poisoned download must fail loudly, not fall through.
                raise
            except Exception:
                # offline fallback: a pinned version with a cached single
                # binary still installs (without companion executables) when
                # resolution/download failed for a network or availability
                # reason (checksum failures are excluded above and always
                # propagate).
                if version not in ["stable", "latest"]:
                    binary_bytes = read_cached_binary(source, version, platform, None)
                if binary_bytes is None:
                    raise
                cache_path = source.cached_binary_path(version, platform)
                logger.warning(
                    f"{source.agent} {version} could not be resolved or "
                    f"downloaded over the network; installing the cached "
                    f"binary at {cache_path} ({platform}) WITHOUT checksum "
                    "verification, because no digest can be obtained "
                    "offline to verify it."
                )
                trace(
                    f"Unable to resolve {source.agent} {version}; using cached "
                    f"single binary ({platform})"
                )
                resolved_version = version
                origin = "cache_unverified"
        else:
            # If we got it from cache, version is already the resolved version
            resolved_version = version
            origin = "cache"

        # write it into the container and return it
        install_path = (
            f"{SANDBOX_INSTALL_DIR}/{source.binary}-{resolved_version}-{platform}"
        )
        if package:
            if source.package_entrypoint is None:
                raise RuntimeError(
                    f"{source.agent} resolved a package archive but the source "
                    "does not define a package_entrypoint"
                )
            binary_path = f"{install_path}/{source.package_entrypoint}"
            # skip write + extract if this version's package is already
            # installed (probe as root to match the extraction below, so a
            # non-traversable install dir can't force re-extraction each call)
            probe = await sandbox.exec(
                bash_command(f"test -x {binary_path}"), user="root"
            )
            if not probe.success:
                # tar preserves mode bits, so companion executables in the
                # archive (e.g. bin/*, codex-resources/*) stay executable; only
                # the entrypoint is chmod'd as a belt-and-suspenders measure
                archive_path = f"{install_path}.tar.gz"
                await sandbox.write_file(archive_path, binary_bytes)
                await sandbox_exec(
                    sandbox,
                    f"mkdir -p {install_path} && "
                    f"tar -xzf {archive_path} -C {install_path} && "
                    f"rm -f {archive_path} && "
                    f"chmod +x {binary_path}",
                    user="root",
                )
        else:
            binary_path = install_path
            await sandbox.write_file(binary_path, binary_bytes)
            await sandbox_exec(sandbox, f"chmod +x {binary_path}", user="root")
        if source.post_install:
            await sandbox_exec(
                sandbox, f"{binary_path} {source.post_install}", user=user
            )
        # recorded only once the install has actually landed, and against the
        # bytes that landed rather than the upstream manifest digest: an agent
        # with a post_download transform installs something other than what it
        # downloaded
        record_agent_binary_install(
            source,
            requested=requested,
            version=resolved_version,
            platform=platform,
            checksum=await anyio.to_thread.run_sync(sha256_checksum, binary_bytes),
            origin=origin,
        )
        return binary_path


async def download_agent_binary_async(
    source: AgentBinarySource,
    version: Literal["stable", "latest"] | str,
    platform: SandboxPlatform,
    logger: Callable[[str], None] | None = None,
) -> AgentBinaryDownload:
    # resolve logger
    logger = logger or print

    # determine version and checksum (cached so concurrent samples don't
    # repeat upstream API calls that may be rate-limited, sharing a failure
    # too so a queue of samples behind a rate-limited call doesn't each
    # retry it in turn)
    resolved = await resolve_agent_binary_version(source, version, platform)
    version = resolved.version
    expected_checksum = resolved.expected_checksum
    download_url = resolved.download_url

    # resolve the cache location (package archives are cached verbatim, so
    # their checksum stays verifiable; single binaries transformed by
    # post_download can't be verified against the download checksum)
    if resolved.package:
        if source.cached_package_path is None:
            raise RuntimeError(
                f"{source.agent} resolved a package archive but the source "
                "does not define a cached_package_path"
            )
        cache_path = source.cached_package_path(version, platform)
        cache_checksum: str | None = expected_checksum
    else:
        cache_path = source.cached_binary_path(version, platform)
        cache_checksum = None if source.post_download else expected_checksum

    binary_data = read_cached_file(cache_path, cache_checksum)
    from_cache = binary_data is not None
    if binary_data is None:
        # not in cache, download and verify checksum
        binary_data = await download_file(download_url)
        if not verify_checksum(binary_data, expected_checksum):
            raise ChecksumMismatchError("Checksum verification failed")

        # apply post-download processing if provided (e.g., extract from tar.gz)
        if not resolved.package and source.post_download is not None:
            binary_data = source.post_download(binary_data)

        # save to cache
        write_cached_file(source, binary_data, cache_path)

        # a package supersedes any single-binary cache entry for the same
        # version (cached by an older inspect_swe): remove it so it can't be
        # served later and doesn't hold an eviction slot
        if resolved.package:
            stale = source.cached_binary_path(version, platform)
            if stale.exists():
                stale.unlink()

        # trace
        logger(f"Downloaded {source.agent} binary: {version} ({platform})")
    else:
        logger(f"Used {source.agent} binary from cache: {version} ({platform})")

    # return data, resolved version, and whether anything was actually fetched
    return AgentBinaryDownload(binary_data, resolved, from_cache)


def read_cached_binary(
    source: AgentBinarySource,
    version: str,
    platform: SandboxPlatform,
    expected_checksum: str | None,
) -> bytes | None:
    return read_cached_file(
        source.cached_binary_path(version, platform), expected_checksum
    )


def read_cached_file(cache_path: Path, expected_checksum: str | None) -> bytes | None:
    # no cached file
    if not cache_path.exists():
        return None

    # read binary
    with open(cache_path, "rb") as f:
        binary_data = f.read()

    if expected_checksum is None or verify_checksum(binary_data, expected_checksum):
        cache_path.touch()
        return binary_data
    else:
        cache_path.unlink()
        return None


def write_cached_file(
    source: AgentBinarySource,
    binary_data: bytes,
    cache_path: Path,
) -> None:
    with open(cache_path, "wb") as f:
        f.write(binary_data)

    _cleanup_binary_cache(source, keep_count=3)


def _cleanup_binary_cache(source: AgentBinarySource, keep_count: int = 5) -> None:
    # get all cached binaries
    cache_files = source.list_cached_binaries()
    if len(cache_files) <= keep_count:
        return

    # remove oldest
    cache_files.sort(key=lambda f: f.stat().st_atime)
    files_to_remove = cache_files[:-keep_count]
    for file_path in files_to_remove:
        file_path.unlink()
