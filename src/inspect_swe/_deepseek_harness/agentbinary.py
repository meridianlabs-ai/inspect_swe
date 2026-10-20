import hashlib
import io
import json
import tarfile
import threading
import zipfile
from pathlib import Path
from typing import Any

from typing_extensions import Literal

from .._util.agentbinary import AgentBinarySource, AgentBinaryVersion
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform

# per-platform wheels holding the standalone `dsh` runtime
DSH_RUNTIME_PACKAGE = "deepseek-harness-runtime-bin"
PYPI_JSON_URL = f"https://pypi.org/pypi/{DSH_RUNTIME_PACKAGE}"

# installed as a whole: the executable finds its sidecars relative to itself
_WHEEL_RUNTIME_DIR = "deepseek_harness_runtime/runtime/"

DSH_ENTRYPOINT = "dsh"


def deepseek_harness_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("deepseek-harness-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        if version in ["stable", "latest"]:
            release = await _fetch_release(None)
            version = str(release["info"]["version"])
        else:
            release = await _fetch_release(version)

        wheel_tag = _wheel_platform_tag(platform)
        wheel = next(
            (
                url
                for url in release.get("urls", [])
                if url.get("packagetype") == "bdist_wheel"
                and str(url.get("filename", "")).endswith(f"-{wheel_tag}.whl")
            ),
            None,
        )
        if wheel is None:
            raise RuntimeError(
                f"No DeepSeek Harness runtime wheel for platform {platform} "
                f"in version {version}"
            )
        return AgentBinaryVersion(
            version, wheel["digests"]["sha256"], wheel["url"], package=True
        )

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        # Never written or read in practice: resolve_version always returns
        # package=True. Defined only because AgentBinarySource requires it.
        return cached_binary_dir / f"deepseek-harness-{version}-{platform}"

    def cached_package_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"deepseek-harness-package-{version}-{platform}.whl"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("deepseek-harness-*"))

    return AgentBinarySource(
        agent="deepseek harness",
        binary="dsh",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=None,
        package_entrypoint=DSH_ENTRYPOINT,
        cached_package_path=cached_package_path,
        package_to_tar_gz=_wheel_to_tar_gz,
    )


def _wheel_platform_tag(platform: SandboxPlatform) -> str:
    # glibc-only wheels; the build would install on musl and then fail at exec
    if platform.endswith("-musl"):
        raise RuntimeError(
            f"DeepSeek Harness publishes glibc-only linux builds; no musl "
            f"runtime is available for {platform}."
        )
    match platform:
        case "linux-x64":
            return "manylinux_2_28_x86_64"
        case "linux-arm64":
            return "manylinux_2_28_aarch64"
        case _:
            raise RuntimeError(f"Unsupported platform for DeepSeek Harness: {platform}")


async def _fetch_release(version: str | None) -> dict[str, Any]:
    url = (
        f"{PYPI_JSON_URL}/json"
        if version is None
        else f"{PYPI_JSON_URL}/{version}/json"
    )
    return dict(json.loads(await download_text_file(url)))


# conversion takes seconds and every sample installs, so memoize per process
_tar_gz_cache_lock = threading.Lock()
_tar_gz_cache: dict[str, bytes] = {}


def _wheel_to_tar_gz(wheel_bytes: bytes) -> bytes:
    key = hashlib.sha256(wheel_bytes).hexdigest()
    with _tar_gz_cache_lock:
        cached = _tar_gz_cache.get(key)
    if cached is not None:
        return cached

    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel,
        tarfile.open(fileobj=output, mode="w:gz", compresslevel=6) as tar,
    ):
        executable: str | None = None
        for info in wheel.infolist():
            if info.is_dir() or not info.filename.startswith(_WHEEL_RUNTIME_DIR):
                continue
            relative = info.filename[len(_WHEEL_RUNTIME_DIR) :]
            name = f"runtime/{relative}"
            data = wheel.read(info)
            # wheels carry unix mode bits in the high word of external_attr
            mode = (info.external_attr >> 16) & 0o777
            is_executable = "/" not in relative and relative.startswith(
                "deepseek-harness-sdk-runtime-"
            )
            if is_executable:
                mode = 0o755
                if not relative.endswith(("-rg", "-spawn-helper")):
                    executable = relative
            _add_file(tar, name, data, mode or 0o644)
        if executable is None:
            raise RuntimeError("DeepSeek Harness wheel has no runtime executable")

        # stable entrypoint (the executable's name embeds the platform)
        launcher = f'#!/bin/sh\nexec "$(dirname "$0")/runtime/{executable}" "$@"\n'
        _add_file(tar, DSH_ENTRYPOINT, launcher.encode(), 0o755)

    result = output.getvalue()
    with _tar_gz_cache_lock:
        _tar_gz_cache[key] = result
    return result


def _add_file(tar: tarfile.TarFile, name: str, data: bytes, mode: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    tar.addfile(info, io.BytesIO(data))
