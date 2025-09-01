import asyncio
import inspect
from typing import Any, Coroutine, Literal, TypeVar, cast

import nest_asyncio  # type: ignore
import sniffio

from .platform import running_in_notebook

T = TypeVar("T")


def is_callable_coroutine(func_or_cls: Any) -> bool:
    if inspect.iscoroutinefunction(func_or_cls):
        return True
    elif callable(func_or_cls):
        return inspect.iscoroutinefunction(func_or_cls.__call__)
    return False


def run_coroutine(coroutine: Coroutine[None, None, T]) -> T:
    if current_async_backend() == "trio":
        raise RuntimeError("run_coroutine cannot be used with trio")

    if running_in_notebook():
        init_nest_asyncio()
        return asyncio.run(coroutine)
    else:
        try:
            # this will throw if there is no running loop
            asyncio.get_running_loop()

            # initialiase nest_asyncio then we are clear to run
            init_nest_asyncio()
            return asyncio.run(coroutine)

        except RuntimeError:
            # No running event loop so we are clear to run
            return asyncio.run(coroutine)


_initialised_nest_asyncio: bool = False


def init_nest_asyncio() -> None:
    global _initialised_nest_asyncio
    if not _initialised_nest_asyncio:
        nest_asyncio.apply()
        _initialised_nest_asyncio = True


def current_async_backend() -> Literal["asyncio", "trio"] | None:
    try:
        return _validate_backend(sniffio.current_async_library().lower())
    except sniffio.AsyncLibraryNotFoundError:
        return None


def _validate_backend(backend: str) -> Literal["asyncio", "trio"]:
    if backend in ["asyncio", "trio"]:
        return cast(Literal["asyncio", "trio"], backend)
    else:
        raise RuntimeError(f"Unknown async backend: {backend}")
