import asyncio
import functools
from typing import Any, Callable, cast

import anyio
import pytest
from acp import RequestError
from acp.client.connection import ClientSideConnection
from inspect_swe.acp.client import (
    ACPError,
    DefaultClient,
    _wrap_connection_methods,
    format_acp_failure,
)
from inspect_swe.acp.transport import ErrorInfo


class PlainAsyncConnection:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.close_calls = 0

    async def initialize(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error

    async def new_session(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error

    async def prompt(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error

    async def cancel(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error

    async def close(self) -> None:
        self.close_calls += 1


F = Callable[..., Any]


def _compat_wrapped_async(method: F) -> F:
    """Mimic ACP's compatibility wrapper around an async method."""

    @functools.wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        return method(self, *args, **kwargs)

    return wrapped


class CompatWrappedConnection:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    @_compat_wrapped_async
    async def initialize(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error

    @_compat_wrapped_async
    async def prompt(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error

    async def close(self) -> None:
        return None


def test_error_info_keeps_live_stderr_tail() -> None:
    info = ErrorInfo(max_stderr_chars=10)
    info.append_stderr("12345")
    assert info.stderr == "12345"

    info.append_stderr("67890")
    assert info.stderr == "1234567890"

    info.append_stderr("XYZ")
    assert info.stderr == "4567890XYZ"


def test_connection_method_wrapping_includes_acp_error_and_stderr() -> None:
    async def run_test() -> None:
        error_info = ErrorInfo()
        conn = PlainAsyncConnection(RequestError(-32000, "Internal error"))

        async def feeder_task() -> None:
            await asyncio.sleep(0.01)
            error_info.append_stderr("The requested model does not exist.\n")

        feeder = asyncio.create_task(feeder_task())
        wrapped = _wrap_connection_methods(
            cast(ClientSideConnection, conn), feeder, error_info
        )

        with pytest.raises(ACPError) as exc_info:
            await wrapped.prompt(prompt=[], session_id="test")

        message = str(exc_info.value)
        assert "ACP prompt failed" in message
        assert "ACP error:\nInternal error" in message
        assert "Adapter stderr:\nThe requested model does not exist." in message

    anyio.run(run_test)


def test_connection_wrapping_catches_compat_wrapped_methods() -> None:
    async def run_test() -> None:
        error_info = ErrorInfo()
        conn = CompatWrappedConnection(RequestError(-32000, "fetch failed"))
        feeder = asyncio.create_task(asyncio.sleep(0))
        wrapped = _wrap_connection_methods(
            cast(ClientSideConnection, conn), feeder, error_info
        )

        with pytest.raises(ACPError, match="ACP initialize failed") as exc_info:
            await wrapped.initialize(protocol_version=1)

        assert "ACP error:\nfetch failed" in str(exc_info.value)
        await feeder

    anyio.run(run_test)


def test_connection_wrapping_does_not_wrap_close() -> None:
    async def run_test() -> None:
        error_info = ErrorInfo()
        conn = PlainAsyncConnection(RequestError(-32000, "Internal error"))
        feeder = asyncio.create_task(asyncio.sleep(0))
        wrapped = _wrap_connection_methods(
            cast(ClientSideConnection, conn), feeder, error_info
        )
        await wrapped.close()
        assert conn.close_calls == 1
        await feeder

    anyio.run(run_test)


def test_format_acp_failure_with_only_acp_error_text() -> None:
    message = format_acp_failure(
        phase="initialize",
        error_info=ErrorInfo(),
        acp_error=RequestError(-32000, "fetch failed"),
    )

    assert "ACP initialize failed" in message
    assert "ACP error:\nfetch failed" in message
    assert "Adapter stderr:" not in message


def test_format_acp_failure_with_only_stderr() -> None:
    error_info = ErrorInfo()
    error_info.exit_code = 1
    error_info.append_stderr("provider-side stderr\n")

    message = format_acp_failure(
        phase="active_session",
        error_info=error_info,
        acp_error=None,
    )

    assert "ACP active_session failed" in message
    assert "ACP error:\nACP adapter process exited unexpectedly." in message
    assert "Adapter stderr:\nprovider-side stderr" in message


def test_default_client_unsupported_capability_error() -> None:
    async def run_test() -> None:
        client = DefaultClient()

        with pytest.raises(RequestError) as exc_info:
            await client.create_terminal(command="pwd", session_id="session-1")

        err = exc_info.value
        assert err.code == -32601
        assert err.data == {"capability": "terminal/create"}
        assert (
            "ACP adapter requested unsupported client capability terminal/create"
            in str(err)
        )

    anyio.run(run_test)
