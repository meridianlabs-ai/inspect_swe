import anyio
import httpx

# These helpers fetch version pointers, manifests, and multi-megabyte agent
# binaries from external CDNs (github.com, code.kimi.com, storage.googleapis.com).
# httpx's default 5-second read timeout is too tight for some of these endpoints
# from CI runners (code.kimi.com's latest.json alone has breached it), and with
# no retry a single transient blip fails the whole eval — so use a generous
# timeout and retry transient failures with backoff.
_TIMEOUT = httpx.Timeout(60.0, connect=30.0)

# delay before each retry (attempts = len + 1)
_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)


async def download_file(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        attempts = len(_RETRY_DELAYS) + 1
        for attempt in range(attempts):
            try:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.content
            except (httpx.TransportError, httpx.HTTPStatusError) as ex:
                # transport errors (timeouts, resets, DNS) and 5xx responses
                # are transient; 4xx responses are permanent
                permanent = (
                    isinstance(ex, httpx.HTTPStatusError)
                    and ex.response.status_code < 500
                )
                if permanent or attempt == attempts - 1:
                    raise
                await anyio.sleep(_RETRY_DELAYS[attempt])
    raise RuntimeError("unreachable")  # satisfies type checker


async def download_text_file(url: str) -> str:
    return (await download_file(url)).decode("utf-8")
