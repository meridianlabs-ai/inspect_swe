import os
from urllib.parse import urlparse

import httpx


def github_api_headers(url: str) -> dict[str, str]:
    """Auth headers for GitHub API requests.

    Unauthenticated requests to api.github.com are limited to 60/hour per IP,
    which shared CI runner IPs routinely exhaust (403 "rate limit exceeded").
    Sending a token raises the limit to 5,000/hour. The token is scoped to
    api.github.com only so it is never sent to other hosts (e.g. release-asset
    downloads that redirect to objects.githubusercontent.com).
    """
    if urlparse(url).hostname == "api.github.com":
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            return {"Authorization": f"Bearer {token}"}
    return {}


async def download_file(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, follow_redirects=True, headers=github_api_headers(url)
        )
        response.raise_for_status()
        return response.content


async def download_text_file(url: str) -> str:
    return (await download_file(url)).decode("utf-8")
